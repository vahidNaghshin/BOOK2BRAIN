"""The Knowledge screen: browse, open a unit with its evidence, review."""

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from database.models import KnowledgeUnit
from tests.extraction_db_helpers import item, make_service, reply, store_library
from tests.ui_helpers import patch_ui

QUOTE = "Entropy is a measure of the disorder of a system."


def knowledge_app() -> None:
    from app.ui.knowledge import render_knowledge_page

    render_knowledge_page()


@pytest.fixture
def library(session: Session, settings, engine: Engine, storage, monkeypatch):
    library = store_library(session)
    library.document.processing_status = "completed"
    session.commit()
    patch_ui(monkeypatch, engine, settings, storage)
    return library


def extract(session: Session, library) -> None:
    def answer(call):
        quote = QUOTE if call.target_id == str(library.passages["c1a"].id) else None
        sources = [{"passage_id": call.target_id, "role": "primary", **({"quote": quote} if quote else {})}]
        flagged = call.target_id == str(library.passages["c1b"].id)
        return reply(item(call, sources=sources, is_uncertain=flagged, uncertainty_notes="Implied." if flagged else None))

    make_service(session, answer)[0].extract_chapter(library.nodes["ch1"].id, ["concept", "principle"])


def open_page() -> AppTest:
    at = AppTest.from_function(knowledge_app, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_a_document_without_knowledge_points_to_extraction(session, library):
    at = open_page()
    assert any("Nothing has been extracted" in i.value for i in at.info) and at.button(key="knowledge_go_extraction")


def test_the_list_shows_counts_and_can_be_filtered(session, library):
    extract(session, library)
    at = open_page()
    assert metrics(at) == {"Knowledge units": "10", "Need attention": "2", "Approved": "0", "Rejected": "0", "Unreviewed": "10"}
    assert len(at.table[0].value) == 10
    assert at.selectbox(key="knowledge_type").options == ["All", "Concept (5)", "Principle (5)"]

    at.selectbox(key="knowledge_type").set_value("principle").run()
    assert {row for row in at.table[0].value["Category"]} == {"principle"}
    at.selectbox(key="knowledge_type").set_value("All").run()
    at.checkbox(key="knowledge_attention").check().run()
    assert len(at.table[0].value) == 2  # the flagged concept and the flagged principle
    at.checkbox(key="knowledge_attention").uncheck().run()
    at.text_input(key="knowledge_search").set_value("second law").run()
    assert set(at.table[0].value["Category"]) == {"principle"}
    at.text_input(key="knowledge_search").set_value("zzz").run()
    assert any("No knowledge matches" in i.value for i in at.info)


def test_opening_a_unit_shows_content_source_and_how_it_was_extracted(session, library):
    extract(session, library)
    at = open_page()
    at.selectbox(key="knowledge_type").set_value("concept").run()
    first = at.selectbox(key="knowledge_unit")
    assert first.value  # the first unit of the page is open
    assert any("Entropy" in s.value for s in at.subheader)
    text = " ".join(m.value for m in at.markdown)
    assert "A measure of the disorder of a system." in text and "Extracted content" in text and "Primary passage" in text
    assert any("The quote appears in the passage" in c.value for c in at.caption)
    assert any(e.label == "How it was extracted" for e in at.expander)
    assert any("local-test" in m.value and "concept_extraction@" in m.value for m in at.markdown)


def test_a_flagged_unit_shows_the_models_doubt(session, library):
    extract(session, library)
    at = open_page()
    at.selectbox(key="knowledge_type").set_value("concept").run()
    at.selectbox(key="knowledge_unit").select_index(1).run()  # the second concept is the flagged one (page 3)
    assert any("flagged this as uncertain" in w.value and "Implied." in w.value for w in at.warning)


def test_approving_and_rejecting_with_a_note(session, library):
    extract(session, library)
    at = open_page()
    unit_id = at.selectbox(key="knowledge_unit").value
    at.text_area(key=f"knowledge_note_{unit_id}").set_value("Looks right.").run()
    at.button(key=f"knowledge_approved_{unit_id}").click().run()
    assert not at.exception
    assert any("Marked as approved" in s.value for s in at.success)
    assert metrics(at)["Approved"] == "1" and metrics(at)["Unreviewed"] == "9"
    assert any("Review note: Looks right." in i.value for i in at.info)
    stored = session.scalars(select(KnowledgeUnit).where(KnowledgeUnit.review_status == "approved")).one()
    assert str(stored.id) == unit_id and stored.unit_metadata["review"]["note"] == "Looks right."

    at.button(key=f"knowledge_rejected_{unit_id}").click().run()
    assert metrics(at)["Rejected"] == "1" and metrics(at)["Approved"] == "0"
    at.selectbox(key="knowledge_review").set_value("rejected").run()
    assert len(at.table[0].value) == 1


def test_the_pages_of_a_long_list_can_be_turned(session, library):
    extract(session, library)
    from app.ui import knowledge

    knowledge.PAGE_SIZE = 4
    try:
        at = open_page()
        assert len(at.table[0].value) == 4
        at.number_input(key="knowledge_page").set_value(3).run()
        assert len(at.table[0].value) == 2
        at.selectbox(key="knowledge_type").set_value("principle").run()  # a new filter starts again on page 1
        assert at.number_input(key="knowledge_page").value == 1
    finally:
        knowledge.PAGE_SIZE = 15


# -- the overview (Day 5): categories, equations, relationships, flags and failures at a glance ----------


def extract_with_everything(session: Session, library) -> None:
    """Concepts, a damaged equation, a relationship, an uncertain unit and one passage that cannot be extracted."""
    from app.extraction.errors import ProviderTimeoutError
    from tests.extraction_db_helpers import add_equation, relationship, relationships_reply

    add_equation(session, library, "c1a", plain_text="dS = dQ / T", validation_status="invalid")
    broken = library.passages["c1b"].id

    def answer(call):
        if call.is_relationship:
            return relationships_reply(relationship(call, "Entropy", "Entropy"))  # no second unit: refused as a self link
        if call.target_id == str(broken):
            raise ProviderTimeoutError("timed out")
        return reply(item(call))

    service, _ = make_service(session, answer, max_retries=1)
    service.extract_chapter(library.nodes["ch1"].id, ["concept", "equation"], limit=2)


def test_the_overview_counts_what_a_reviewer_needs_without_opening_each_unit(session, library):
    from app.repositories.knowledge_repository import KnowledgeFilter
    from app.services.knowledge_service import KnowledgeService

    extract_with_everything(session, library)

    overview = KnowledgeService(session).overview(KnowledgeFilter(book_id=library.book.id, document_id=library.document.id))

    assert overview.by_type["concept"] == 1 and overview.by_type["equation"] >= 1
    assert overview.equation_status == {"needs_review": 1}, "Phase 1 marked the equation invalid"
    assert overview.open_flags.get("equation_needs_review") == 1
    assert overview.attempt_outcomes.get("provider_error", 0) >= 2, "the failed passage was tried and retried, and every call is counted"
    failure = next(f for f in overview.failed_extractions if f.category == "concept")
    assert failure.failure_code == "provider_timeout" and failure.page_start == library.passages["c1b"].page_start
    assert failure.attempts == 2 and failure.prompt and failure.model


def test_the_screen_shows_the_overview_with_failures_open(session, library):
    extract_with_everything(session, library)

    at = open_page()

    [expander] = [e for e in at.expander if e.label.startswith("Overview")]
    assert "2 failed extraction(s)" in expander.label and "open review flag(s)" in expander.label
    assert expander.proto.expanded, "failures are shown without having to ask"
    shown = [str(df.value.to_dict("records")) for df in at.dataframe]
    assert any("provider_timeout" in text for text in shown) and any("Equation needs review" in text for text in shown)
    assert any("Unresolved extraction error" in text for text in shown), "unresolved failures are visible as review flags too"
    assert len(at.table[0].value) >= 2, "the unit list is unchanged by the overview"
