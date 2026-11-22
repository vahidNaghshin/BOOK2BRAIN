"""The Day 4 additions to the screens: source checks, related units, equation capture, flags, attempts, passage status and resume."""

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from app.extraction.errors import ProviderTimeoutError
from app.knowledge.schemas import KnowledgeRelationshipCreate
from app.services.relationship_service import RelationshipService
from database.models import KnowledgeUnit, ReviewFlag
from tests.extraction_db_helpers import Library, add_equation, item, make_service, one_item, reply, store_library
from tests.extraction_helpers import hosted_config, make_config
from tests.knowledge_helpers import equation_data
from tests.test_extraction_ui import SyncRunner, extraction_app, metrics
from tests.test_knowledge_ui import knowledge_app
from tests.ui_helpers import patch_ui

QUOTE = "Entropy is a measure of the disorder of a system."


@pytest.fixture
def library(session: Session, settings, engine: Engine, storage, monkeypatch) -> Library:
    lib = store_library(session)
    lib.document.processing_status = "completed"
    session.commit()
    patch_ui(monkeypatch, engine, settings, storage)
    return lib


def open_knowledge() -> AppTest:
    at = AppTest.from_function(knowledge_app, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(at: AppTest) -> str:
    return " ".join([*(m.value for m in at.markdown), *(c.value for c in at.caption), *(w.value for w in at.warning), *(s.value for s in at.success)])


def cells(at: AppTest) -> str:
    return " ".join(str(t.value.to_dict()) for t in at.table)


def extract(session, library, categories=("concept", "principle"), answer=one_item, **options):
    service, _ = make_service(session, answer, **options)
    service.extract_section(library.nodes["ss1"].id, list(categories))
    return service


# ---------------------------------------------------------------- Knowledge: sources


def test_a_unit_shows_how_its_source_was_checked_and_can_be_checked_again(session, library):
    extract(session, library, ("concept",), lambda call: reply(item(call, sources=[{"passage_id": call.target_id, "role": "primary", "quote": QUOTE}])))
    at = open_knowledge()
    text = texts(at)
    assert "Source check: **verified**" in text and "extracted by a model" in text and "✅ verified" in text and "The quote appears in the passage" in text

    unit = session.scalars(select(KnowledgeUnit)).one()
    unit.page_start = unit.page_end = 40  # the pages no longer match the source
    session.commit()
    at.button(key=f"knowledge_verify_{unit.id}").click().run()
    assert not at.exception
    assert any("Sources verified: 2 problem(s) found, 2 flag(s) raised." in s.value for s in at.success)
    text = texts(at)
    assert "Source check: **problems found**" in text and "page inconsistent" in text and "outside the cited passages" in text


def test_a_unit_whose_passages_were_regenerated_still_shows_its_source_snapshot(session, library):
    extract(session, library, ("concept",), lambda call: reply(item(call, sources=[{"passage_id": call.target_id, "role": "primary", "quote": QUOTE}])))
    from database.models import Passage

    for passage in session.scalars(select(Passage)):
        session.delete(passage)
    session.commit()
    at = open_knowledge()
    assert any("no longer exist" in w.value for w in at.warning)
    assert any(e.label == "Source snapshot taken when the unit was stored" for e in at.expander)
    assert QUOTE in cells(at) and "Statistical view" in cells(at)  # the snapshot keeps the quote and where it was


# ---------------------------------------------------------------- Knowledge: relationships, flags, attempts


def test_related_units_flags_and_attempts_are_shown(session, library):
    extract(session, library, ("concept", "principle"), lambda call: reply(item(call, is_uncertain=True, uncertainty_notes="Implied.")))
    units = {u.knowledge_type: u for u in session.scalars(select(KnowledgeUnit))}
    RelationshipService(session).create(
        KnowledgeRelationshipCreate(
            book_id=library.book.id, source_unit_id=units["principle"].id, target_unit_id=units["concept"].id, relationship_type="depends_on",
            origin="extracted", explanation="The law needs entropy.", source_passage_ids=[library.passages["ss1a"].id], evidence_quote=QUOTE,
        )
    )
    at = open_knowledge()
    at.selectbox(key="knowledge_type").set_value("concept").run()
    table = cells(at)
    assert "Related knowledge" in texts(at) and "depends on" in table and "evidence from the book" in table and "The law needs entropy." in table
    assert "uncertain extraction" in texts(at) and "Implied." in texts(at)
    assert any(e.label == "How it was extracted" for e in at.expander)
    assert "needs review" in table  # the attempt's validation outcome
    assert "test-model:1b" in table and "concept_extraction@" in table


def test_a_suggested_relationship_is_labelled_as_needing_review(session, library):
    extract(session, library)
    units = {u.knowledge_type: u for u in session.scalars(select(KnowledgeUnit))}
    RelationshipService(session).create(
        KnowledgeRelationshipCreate(book_id=library.book.id, source_unit_id=units["principle"].id, target_unit_id=units["concept"].id, relationship_type="related_to", origin="ai_suggested")
    )
    at = open_knowledge()
    at.selectbox(key="knowledge_type").set_value("principle").run()
    assert "AI suggestion, needs review" in cells(at)


def test_a_flag_can_be_resolved_from_the_screen(session, library):
    extract(session, library, ("concept",), lambda call: reply(item(call, is_uncertain=True, uncertainty_notes="Implied.")))
    at = open_knowledge()
    flag = session.scalars(select(ReviewFlag)).one()
    assert "uncertain extraction" in texts(at)
    at.button(key=f"knowledge_flag_{flag.id}").click().run()
    assert not at.exception and "uncertain extraction" not in texts(at)
    session.expire_all()
    assert session.get(ReviewFlag, flag.id).status == "resolved"


# ---------------------------------------------------------------- Knowledge: equations


def test_an_equation_is_shown_as_captured_from_the_source(session, library):
    library.passages["ss1a"].text = "Before the equation. dS = dQ / T. After the equation."
    session.commit()
    eq = add_equation(session, library, "ss1a")
    add_equation(session, library, "ss1a", n=1, latex="", original_latex=None, plain_text=None)
    service, _ = make_service(session, lambda call: reply(item(call, content=equation_data())))
    service.extract_section(library.nodes["ss1"].id, ["equation"])
    at = open_knowledge()
    at.selectbox(key="knowledge_type").set_value("equation").run()
    at.selectbox(key="knowledge_unit").select_index(0).run()
    text = texts(at)
    assert "Equation as captured from the source" in text and "Capture status: **Complete**" in text and "never overwritten" in text
    assert eq.crop_reference in text and "Original text" in text and "Surrounding text" in [e.label for e in at.expander]
    labels = [o for o in at.selectbox(key="knowledge_unit").options]
    at.selectbox(key="knowledge_unit").select_index(len(labels) - 1).run()
    text = texts(at)
    assert "Capture status: **Image only**" in text and "copied from the source (no model)" in text and "Only an image of the equation is available." in text


# ---------------------------------------------------------------- Extraction: passages and resume


@pytest.fixture
def extraction(library, engine, monkeypatch):
    def setup(answer, **options):
        options.setdefault("configs", [make_config(), hosted_config()])
        runner = SyncRunner(engine, answer=answer, **options)
        monkeypatch.setattr("app.ui.extraction.get_job_runner", lambda: runner)
        monkeypatch.setattr("app.ui.extraction._service", lambda s: make_service(s, answer=answer, **options)[0])
        at = AppTest.from_function(extraction_app, default_timeout=30)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    return setup


def test_the_extraction_screen_can_ask_for_relationships_and_shows_the_extra_calls(extraction):
    at = extraction(one_item)
    at.multiselect(key="extraction_categories").set_value(["concept"]).run()
    at.number_input(key="extraction_limit").set_value(2).run()
    assert metrics(at)["Model calls"] == "2" and not any("find relationships" in c.value for c in at.caption)
    at.checkbox(key="extraction_relationships").check().run()
    assert any("Plus up to 2 calls to find relationships." in c.value for c in at.caption)


def test_passage_status_and_resume_are_shown_for_a_run_with_a_failed_passage(session, library, extraction):
    failing = str(library.passages["c1b"].id)
    state = {"fail": True}

    def answer(call):
        return ProviderTimeoutError("slow") if state["fail"] and call.target_id == failing else one_item(call)

    at = extraction(answer, max_retries=0)
    at.multiselect(key="extraction_categories").set_value(["concept"]).run()
    at.number_input(key="extraction_limit").set_value(3).run()
    at.button(key="extraction_start").click().run()
    assert not at.exception
    assert any("Failed passages" == m.label and m.value == "1" for m in at.metric)
    expander = next(e for e in at.expander if e.label.startswith("Passages:"))
    assert "1 failed" in expander.label and "2 completed" in expander.label
    assert any("Resuming continues with 1 failed, 0 not reached" in i.value for i in at.info)
    assert "provider_timeout: slow" in cells(at)
    resume = next(b for b in at.button if b.key and b.key.startswith("extraction_resume_"))

    state["fail"] = False
    resume.click().run()
    assert not at.exception
    assert metrics(at)["Failed passages"] == "0" and not any(b.key and b.key.startswith("extraction_resume_") for b in at.button)
    assert any(e.label.startswith("Passages:") and "3 completed" in e.label for e in at.expander)
    assert session.scalars(select(KnowledgeUnit)).all().__len__() == 3
