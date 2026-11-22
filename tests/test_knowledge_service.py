"""Browsing and reviewing extracted knowledge (test database; the knowledge comes from the mocked pipeline)."""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services.errors import InvalidReviewError, KnowledgeNotFoundError
from app.services.knowledge_service import KnowledgeFilter, KnowledgeService
from database.models import KnowledgeUnit, Passage
from tests.extraction_db_helpers import Library, item, make_service, reply, store_library

QUOTE = "Entropy is a measure of the disorder of a system."


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


def extracted(session: Session, library: Library):
    """Chapter 1 (5 passages): concepts everywhere; passage s1a also has a principle; quotes are right for c1a, wrong for c1b."""

    def answer(call):
        quote = {str(library.passages["c1a"].id): QUOTE, str(library.passages["c1b"].id): "Words that are not in the passage."}.get(call.target_id)
        sources = [{"passage_id": call.target_id, "role": "primary", **({"quote": quote} if quote else {})}]
        uncertain = call.target_id == str(library.passages["s1b"].id)
        return reply(item(call, sources=sources, is_uncertain=uncertain, uncertainty_notes="Implied." if uncertain else None, basis="inferred" if uncertain else "explicit"))

    service, _ = make_service(session, answer)
    service.extract_chapter(library.nodes["ch1"].id, ["concept", "principle"])
    return service


def flt(library: Library, **fields) -> KnowledgeFilter:
    return KnowledgeFilter(book_id=library.book.id, **fields)


def test_units_are_listed_in_reading_order_with_pagination(session, library):
    extracted(session, library)
    service = KnowledgeService(session)
    page = service.list_units(flt(library), page=1, page_size=4)
    assert (page.total, page.pages, page.page, len(page.items)) == (10, 3, 1, 4)
    pages = [i.page_start for i in service.list_units(flt(library), page=1, page_size=10).items]
    assert pages == sorted(pages)
    last = service.list_units(flt(library), page=3, page_size=4)
    assert len(last.items) == 2
    assert service.list_units(flt(library), page=99, page_size=4).page == 3 and service.list_units(flt(library), page=0).page == 1
    first = page.items[0]
    assert first.title == "Entropy" and first.review_status == "unreviewed" and first.summary.startswith("A measure")


def test_the_list_can_be_filtered(session, library):
    service = extracted(session, library)
    knowledge = KnowledgeService(session)
    assert knowledge.list_units(flt(library, knowledge_type="principle")).total == 5
    assert knowledge.list_units(flt(library, knowledge_type="concept", uncertain_only=True)).total == 2  # the model's doubt, and the quote that is not in the passage
    assert knowledge.list_units(flt(library, validation_status="uncertain")).total == 4  # two concepts and two principles
    assert knowledge.list_units(flt(library, review_status="approved")).total == 0
    assert knowledge.list_units(flt(library, document_id=library.document.id)).total == 10
    assert knowledge.list_units(flt(library, document_id=uuid.uuid4())).total == 0
    run_id = service.list_runs(library.book.id)[0].run.id
    assert knowledge.list_units(flt(library, run_id=run_id)).total == 10 and knowledge.list_units(flt(library, run_id=uuid.uuid4())).total == 0
    assert knowledge.list_units(flt(library, search="measure of the disorder")).total == 5  # found in the concepts' text
    assert knowledge.list_units(flt(library, search="second law")).total == 5  # principles, found by title
    assert knowledge.list_units(flt(library, search="100%_")).total == 0  # wildcard characters are literal
    assert knowledge.list_units(flt(library, search="   ")).total == 10


def test_the_summary_counts_by_category_review_and_validation(session, library):
    extracted(session, library)
    service = KnowledgeService(session)
    summary = service.summary(flt(library))
    assert summary.total == 10 and summary.by_type == {"concept": 5, "principle": 5}
    assert summary.by_review == {"unreviewed": 10} and summary.needing_attention == summary.total - summary.by_validation.get("valid", 0)
    service.review(service.list_units(flt(library)).items[0].id, "approved")
    assert service.summary(flt(library)).by_review == {"unreviewed": 9, "approved": 1}


def test_a_unit_is_shown_with_its_content_sources_and_where_it_comes_from(session, library):
    extracted(session, library)
    service = KnowledgeService(session)
    target = library.passages["c1a"]
    unit = next(i for i in service.list_units(flt(library, knowledge_type="concept")).items if i.page_start == target.page_start)
    detail = service.detail(unit.id)

    assert detail.unit.content.name == "Entropy" and detail.unit.knowledge_type.value == "concept"
    assert (detail.chapter_title, detail.section_title) == ("Entropy", None)
    [source] = detail.sources
    assert (source.passage_id, source.role, source.page_start, source.quote, source.quote_found) == (target.id, "primary", 2, QUOTE, True)
    assert source.text == target.text

    mismatch = next(i for i in service.list_units(flt(library, knowledge_type="concept")).items if i.page_start == library.passages["c1b"].page_start)
    assert service.detail(mismatch.id).sources[0].quote_found is False
    plain = next(i for i in service.list_units(flt(library, knowledge_type="concept")).items if i.page_start == library.passages["s1a"].page_start)
    assert service.detail(plain.id).sources[0].quote_found is None and service.detail(plain.id).section_title == "The second law"


def test_a_unit_shows_how_it_was_extracted(session, library):
    extracted(session, library)
    service = KnowledgeService(session)
    page = library.passages["s1b"].page_start
    uncertain = service.detail(next(i for i in service.list_units(flt(library, knowledge_type="concept"), page_size=10).items if i.page_start == page).id)
    execution = uncertain.execution
    assert (execution.model_configuration_id, execution.provider, execution.model_id) == ("local-test", "ollama", "test-model:1b")
    assert execution.prompt.startswith("concept_extraction@") and execution.attempts == 1 and execution.status == "needs_review"
    assert execution.review_reasons and uncertain.unit.uncertainty.is_uncertain and uncertain.unit.uncertainty.notes == "Implied."
    assert uncertain.unit.metadata["basis"] == "inferred"


def test_the_same_title_in_other_passages_is_pointed_out(session, library):
    extracted(session, library)
    service = KnowledgeService(session)
    concepts = service.list_units(flt(library, knowledge_type="concept"), page_size=10).items
    last = service.detail(concepts[-1].id)
    assert len(last.same_title_as) == 4
    assert {i.id for i in service.items(last.same_title_as)} == {c.id for c in concepts[:4]}
    assert service.detail(concepts[0].id).same_title_as == []


def test_a_unit_whose_passages_were_regenerated_is_still_readable(session, library):
    extracted(session, library)
    for passage in session.scalars(select(Passage)):
        session.delete(passage)
    session.commit()
    session.expire_all()
    service = KnowledgeService(session)
    unit = service.list_units(flt(library)).items[0]
    detail = service.detail(unit.id)
    assert detail.sources == [] and detail.unit.page_start is not None and detail.execution is not None


def test_missing_units_are_reported(session, library):
    service = KnowledgeService(session)
    with pytest.raises(KnowledgeNotFoundError):
        service.detail(uuid.uuid4())
    with pytest.raises(KnowledgeNotFoundError):
        service.review(uuid.uuid4(), "approved")


def test_review_decisions_are_saved_with_a_note_and_can_be_undone(session, library):
    extracted(session, library)
    service = KnowledgeService(session)
    unit_id = service.list_units(flt(library)).items[0].id

    approved = service.review(unit_id, "approved", "  Matches the text.  ")
    assert approved.unit.review_status.value == "approved" and approved.review_note == "Matches the text." and approved.reviewed_at
    record = session.get(KnowledgeUnit, unit_id)
    assert record.review_status == "approved" and record.unit_metadata["review"]["status"] == "approved"
    assert record.unit_metadata["basis"] == "explicit"  # other metadata is kept

    rejected = service.review(unit_id, "rejected")
    assert rejected.unit.review_status.value == "rejected" and rejected.review_note is None
    assert service.review(unit_id, "needs_changes", "Definition too broad.").review_note == "Definition too broad."
    reset = service.review(unit_id, "unreviewed")
    assert reset.unit.review_status.value == "unreviewed" and reset.review_note is None and "review" not in session.get(KnowledgeUnit, unit_id).unit_metadata
    with pytest.raises(InvalidReviewError, match="must be one of"):
        service.review(unit_id, "maybe")
