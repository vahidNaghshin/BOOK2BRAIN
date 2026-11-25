"""Day 5: the whole of Phase 2 on a book that Phase 1 really processed, and recovery from failures.

The book is generated as a DOCX (with real OMML equations), uploaded and processed by the real Phase 1
pipeline: parsing, equations, structure and passages. Only the language model is replaced by a scripted
one. Nothing in the pipeline, the persistence or the provenance checks is bypassed.
"""

import io
import json
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.extraction.errors import ProviderTimeoutError
from app.knowledge.types import EquationCaptureStatus
from app.repositories.knowledge_repository import KnowledgeFilter
from app.schemas import BookMetadata
from app.services.equation_service import EquationService
from app.services.knowledge_extraction_service import NothingToExtractError
from app.services.knowledge_service import KnowledgeService
from app.services.pipeline_service import ProcessingPipeline
from database.models import (
    Equation,
    EquationSource,
    ExtractionAttempt,
    ExtractionExecution,
    ExtractionPassageRecord,
    KnowledgeRelationship,
    KnowledgeUnit,
    Passage,
    StructuralNode,
)
from tests.extraction_db_helpers import Call, item, make_service, relationship, relationships_reply, reply
from tests.factories import make_docx_with_equations
from tests.knowledge_helpers import equation_data

PARAGRAPH = (
    "Retention fades with time unless a memory is revisited, and the pace of fading depends on how well the material "
    "was learned in the first place. "
) * 4


def book_items(chapters: int = 3) -> list[tuple]:
    items: list[tuple] = []
    for number in range(1, chapters + 1):
        items += [("heading", f"Chapter {number}: Part {number}"), ("paragraph", PARAGRAPH), ("paragraph", PARAGRAPH)]
        if number == 1:
            items.append(("display", "force", "(1.1)"))
        if number == 2:
            items.append(("inline", "The energy is ", "energy", " in joules for the body."))
        items.append(("paragraph", PARAGRAPH))
    return items


@pytest.fixture
def book(session: Session, document_service, storage, settings):
    """A processed book: one document, three chapters, one passage each, two Phase 1 equations."""
    document = document_service.upload(
        file=io.BytesIO(make_docx_with_equations(book_items())), filename="learning.docx", metadata=BookMetadata(title="Learning", author="A. Writer")
    ).document
    run = ProcessingPipeline(session, storage, settings).process_document(document.id)
    assert run.status == "completed"
    session.rollback()
    return document


def passages_of(session, document) -> list[Passage]:
    return list(session.scalars(select(Passage).where(Passage.document_id == document.id).order_by(Passage.document_sequence)))


def phase_1_state(session, document):
    session.rollback()
    return (
        [(p.id, p.text, p.page_start, p.document_sequence) for p in passages_of(session, document)],
        [(e.id, e.latex, e.plain_text, e.page_number) for e in session.scalars(select(Equation).where(Equation.document_id == document.id).order_by(Equation.sequence_number))],
        [(n.id, n.title) for n in session.scalars(select(StructuralNode).where(StructuralNode.document_id == document.id).order_by(StructuralNode.sequence))],
    )


def knowledge_answer(session: Session):
    """A concept per passage, the first Phase 1 equation as the model reads it, and a relationship between them."""

    def answer(call: Call):
        if call.is_relationship:
            titles = {u["title"] for u in call.units}
            if {"Force law", "Retention"} <= titles:
                return relationships_reply(relationship(call, "Force law", "Retention", "describes"))
            return relationships_reply()
        if call.category.value == "concept":
            return reply(item(call, content={"name": "Retention"}))
        if call.category.value == "equation" and session.get(Passage, uuid.UUID(call.target_id)).document_sequence == 0:
            return reply(item(call, content=equation_data(name="Force law", original_expression="F=ma", latex="F=ma")))
        return reply()

    return answer


# -- the complete workflow -------------------------------------------------------------------------


def test_the_complete_workflow_from_a_processed_book_to_inspectable_knowledge(session, book):
    before = phase_1_state(session, book)
    passages = passages_of(session, book)
    assert len(passages) == 3 and len(before[1]) == 2, "Phase 1 produced three passages and two equations"
    service, providers = make_service(session, knowledge_answer(session))

    # 1-3. an existing processed book, a scope and an active model
    [model] = [m for m in service.available_models() if m.id == "local-test"]
    assert model.ready and model.active
    chapter = session.scalar(select(StructuralNode).where(StructuralNode.document_id == book.id, StructuralNode.title == "Part 2"))
    scope = service.chapter_scope(chapter.id)
    plan = service.plan(scope, ["concept", "equation"], model_id="local-test", extract_relationships=True)
    assert (plan.total_passages, plan.model_calls, plan.relationship_calls, plan.skipped) == (1, 2, 1, 0)

    # 4-14. start a run and let it work through retrieval, context, prompts, validation and persistence
    events = []
    run_id = service.create_run(scope, ["concept", "equation"], model_id="local-test", extract_relationships=True)
    assert service.get_run_status(run_id).status == "pending"
    status = service.execute_run(run_id, events.append)
    assert status.status in ("completed", "needs_review") and status.run.passages_processed == 1
    assert [e.step for e in events] == ["knowledge", "knowledge", "relationships"]
    sent = providers["local-test"].requests[0].messages[0].content
    assert "Part 2" in sent and "Learning" in sent, "the model is told which book and chapter it reads"
    assert 'role="target"' in sent

    # the rest of the book, then everything is inspectable
    rest = service.extract_book(book.book_id, ["concept", "equation"], extract_relationships=True)
    assert rest.status in ("completed", "needs_review")
    session.rollback()
    f = KnowledgeFilter(book_id=book.book_id, document_id=book.id)
    knowledge = KnowledgeService(session)
    summary = knowledge.summary(f)
    assert summary.by_type == {"concept": 3, "equation": 2}, "one concept per passage; the two Phase 1 equations, each kept once"

    # 10-12. sources, equations and relationships
    concept = knowledge.list_units(KnowledgeFilter(book_id=book.book_id, knowledge_type="concept")).items[0]
    detail = knowledge.detail(concept.id)
    [source] = detail.sources
    assert source.role == "primary" and "Retention fades" in source.text and source.page_start is None, "a DOCX has no pages and none are invented"
    assert detail.unit.chapter_node_id is not None and detail.unit.page_status.value == "unknown"
    equation_service = EquationService(session)
    captures = {
        u.title: equation_service.get_capture(u.id)
        for u in session.scalars(select(KnowledgeUnit).where(KnowledgeUnit.knowledge_type == "equation"))
    }
    assert captures["Force law"].capture_status == EquationCaptureStatus.COMPLETE and captures["Force law"].original_text == "F=ma"
    [from_source] = [c for title, c in captures.items() if title != "Force law"]
    assert from_source.capture_status in (EquationCaptureStatus.COMPLETE, EquationCaptureStatus.TEXT_ONLY), \
        "the equation the model did not read is kept from Phase 1, not dropped"
    assert session.scalar(select(func.count()).select_from(KnowledgeRelationship)) >= 1

    # 13. model, prompt and configuration are recorded on every execution and attempt
    for execution in session.scalars(select(ExtractionExecution)):
        assert execution.model_configuration_id == "local-test" and execution.prompt_name and execution.prompt_version
        assert execution.model_configuration["model"] == "test-model:1b" and "test-value" not in json.dumps(execution.model_configuration)
    assert all(a.model_configuration_id == "local-test" and a.prompt_version for a in session.scalars(select(ExtractionAttempt)))

    # 14. passage and run statuses
    assert {r.status for r in session.scalars(select(ExtractionPassageRecord))} <= {"completed", "needs_review"}

    # nothing is repeated, and Phase 1 is exactly as it was
    assert service.plan(service.book_scope(book.book_id), ["concept", "equation"], model_id="local-test").model_calls == 0
    with pytest.raises(NothingToExtractError):
        service.extract_book(book.book_id, ["concept", "equation"])
    assert phase_1_state(session, book) == before


def test_one_phase_1_equation_is_never_attached_to_a_reading_of_another(session, book):
    """Regression: the three equations of a real extraction were all bound to the only Phase 1 equation of the passage."""

    def answer(call):
        if call.category.value == "equation":
            return reply(item(call, content=equation_data(name="Something else", original_expression="P(y|x) = exp(-bF)/Z", latex="P(y|x)=exp(-bF)/Z")))
        return reply()

    service, _ = make_service(session, answer)
    service.extract_book(book.book_id, ["equation"])
    session.rollback()

    sources = list(session.scalars(select(EquationSource)))
    model_read = [s for s in sources if s.capture_status == "needs_review" and s.equation_id is None]
    assert model_read, "an unrelated reading is not tied to a Phase 1 equation"
    assert all(s.original_text is None for s in model_read)
    kept = [u for u in session.scalars(select(KnowledgeUnit)) if u.origin == "source_derived"]
    assert len(kept) == 2, "both Phase 1 equations survive as units of their own"


# -- failure and recovery ----------------------------------------------------------------------------


class Flaky:
    """Fails per passage as scripted: 'timeout' always, 'malformed' once, 'interrupt' once, anything else answers."""

    def __init__(self, session: Session, plan: dict[int, str]) -> None:
        self.session, self.plan, self.seen = session, dict(plan), {}
        self.targets: list[int] = []

    def __call__(self, call: Call):
        order = self.session.get(Passage, uuid.UUID(call.target_id)).document_sequence
        self.targets.append(order)
        mode = self.plan.get(order)
        self.seen[order] = self.seen.get(order, 0) + 1
        if mode == "timeout":
            raise ProviderTimeoutError("provider timed out")
        if mode == "malformed" and self.seen[order] == 1:
            return "this is {not json"
        if mode == "interrupt" and self.seen[order] == 1:
            raise KeyboardInterrupt
        return reply(item(call, content={"name": f"Concept {order}"}))


def units(session) -> list[KnowledgeUnit]:
    session.rollback()
    return list(session.scalars(select(KnowledgeUnit).order_by(KnowledgeUnit.title)))


def test_an_interrupted_and_partly_failed_run_resumes_without_repeating_or_duplicating(session, book):
    flaky = Flaky(session, {1: "interrupt", 2: "malformed"})
    service, _ = make_service(session, flaky, max_retries=1)
    run_id = service.create_run(service.book_scope(book.book_id), ["concept"], model_id="local-test")

    # the application is interrupted while the second passage is being processed
    with pytest.raises(KeyboardInterrupt):
        service.execute_run(run_id)
    interrupted = service.get_run_status(run_id)
    assert interrupted.status == "cancelled" and interrupted.run.units_extracted == 1
    assert [u.title for u in units(session)] == ["Concept 0"], "the finished passage keeps its result"
    assert service.get_resume_info(run_id).can_resume

    # resuming continues with the passage that was cut off and the one never reached; a malformed answer is corrected
    flaky.targets.clear()
    resumed = service.execute_run(run_id)
    assert flaky.targets[0] == 1 and 0 not in flaky.targets, "the finished passage is not asked again"
    assert resumed.status in ("completed", "needs_review") and resumed.failed_passages == 0
    assert [u.title for u in units(session)] == ["Concept 0", "Concept 1", "Concept 2"]
    assert resumed.run.validation_failure_count >= 1 and resumed.run.retry_count >= 1, "the malformed answer was counted and corrected"

    # nothing left to do: resuming again changes nothing
    before = len(units(session))
    with pytest.raises(Exception, match="already finished"):
        service.execute_run(run_id)
    assert len(units(session)) == before


def test_a_failed_passage_is_visible_then_recovered_and_every_attempt_is_kept(session, book):
    flaky = Flaky(session, {1: "timeout"})
    service, _ = make_service(session, flaky, max_retries=2)
    run_id = service.create_run(service.book_scope(book.book_id), ["concept"], model_id="local-test")

    first = service.execute_run(run_id)

    assert first.status == "needs_review" and first.failed_passages == 1 and first.run.units_extracted == 2
    [failed] = [p for p in service.get_passage_status(run_id) if p.status == "failed"]
    assert failed.sequence == 1 and failed.error_code == "provider_timeout" and failed.attempts == 3, "retry limit of 2 means three calls, all recorded"
    attempts = list(session.scalars(select(ExtractionAttempt).where(ExtractionAttempt.passage_id == failed.passage_id)))
    assert len(attempts) == 3 and all(a.status == "provider_error" and a.provider_error for a in attempts)
    overview = KnowledgeService(session).overview(KnowledgeFilter(book_id=book.book_id))
    assert [f.failure_code for f in overview.failed_extractions] == ["provider_timeout"], "the unresolved failure is on the review overview"
    assert overview.open_flags.get("unresolved_extraction_error") == 1

    # the provider comes back; only the failed passage is repeated and the earlier attempts stay
    flaky.plan.clear()
    flaky.targets.clear()
    second = service.execute_run(run_id)

    assert flaky.targets == [1], "only the failed passage was asked again"
    assert second.failed_passages == 0 and second.run.units_extracted == 3
    assert [u.title for u in units(session)] == ["Concept 0", "Concept 1", "Concept 2"]
    assert session.scalar(select(func.count()).select_from(ExtractionAttempt).where(ExtractionAttempt.passage_id == failed.passage_id)) == 4
    assert session.scalar(select(func.count()).select_from(ExtractionPassageRecord).where(ExtractionPassageRecord.extraction_run_id == run_id)) == 3


def test_reprocessing_a_finished_passage_needs_an_explicit_request_and_adds_no_duplicate_unit(session, book):
    service, providers = make_service(session, Flaky(session, {}))
    service.extract_book(book.book_id, ["concept"])
    calls = len(providers["local-test"].requests)

    with pytest.raises(NothingToExtractError):
        service.extract_book(book.book_id, ["concept"])
    assert len(providers["local-test"].requests) == calls

    forced = service.extract_book(book.book_id, ["concept"], force=True)

    assert len(providers["local-test"].requests) == 2 * calls, "forcing asks the model again"
    assert forced.run.duplicate_count == 3 and len(units(session)) == 3, "and the repeated answers are recognised as the units already stored"
