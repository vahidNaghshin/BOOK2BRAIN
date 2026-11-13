"""The processing pipeline end to end: stages, status, failure, retry/resume, idempotency.

These tests drive `ProcessingPipeline` with a scripted parser (so they are fast and can fail on
demand) over real database tables; the real PDF pipeline is exercised in test_end_to_end.py.
"""

import io
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.parsing.errors import CorruptDocumentError
from app.schemas import BookMetadata, DocumentRead
from app.services.document_service import DocumentService
from app.services.errors import ProcessingNotAllowedError
from app.services.pipeline_service import ProcessingPipeline
from app.services.storage_service import LocalStorage
from database.models import (
    Document,
    ExtractedBlock,
    Paragraph,
    Passage,
    PassageParagraph,
    ProcessingError,
    ProcessingRun,
    StructuralNode,
    StructureAnalysis,
)
from tests.factories import make_pdf
from tests.fakes import StubParser, StubParserFactory
from tests.structure_helpers import content_blocks, numbered_textbook


@pytest.fixture
def settings(test_database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        database_url=test_database_url,
        storage_root=tmp_path / "storage",
        max_upload_size_mb=5,
        equation_extraction_enabled=False,
        _env_file=None,
    )


@pytest.fixture
def book_factory() -> StubParserFactory:
    document = numbered_textbook().build()
    return StubParserFactory(StubParser(content_blocks(document), page_count=document.page_count))


def upload(service: DocumentService, seed: str, *, title: str = "Sample Book", author: str | None = "A. Writer") -> DocumentRead:
    return service.upload(
        file=io.BytesIO(make_pdf(seed)), filename=f"{seed}.pdf", metadata=BookMetadata(title=title, author=author)
    ).document


def make_pipeline(session: Session, storage: LocalStorage, settings: Settings, factory) -> ProcessingPipeline:
    return ProcessingPipeline(session, storage, settings, factory)


def count(session: Session, model, **filters) -> int:
    session.rollback()
    stmt = select(func.count()).select_from(model)
    for column, value in filters.items():
        stmt = stmt.where(getattr(model, column) == value)
    return session.scalar(stmt) or 0


def counts(session: Session, document_id: uuid.UUID) -> dict[str, int]:
    return {
        "blocks": count(session, ExtractedBlock, document_id=document_id),
        "nodes": count(session, StructuralNode, document_id=document_id),
        "paragraphs": count(session, Paragraph, document_id=document_id),
        "passages": count(session, Passage, document_id=document_id),
        "links": session.scalar(text("select count(*) from passage_paragraphs")) or 0,
        "runs": count(session, ProcessingRun, document_id=document_id),
    }


def state(engine: Engine, document_id: uuid.UUID) -> dict:
    """What another user would see: committed state, read through a separate connection."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT d.processing_status AS document_status, r.status AS run_status, r.stage, r.retry_count, "
                "r.error_message FROM documents d LEFT JOIN processing_runs r ON r.document_id = d.id "
                "WHERE d.id = :id ORDER BY r.created_at DESC LIMIT 1"
            ),
            {"id": document_id},
        ).one()
    return dict(row._mapping)


# -- the happy path -----------------------------------------------------------------------------


def test_processing_a_book_runs_every_stage_and_completes(
    session: Session, document_service: DocumentService, storage, settings, book_factory, engine
) -> None:
    document = upload(document_service, "a1")
    assert state(engine, document.id)["document_status"] == "uploaded"

    run = make_pipeline(session, storage, settings, book_factory).process_document(document.id)

    assert run.status == "completed" and run.stage == "completed"
    assert run.retry_count == 0 and run.error_message is None
    assert run.started_at and run.completed_at and run.updated_at >= run.started_at
    assert run.book_id == document.book_id and run.document_id == document.id
    after = state(engine, document.id)
    assert (after["document_status"], after["run_status"], after["stage"]) == ("completed", "completed", "completed")
    found = counts(session, document.id)
    assert found["blocks"] == 33 and found["nodes"] == 16 and found["paragraphs"] > 0 and found["passages"] > 0
    assert found["runs"] == 1


def test_stage_by_stage_progress_is_visible_while_running(
    session: Session, document_service, storage, settings, book_factory, engine
) -> None:
    from app.parsing.models import ParseStage

    document = upload(document_service, "a2")
    seen: list[tuple[str, str | None, str]] = []
    scripted = numbered_textbook().build()
    book_factory = StubParserFactory(
        StubParser(
            content_blocks(scripted),
            page_count=scripted.page_count,
            stages=[(ParseStage.PARSING, 0, 1), (ParseStage.PARSING, 1, 1)],
        )
    )

    def watch(stage: str, current: int, total: int) -> None:
        committed = state(engine, document.id)
        seen.append((stage, committed["stage"], committed["document_status"]))

    make_pipeline(session, storage, settings, book_factory).process_document(document.id, on_progress=watch)

    order = list(dict.fromkeys(stage for stage, _, _ in seen))
    assert order == [
        "document_registration", "parsing", "normalizing", "structural_detection",
        "paragraph_identification", "passage_segmentation", "persistence", "validation",
    ]
    # The stage another session reads is the stage being reported, and the document is not
    # "completed" at any point before the end.
    assert all(reported == committed for reported, committed, _ in seen if reported != "parsing")
    assert all(status != "completed" for _, _, status in seen)


def test_stage_results_are_recorded_for_every_stage(
    session: Session, document_service, storage, settings, book_factory
) -> None:
    document = upload(document_service, "a3")

    make_pipeline(session, storage, settings, book_factory).process_document(document.id)

    run = session.scalar(select(ProcessingRun).where(ProcessingRun.document_id == document.id))
    results = run.stage_results
    for stage in ("document_registration", "parsing", "text_extraction", "structural_detection",
                  "paragraph_identification", "passage_segmentation", "persistence", "validation", "completed"):
        assert results[stage]["status"] == "completed", stage
    assert results["document_registration"]["book_id"] == str(document.book_id)
    assert results["text_extraction"]["block_count"] == 33
    assert results["structural_detection"]["node_count"] == 16
    assert results["passage_segmentation"]["passage_count"] == counts(session, document.id)["passages"]
    assert results["validation"]["errors"] == 0


def test_a_completed_document_is_not_processed_again(
    session: Session, document_service, storage, settings, book_factory
) -> None:
    document = upload(document_service, "a4")
    pipeline = make_pipeline(session, storage, settings, book_factory)
    pipeline.process_document(document.id)

    with pytest.raises(ProcessingNotAllowedError, match="already been processed"):
        pipeline.process_document(document.id)
    assert counts(session, document.id)["runs"] == 1


def test_a_second_pipeline_on_a_fresh_session_sees_everything_persisted(
    session: Session, document_service, storage, settings, book_factory, engine
) -> None:
    document = upload(document_service, "a5")
    make_pipeline(session, storage, settings, book_factory).process_document(document.id)
    expected = counts(session, document.id)

    with Session(engine) as restarted:  # as after an application restart
        assert counts(restarted, document.id) == expected
        assert restarted.get(Document, document.id).processing_status == "completed"
        assert restarted.scalar(select(StructureAnalysis).where(StructureAnalysis.document_id == document.id)).structure_status == "completed"


# -- failure, retry and resume --------------------------------------------------------------------


def break_structure(monkeypatch, times: int = 1):
    """Make structural detection fail the first `times` calls, then work again."""
    from app.structure import detector

    original = detector.StructureDetector.detect
    calls = {"n": 0}

    def flaky(self, document, on_stage=None):
        calls["n"] += 1
        if calls["n"] <= times:
            raise RuntimeError("detector exploded")
        return original(self, document, on_stage)

    monkeypatch.setattr("app.services.structure_service.StructureDetector.detect", flaky)
    return calls


def test_a_structural_failure_keeps_the_text_and_is_reported_at_its_stage(
    session: Session, document_service, storage, settings, book_factory, engine, monkeypatch
) -> None:
    document = upload(document_service, "b1")
    break_structure(monkeypatch)

    run = make_pipeline(session, storage, settings, book_factory).process_document(document.id)

    assert run.status == "failed" and run.stage == "structural_detection"
    assert run.error_message == "Structural detection failed."
    after = state(engine, document.id)
    assert (after["document_status"], after["run_status"], after["stage"]) == ("failed", "failed", "structural_detection")
    found = counts(session, document.id)
    assert found["blocks"] == 33  # the extracted text is preserved
    assert found["nodes"] == found["paragraphs"] == found["passages"] == 0
    error = session.scalar(select(ProcessingError).where(ProcessingError.document_id == document.id))
    assert (error.stage, error.retry_attempt, error.error_type) == ("structural_detection", 0, "structure_failed")
    assert "detector exploded" in error.details["traceback"]  # developers get the cause, users do not
    results = session.get(ProcessingRun, run.id).stage_results
    assert results["text_extraction"]["status"] == "completed" and results["structural_detection"]["status"] == "failed"


def test_retry_resumes_the_failed_run_without_parsing_again(
    session: Session, document_service, storage, settings, book_factory, engine, monkeypatch
) -> None:
    document = upload(document_service, "b2")
    break_structure(monkeypatch)
    pipeline = make_pipeline(session, storage, settings, book_factory)
    failed = pipeline.process_document(document.id)
    parses = len(book_factory.requested)

    retried = pipeline.process_document(document.id)  # the retry action

    assert retried.id == failed.id  # the same run was resumed, not duplicated
    assert retried.status == "completed" and retried.stage == "completed"
    assert retried.retry_count == 1 and retried.error_message is None
    assert len(book_factory.requested) == parses  # the file was not parsed again
    found = counts(session, document.id)
    assert found["runs"] == 1 and found["blocks"] == 33 and found["nodes"] == 16 and found["passages"] > 0
    assert state(engine, document.id)["document_status"] == "completed"
    # The failure stays on record, with the attempt it happened in.
    errors = list(session.scalars(select(ProcessingError).where(ProcessingError.document_id == document.id)))
    assert [(e.stage, e.retry_attempt) for e in errors] == [("structural_detection", 0)]


def test_recovery_leaves_no_duplicates_and_survives_a_restart(
    session: Session, document_service, storage, settings, book_factory, engine, monkeypatch
) -> None:
    # Upload, parse successfully, fail structure, confirm failure and kept text, retry, confirm
    # structure and passages, no duplicates, then look again from a brand new session.
    document = upload(document_service, "b3")
    break_structure(monkeypatch)
    pipeline = make_pipeline(session, storage, settings, book_factory)
    pipeline.process_document(document.id)
    assert state(engine, document.id)["run_status"] == "failed"
    assert counts(session, document.id)["blocks"] == 33

    pipeline.process_document(document.id)

    found = counts(session, document.id)
    nodes = list(session.scalars(select(StructuralNode).where(StructuralNode.document_id == document.id)))
    assert len(nodes) == 16 and len({n.sequence for n in nodes}) == 16
    keys = [(n.parent_id, n.node_type, n.title, n.number) for n in nodes if not (n.node_metadata or {}).get("synthetic")]
    assert len(keys) == len(set(keys))  # no duplicate chapters or sections
    paragraph_keys = list(session.execute(select(Paragraph.structural_node_id, Paragraph.sequence).where(Paragraph.document_id == document.id)))
    assert len(paragraph_keys) == len(set(paragraph_keys)) == found["paragraphs"]
    passage_keys = list(session.execute(select(Passage.structural_node_id, Passage.sequence).where(Passage.document_id == document.id)))
    assert len(passage_keys) == len(set(passage_keys)) == found["passages"]
    assert found["runs"] == 1

    with Session(engine) as restarted:
        assert counts(restarted, document.id) == found
        assert restarted.get(Document, document.id).processing_status == "completed"


def test_a_passage_segmentation_failure_is_retried_without_parsing_or_duplicates(
    session: Session, document_service, storage, settings, book_factory, monkeypatch
) -> None:
    document = upload(document_service, "b4")
    from app.services import structure_service

    original = structure_service.generate_passages
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("segmenter exploded")
        return original(*args, **kwargs)

    monkeypatch.setattr(structure_service, "generate_passages", flaky)
    pipeline = make_pipeline(session, storage, settings, book_factory)

    failed = pipeline.process_document(document.id)
    assert failed.status == "failed" and failed.stage == "passage_segmentation"
    assert failed.error_message == "Passage segmentation failed."
    assert counts(session, document.id)["blocks"] == 33 and counts(session, document.id)["passages"] == 0

    done = pipeline.process_document(document.id)

    assert done.status == "completed" and done.id == failed.id
    assert len(book_factory.requested) == 1
    assert counts(session, document.id)["passages"] > 0


def test_a_persistence_failure_saves_nothing_partial(
    session: Session, document_service, storage, settings, book_factory, monkeypatch
) -> None:
    document = upload(document_service, "b5")

    def boom(self, document, passages):
        raise RuntimeError("disk full")

    monkeypatch.setattr("app.repositories.structure_repository.StructureRepository.replace_passages", boom)
    run = make_pipeline(session, storage, settings, book_factory).process_document(document.id)

    assert run.status == "failed" and run.stage == "persistence"
    assert run.error_message == "Database persistence failed."
    found = counts(session, document.id)
    assert found["nodes"] == found["paragraphs"] == found["passages"] == 0  # one transaction: all or nothing
    assert found["blocks"] == 33


def test_failed_validation_never_reports_completion_and_resumes_at_validation(
    session: Session, document_service, storage, settings, book_factory, engine, monkeypatch
) -> None:
    document = upload(document_service, "b6")
    from app.services.integrity_service import IntegrityValidator
    from app.structure.models import Severity, ValidationIssue

    real = IntegrityValidator.validate
    state_ = {"fail": True}

    def validate(self, document_id, run_id=None):
        if state_["fail"]:
            return [ValidationIssue(Severity.ERROR, "synthetic_failure", "Something is inconsistent.", "document")]
        return real(self, document_id, run_id)

    monkeypatch.setattr(IntegrityValidator, "validate", validate)
    detections = break_structure(monkeypatch, times=0)
    pipeline = make_pipeline(session, storage, settings, book_factory)

    failed = pipeline.process_document(document.id)

    assert failed.status == "failed" and failed.stage == "validation"
    assert "Data integrity validation failed" in failed.error_message
    assert state(engine, document.id)["document_status"] == "failed"
    assert counts(session, document.id)["passages"] > 0  # the data stays, so it can be inspected
    results = session.get(ProcessingRun, failed.id).stage_results["validation"]
    assert results["status"] == "failed" and results["errors"] == 1
    error = session.scalar(select(ProcessingError).where(ProcessingError.document_id == document.id))
    assert error.stage == "validation" and error.details["issues"][0]["code"] == "synthetic_failure"

    state_["fail"] = False
    done = pipeline.process_document(document.id)

    assert done.status == "completed" and done.id == failed.id
    assert detections["n"] == 1  # structure was not detected a second time


def test_a_parse_failure_starts_a_new_run_on_retry(
    session: Session, document_service, storage, settings, engine
) -> None:
    document = upload(document_service, "b7")
    parser = StubParser(error=CorruptDocumentError("Not a valid PDF."))
    factory = StubParserFactory(parser)
    pipeline = make_pipeline(session, storage, settings, factory)

    failed = pipeline.process_document(document.id)
    assert failed.status == "failed" and failed.stage == "parsing"
    assert counts(session, document.id)["blocks"] == 0

    parser.error = None
    parser.blocks = content_blocks(numbered_textbook().build())
    parser.page_count = 9
    retried = pipeline.process_document(document.id)

    assert retried.id != failed.id and retried.status == "completed"
    assert retried.retry_count == 1
    assert counts(session, document.id)["runs"] == 2  # the failed run stays as history
    assert session.get(ProcessingRun, failed.id).status == "failed"


def test_a_missing_original_file_fails_at_registration(
    session: Session, document_service, storage, settings, book_factory
) -> None:
    document = upload(document_service, "b8")
    storage.delete(document.file_path)

    run = make_pipeline(session, storage, settings, book_factory).process_document(document.id)

    assert run.status == "failed" and run.stage == "document_registration"
    assert "missing from storage" in run.error_message
    assert book_factory.requested == []


def test_empty_extraction_is_a_text_failure_not_a_success(
    session: Session, document_service, storage, settings
) -> None:
    document = upload(document_service, "b9")
    factory = StubParserFactory(StubParser([], page_count=3))

    run = make_pipeline(session, storage, settings, factory).process_document(document.id)

    assert run.status == "failed"
    assert counts(session, document.id)["nodes"] == 0
    assert session.get(Document, document.id).processing_status == "failed"


def test_processing_errors_carry_run_book_and_document(
    session: Session, document_service, storage, settings, book_factory, monkeypatch
) -> None:
    document = upload(document_service, "b10")
    break_structure(monkeypatch)

    run = make_pipeline(session, storage, settings, book_factory).process_document(document.id)

    error = session.scalar(select(ProcessingError).where(ProcessingError.processing_run_id == run.id))
    assert error.document_id == document.id and error.created_at
    assert session.get(ProcessingRun, run.id).book_id == document.book_id


# -- scanned pages ---------------------------------------------------------------------------------


def test_unreadable_scanned_pages_complete_with_a_warning_not_silently(
    session: Session, document_service, storage, settings
) -> None:
    from app.parsing.base import Extraction
    from app.parsing.models import OcrSummary

    class PartialParser(StubParser):
        def _extract(self, path, on_progress):
            blocks = content_blocks(numbered_textbook().build())
            ocr = OcrSummary(engine="fake", language="eng", dpi=300, pages_processed=[2], pages_failed=[3])
            return Extraction(blocks=blocks, page_count=9, ocr=ocr, partial=True)

    document = upload(document_service, "c1")
    run = make_pipeline(session, storage, settings, StubParserFactory(PartialParser())).process_document(document.id)

    assert run.status == "completed"
    assert run.summary.status.value == "partial" and run.summary.ocr.pages_failed == [3]
    results = session.get(ProcessingRun, run.id).stage_results
    assert results["ocr"]["status"] == "partial" and results["ocr"]["pages_failed"] == [3]
    codes = [i["code"] for i in results["validation"]["issues"]]
    assert "incomplete_extraction" in codes


# -- several books and editions -------------------------------------------------------------------


def test_two_editions_of_a_book_keep_separate_structure_and_pages(
    session: Session, document_service, storage, settings
) -> None:
    first = upload(document_service, "d1", title="Dune", author="Frank Herbert")
    second = upload(document_service, "d2", title="Dune", author="Frank Herbert")
    assert first.book_id == second.book_id and first.id != second.id

    shifted = numbered_textbook()
    for block in shifted.blocks:
        block.page = (block.page or 0) + 10  # the second edition paginates differently
    factories = {
        first.id: StubParserFactory(StubParser(content_blocks(numbered_textbook().build()), page_count=9)),
        second.id: StubParserFactory(StubParser(content_blocks(shifted.build(page_count=19)), page_count=19)),
    }
    for document in (first, second):
        assert make_pipeline(session, storage, settings, factories[document.id]).process_document(document.id).status == "completed"

    for document, offset in ((first, 0), (second, 10)):
        nodes = list(session.scalars(select(StructuralNode).where(StructuralNode.document_id == document.id)))
        passages = list(session.scalars(select(Passage).where(Passage.document_id == document.id)))
        assert len(nodes) == 16 and passages
        assert all(n.book_id == document.book_id for n in nodes)
        pages = [p.page_start for p in passages if p.page_start]
        assert min(pages) > offset and max(pages) <= 9 + offset
    assert count(session, StructuralNode) == 32 and count(session, StructureAnalysis) == 2


def test_a_failure_in_one_document_does_not_touch_another(
    session: Session, document_service, storage, settings, book_factory, monkeypatch, engine
) -> None:
    good = upload(document_service, "e1", title="Good Book", author=None)
    bad = upload(document_service, "e2", title="Bad Book", author=None)
    pipeline = make_pipeline(session, storage, settings, book_factory)
    assert pipeline.process_document(good.id).status == "completed"
    good_counts = counts(session, good.id)

    break_structure(monkeypatch)
    assert pipeline.process_document(bad.id).status == "failed"

    assert state(engine, good.id)["document_status"] == "completed"
    assert state(engine, bad.id)["document_status"] == "failed"
    assert counts(session, good.id) == good_counts


def test_unrelated_books_are_processed_independently(
    session: Session, document_service, storage, settings, book_factory
) -> None:
    one = upload(document_service, "f1", title="Book One", author=None)
    two = upload(document_service, "f2", title="Book Two", author=None)
    pipeline = make_pipeline(session, storage, settings, book_factory)

    pipeline.process_document(one.id)
    assert count(session, StructuralNode, document_id=two.id) == 0  # processing one touches nothing else
    pipeline.process_document(two.id)

    assert count(session, StructuralNode, document_id=one.id) == count(session, StructuralNode, document_id=two.id) == 16
    assert count(session, ProcessingRun) == 2
