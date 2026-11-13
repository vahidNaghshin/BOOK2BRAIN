"""Day 5 schema additions (stages, retry bookkeeping, publisher/year) and the library read model."""

import io
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.schemas import BookMetadata
from app.services.job_runner import ProcessingJobRunner
from app.services.library_service import LibraryService
from app.services.pipeline_service import ProcessingPipeline
from database.models import (
    RESUMABLE_STAGES,
    Book,
    ProcessingError,
    ProcessingRun,
    ProcessingStage,
)
from database.models.base import utcnow
from tests.factories import make_pdf
from tests.fakes import StubParser, StubParserFactory
from tests.structure_helpers import content_blocks, numbered_textbook
from tests.test_models import make_book, make_document
from tests.test_pipeline import break_structure, upload


@pytest.fixture
def settings(test_database_url, tmp_path) -> Settings:
    return Settings(
        database_url=test_database_url, storage_root=tmp_path / "storage", max_upload_size_mb=5,
        equation_extraction_enabled=False, _env_file=None,
    )


@pytest.fixture
def session_factory(engine):
    from contextlib import contextmanager

    @contextmanager
    def scope():
        local = Session(bind=engine, autoflush=False, expire_on_commit=False)
        try:
            yield local
        finally:
            local.close()

    return scope


@pytest.fixture
def document(session: Session):
    value = make_document(make_book())
    session.add(value)
    session.commit()
    return value


def test_every_pipeline_stage_is_accepted_and_unknown_ones_are_not(session: Session, document) -> None:
    for stage in ProcessingStage:
        session.add(ProcessingRun(document_id=document.id, status="processing", stage=stage.value))
    session.commit()
    assert len(session.scalars(select(ProcessingRun)).all()) == len(ProcessingStage)

    session.add(ProcessingRun(document_id=document.id, status="processing", stage="making_coffee"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_stage_order_covers_the_required_stages() -> None:
    required = {
        "document_registration", "parsing", "text_extraction", "ocr", "structural_detection",
        "paragraph_identification", "passage_segmentation", "persistence", "validation", "completed",
    }
    assert required <= {stage.value for stage in ProcessingStage}
    assert {s.value for s in RESUMABLE_STAGES} == {
        "equation_extraction", "structural_detection", "paragraph_identification",
        "passage_segmentation", "persistence", "validation",
    }


def test_run_bookkeeping_defaults_and_constraints(session: Session, document) -> None:
    run = ProcessingRun(document_id=document.id)
    session.add(run)
    session.commit()

    assert run.retry_count == 0 and run.text_completed_at is None and run.stage_results is None
    assert run.updated_at is not None and run.book_id == document.book_id
    before = run.updated_at
    run.status = "processing"
    session.commit()
    assert run.updated_at > before

    session.add(ProcessingRun(document_id=document.id, retry_count=-1))
    with pytest.raises(IntegrityError):
        session.commit()


def test_errors_record_stage_and_attempt(session: Session, document) -> None:
    run = ProcessingRun(document_id=document.id)
    session.add(run)
    session.flush()
    session.add(ProcessingError(document_id=document.id, processing_run_id=run.id, error_type="x", error_message="m"))
    session.commit()

    error = session.scalar(select(ProcessingError))
    assert error.stage is None and error.retry_attempt == 0


def test_publisher_and_year_are_stored_and_the_year_is_bounded(session: Session) -> None:
    session.add(Book(title="Dune", publisher="Chilton", publication_year=1965))
    session.commit()
    assert session.scalar(select(Book)).publication_year == 1965

    session.add(Book(title="Bad", publication_year=5))
    with pytest.raises(IntegrityError):
        session.commit()


# -- library read model -------------------------------------------------------------------------


def test_the_library_read_model_reports_each_documents_latest_state(session: Session, document_service, storage, settings, monkeypatch) -> None:
    good = upload(document_service, "l1", title="Good", author=None)
    bad = upload(document_service, "l2", title="Bad", author=None)
    fresh = upload(document_service, "l3", title="Fresh", author=None)
    built = numbered_textbook().build()
    factory = StubParserFactory(StubParser(content_blocks(built), page_count=built.page_count))
    ProcessingPipeline(session, storage, settings, factory).process_document(good.id)
    with monkeypatch.context() as m:
        break_structure(m)
        ProcessingPipeline(session, storage, settings, factory).process_document(bad.id)

    books = {b.book.title: b for b in LibraryService(session).list_books()}

    done = books["Good"].documents[0]
    assert (done.run_status, done.stage, done.is_processed, done.run_count) == ("completed", "completed", True, 1)
    assert done.node_count == 16 and done.passage_count > 0 and not done.extraction_incomplete
    broke = books["Bad"].documents[0]
    assert (broke.run_status, broke.stage, broke.is_processed) == ("failed", "structural_detection", False)
    assert broke.error_message == "Structural detection failed." and broke.passage_count == 0
    new = books["Fresh"].documents[0]
    assert (new.run_status, new.run_count, new.is_active) == (None, 0, False)
    assert fresh.id == new.document.id


# -- resume paths --------------------------------------------------------------------------------


def test_a_stale_interrupted_run_with_saved_text_is_resumed_not_restarted(
    session: Session, document_service, storage, settings, monkeypatch
) -> None:
    document = upload(document_service, "s1")
    built = numbered_textbook().build()
    factory = StubParserFactory(StubParser(content_blocks(built), page_count=built.page_count))
    pipeline = ProcessingPipeline(session, storage, settings, factory)
    with monkeypatch.context() as m:
        break_structure(m)
        failed = pipeline.process_document(document.id)
    # The app was restarted mid-run: the run looks "processing" for longer than the stale limit.
    run = session.get(ProcessingRun, failed.id)
    run.status = "processing"
    run.started_at = utcnow() - timedelta(minutes=settings.processing_stale_minutes + 5)
    session.get(type(session.get(ProcessingRun, failed.id).document), document.id).processing_status = "processing"
    session.commit()

    done = ProcessingPipeline(session, storage, settings, factory).process_document(document.id)

    assert done.id == failed.id and done.status == "completed"
    assert len(factory.requested) == 1


def test_the_job_runner_resumes_a_failed_document_in_the_background(
    session: Session, document_service, storage, settings, session_factory, monkeypatch
) -> None:
    document = upload(document_service, "j1")
    built = numbered_textbook().build()
    factory = StubParserFactory(StubParser(content_blocks(built), page_count=built.page_count))
    runner = ProcessingJobRunner(settings, session_factory=session_factory, storage=storage, parser_factory=factory)
    with monkeypatch.context() as m:
        break_structure(m)
        first = runner.submit(document.id)
        runner.shutdown(wait=True)
    session.rollback()
    assert session.get(ProcessingRun, first).status == "failed"

    runner = ProcessingJobRunner(settings, session_factory=session_factory, storage=storage, parser_factory=factory)
    second = runner.retry(document.id)
    runner.shutdown(wait=True)

    session.rollback()
    assert second == first
    run = session.get(ProcessingRun, first)
    assert (run.status, run.stage, run.retry_count) == ("completed", "completed", 1)
    assert len(factory.requested) == 1
