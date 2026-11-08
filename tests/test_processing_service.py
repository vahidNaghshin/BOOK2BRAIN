import io
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.config import Settings
from app.parsing.errors import CorruptDocumentError, EncryptedDocumentError
from app.parsing.models import BlockType, ParseStage
from app.parsing.parser_factory import ParserFactory
from app.schemas import BookMetadata, DocumentRead
from app.services.book_service import BookService
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import DocumentService
from app.services.errors import (
    DocumentNotFoundError,
    ProcessingNotAllowedError,
    ProcessingServiceError,
)
from app.services.job_runner import ProcessingJobRunner
from app.services.storage_service import LocalStorage
from database.models import (
    Document,
    ExtractedBlock,
    ProcessingError,
    ProcessingRun,
    ProcessingRunStatus,
)
from database.models.base import utcnow
from tests.conftest import stored_files
from tests.factories import (
    make_corrupt_pdf,
    make_docx_document,
    make_docx_with_broken_xml,
    make_epub_book,
    make_mixed_pdf,
    make_text_pdf,
)
from tests.fakes import FakeOcrEngine, StubParser, StubParserFactory, block, ocr_result

LONG = "This is a sentence that is long enough to count as meaningful extracted content."


@pytest.fixture
def settings(test_database_url: str, tmp_path: Path) -> Settings:
    """Like the shared fixture, but with room for page-sized scan images."""
    return Settings(
        database_url=test_database_url,
        storage_root=tmp_path / "storage",
        max_upload_size_mb=20,
        _env_file=None,
    )


@pytest.fixture
def upload(document_service: DocumentService):
    counter = iter(range(1000))

    def _upload(data: bytes, filename: str, title: str = "Test Book") -> DocumentRead:
        # Each call is a distinct book so hashes and identities never collide.
        result = document_service.upload(
            file=io.BytesIO(data),
            filename=filename,
            metadata=BookMetadata(title=f"{title} {next(counter)}"),
        )
        return result.document

    return _upload


@pytest.fixture
def make_service(session: Session, storage: LocalStorage, settings: Settings):
    def _make(parser_factory=None) -> DocumentProcessingService:
        return DocumentProcessingService(session, storage, settings, parser_factory)

    return _make


@pytest.fixture
def service(make_service) -> DocumentProcessingService:
    return make_service()


def db_state(engine: Engine, document_id: uuid.UUID) -> dict:
    """Read committed state through a separate connection, as another user would see it."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT d.processing_status AS document_status, r.status AS run_status, r.stage, "
                "r.progress_current, r.progress_total FROM documents d "
                "LEFT JOIN processing_runs r ON r.document_id = d.id "
                "WHERE d.id = :id ORDER BY r.created_at DESC LIMIT 1"
            ),
            {"id": document_id},
        ).one()
    return dict(row._mapping)


def count(session: Session, model: type) -> int:
    session.rollback()
    return session.scalar(select(func.count()).select_from(model)) or 0


def blocks_of(session: Session, document_id: uuid.UUID) -> list[ExtractedBlock]:
    session.rollback()
    stmt = select(ExtractedBlock).where(ExtractedBlock.document_id == document_id)
    return list(session.scalars(stmt.order_by(ExtractedBlock.sequence_number)))


def stub_factory(**kwargs) -> StubParserFactory:
    return StubParserFactory(StubParser(**kwargs))


# -- successful processing and persistence -------------------------------------------


def test_docx_is_processed_and_persisted(upload, service, session) -> None:
    document = upload(
        make_docx_document(
            heading="Annual Report",
            paragraphs=[LONG],
            bullets=["Bullet one"],
            table=[["Name", "Value"], ["Alpha", "42"]],
            closing="The closing paragraph of the document.",
        ),
        "report.docx",
    )

    run = service.process(document.id)

    assert run.status == "completed" and run.error_message is None
    assert run.summary.character_count > 0 and run.summary.page_count is None
    assert run.summary.block_count == 5 and run.summary.extraction_method == "docling"
    session.rollback()
    assert session.get(Document, document.id).processing_status == "completed"

    stored = blocks_of(session, document.id)
    assert [b.block_type for b in stored] == ["heading", "paragraph", "list", "table", "paragraph"]
    assert [b.sequence_number for b in stored] == [0, 1, 2, 3, 4]
    assert stored[3].text == "Name | Value\nAlpha | 42"
    assert all(b.page_number is None for b in stored)
    assert stored[0].block_metadata == {"heading_level": 1}
    assert stored[1].section_title == "Annual Report"
    assert stored[1].source_location["docling_ref"].startswith("#/texts/")
    assert run.summary.character_count == sum(len(b.text) for b in stored)


def test_epub_chapter_references_are_persisted(upload, service, session) -> None:
    epub = make_epub_book(
        [
            ("ch1.xhtml", "Chapter One", "<h1>Start</h1><p>First chapter text goes here.</p>"),
            ("ch2.xhtml", "Chapter Two", "<h1>Next</h1><p>Second chapter text goes here.</p>"),
        ],
        spine_order=[1, 0],
    )
    document = upload(epub, "book.epub")

    run = service.process(document.id)

    assert run.status == "completed" and run.summary.extraction_method == "ebooklib"
    stored = blocks_of(session, document.id)
    assert [b.text for b in stored if b.block_type == "heading"] == ["Next", "Start"]  # spine order
    assert {b.source_location["href"] for b in stored} == {"ch1.xhtml", "ch2.xhtml"}
    assert stored[0].source_location["chapter_title"] == "Chapter Two"
    assert all(b.page_number is None for b in stored)


def test_pdf_is_processed_with_ocr_and_page_numbers_persisted(
    upload, make_service, session, settings, docling_ready
) -> None:
    engine = FakeOcrEngine([ocr_result("Text recovered from the scanned second page.", confidence=88.0)])
    service = make_service(ParserFactory(settings, ocr_engine=engine))
    document = upload(make_mixed_pdf([("text", [LONG]), ("scan", ["x"]), ("blank", [])]), "mixed.pdf")

    run = service.process(document.id)

    assert run.status == "completed"
    assert run.summary.page_count == 3
    assert run.summary.ocr.pages_processed == [2] and run.summary.ocr.mean_confidence == 88.0
    stored = blocks_of(session, document.id)
    assert [(b.page_number, b.text) for b in stored] == [
        (1, LONG),
        (2, "Text recovered from the scanned second page."),
    ]
    assert stored[1].block_metadata["ocr"] is True
    assert stored[1].source_location["method"] == "ocr"


def test_original_text_is_kept_only_where_normalization_changed_it(upload, make_service, session) -> None:
    factory = stub_factory(
        blocks=[block("Clean sentence that needs nothing at all.\n"), block("Messy\u00a0 sentence \ufb01xed with enough text.")]
    )
    document = upload(make_docx_document(), "x.docx")

    make_service(factory).process(document.id)

    first, second = blocks_of(session, document.id)
    assert first.text == "Clean sentence that needs nothing at all." and first.raw_text is not None
    assert second.text == "Messy sentence fixed with enough text."
    assert second.raw_text == "Messy\u00a0 sentence \ufb01xed with enough text."


def test_processing_does_not_touch_the_stored_original_or_leave_temp_files(
    upload, service, storage, settings
) -> None:
    document = upload(make_docx_document(paragraphs=[LONG]), "x.docx")
    before = stored_files(settings.storage_root)

    service.process(document.id)

    after = stored_files(settings.storage_root)
    assert after == before
    assert storage.exists(document.file_path)
    assert not any((settings.storage_root / "temporary").glob("processing-*"))


# -- status workflow -----------------------------------------------------------------


def test_status_moves_through_pending_processing_and_completed(upload, make_service, engine) -> None:
    seen: dict[str, dict] = {}

    def observe(_path: Path) -> None:
        seen["during_parse"] = db_state(engine, document.id)

    factory = stub_factory(blocks=[block(LONG)], observer=observe)
    service = make_service(factory)
    document = upload(make_docx_document(), "x.docx")
    assert db_state(engine, document.id)["document_status"] == "uploaded"

    run_id = service.start(document.id)
    seen["queued"] = db_state(engine, document.id)
    service.execute(run_id)
    seen["finished"] = db_state(engine, document.id)

    assert seen["queued"]["document_status"] == "pending" and seen["queued"]["run_status"] == "pending"
    assert seen["during_parse"]["document_status"] == "processing"
    assert seen["during_parse"]["run_status"] == "processing" and seen["during_parse"]["stage"] == "parsing"
    assert seen["finished"]["document_status"] == "completed" and seen["finished"]["run_status"] == "completed"
    assert seen["finished"]["stage"] is None and seen["finished"]["progress_total"] is None


def test_stages_and_progress_are_visible_while_running(upload, make_service, engine) -> None:
    observed: list[tuple[str, int, int, str | None]] = []
    stages = [(ParseStage.PARSING, 0, 1), (ParseStage.PARSING, 1, 1), (ParseStage.OCR, 1, 4), (ParseStage.OCR, 4, 4)]
    service = make_service(stub_factory(blocks=[block(LONG)], stages=stages))
    document = upload(make_docx_document(), "x.docx")

    def watch(stage: str, current: int, total: int) -> None:
        observed.append((stage, current, total, db_state(engine, document.id)["stage"]))

    service.process(document.id, on_progress=watch)

    assert [(s, c, t) for s, c, t, _ in observed] == [
        ("parsing", 0, 1), ("parsing", 1, 1), ("ocr", 1, 4), ("ocr", 4, 4), ("normalizing", 0, 1), ("normalizing", 1, 1),
    ]
    # Each update was already committed (visible to another connection) when the callback fired.
    assert [seen_stage for *_, seen_stage in observed] == ["parsing", "parsing", "ocr", "ocr", "normalizing", "normalizing"]


def test_library_shows_the_processing_status(upload, service, session) -> None:
    document = upload(make_docx_document(paragraphs=[LONG]), "x.docx")
    statuses = lambda: [d.processing_status for b in BookService(session).list_books() for d in b.documents]  # noqa: E731
    assert statuses() == ["uploaded"]

    service.process(document.id)
    session.rollback()

    assert statuses() == ["completed"]


# -- failures, error recording and retry ---------------------------------------------


def test_parser_error_fails_the_run_and_records_diagnostics(upload, make_service, session) -> None:
    error = CorruptDocumentError("The file is damaged.", details={"reason": "bad xref"})
    service = make_service(stub_factory(error=error, stages=[(ParseStage.PARSING, 0, 1)]))
    document = upload(make_docx_document(), "x.docx")

    run = service.process(document.id)

    assert run.status == "failed" and run.error_message == "The file is damaged."
    assert run.completed_at is not None and run.summary is None
    session.rollback()
    assert session.get(Document, document.id).processing_status == "failed"
    (saved,) = session.scalars(select(ProcessingError)).all()
    assert saved.error_type == "corrupt_document" and saved.error_message == "The file is damaged."
    assert saved.document_id == document.id and saved.processing_run_id == run.id
    assert saved.details == {"reason": "bad xref", "stage": "parsing"}
    assert blocks_of(session, document.id) == []


def test_real_corrupt_files_fail_cleanly_end_to_end(upload, service, session, docling_ready) -> None:
    for data, name, code in [
        (make_docx_with_broken_xml(), "broken.docx", "corrupt_document"),
        (make_corrupt_pdf(), "broken.pdf", "corrupt_document"),
    ]:
        document = upload(data, name)

        run = service.process(document.id)

        assert run.status == "failed"
        session.rollback()
        codes = [e.error_type for e in session.scalars(select(ProcessingError).where(ProcessingError.document_id == document.id))]
        assert codes == [code]
        assert blocks_of(session, document.id) == []


def test_unexpected_exception_is_recorded_with_traceback(upload, make_service, session) -> None:
    service = make_service(stub_factory(exception=RuntimeError("kaboom")))
    document = upload(make_docx_document(), "x.docx")

    run = service.process(document.id)

    assert run.status == "failed" and "RuntimeError" in run.error_message
    session.rollback()
    (saved,) = session.scalars(select(ProcessingError)).all()
    assert saved.error_type == "unexpected_error"
    assert saved.details["exception"] == "RuntimeError" and "kaboom" in saved.details["traceback"]
    assert session.get(Document, document.id).processing_status == "failed"


def test_missing_stored_file_is_recorded_as_a_storage_error(upload, service, storage, session) -> None:
    document = upload(make_docx_document(), "x.docx")
    storage.delete(document.file_path)

    run = service.process(document.id)

    assert run.status == "failed"
    session.rollback()
    (saved,) = session.scalars(select(ProcessingError)).all()
    assert saved.error_type == "storage_error"


def test_document_without_meaningful_content_is_not_marked_completed(upload, make_service, session) -> None:
    service = make_service(stub_factory(blocks=[block("Hi"), block("  ")]))
    document = upload(make_docx_document(), "x.docx")

    run = service.process(document.id)

    assert run.status == "failed"
    session.rollback()
    assert session.get(Document, document.id).processing_status == "failed"
    (saved,) = session.scalars(select(ProcessingError)).all()
    assert saved.error_type == "no_meaningful_content"
    assert saved.details["character_count"] == 2 and saved.details["minimum"] == 20
    assert blocks_of(session, document.id) == []


def test_unavailable_ocr_fails_the_run_in_the_ocr_stage_with_an_actionable_error(
    upload, make_service, settings, session, docling_ready
) -> None:
    engine = FakeOcrEngine(unavailable="Tesseract OCR is not installed. Run `brew install tesseract`.")
    service = make_service(ParserFactory(settings, ocr_engine=engine))
    document = upload(make_mixed_pdf([("text", [LONG]), ("scan", ["x"])]), "needs-ocr.pdf")

    run = service.process(document.id)

    assert run.status == "failed" and run.stage == "ocr"
    assert "brew install tesseract" in run.error_message
    session.rollback()
    (saved,) = session.scalars(select(ProcessingError)).all()
    assert saved.error_type == "ocr_unavailable" and saved.details["stage"] == "ocr"
    assert blocks_of(session, document.id) == []  # no partial result without the scanned pages

    # Once OCR works, the same document can be retried successfully.
    service.parser_factory = ParserFactory(settings, ocr_engine=FakeOcrEngine([ocr_result("Recovered page two text.")]))
    assert service.process(document.id).status == "completed"
    assert [b.page_number for b in blocks_of(session, document.id)] == [1, 2]


def test_encrypted_pdf_failure_has_a_clear_user_message(upload, make_service, settings, session) -> None:
    service = make_service(stub_factory(error=EncryptedDocumentError("The PDF is password-protected.")))
    document = upload(make_docx_document(), "x.docx")

    run = service.process(document.id)

    assert "password-protected" in run.error_message


def test_failed_document_can_be_retried_and_history_is_kept(upload, make_service, session) -> None:
    service = make_service(stub_factory(error=CorruptDocumentError("First attempt failed.")))
    document = upload(make_docx_document(), "x.docx")
    first = service.process(document.id)
    assert first.status == "failed"

    service.parser_factory = stub_factory(blocks=[block(LONG)])
    second = service.process(document.id)

    assert second.status == "completed" and second.id != first.id
    info = service.get_info(document.id)
    assert info.document_status == "completed" and info.run_count == 2
    assert info.latest_run.id == second.id
    assert [e.error_type for e in info.errors] == ["corrupt_document"]  # the first failure is preserved
    assert count(session, ProcessingRun) == 2
    assert len(blocks_of(session, document.id)) == 1


def test_retry_after_a_failure_does_not_duplicate_blocks(upload, make_service, session) -> None:
    document = upload(make_docx_document(), "x.docx")
    service = make_service(stub_factory(error=CorruptDocumentError("nope")))
    service.process(document.id)
    service.parser_factory = stub_factory(blocks=[block(LONG), block(LONG + " Again.")])

    service.process(document.id)

    assert [b.sequence_number for b in blocks_of(session, document.id)] == [0, 1]


def test_interruption_marks_the_run_failed_and_propagates(upload, make_service, session) -> None:
    class Interrupted(BaseException):
        pass

    service = make_service(stub_factory(exception=Interrupted()))
    document = upload(make_docx_document(), "x.docx")

    with pytest.raises(Interrupted):
        service.process(document.id)

    session.rollback()
    assert session.get(Document, document.id).processing_status == "failed"
    assert session.scalars(select(ProcessingRun.status)).one() == "failed"


# -- start rules ----------------------------------------------------------------------


def test_unknown_document_cannot_be_processed(service) -> None:
    with pytest.raises(DocumentNotFoundError):
        service.start(uuid.uuid4())
    with pytest.raises(DocumentNotFoundError):
        service.get_info(uuid.uuid4())


def test_completed_document_cannot_be_started_again(upload, make_service, session) -> None:
    service = make_service(stub_factory(blocks=[block(LONG)]))
    document = upload(make_docx_document(), "x.docx")
    service.process(document.id)

    with pytest.raises(ProcessingNotAllowedError, match="already been processed"):
        service.start(document.id)

    assert count(session, ProcessingRun) == 1


def test_document_already_being_processed_cannot_be_started_twice(upload, service, session) -> None:
    document = upload(make_docx_document(), "x.docx")
    service.start(document.id)

    with pytest.raises(ProcessingNotAllowedError, match="already being processed"):
        service.start(document.id)

    assert count(session, ProcessingRun) == 1


def test_a_run_cannot_be_executed_twice(upload, make_service) -> None:
    service = make_service(stub_factory(blocks=[block(LONG)]))
    document = upload(make_docx_document(), "x.docx")
    run_id = service.start(document.id)
    service.execute(run_id)

    with pytest.raises(ProcessingNotAllowedError, match="already been started"):
        service.execute(run_id)
    with pytest.raises(ProcessingServiceError):
        service.execute(uuid.uuid4())


@pytest.mark.parametrize("stuck_status", ["pending", "processing"])
def test_an_interrupted_run_is_recovered_and_can_be_retried(
    upload, make_service, session, settings, stuck_status: str
) -> None:
    service = make_service(stub_factory(blocks=[block(LONG)]))
    document = upload(make_docx_document(), "x.docx")
    stuck_id = service.start(document.id)
    old = utcnow() - timedelta(minutes=settings.processing_stale_minutes + 5)
    session.execute(
        update(ProcessingRun).where(ProcessingRun.id == stuck_id).values(status=stuck_status, created_at=old, started_at=old)
    )
    session.execute(update(Document).where(Document.id == document.id).values(processing_status=stuck_status))
    session.commit()
    assert service.get_info(document.id).is_stale is True

    run = service.process(document.id)

    assert run.status == "completed"
    session.rollback()
    stuck = session.get(ProcessingRun, stuck_id)
    assert stuck.status == "failed" and "interrupted" in stuck.error_message
    assert [e.error_type for e in session.scalars(select(ProcessingError))] == ["interrupted"]


def test_a_recent_run_is_not_treated_as_stale(upload, service) -> None:
    document = upload(make_docx_document(), "x.docx")
    service.start(document.id)

    info = service.get_info(document.id)

    assert info.document_status == "pending" and info.is_stale is False


# -- transactions and cleanup ---------------------------------------------------------


def test_failure_while_saving_leaves_no_partial_blocks(upload, make_service, session, monkeypatch) -> None:
    service = make_service(stub_factory(blocks=[block(LONG), block(LONG + " Two."), block(LONG + " Three.")]))
    original = service.blocks.replace_for_document

    def insert_then_crash(document_id, rows):  # type: ignore[no-untyped-def]
        original(document_id, rows)  # rows are inserted inside the open transaction...
        raise RuntimeError("database connection lost mid-save")  # ...then the save fails

    monkeypatch.setattr(service.blocks, "replace_for_document", insert_then_crash)
    document = upload(make_docx_document(), "x.docx")

    run = service.process(document.id)

    assert run.status == "failed"
    assert blocks_of(session, document.id) == []  # rolled back, nothing partial persisted
    session.rollback()
    assert session.get(Document, document.id).processing_status == "failed"
    assert [e.error_type for e in session.scalars(select(ProcessingError))] == ["unexpected_error"]


def test_a_database_constraint_violation_while_saving_rolls_everything_back(
    upload, make_service, session
) -> None:
    """Rows before the offending one are inserted inside the transaction, then must vanish."""
    blocks = [block(LONG, page_number=1), block(LONG + " Two.", page_number=1), block(LONG + " Bad.", page_number=0)]
    service = make_service(stub_factory(blocks=blocks))  # page_number 0 violates a CHECK constraint
    document = upload(make_docx_document(), "x.docx")

    run = service.process(document.id)

    assert run.status == "failed"
    assert blocks_of(session, document.id) == []
    session.rollback()
    assert session.get(Document, document.id).processing_status == "failed"
    assert [e.error_type for e in session.scalars(select(ProcessingError))] == ["unexpected_error"]


def test_stale_blocks_are_replaced_not_accumulated(upload, service, session) -> None:
    document = upload(make_docx_document(), "x.docx")
    rows = [{"sequence_number": 0, "block_type": "paragraph", "text": "old"}]
    service.blocks.replace_for_document(document.id, rows)
    service.blocks.replace_for_document(document.id, [{**rows[0], "text": "new"}])
    session.commit()

    assert [b.text for b in blocks_of(session, document.id)] == ["new"]


# -- reading results ------------------------------------------------------------------


def test_blocks_can_be_paged_and_filtered_by_page(upload, make_service, settings, session, docling_ready) -> None:
    pages = [[f"Content of original page {n} with enough words to be extracted."] for n in range(1, 5)]
    service = make_service(ParserFactory(settings, ocr_engine=FakeOcrEngine()))
    document = upload(make_text_pdf(pages, headings=False), "paged.pdf")
    service.process(document.id)

    assert service.count_blocks(document.id) == 4
    assert [b.page_number for b in service.list_blocks(document.id, limit=2)] == [1, 2]
    assert [b.page_number for b in service.list_blocks(document.id, offset=2, limit=10)] == [3, 4]
    only_three = service.list_blocks(document.id, page_number=3)
    assert [b.text for b in only_three] == ["Content of original page 3 with enough words to be extracted."]
    assert service.count_blocks(document.id, page_number=3) == 1


def test_get_info_for_an_unprocessed_document(upload, service) -> None:
    document = upload(make_docx_document(), "x.docx")

    info = service.get_info(document.id)

    assert info.document_status == "uploaded" and info.latest_run is None
    assert info.errors == [] and info.run_count == 0 and info.is_stale is False


def test_summary_round_trips_through_the_database(upload, make_service, session) -> None:
    service = make_service(stub_factory(blocks=[block(LONG, block_type=BlockType.HEADING)], page_count=7))
    document = upload(make_docx_document(), "x.docx")
    service.process(document.id)
    session.rollback()

    info = service.get_info(document.id)

    summary = info.latest_run.summary
    assert summary.page_count == 7 and summary.block_count == 1
    assert summary.character_count == len(LONG) and summary.status.value == "success"


# -- background job runner ------------------------------------------------------------


@pytest.fixture
def session_factory(engine: Engine):
    @contextmanager
    def _factory() -> Iterator[Session]:
        db_session = Session(bind=engine, autoflush=False, expire_on_commit=False)
        try:
            yield db_session
        finally:
            db_session.close()

    return _factory


def test_job_runner_processes_in_the_background(upload, session, session_factory, storage, settings, engine) -> None:
    runner = ProcessingJobRunner(
        settings, session_factory=session_factory, storage=storage, parser_factory=stub_factory(blocks=[block(LONG)])
    )
    document = upload(make_docx_document(), "x.docx")

    run_id = runner.submit(document.id)
    runner.shutdown(wait=True)

    state = db_state(engine, document.id)
    assert state["document_status"] == "completed" and state["run_status"] == "completed"
    session.rollback()
    assert session.get(ProcessingRun, run_id).status == "completed"
    assert len(blocks_of(session, document.id)) == 1


def test_job_runner_rejects_a_second_submit_while_the_first_is_running(
    upload, session_factory, storage, settings, engine
) -> None:
    started, release = threading.Event(), threading.Event()

    def hold(_path: Path) -> None:
        started.set()
        assert release.wait(timeout=20)

    runner = ProcessingJobRunner(
        settings,
        session_factory=session_factory,
        storage=storage,
        parser_factory=stub_factory(blocks=[block(LONG)], observer=hold),
    )
    document = upload(make_docx_document(), "x.docx")
    try:
        runner.submit(document.id)
        assert started.wait(timeout=20)  # the worker is now inside the parser: status is "processing"
        assert db_state(engine, document.id)["document_status"] == "processing"

        with pytest.raises(ProcessingNotAllowedError, match="already being processed"):
            runner.submit(document.id)
    finally:
        release.set()
        runner.shutdown(wait=True)

    assert db_state(engine, document.id)["document_status"] == "completed"


def test_job_runner_records_a_failure_without_raising(upload, session_factory, storage, settings, engine) -> None:
    runner = ProcessingJobRunner(
        settings, session_factory=session_factory, storage=storage,
        parser_factory=stub_factory(error=CorruptDocumentError("Damaged file.")),
    )
    document = upload(make_docx_document(), "x.docx")

    runner.submit(document.id)
    runner.shutdown(wait=True)

    state = db_state(engine, document.id)
    assert state["document_status"] == "failed" and state["run_status"] == "failed"
