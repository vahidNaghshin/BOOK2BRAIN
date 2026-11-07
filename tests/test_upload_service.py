import io
import threading
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.schemas import BookMetadata
from app.services.document_service import DocumentService
from app.services.errors import (
    DuplicateDocumentError,
    EmptyFileError,
    FileTooLargeError,
    InvalidFileError,
    StorageError,
    UnsupportedFileTypeError,
    UploadFailedError,
)
from app.services.storage_service import LocalStorage
from database.models import Book, Document
from tests.conftest import stored_files
from tests.factories import make_docx, make_epub, make_pdf


def upload(service: DocumentService, data: bytes, filename: str, metadata: BookMetadata, **kwargs):
    return service.upload(file=io.BytesIO(data), filename=filename, metadata=metadata, **kwargs)


def count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def assert_nothing_persisted(session: Session, storage_root: Path) -> None:
    session.rollback()
    assert count(session, Book) == 0
    assert count(session, Document) == 0
    assert stored_files(storage_root) == []


# --- success ---------------------------------------------------------------


def test_successful_pdf_upload(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    data = make_pdf("success")

    result = upload(document_service, data, "Dune (1965).pdf", metadata, declared_mime_type="application/pdf")

    assert result.book_created is True
    assert result.book.title == "Dune" and result.book.author == "Frank Herbert"
    document = result.document
    assert document.original_filename == "Dune (1965).pdf"
    assert document.file_format == "pdf"
    assert document.mime_type == "application/pdf"
    assert document.file_size == len(data)
    assert len(document.sha256_hash) == 64
    assert document.processing_status == "uploaded"
    assert document.book_id == result.book.id

    assert storage.exists(document.file_path)
    with storage.open(document.file_path) as stored:
        assert stored.read() == data

    session.expire_all()
    assert count(session, Book) == 1 and count(session, Document) == 1


@pytest.mark.parametrize(
    ("data", "filename", "expected_format"),
    [
        (make_pdf("p"), "book.pdf", "pdf"),
        (make_epub("e"), "book.epub", "epub"),
        (make_docx("d"), "book.docx", "docx"),
    ],
)
def test_all_supported_formats_upload(
    document_service: DocumentService, metadata: BookMetadata, data: bytes, filename: str, expected_format: str
) -> None:
    result = upload(document_service, data, filename, metadata)

    assert result.document.file_format == expected_format
    assert result.document.file_path.endswith(f".{expected_format}")


def test_all_metadata_fields_are_saved(document_service: DocumentService, session: Session) -> None:
    metadata = BookMetadata(
        title="Dune",
        subtitle="Book One",
        author="Frank Herbert",
        isbn="978-0-441-17271-9",
        language="en",
        description="Desert planet.",
    )

    result = upload(document_service, make_pdf("meta"), "dune.pdf", metadata)

    session.expire_all()
    book = session.get(Book, result.book.id)
    assert (book.title, book.subtitle, book.author) == ("Dune", "Book One", "Frank Herbert")
    assert (book.isbn, book.language, book.description) == ("9780441172719", "en", "Desert planet.")


def test_progress_callback_reports_stages_in_order(
    document_service: DocumentService, metadata: BookMetadata
) -> None:
    events: list[tuple[str, float]] = []

    upload(document_service, make_pdf("progress"), "dune.pdf", metadata, on_progress=lambda s, f: events.append((s, f)))

    fractions = [fraction for _, fraction in events]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert events[0][0] == "Validating file" and events[-1][0] == "Done"


def test_stored_file_uses_safe_name_not_user_filename(
    document_service: DocumentService, storage: LocalStorage, metadata: BookMetadata
) -> None:
    result = upload(document_service, make_pdf("safe"), "../../etc/evil name.pdf", metadata)

    document = result.document
    assert document.original_filename == "evil name.pdf"
    assert document.stored_filename == f"{document.id}.pdf"
    assert document.file_path == f"original/{document.id}/{document.id}.pdf"
    assert [p.relative_to(storage.root).as_posix() for p in stored_files(storage.root)] == [document.file_path]


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize("filename", ["notes.txt", "book.exe", "book.doc", "book"])
def test_unsupported_extension_is_rejected(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata, filename: str
) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        upload(document_service, make_pdf(), filename, metadata)

    assert_nothing_persisted(session, storage.root)


def test_oversized_file_is_rejected(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    too_big = b"%PDF-1.4\n" + b"x" * (1024 * 1024)  # test limit is 1 MB

    with pytest.raises(FileTooLargeError, match="1 MB"):
        upload(document_service, too_big, "big.pdf", metadata)

    assert_nothing_persisted(session, storage.root)


def test_file_exactly_at_limit_is_accepted(document_service: DocumentService, metadata: BookMetadata) -> None:
    header = b"%PDF-1.4\n"
    data = header + b"x" * (1024 * 1024 - len(header))

    assert upload(document_service, data, "exact.pdf", metadata).document.file_size == 1024 * 1024


def test_empty_file_is_rejected(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    with pytest.raises(EmptyFileError):
        upload(document_service, b"", "empty.pdf", metadata)

    assert_nothing_persisted(session, storage.root)


@pytest.mark.parametrize(
    ("data", "filename"),
    [
        (b"this is plain text pretending to be a pdf", "fake.pdf"),
        (make_epub(), "epub-as.docx"),
        (make_docx(), "docx-as.epub"),
        (make_pdf(), "pdf-as.epub"),
        (b"PK\x03\x04garbage", "corrupt.docx"),
    ],
    ids=["text-as-pdf", "epub-as-docx", "docx-as-epub", "pdf-as-epub", "corrupt-zip"],
)
def test_content_must_match_extension(
    document_service: DocumentService,
    session: Session,
    storage: LocalStorage,
    metadata: BookMetadata,
    data: bytes,
    filename: str,
) -> None:
    with pytest.raises(InvalidFileError):
        upload(document_service, data, filename, metadata)

    assert_nothing_persisted(session, storage.root)


def test_contradicting_mime_type_is_rejected(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    with pytest.raises(InvalidFileError):
        upload(document_service, make_pdf(), "book.pdf", metadata, declared_mime_type="image/png")

    assert_nothing_persisted(session, storage.root)


def test_correct_content_is_accepted_even_with_generic_mime_type(
    document_service: DocumentService, metadata: BookMetadata
) -> None:
    result = upload(document_service, make_epub(), "book.epub", metadata, declared_mime_type="application/octet-stream")

    assert result.document.mime_type == "application/epub+zip"


# --- duplicates ------------------------------------------------------------


def test_duplicate_file_is_rejected_without_new_records_or_copies(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    data = make_pdf("dup")
    first = upload(document_service, data, "dune.pdf", metadata)
    files_before = stored_files(storage.root)

    with pytest.raises(DuplicateDocumentError) as exc_info:
        upload(document_service, data, "renamed.pdf", metadata)

    error = exc_info.value
    assert error.existing.id == first.document.id
    assert error.book_title == "Dune"
    assert "already been uploaded" in str(error)
    assert count(session, Document) == 1 and count(session, Book) == 1
    assert stored_files(storage.root) == files_before


def test_duplicate_is_detected_even_under_different_book_metadata(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    data = make_pdf("dup-other-book")
    first = upload(document_service, data, "dune.pdf", metadata)

    with pytest.raises(DuplicateDocumentError):
        upload(document_service, data, "other.pdf", BookMetadata(title="Something Else"))

    session.rollback()
    assert count(session, Book) == 1  # no book created for the rejected duplicate
    assert session.get(Document, first.document.id) is not None


def test_concurrent_duplicate_is_translated_from_integrity_error(
    document_service: DocumentService,
    session: Session,
    storage: LocalStorage,
    metadata: BookMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If two uploads race past the pre-check, the unique hash still prevents a duplicate."""
    data = make_pdf("race")
    upload(document_service, data, "dune.pdf", metadata)
    files_before = stored_files(storage.root)
    monkeypatch.setattr(document_service, "_raise_if_duplicate", lambda _sha: None)

    with pytest.raises(DuplicateDocumentError):
        upload(document_service, data, "dune-again.pdf", metadata)

    assert count(session, Document) == 1
    assert stored_files(storage.root) == files_before


def test_parallel_uploads_of_same_file_store_exactly_one_copy(
    session: Session, engine, settings: Settings, storage: LocalStorage, metadata: BookMetadata
) -> None:
    data = make_pdf("parallel")
    outcomes: list[str] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        worker_session = Session(bind=engine, autoflush=False, expire_on_commit=False)
        service = DocumentService(worker_session, LocalStorage(settings.storage_root), settings)
        barrier.wait()
        try:
            upload(service, data, "dune.pdf", metadata)
            outcomes.append("ok")
        except DuplicateDocumentError:
            outcomes.append("duplicate")
        finally:
            worker_session.close()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcomes) == ["duplicate", "duplicate", "duplicate", "ok"]
    session.rollback()
    assert count(session, Document) == 1 and count(session, Book) == 1
    assert len(stored_files(storage.root)) == 1


# --- books and formats -----------------------------------------------------


def test_different_formats_attach_to_the_same_book(
    document_service: DocumentService, session: Session, metadata: BookMetadata
) -> None:
    pdf = upload(document_service, make_pdf("fmt"), "dune.pdf", metadata)
    epub = upload(document_service, make_epub("fmt"), "dune.epub", metadata)

    assert pdf.book_created is True and epub.book_created is False
    assert pdf.book.id == epub.book.id
    assert pdf.document.id != epub.document.id
    assert count(session, Book) == 1 and count(session, Document) == 2
    assert {d.file_format for d in epub.book.documents} == {"pdf", "epub"}


def test_book_matching_ignores_case_and_extra_whitespace(
    document_service: DocumentService, session: Session
) -> None:
    first = upload(document_service, make_pdf("case"), "a.pdf", BookMetadata(title="Dune", author="Frank Herbert"))
    second = upload(
        document_service, make_epub("case"), "a.epub", BookMetadata(title="  DUNE ", author="frank   herbert")
    )

    assert first.book.id == second.book.id


def test_matching_existing_book_does_not_overwrite_its_metadata(
    document_service: DocumentService, session: Session
) -> None:
    first = upload(
        document_service, make_pdf("keep"), "a.pdf", BookMetadata(title="Dune", author="Frank Herbert", language="en")
    )
    upload(
        document_service,
        make_epub("keep"),
        "a.epub",
        BookMetadata(title="Dune", author="Frank Herbert", language="fr", description="changed"),
    )

    session.expire_all()
    book = session.get(Book, first.book.id)
    assert book.language == "en" and book.description is None


def test_same_title_by_different_author_is_a_different_book(
    document_service: DocumentService, session: Session
) -> None:
    upload(document_service, make_pdf("a1"), "a.pdf", BookMetadata(title="Dune", author="Frank Herbert"))
    upload(document_service, make_pdf("a2"), "b.pdf", BookMetadata(title="Dune", author="Someone Else"))

    assert count(session, Book) == 2


# --- failure handling and cleanup ------------------------------------------


class FailingStorage:
    """Storage whose save fails after partially doing its job."""

    def save(self, key: str, source) -> int:
        raise StorageError("disk full")

    def delete(self, key: str) -> None: ...


def test_storage_failure_creates_no_database_records(
    session: Session, settings: Settings, storage: LocalStorage, metadata: BookMetadata
) -> None:
    service = DocumentService(session, FailingStorage(), settings)

    with pytest.raises(StorageError, match="disk full"):
        upload(service, make_pdf("storage-fail"), "dune.pdf", metadata)

    assert_nothing_persisted(session, storage.root)


def test_unexpected_storage_error_is_reported_and_leaves_no_records(
    session: Session, settings: Settings, storage: LocalStorage, metadata: BookMetadata
) -> None:
    class CrashingStorage(FailingStorage):
        def save(self, key: str, source) -> int:
            raise RuntimeError("kaboom")

    service = DocumentService(session, CrashingStorage(), settings)

    with pytest.raises(UploadFailedError):
        upload(service, make_pdf("crash"), "dune.pdf", metadata)

    assert_nothing_persisted(session, storage.root)


def test_commit_failure_removes_the_stored_file(
    document_service: DocumentService,
    session: Session,
    storage: LocalStorage,
    metadata: BookMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_commit() -> None:
        raise RuntimeError("database went away")

    monkeypatch.setattr(session, "commit", failing_commit)

    with pytest.raises(UploadFailedError):
        upload(document_service, make_pdf("commit-fail"), "dune.pdf", metadata)

    assert stored_files(storage.root) == []
    session.rollback()
    assert count(session, Document) == 0 and count(session, Book) == 0


def test_size_mismatch_after_save_is_rolled_back_and_cleaned_up(
    session: Session, settings: Settings, storage: LocalStorage, metadata: BookMetadata
) -> None:
    class ShortWriteStorage(LocalStorage):
        def save(self, key: str, source) -> int:
            super().save(key, source)
            return 1

    service = DocumentService(session, ShortWriteStorage(settings.storage_root), settings)

    with pytest.raises(StorageError, match="size"):
        upload(service, make_pdf("short"), "dune.pdf", metadata)

    assert_nothing_persisted(session, storage.root)


def test_failed_cleanup_is_logged_and_original_error_still_raised(
    session: Session,
    settings: Settings,
    storage: LocalStorage,
    metadata: BookMetadata,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class StubbornStorage(LocalStorage):
        def delete(self, key: str) -> None:
            raise StorageError("cannot delete")

    service = DocumentService(session, StubbornStorage(settings.storage_root), settings)
    monkeypatch.setattr(session, "commit", lambda: (_ for _ in ()).throw(RuntimeError("db down")))

    with pytest.raises(UploadFailedError):
        upload(service, make_pdf("stubborn"), "dune.pdf", metadata)

    assert "Orphaned file left in storage" in caplog.text


def test_service_is_usable_after_a_failed_upload(
    document_service: DocumentService, session: Session, metadata: BookMetadata
) -> None:
    with pytest.raises(EmptyFileError):
        upload(document_service, b"", "empty.pdf", metadata)

    result = upload(document_service, make_pdf("recover"), "dune.pdf", metadata)

    assert result.book_created is True
    assert count(session, Document) == 1
