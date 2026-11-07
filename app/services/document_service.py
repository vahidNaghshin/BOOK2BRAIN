import logging
import uuid
from collections.abc import Callable
from typing import BinaryIO

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.repositories.document_repository import DocumentRepository
from app.schemas import BookMetadata, BookRead, DocumentRead, UploadResult
from app.services.book_service import BookService
from app.services.errors import (
    DuplicateDocumentError,
    InvalidFileError,
    StorageError,
    UploadError,
    UploadFailedError,
)
from app.services.file_validation import (
    check_declared_mime_type,
    detect_format,
    format_from_filename,
    measure_stream,
    sanitize_filename,
)
from app.services.storage_service import StorageBackend, build_original_key
from database.models import Document, ProcessingStatus

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


class DocumentService:
    def __init__(
        self,
        session: Session,
        storage: StorageBackend,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.storage = storage
        self.settings = settings or get_settings()
        self.books = BookService(session)
        self.documents = DocumentRepository(session)

    def upload(
        self,
        *,
        file: BinaryIO,
        filename: str,
        metadata: BookMetadata,
        declared_mime_type: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> UploadResult:
        """Validate and store an uploaded book file and record it in the database.

        `file` must be seekable. On any failure nothing is persisted: no database
        rows and no stored file.
        """
        progress = on_progress or (lambda _stage, _fraction: None)

        progress("Validating file", 0.1)
        safe_name = sanitize_filename(filename)
        expected_format = format_from_filename(safe_name)
        check_declared_mime_type(expected_format, declared_mime_type)
        sha256, size = measure_stream(file, self.settings.max_upload_size_bytes)
        detected_format = detect_format(file)
        if detected_format != expected_format:
            raise InvalidFileError(
                f"The file content looks like {detected_format.extension}, "
                f"but its extension is {expected_format.extension}."
            )

        document_id = uuid.uuid4()
        stored_filename = f"{document_id}{detected_format.extension}"
        key = build_original_key(document_id, stored_filename)
        stored = False

        try:
            progress("Checking for duplicates", 0.3)
            self._raise_if_duplicate(sha256)

            progress("Saving book record", 0.45)
            book, book_created = self.books.get_or_create(metadata)
            document = self.documents.add(
                Document(
                    id=document_id,
                    book_id=book.id,
                    original_filename=safe_name,
                    stored_filename=stored_filename,
                    file_path=key,
                    file_format=detected_format.value,
                    mime_type=detected_format.mime_type,
                    file_size=size,
                    sha256_hash=sha256,
                    processing_status=ProcessingStatus.UPLOADED.value,
                )
            )

            progress("Storing file", 0.6)
            file.seek(0)
            written = self.storage.save(key, file)
            stored = True
            if written != size:
                raise StorageError("The stored file size does not match the uploaded file.")

            progress("Finalizing", 0.9)
            self.session.commit()
        except Exception as exc:
            self.session.rollback()
            if stored:
                self._discard_stored_file(key)
            translated = self._translate(exc, sha256)
            if translated is exc:
                raise
            raise translated from exc

        logger.info(
            "Uploaded document %s (%s, %d bytes, sha256=%s) for book %s",
            document.id, detected_format.value, size, sha256, book.id,
        )
        progress("Done", 1.0)
        return UploadResult(
            book=BookRead.model_validate(book),
            document=DocumentRead.model_validate(document),
            book_created=book_created,
        )

    def _raise_if_duplicate(self, sha256: str) -> None:
        existing = self.documents.get_by_hash(sha256)
        if existing is not None:
            logger.info("Rejected duplicate upload of document %s", existing.id)
            raise DuplicateDocumentError(
                DocumentRead.model_validate(existing), existing.book.title
            )

    def _discard_stored_file(self, key: str) -> None:
        try:
            self.storage.delete(key)
        except StorageError:
            logger.exception("Orphaned file left in storage after failed upload: %s", key)

    def _translate(self, exc: Exception, sha256: str) -> UploadError:
        """Map an internal failure to a user-facing UploadError (never swallowing it)."""
        if isinstance(exc, UploadError):
            if not isinstance(exc, DuplicateDocumentError):
                logger.warning("Upload rejected: %s", exc)
            return exc
        if isinstance(exc, IntegrityError):
            # A concurrent upload of the same file won the race for the unique hash.
            existing = self.documents.get_by_hash(sha256)
            if existing is not None:
                logger.info("Concurrent duplicate upload detected for sha256=%s", sha256)
                return DuplicateDocumentError(
                    DocumentRead.model_validate(existing), existing.book.title
                )
        logger.exception("Upload failed unexpectedly")
        return UploadFailedError("The upload failed and nothing was saved. Please try again.")
