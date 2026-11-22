from app.schemas import DocumentRead


class UploadError(Exception):
    """Base class for upload failures. The message is safe to show to users."""


class UnsupportedFileTypeError(UploadError):
    pass


class FileTooLargeError(UploadError):
    pass


class EmptyFileError(UploadError):
    pass


class InvalidFileError(UploadError):
    """The file content does not match its extension or declared type."""


class DuplicateDocumentError(UploadError):
    def __init__(self, existing: DocumentRead, book_title: str) -> None:
        self.existing = existing
        self.book_title = book_title
        super().__init__(
            f"This file has already been uploaded as '{existing.original_filename}' "
            f"for the book '{book_title}'. No new copy was saved."
        )


class StorageError(UploadError):
    pass


class UploadFailedError(UploadError):
    """Unexpected failure; nothing was saved. Details are in the logs."""


class ProcessingServiceError(Exception):
    """Base class for document-processing errors. The message is safe to show to users."""


class DocumentNotFoundError(ProcessingServiceError):
    pass


class ProcessingNotAllowedError(ProcessingServiceError):
    """The document is in a state that does not allow starting processing."""


class StructureServiceError(ProcessingServiceError):
    """Structure detection or passage generation could not run or failed."""


class StructureNotAllowedError(StructureServiceError):
    """The document is not in a state that allows structure detection (it must be processed)."""


class ExtractionRunError(Exception):
    """A knowledge extraction run could not be started or continued. The message is safe to show to users."""


class ExtractionScopeError(ExtractionRunError):
    """The requested book, chapter or section does not exist or is of the wrong kind."""


class KnowledgeNotFoundError(Exception):
    """The knowledge unit does not exist (any more). The message is safe to show to users."""


class InvalidReviewError(Exception):
    """The review status is not one of the allowed values."""


class RelationshipError(Exception):
    """A relationship cannot be stored or found. The message is safe to show to users."""
