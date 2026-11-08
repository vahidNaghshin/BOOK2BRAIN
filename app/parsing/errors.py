from typing import Any

from app.parsing.models import ParsingIssue


class ParserError(Exception):
    """A problem with the document itself. Parsers turn these into a failed ParsingResult."""

    code = "parser_error"

    def __init__(
        self,
        message: str,
        *,
        page_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.page_number = page_number
        self.details = details or {}

    def to_issue(self) -> ParsingIssue:
        return ParsingIssue(
            code=self.code,
            message=self.message,
            page_number=self.page_number,
            details=self.details,
        )


class CorruptDocumentError(ParserError):
    code = "corrupt_document"


class EncryptedDocumentError(ParserError):
    code = "encrypted_document"


class UnsupportedDocumentError(ParserError):
    code = "unsupported_document"


class EmptyDocumentError(ParserError):
    code = "empty_document"


class OcrUnavailableError(ParserError):
    """The OCR engine cannot be used at all (not installed, language missing)."""

    code = "ocr_unavailable"


class OcrPageError(Exception):
    """OCR failed for a single page; other pages can still be processed."""
