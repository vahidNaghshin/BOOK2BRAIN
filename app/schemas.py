import re
import unicodedata
import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from app.parsing.models import ExtractionStatus, OcrSummary, ParsingIssue

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Collapse whitespace and normalise unicode so identical-looking text compares equal."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", value)).strip()


def _isbn_checksum_ok(isbn: str) -> bool:
    if len(isbn) == 10:
        if not re.fullmatch(r"\d{9}[\dX]", isbn):
            return False
        total = sum((10 - i) * (10 if c == "X" else int(c)) for i, c in enumerate(isbn))
        return total % 11 == 0
    if len(isbn) == 13:
        if not isbn.isdigit():
            return False
        total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(isbn))
        return total % 10 == 0
    return False


class BookMetadata(BaseModel):
    """User-supplied metadata for a book. Blank optional fields become None."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    author: Annotated[str, StringConstraints(max_length=500)] | None = None
    subtitle: Annotated[str, StringConstraints(max_length=500)] | None = None
    isbn: str | None = None
    language: Annotated[str, StringConstraints(max_length=64)] | None = None
    description: Annotated[str, StringConstraints(max_length=20000)] | None = None

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: object) -> object:
        return normalize_text(value) if isinstance(value, str) else value

    @field_validator("author", "subtitle", mode="before")
    @classmethod
    def _clean_names(cls, value: object) -> object:
        return normalize_text(value) if isinstance(value, str) else value

    @field_validator("author", "subtitle", "isbn", "language", "description", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("isbn")
    @classmethod
    def _validate_isbn(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"[\s-]", "", value).upper()
        if not _isbn_checksum_ok(cleaned):
            raise ValueError("ISBN must be a valid ISBN-10 or ISBN-13")
        return cleaned


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    book_id: uuid.UUID
    original_filename: str
    stored_filename: str
    file_path: str
    file_format: str
    mime_type: str
    file_size: int
    sha256_hash: str
    processing_status: str
    created_at: datetime
    updated_at: datetime


class BookRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    subtitle: str | None
    author: str | None
    description: str | None
    language: str | None
    isbn: str | None
    created_at: datetime
    updated_at: datetime
    documents: list[DocumentRead] = Field(default_factory=list)


class UploadResult(BaseModel):
    book: BookRead
    document: DocumentRead
    book_created: bool


class ProcessingSummary(BaseModel):
    """What a completed run extracted. Stored as JSON on the processing run."""

    extraction_method: str
    status: ExtractionStatus
    extracted_at: datetime
    page_count: int | None = None
    character_count: int
    block_count: int
    ocr: OcrSummary | None = None
    warnings: list[ParsingIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessingRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    status: str
    stage: str | None
    progress_current: int | None
    progress_total: int | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    created_at: datetime
    # Equation extraction has its own status; it never changes the run or document status.
    equation_status: str = "not_started"
    equation_summary: dict[str, Any] | None = None
    summary: ProcessingSummary | None = None


class ProcessingErrorRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    processing_run_id: uuid.UUID | None
    error_type: str
    error_message: str
    details: dict[str, Any] | None
    created_at: datetime


class ProcessingInfo(BaseModel):
    document_id: uuid.UUID
    document_status: str
    latest_run: ProcessingRunRead | None = None
    errors: list[ProcessingErrorRead] = Field(default_factory=list)
    run_count: int = 0
    is_stale: bool = False
    equation_is_stale: bool = False


class ExtractedBlockRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    sequence_number: int
    block_type: str
    text: str
    raw_text: str | None
    page_number: int | None
    section_title: str | None
    source_location: dict[str, Any] | None
    # The ORM attribute is `block_metadata` ("metadata" is reserved on declarative classes).
    metadata: dict[str, Any] | None = Field(
        default=None, validation_alias=AliasChoices("block_metadata", "metadata")
    )
    created_at: datetime


class EquationRead(BaseModel):
    """A stored equation, for display. `latex` is normalized; `original_latex` is as extracted."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    document_id: uuid.UUID
    extracted_block_id: uuid.UUID | None
    sequence_number: int
    latex: str
    original_latex: str | None
    plain_text: str | None
    equation_type: str
    equation_category: str | None
    equation_number: str | None
    page_number: int | None
    section_title: str | None
    surrounding_text: str | None
    source_location: dict[str, Any] | None
    bounding_box: dict[str, Any] | None
    crop_reference: str | None
    extraction_method: str
    confidence: float | None
    validation_status: str
    validation_warnings: list[str] | None
    metadata: dict[str, Any] | None = Field(
        default=None, validation_alias=AliasChoices("equation_metadata", "metadata")
    )
    created_at: datetime
