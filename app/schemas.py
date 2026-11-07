import re
import unicodedata
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

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
