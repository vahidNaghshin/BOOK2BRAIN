from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from database.models.book import Book
    from database.models.processing import ProcessingError, ProcessingRun


class ProcessingStatus(enum.StrEnum):
    UPLOADED = "uploaded"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def _in_list(column: str, values: type[enum.StrEnum]) -> str:
    quoted = ", ".join(f"'{member.value}'" for member in values)
    return f"{column} IN ({quoted})"


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An uploaded physical file belonging to a book."""

    __tablename__ = "documents"

    book_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(255))
    # Storage key relative to the configured storage root.
    file_path: Mapped[str] = mapped_column(String(1024), unique=True)
    file_format: Mapped[str] = mapped_column(String(16))
    mime_type: Mapped[str] = mapped_column(String(255))
    file_size: Mapped[int] = mapped_column(BigInteger)
    sha256_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    processing_status: Mapped[str] = mapped_column(
        String(20),
        default=ProcessingStatus.UPLOADED.value,
        server_default=ProcessingStatus.UPLOADED.value,
        index=True,
    )

    book: Mapped[Book] = relationship(back_populates="documents")
    processing_runs: Mapped[list[ProcessingRun]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProcessingRun.created_at",
    )
    processing_errors: Mapped[list[ProcessingError]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProcessingError.created_at",
    )

    __table_args__ = (
        CheckConstraint("file_size > 0", name="file_size_positive"),
        CheckConstraint(
            _in_list("processing_status", ProcessingStatus), name="processing_status_valid"
        ),
    )
