from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, UUIDPrimaryKeyMixin, utcnow

if TYPE_CHECKING:
    from database.models.document import Document

# Must match app.parsing.models.BlockType (a test enforces this).
BLOCK_TYPES = ("paragraph", "heading", "table", "list", "caption", "equation", "other")


class ExtractedBlock(UUIDPrimaryKeyMixin, Base):
    """A normalized piece of text extracted from a document, with its source location."""

    __tablename__ = "extracted_blocks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    sequence_number: Mapped[int] = mapped_column(Integer)
    block_type: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text)
    # Text as extracted, before normalization. NULL when normalization changed nothing.
    raw_text: Mapped[str | None] = mapped_column(Text)
    # Only set when the source format has reliable pages (PDF). Never invented.
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(String(1000))
    source_location: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # "metadata" is reserved on declarative classes, hence the attribute name.
    block_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="extracted_blocks")

    __table_args__ = (
        UniqueConstraint(
            "document_id", "sequence_number", name="uq_extracted_blocks_document_sequence"
        ),
        Index("ix_extracted_blocks_document_page", "document_id", "page_number"),
        CheckConstraint(
            "block_type IN (" + ", ".join(f"'{t}'" for t in BLOCK_TYPES) + ")",
            name="block_type_valid",
        ),
        CheckConstraint("sequence_number >= 0", name="sequence_number_non_negative"),
        CheckConstraint("page_number IS NULL OR page_number >= 1", name="page_number_positive"),
    )
