from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
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
    from database.models.extracted_block import ExtractedBlock

# These must match app.parsing.equation.equation_models (a test enforces it).
EQUATION_TYPES = ("inline", "display", "numbered", "unknown")
VALIDATION_STATUSES = ("pending", "valid", "uncertain", "invalid")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


class Equation(UUIDPrimaryKeyMixin, Base):
    """A mathematical equation extracted from a document, with its source location.

    The model is deliberately open-ended: `equation_category` and the JSON `metadata` leave
    room for variable definitions, units, assumptions and executable forms later.
    """

    __tablename__ = "equations"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    # The text block this equation corresponds to, when one exists. Blocks are replaced when a
    # document is reprocessed, so the link is cleared (not the equation) if the block goes.
    extracted_block_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extracted_blocks.id", ondelete="SET NULL"), index=True
    )
    sequence_number: Mapped[int] = mapped_column(Integer)
    # Stable identity of this occurrence: document + source location + position + content.
    identity_hash: Mapped[str] = mapped_column(String(64))
    # Normalized LaTeX. Empty when recognition failed (the failure is then recorded, not hidden).
    latex: Mapped[str] = mapped_column(Text)
    # The representation as the extractor produced it (before normalization).
    original_latex: Mapped[str | None] = mapped_column(Text)
    plain_text: Mapped[str | None] = mapped_column(Text)
    equation_type: Mapped[str] = mapped_column(String(20), default="unknown")
    equation_category: Mapped[str | None] = mapped_column(String(100))
    equation_number: Mapped[str | None] = mapped_column(String(50), index=True)
    # Only set when the source has reliable pages (PDF). Never invented.
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(String(1000))
    surrounding_text: Mapped[str | None] = mapped_column(Text)
    source_location: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Storage key (relative to the storage root) of the image region used for recognition.
    crop_reference: Mapped[str | None] = mapped_column(String(1024))
    extraction_method: Mapped[str] = mapped_column(String(50))
    # Only what the recognition engine reported (0-1). NULL when it reported nothing.
    confidence: Mapped[float | None] = mapped_column(Float)
    validation_status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    validation_warnings: Mapped[list[str] | None] = mapped_column(JSONB)
    # "metadata" is reserved on declarative classes, hence the attribute name.
    equation_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="equations")
    extracted_block: Mapped[ExtractedBlock | None] = relationship()

    __table_args__ = (
        UniqueConstraint("document_id", "sequence_number", name="uq_equations_document_sequence"),
        UniqueConstraint("document_id", "identity_hash", name="uq_equations_document_identity"),
        Index("ix_equations_document_page", "document_id", "page_number"),
        CheckConstraint(_in_list("equation_type", EQUATION_TYPES), name="equation_type_valid"),
        CheckConstraint(
            _in_list("validation_status", VALIDATION_STATUSES), name="validation_status_valid"
        ),
        CheckConstraint("sequence_number >= 0", name="sequence_number_non_negative"),
        CheckConstraint("page_number IS NULL OR page_number >= 1", name="page_number_positive"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"
        ),
    )
