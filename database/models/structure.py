from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
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

from database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utcnow

if TYPE_CHECKING:
    from database.models.book import Book
    from database.models.document import Document

# Where a page range comes from. Never invented: "unknown" means the format has no reliable
# pages (EPUB, DOCX); "partial" means only some of the source blocks carried a page.
PAGE_STATUSES = ("exact", "partial", "unknown")


class StructureStatus(enum.StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class StructuralNode(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One element of a document's logical structure: front/back matter, part, chapter, section...

    Nodes belong to a *document* (not only the logical book) because different editions have
    different pagination and structure. `node_type` is deliberately not a database enum so new
    types can be added without a migration; unknown headings use "other_heading".
    """

    __tablename__ = "structural_nodes"

    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("structural_nodes.id", ondelete="CASCADE"), index=True
    )
    node_type: Mapped[str] = mapped_column(String(40))
    # Normalised title (without "Chapter 3:"-style labels); original_title is the heading as written.
    title: Mapped[str] = mapped_column(String(1000))
    original_title: Mapped[str | None] = mapped_column(String(1000))
    number: Mapped[str | None] = mapped_column(String(50))
    # Depth in the tree: 0 for top-level nodes, parent.level + 1 otherwise.
    level: Mapped[int] = mapped_column(Integer)
    # Position in document order, over all nodes of the document.
    sequence: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    # Offsets into the document's canonical text (the extracted blocks joined by a blank line).
    start_offset: Mapped[int | None] = mapped_column(Integer)
    end_offset: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)
    # "metadata" is reserved on declarative classes, hence the attribute name.
    node_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)

    document: Mapped[Document] = relationship(back_populates="structural_nodes")
    book: Mapped[Book] = relationship()
    parent: Mapped[StructuralNode | None] = relationship(
        back_populates="children", remote_side="StructuralNode.id"
    )
    children: Mapped[list[StructuralNode]] = relationship(
        back_populates="parent",
        passive_deletes=True,
        order_by="StructuralNode.sequence",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "sequence", name="uq_structural_nodes_document_sequence"),
        Index("ix_structural_nodes_document_type", "document_id", "node_type"),
        Index("ix_structural_nodes_document_parent", "document_id", "parent_id"),
        CheckConstraint("level >= 0", name="level_non_negative"),
        CheckConstraint("sequence >= 0", name="sequence_non_negative"),
        CheckConstraint("page_start IS NULL OR page_start >= 1", name="page_start_positive"),
        CheckConstraint(
            "page_end IS NULL OR page_start IS NULL OR page_end >= page_start", name="page_order"
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"
        ),
    )


class Paragraph(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A paragraph-level unit of source text, kept independently of processing passages."""

    __tablename__ = "paragraphs"

    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    structural_node_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("structural_nodes.id", ondelete="CASCADE")
    )
    # Position inside the structural node, and over the whole document.
    sequence: Mapped[int] = mapped_column(Integer)
    document_sequence: Mapped[int] = mapped_column(Integer)
    # paragraph, list, table, equation, caption, heading (a rejected heading kept as text), other
    kind: Mapped[str] = mapped_column(String(20), default="paragraph")
    text: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    page_status: Mapped[str] = mapped_column(String(10), default="unknown")
    start_offset: Mapped[int | None] = mapped_column(Integer)
    end_offset: Mapped[int | None] = mapped_column(Integer)
    # Source blocks, bounding boxes, equation/table/figure references.
    source_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)

    structural_node: Mapped[StructuralNode] = relationship()

    __table_args__ = (
        UniqueConstraint("structural_node_id", "sequence", name="uq_paragraphs_node_sequence"),
        UniqueConstraint(
            "document_id", "document_sequence", name="uq_paragraphs_document_sequence"
        ),
        CheckConstraint("page_status IN ('exact', 'partial', 'unknown')", name="page_status_valid"),
        CheckConstraint("page_start IS NULL OR page_start >= 1", name="page_start_positive"),
        CheckConstraint(
            "page_end IS NULL OR page_start IS NULL OR page_end >= page_start", name="page_order"
        ),
    )


class Passage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A processing unit derived from paragraphs, sized for downstream AI work.

    Passages are regenerable: changing the size settings replaces them without touching the
    structural nodes or paragraphs they are derived from.
    """

    __tablename__ = "passages"

    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    structural_node_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("structural_nodes.id", ondelete="CASCADE")
    )
    text: Mapped[str] = mapped_column(Text)
    # Position inside the structural node, and over the whole document.
    sequence: Mapped[int] = mapped_column(Integer)
    document_sequence: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    page_status: Mapped[str] = mapped_column(String(10), default="unknown")
    token_count: Mapped[int] = mapped_column(Integer)
    start_offset: Mapped[int | None] = mapped_column(Integer)
    end_offset: Mapped[int | None] = mapped_column(Integer)
    # Overlap with the previous passage, referenced blocks, equations, tables, figures, bounding boxes.
    passage_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)

    structural_node: Mapped[StructuralNode] = relationship()
    paragraph_links: Mapped[list[PassageParagraph]] = relationship(
        back_populates="passage",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="PassageParagraph.position",
    )

    __table_args__ = (
        UniqueConstraint("structural_node_id", "sequence", name="uq_passages_node_sequence"),
        UniqueConstraint("document_id", "document_sequence", name="uq_passages_document_sequence"),
        CheckConstraint("page_status IN ('exact', 'partial', 'unknown')", name="page_status_valid"),
        CheckConstraint("token_count >= 0", name="token_count_non_negative"),
        CheckConstraint("page_start IS NULL OR page_start >= 1", name="page_start_positive"),
        CheckConstraint(
            "page_end IS NULL OR page_start IS NULL OR page_end >= page_start", name="page_order"
        ),
    )


class PassageParagraph(Base):
    """Which paragraphs a passage was built from (many-to-many, ordered).

    `is_overlap` marks paragraphs present only because the passage repeats the end of the previous
    one. `text_start`/`text_end` give the character range used when a long paragraph was split
    or only its last sentences were repeated; both are NULL when the whole paragraph is used.
    """

    __tablename__ = "passage_paragraphs"

    passage_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("passages.id", ondelete="CASCADE"), primary_key=True
    )
    # A paragraph can appear twice in one passage (the repeated end of a long paragraph and its
    # next part), so the position, not the paragraph, completes the key.
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    paragraph_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("paragraphs.id", ondelete="CASCADE"), index=True
    )
    is_overlap: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    text_start: Mapped[int | None] = mapped_column(Integer)
    text_end: Mapped[int | None] = mapped_column(Integer)

    passage: Mapped[Passage] = relationship(back_populates="paragraph_links")

    __table_args__ = (
        CheckConstraint("position >= 0", name="position_non_negative"),
        CheckConstraint(
            "text_start IS NULL OR text_end IS NULL OR text_end > text_start", name="text_range_valid"
        ),
    )


class StructureAnalysis(UUIDPrimaryKeyMixin, Base):
    """Status, summary and validation findings of the structure and passage steps of a document.

    One row per document. Structure detection and passage generation are tracked separately
    because passages can be regenerated without repeating detection.
    """

    __tablename__ = "structure_analyses"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), unique=True
    )
    structure_status: Mapped[str | None] = mapped_column(String(20))
    structure_error: Mapped[str | None] = mapped_column(Text)
    structure_detected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    structure_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    structure_issues: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    passage_status: Mapped[str | None] = mapped_column(String(20))
    passage_error: Mapped[str | None] = mapped_column(Text)
    passage_generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    passage_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    passage_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    passage_issues: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="structure_analysis")

    __table_args__ = (
        CheckConstraint(
            "structure_status IS NULL OR structure_status IN ('completed', 'failed')",
            name="structure_status_valid",
        ),
        CheckConstraint(
            "passage_status IS NULL OR passage_status IN ('completed', 'failed')",
            name="passage_status_valid",
        ),
    )
