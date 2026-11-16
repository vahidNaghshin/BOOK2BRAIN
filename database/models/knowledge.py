from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, utcnow
from database.models.structure import PAGE_STATUSES

# These must match app.knowledge.types (a test enforces it).
KNOWLEDGE_TYPES = ("concept", "principle", "method", "framework", "evidence", "example", "equation")
KNOWLEDGE_VALIDATION_STATUSES = ("pending", "valid", "uncertain", "invalid")
REVIEW_STATUSES = ("unreviewed", "approved", "rejected", "needs_changes")
RELATIONSHIP_TYPES = (
    "supports", "illustrates", "implements", "describes", "part_of", "depends_on", "contradicts", "related_to",
)
EXTRACTION_RUN_STATUSES = ("pending", "processing", "completed", "failed", "cancelled")
SOURCE_ROLES = ("primary", "supporting")

# Version of the typed content stored in KnowledgeUnit.details (app.knowledge.schemas).
CONTENT_SCHEMA_VERSION = 1


def _in_list(column: str, values: Iterable[str]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


class ExtractionRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One execution of knowledge extraction over a book, or over one document or section of it.

    Independent of the Phase 1 ProcessingRun. Knowledge units keep their link to the run that
    produced them, but outlive it: deleting a run never deletes knowledge.
    """

    __tablename__ = "extraction_runs"

    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    # Restricts the run to a chapter or section. Cleared if the structure is regenerated.
    structural_node_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("structural_nodes.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending")
    passages_processed: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    units_extracted: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    failure_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # Free text: nothing here depends on a particular LLM provider.
    model_name: Mapped[str | None] = mapped_column(String(200))
    prompt_version: Mapped[str | None] = mapped_column(String(100))
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    knowledge_units: Mapped[list[KnowledgeUnit]] = relationship(
        back_populates="extraction_run", passive_deletes=True
    )

    __table_args__ = (
        CheckConstraint(_in_list("status", EXTRACTION_RUN_STATUSES), name="status_valid"),
        CheckConstraint(
            "passages_processed >= 0 AND units_extracted >= 0 AND failure_count >= 0", name="counts_non_negative"
        ),
        CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at", name="completion_order"
        ),
        Index("ix_extraction_runs_book_status", "book_id", "status"),
    )


class KnowledgeUnit(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One piece of extracted knowledge of any category (concept, principle, method, ...).

    `details` holds the complete typed content, validated by the Pydantic model of the unit's
    knowledge type (app.knowledge.schemas). `title`, `content` and `summary` are projections of
    it, kept as columns so they can be listed, searched and indexed without opening the JSON.

    A unit is backed by any number of passages (KnowledgeUnitPassage). The page range is stored
    on the unit as well, so provenance survives when passages are regenerated.
    """

    __tablename__ = "knowledge_units"

    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="SET NULL"), index=True
    )
    # Where the knowledge sits in the book. Structure is regenerable, so these are cleared (not
    # the unit) if the node goes.
    chapter_node_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("structural_nodes.id", ondelete="SET NULL"), index=True
    )
    section_node_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("structural_nodes.id", ondelete="SET NULL"), index=True
    )
    # The Phase 1 equation record an equation unit was built from, when there is one.
    equation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("equations.id", ondelete="SET NULL"), index=True
    )

    knowledge_type: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB)
    schema_version: Mapped[int] = mapped_column(Integer, default=CONTENT_SCHEMA_VERSION, server_default="1")

    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    page_status: Mapped[str] = mapped_column(String(10), default="unknown", server_default="unknown")

    confidence: Mapped[float | None] = mapped_column(Float)
    is_uncertain: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    uncertainty_notes: Mapped[str | None] = mapped_column(Text)
    validation_status: Mapped[str] = mapped_column(String(20), default="pending", server_default="pending")
    review_status: Mapped[str] = mapped_column(String(20), default="unreviewed", server_default="unreviewed")
    # Extra structured information that is not part of the typed content (extractor notes, tags).
    unit_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    extraction_run: Mapped[ExtractionRun | None] = relationship(back_populates="knowledge_units")
    sources: Mapped[list[KnowledgeUnitPassage]] = relationship(
        back_populates="knowledge_unit",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="KnowledgeUnitPassage.position",
    )
    outgoing_relationships: Mapped[list[KnowledgeRelationship]] = relationship(
        primaryjoin="KnowledgeUnit.id == KnowledgeRelationship.source_unit_id",
        foreign_keys="[KnowledgeRelationship.source_unit_id]",
        viewonly=True,
    )
    incoming_relationships: Mapped[list[KnowledgeRelationship]] = relationship(
        primaryjoin="KnowledgeUnit.id == KnowledgeRelationship.target_unit_id",
        foreign_keys="[KnowledgeRelationship.target_unit_id]",
        viewonly=True,
    )

    __table_args__ = (
        # Lets relationships reference (unit, book) so both ends are guaranteed to share a book.
        UniqueConstraint("id", "book_id", name="uq_knowledge_units_id_book_id"),
        Index("ix_knowledge_units_book_type", "book_id", "knowledge_type"),
        Index("ix_knowledge_units_book_review", "book_id", "review_status"),
        CheckConstraint(_in_list("knowledge_type", KNOWLEDGE_TYPES), name="knowledge_type_valid"),
        CheckConstraint(_in_list("validation_status", KNOWLEDGE_VALIDATION_STATUSES), name="validation_status_valid"),
        CheckConstraint(_in_list("review_status", REVIEW_STATUSES), name="review_status_valid"),
        CheckConstraint(_in_list("page_status", PAGE_STATUSES), name="page_status_valid"),
        CheckConstraint("length(btrim(title)) > 0", name="title_not_blank"),
        CheckConstraint("length(btrim(content)) > 0", name="content_not_blank"),
        CheckConstraint("jsonb_typeof(details) = 'object'", name="details_is_object"),
        CheckConstraint("schema_version >= 1", name="schema_version_positive"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"),
        CheckConstraint("page_start IS NULL OR page_start >= 1", name="page_start_positive"),
        CheckConstraint(
            "page_end IS NULL OR page_start IS NULL OR page_end >= page_start", name="page_order"
        ),
        CheckConstraint("uncertainty_notes IS NULL OR is_uncertain", name="uncertainty_notes_need_flag"),
        CheckConstraint("equation_id IS NULL OR knowledge_type = 'equation'", name="equation_link_type"),
    )


class KnowledgeUnitPassage(Base):
    """A passage that supports a knowledge unit. A unit has one or more of them."""

    __tablename__ = "knowledge_unit_passages"

    knowledge_unit_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_units.id", ondelete="CASCADE"), primary_key=True
    )
    # Passages are regenerable; if one is replaced the link goes, the knowledge unit stays.
    passage_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("passages.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(10), default="supporting", server_default="supporting")
    # The exact words in the passage that back the unit, when the extractor reports them.
    quote: Mapped[str | None] = mapped_column(Text)

    knowledge_unit: Mapped[KnowledgeUnit] = relationship(back_populates="sources")

    __table_args__ = (
        UniqueConstraint("knowledge_unit_id", "position", name="uq_knowledge_unit_passages_position"),
        Index(
            "uq_knowledge_unit_passages_primary",
            "knowledge_unit_id",
            unique=True,
            postgresql_where=text("role = 'primary'"),
        ),
        CheckConstraint("position >= 0", name="position_non_negative"),
        CheckConstraint(_in_list("role", SOURCE_ROLES), name="role_valid"),
    )


class KnowledgeRelationship(UUIDPrimaryKeyMixin, Base):
    """A typed, directed link between two knowledge units of the same book.

    Read as "source <relationship_type> target": an evidence unit *supports* a principle, an
    example *illustrates* a method, a method *implements* a framework, an equation *describes*
    a concept.
    """

    __tablename__ = "knowledge_relationships"

    book_id: Mapped[uuid.UUID] = mapped_column(index=True)
    source_unit_id: Mapped[uuid.UUID] = mapped_column()
    target_unit_id: Mapped[uuid.UUID] = mapped_column(index=True)
    relationship_type: Mapped[str] = mapped_column(String(20))
    explanation: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    # Read-only: the ids are set explicitly (both ends share book_id, which an ORM sync would fight over).
    source: Mapped[KnowledgeUnit] = relationship(
        primaryjoin="KnowledgeRelationship.source_unit_id == KnowledgeUnit.id",
        foreign_keys="[KnowledgeRelationship.source_unit_id]",
        viewonly=True,
    )
    target: Mapped[KnowledgeUnit] = relationship(
        primaryjoin="KnowledgeRelationship.target_unit_id == KnowledgeUnit.id",
        foreign_keys="[KnowledgeRelationship.target_unit_id]",
        viewonly=True,
    )
    sources: Mapped[list[KnowledgeRelationshipPassage]] = relationship(
        back_populates="knowledge_relationship",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="KnowledgeRelationshipPassage.position",
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["source_unit_id", "book_id"],
            ["knowledge_units.id", "knowledge_units.book_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["target_unit_id", "book_id"],
            ["knowledge_units.id", "knowledge_units.book_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "source_unit_id", "target_unit_id", "relationship_type", name="uq_knowledge_relationships_edge"
        ),
        CheckConstraint(_in_list("relationship_type", RELATIONSHIP_TYPES), name="relationship_type_valid"),
        CheckConstraint("source_unit_id <> target_unit_id", name="no_self_relationship"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"),
    )


class KnowledgeRelationshipPassage(Base):
    """A passage that shows a relationship (where the book connects the two units)."""

    __tablename__ = "knowledge_relationship_passages"

    relationship_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_relationships.id", ondelete="CASCADE"), primary_key=True
    )
    passage_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("passages.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    position: Mapped[int] = mapped_column(Integer)

    knowledge_relationship: Mapped[KnowledgeRelationship] = relationship(back_populates="sources")

    __table_args__ = (
        UniqueConstraint("relationship_id", "position", name="uq_knowledge_relationship_passages_position"),
        CheckConstraint("position >= 0", name="position_non_negative"),
    )
