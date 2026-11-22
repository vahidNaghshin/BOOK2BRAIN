from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
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
    "derived_from", "example_of", "applies_to",
)
RELATIONSHIP_ORIGINS = ("extracted", "ai_suggested", "manual")
KNOWLEDGE_ORIGINS = ("ai_extracted", "ai_interpretation", "ai_reconstructed", "source_derived", "human")
SOURCE_VERIFICATION_STATUSES = ("unverified", "verified", "invalid")
SOURCE_REFERENCE_TYPES = ("text", "equation", "figure", "table", "other")
EQUATION_CAPTURE_STATUSES = ("complete", "incomplete", "ambiguous", "image_only", "text_only", "needs_review")
MATH_VALIDATION_STATUSES = ("not_validated", "valid", "invalid", "inconclusive")
REVIEW_FLAG_CODES = (
    "invalid_source_reference", "source_quote_mismatch", "page_inconsistent", "no_source", "incomplete_equation",
    "ambiguous_equation", "equation_needs_review", "low_confidence", "uncertain_extraction", "inferred_content",
    "unsupported_claim", "potential_duplicate", "repeated_validation_failure", "unresolved_extraction_error",
    "ai_suggested_relationship", "context_reduced",
)
FLAG_SEVERITIES = ("info", "warning", "error")
FLAG_STATUSES = ("open", "resolved", "dismissed")
PASSAGE_STATUSES = ("pending", "processing", "completed", "failed", "needs_review")
VALIDATION_OUTCOMES = (
    "schema_valid", "schema_invalid", "source_invalid", "semantic_concerns", "unsupported_content", "duplicate",
    "needs_review", "not_validated",
)
ATTEMPT_STATUSES = ("completed", "provider_error")
# What an execution can be about: a knowledge category, or the relationships between units of a passage.
EXECUTION_KINDS = KNOWLEDGE_TYPES + ("relationship",)
EXTRACTION_RUN_STATUSES = ("pending", "processing", "completed", "needs_review", "failed", "cancelled")
EXTRACTION_SCOPES = ("book", "document", "chapter", "section")
SOURCE_ROLES = ("primary", "supporting")
EXECUTION_STATUSES = ("succeeded", "needs_review", "failed")
ATTEMPT_OUTCOMES = ("valid", "invalid", "provider_error")
ATTEMPT_KINDS = ("initial", "correction", "retry")

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
    # What the run covers: the book, one document, or a chapter/section (structural_node_id).
    scope_type: Mapped[str | None] = mapped_column(String(20))
    # The model configuration chosen for the run, as it was when the run started. NULL when
    # categories use different models (see `parameters`). Holds no credentials.
    model_configuration_id: Mapped[str | None] = mapped_column(String(100))
    provider: Mapped[str | None] = mapped_column(String(50))
    model_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Progress and totals. `passages_processed` counts passages with at least one finished extraction.
    total_passages: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    needs_review_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    skipped_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    duplicate_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    relationships_extracted: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    validation_failure_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)

    knowledge_units: Mapped[list[KnowledgeUnit]] = relationship(
        back_populates="extraction_run", passive_deletes=True
    )
    executions: Mapped[list[ExtractionExecution]] = relationship(
        back_populates="extraction_run", cascade="all, delete-orphan", passive_deletes=True
    )
    passage_records: Mapped[list[ExtractionPassageRecord]] = relationship(
        back_populates="extraction_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ExtractionPassageRecord.document_sequence",
    )

    __table_args__ = (
        CheckConstraint(_in_list("status", EXTRACTION_RUN_STATUSES), name="status_valid"),
        CheckConstraint(
            "passages_processed >= 0 AND units_extracted >= 0 AND failure_count >= 0", name="counts_non_negative"
        ),
        CheckConstraint(
            "scope_type IS NULL OR " + _in_list("scope_type", EXTRACTION_SCOPES), name="scope_type_valid"
        ),
        CheckConstraint(
            "total_passages >= 0 AND needs_review_count >= 0 AND skipped_count >= 0 AND duplicate_count >= 0 "
            "AND attempt_count >= 0 AND retry_count >= 0 AND validation_failure_count >= 0 "
            "AND relationships_extracted >= 0",
            name="progress_counts_non_negative",
        ),
        CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at", name="completion_order"
        ),
        Index("ix_extraction_runs_book_status", "book_id", "status"),
    )


class ExtractionExecution(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One call of the extraction service: one passage, one knowledge category, any number of attempts.

    Records the model configuration actually used (a snapshot, so later changes to the models
    file never rewrite history), the prompt, the generation parameters, usage and the outcome.
    Holds no credentials. The passage link is cleared if passages are regenerated; the execution
    stays. Knowledge units keep a link to the execution that produced them.
    """

    __tablename__ = "extraction_executions"

    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="CASCADE"), index=True
    )
    passage_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("passages.id", ondelete="SET NULL"), index=True
    )
    knowledge_type: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20))
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    review_reasons: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    items_extracted: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    model_configuration_id: Mapped[str | None] = mapped_column(String(100))
    provider: Mapped[str | None] = mapped_column(String(50))
    model_id: Mapped[str | None] = mapped_column(String(200))
    model_version: Mapped[str | None] = mapped_column(String(100))
    model_configuration: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    prompt_name: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    generation_parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    validation_failures: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    context_passages_dropped: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    # How the input was prepared: neighbours requested and sent, what was dropped and why, estimated tokens.
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    quality: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default="[]")
    failure_code: Mapped[str | None] = mapped_column(String(100))
    failure_message: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    # The last response received, kept for debugging (truncated by the service).
    raw_response: Mapped[str | None] = mapped_column(Text)

    extraction_run: Mapped[ExtractionRun] = relationship(back_populates="executions")
    attempts: Mapped[list[ExtractionAttempt]] = relationship(
        back_populates="execution",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ExtractionAttempt.attempt_number",
    )

    __table_args__ = (
        CheckConstraint(_in_list("knowledge_type", EXECUTION_KINDS), name="knowledge_type_valid"),
        CheckConstraint(_in_list("status", EXECUTION_STATUSES), name="status_valid"),
        CheckConstraint(
            "attempt_count >= 0 AND retry_count >= 0 AND validation_failures >= 0 AND items_extracted >= 0 "
            "AND duration_ms >= 0",
            name="counts_non_negative",
        ),
        CheckConstraint("completed_at >= started_at", name="completion_order"),
        CheckConstraint("status <> 'failed' OR needs_review", name="failed_needs_review"),
        Index("ix_extraction_executions_passage_type", "passage_id", "knowledge_type"),
    )


class ExtractionAttempt(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One request to the model within an execution, with the model and prompt actually used.

    Kept per attempt because the model identifier reported by the provider, the prompt and the
    outcome can differ between attempts of the same execution.
    """

    __tablename__ = "extraction_attempts"

    execution_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extraction_executions.id", ondelete="CASCADE"), index=True
    )
    # Denormalised from the execution so attempts can be queried by run or passage directly.
    extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="CASCADE"), index=True
    )
    passage_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("passages.id", ondelete="SET NULL"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20))
    # What happened to the call: it returned an answer ("completed") or the provider failed. Whether the
    # answer was any good is `validation_outcome`; `outcome` is the older combined value, kept for history.
    status: Mapped[str] = mapped_column(String(20), default="completed", server_default="completed")
    outcome: Mapped[str] = mapped_column(String(20))
    # The most serious result of checking the answer, and every result that applied.
    validation_outcome: Mapped[str] = mapped_column(String(30), default="not_validated", server_default="not_validated")
    validation_outcomes: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    # The configuration exactly as it was for this call (no credentials), whatever it is today.
    model_configuration: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    model_configuration_id: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(50))
    model_id: Mapped[str] = mapped_column(String(200))
    model_version: Mapped[str | None] = mapped_column(String(100))
    prompt_name: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(50))
    generation_parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    finish_reason: Mapped[str | None] = mapped_column(String(50))
    raw_response: Mapped[str | None] = mapped_column(Text)
    provider_error_code: Mapped[str | None] = mapped_column(String(100))
    provider_error: Mapped[str | None] = mapped_column(Text)
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default="[]")

    execution: Mapped[ExtractionExecution] = relationship(back_populates="attempts")

    __table_args__ = (
        UniqueConstraint("execution_id", "attempt_number", name="uq_extraction_attempts_number"),
        CheckConstraint("attempt_number >= 1", name="attempt_number_positive"),
        CheckConstraint(_in_list("kind", ATTEMPT_KINDS), name="kind_valid"),
        CheckConstraint(_in_list("outcome", ATTEMPT_OUTCOMES), name="outcome_valid"),
        CheckConstraint(_in_list("status", ATTEMPT_STATUSES), name="status_valid"),
        CheckConstraint(_in_list("validation_outcome", VALIDATION_OUTCOMES), name="validation_outcome_valid"),
        CheckConstraint("completed_at >= started_at", name="completion_order"),
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
    # The model call that produced the unit (model, prompt and parameters used). Kept if it is removed.
    extraction_execution_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_executions.id", ondelete="SET NULL"), index=True
    )
    # Hash of the unit's category, primary source passage and normalised content (see
    # app.services.knowledge_fingerprint). Identical extractions of the same passage share it.
    content_fingerprint: Mapped[str | None] = mapped_column(String(64))
    # What the content is. The original source stays authoritative: a unit never replaces its passages.
    origin: Mapped[str] = mapped_column(String(20), default="ai_extracted", server_default="ai_extracted")
    # Result of the last source verification (app.services.provenance_service).
    source_status: Mapped[str] = mapped_column(String(20), default="unverified", server_default="unverified")
    # The source as it was when the unit was stored, so provenance survives regenerated passages:
    # [{passage_id, document_id, chapter, section, pages, excerpt, excerpt_is_quote}].
    source_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default="[]")
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
        Index("ix_knowledge_units_book_fingerprint", "book_id", "content_fingerprint"),
        Index("ix_knowledge_units_book_source_status", "book_id", "source_status"),
        CheckConstraint(_in_list("origin", KNOWLEDGE_ORIGINS), name="origin_valid"),
        CheckConstraint(_in_list("source_status", SOURCE_VERIFICATION_STATUSES), name="source_status_valid"),
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
    reference_type: Mapped[str] = mapped_column(String(10), default="text", server_default="text")
    # Where in the book the reference sits, copied from the passage when the unit is stored.
    structural_node_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("structural_nodes.id", ondelete="SET NULL"), index=True
    )
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    # Character range of the quote inside the passage text, when the quote was found word for word.
    quote_start: Mapped[int | None] = mapped_column(Integer)
    quote_end: Mapped[int | None] = mapped_column(Integer)
    location: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # e.g. a bounding box for figures and equations
    verification_status: Mapped[str] = mapped_column(String(10), default="unverified", server_default="unverified")
    verification_issues: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default="[]")
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

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
        CheckConstraint(_in_list("reference_type", SOURCE_REFERENCE_TYPES), name="reference_type_valid"),
        CheckConstraint(_in_list("verification_status", SOURCE_VERIFICATION_STATUSES), name="verification_status_valid"),
        CheckConstraint(
            "(quote_start IS NULL) = (quote_end IS NULL) AND (quote_start IS NULL OR (quote_start >= 0 AND quote_end > quote_start))",
            name="quote_range_valid",
        ),
        CheckConstraint("page_start IS NULL OR page_start >= 1", name="page_start_positive"),
        CheckConstraint("page_end IS NULL OR page_start IS NULL OR page_end >= page_start", name="page_order"),
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
    # The model call that proposed it (model, prompt, parameters).
    extraction_execution_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_executions.id", ondelete="SET NULL"), index=True
    )
    # "extracted": the model quoted evidence that was found in a source passage. "ai_suggested": a model
    # proposal without verified evidence, to be reviewed. "manual": created by a person or by code.
    origin: Mapped[str] = mapped_column(String(15), default="manual", server_default="manual")
    review_status: Mapped[str] = mapped_column(String(15), default="unreviewed", server_default="unreviewed")
    evidence_quote: Mapped[str | None] = mapped_column(Text)
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
        CheckConstraint(_in_list("origin", RELATIONSHIP_ORIGINS), name="origin_valid"),
        CheckConstraint(_in_list("review_status", REVIEW_STATUSES), name="review_status_valid"),
        Index("ix_knowledge_relationships_book_origin", "book_id", "origin"),
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


class ExtractionPassageRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Where one passage stands in an extraction run (all the categories of the run together).

    Created for every selected passage when the run is created, so "never processed" is a recorded state
    and a resumed run knows exactly what is left. Its status is derived from the passage's executions.
    """

    __tablename__ = "extraction_passage_records"

    extraction_run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("extraction_runs.id", ondelete="CASCADE"))
    passage_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("passages.id", ondelete="SET NULL"), index=True)
    # Reading order, kept so the record stays ordered when the passage is regenerated away.
    document_sequence: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    status: Mapped[str] = mapped_column(String(15), default="pending", server_default="pending")
    execution_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    validation_failures: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    units_extracted: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)

    extraction_run: Mapped[ExtractionRun] = relationship(back_populates="passage_records")

    __table_args__ = (
        UniqueConstraint("extraction_run_id", "passage_id", name="uq_extraction_passage_records_passage"),
        CheckConstraint(_in_list("status", PASSAGE_STATUSES), name="status_valid"),
        CheckConstraint(
            "execution_count >= 0 AND attempt_count >= 0 AND retry_count >= 0 AND validation_failures >= 0 "
            "AND units_extracted >= 0",
            name="counts_non_negative",
        ),
        CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at", name="completion_order"
        ),
        Index("ix_extraction_passage_records_run_status", "extraction_run_id", "status"),
    )


class ReviewFlag(UUIDPrimaryKeyMixin, Base):
    """Something that needs a human look, and why. Attached to one knowledge unit, attempt or relationship.

    Flags are findings, not decisions: this is not a review workflow. An open flag with the same code and
    key on the same target is never created twice, so verifying or resuming again does not pile them up.
    """

    __tablename__ = "review_flags"

    book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    knowledge_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("knowledge_units.id", ondelete="CASCADE"), index=True
    )
    extraction_attempt_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_attempts.id", ondelete="CASCADE"), index=True
    )
    relationship_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("knowledge_relationships.id", ondelete="CASCADE"), index=True
    )
    extraction_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_runs.id", ondelete="SET NULL"), index=True
    )
    code: Mapped[str] = mapped_column(String(50))
    severity: Mapped[str] = mapped_column(String(10), default="warning", server_default="warning")
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    # Tells apart flags of one code on one target (for example one per source passage).
    dedupe_key: Mapped[str] = mapped_column(String(200), default="", server_default="")
    # Who raised it: the system's own checks, or the model's own report of uncertainty.
    raised_by: Mapped[str] = mapped_column(String(10), default="system", server_default="system")
    status: Mapped[str] = mapped_column(String(10), default="open", server_default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(_in_list("code", REVIEW_FLAG_CODES), name="code_valid"),
        CheckConstraint(_in_list("severity", FLAG_SEVERITIES), name="severity_valid"),
        CheckConstraint(_in_list("status", FLAG_STATUSES), name="status_valid"),
        CheckConstraint("raised_by IN ('system', 'model', 'human')", name="raised_by_valid"),
        CheckConstraint(
            "num_nonnulls(knowledge_unit_id, extraction_attempt_id, relationship_id) = 1", name="exactly_one_target"
        ),
        CheckConstraint("(status = 'open') = (resolved_at IS NULL)", name="resolved_when_not_open"),
        Index(
            "uq_review_flags_open_unit", "knowledge_unit_id", "code", "dedupe_key", unique=True,
            postgresql_where=text("status = 'open' AND knowledge_unit_id IS NOT NULL"),
        ),
        Index(
            "uq_review_flags_open_attempt", "extraction_attempt_id", "code", "dedupe_key", unique=True,
            postgresql_where=text("status = 'open' AND extraction_attempt_id IS NOT NULL"),
        ),
        Index(
            "uq_review_flags_open_relationship", "relationship_id", "code", "dedupe_key", unique=True,
            postgresql_where=text("status = 'open' AND relationship_id IS NOT NULL"),
        ),
        Index("ix_review_flags_book_status", "book_id", "status"),
    )


class EquationSource(Base):
    """What the source gave us for an equation unit, kept apart from what the model extracted.

    The values here are copied from Phase 1 data (the stored equation and the passage text), never written
    by a model, and not overwritten later. `KnowledgeUnit.details` holds the model's structured reading;
    this holds the original, its context and its image reference, and says what was actually captured.
    """

    __tablename__ = "equation_sources"

    knowledge_unit_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_units.id", ondelete="CASCADE"), primary_key=True
    )
    equation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("equations.id", ondelete="SET NULL"), index=True
    )
    passage_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("passages.id", ondelete="SET NULL"), index=True)
    structural_node_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("structural_nodes.id", ondelete="SET NULL")
    )
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    original_text: Mapped[str | None] = mapped_column(Text)  # the equation as extracted from the source
    source_latex: Mapped[str | None] = mapped_column(Text)  # LaTeX where the source extraction produced it
    # A reference (path) to the stored image of the equation. The image itself is not kept in the database.
    image_reference: Mapped[str | None] = mapped_column(String(1024))
    source_location: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    surrounding_text: Mapped[str | None] = mapped_column(Text)
    context_before: Mapped[str | None] = mapped_column(Text)  # passage text before the equation
    context_after: Mapped[str | None] = mapped_column(Text)  # passage text after it
    capture_status: Mapped[str] = mapped_column(String(15))
    capture_notes: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())

    __table_args__ = (
        CheckConstraint(_in_list("capture_status", EQUATION_CAPTURE_STATUSES), name="capture_status_valid"),
        Index("ix_equation_sources_status", "capture_status"),
    )


class EquationReconstruction(UUIDPrimaryKeyMixin, Base):
    """Room for a future phase to store an AI reconstruction of an equation, apart from the original.

    Nothing writes here yet. Keeping it a separate table means a reconstruction can never replace the
    original expression, can be repeated by several models, and carries its own model, prompt, validation
    and review state.
    """

    __tablename__ = "equation_reconstructions"

    knowledge_unit_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_units.id", ondelete="CASCADE"), index=True
    )
    reconstructed_expression: Mapped[str | None] = mapped_column(Text)
    reconstructed_latex: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    model_configuration_id: Mapped[str | None] = mapped_column(String(100))
    provider: Mapped[str | None] = mapped_column(String(50))
    model_id: Mapped[str | None] = mapped_column(String(200))
    model_version: Mapped[str | None] = mapped_column(String(100))
    prompt_name: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(50))
    generation_parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default="{}")
    math_validation_status: Mapped[str] = mapped_column(String(15), default="not_validated", server_default="not_validated")
    review_status: Mapped[str] = mapped_column(String(15), default="unreviewed", server_default="unreviewed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())

    __table_args__ = (
        CheckConstraint(_in_list("math_validation_status", MATH_VALIDATION_STATUSES), name="math_validation_status_valid"),
        CheckConstraint(_in_list("review_status", REVIEW_STATUSES), name="review_status_valid"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"),
        CheckConstraint(
            "reconstructed_expression IS NOT NULL OR reconstructed_latex IS NOT NULL", name="has_content"
        ),
    )
