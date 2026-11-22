"""Common knowledge schemas: the envelope around typed content, relationships and extraction runs.

Everything that is the same for all seven categories is here (identity, sources, location,
confidence, uncertainty, status, metadata). What differs per category is in
app.knowledge.content. The `...Create` models are what an extractor produces; the `...Read`
models add what the database assigns.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, TypeAdapter, model_validator

from app.knowledge.content import AnyKnowledgeContent, EquationContent, KnowledgeContent
from app.knowledge.types import (
    ExtractionRunStatus,
    KnowledgeOrigin,
    KnowledgeType,
    PageStatus,
    RelationshipOrigin,
    RelationshipType,
    ReviewStatus,
    SourceVerificationStatus,
    SourceRole,
    ValidationStatus,
)

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Quote = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5_000)]
Note = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5_000)]
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Page = Annotated[int, Field(ge=1)]

MAX_SOURCES = 50

_CONTENT_ADAPTER: TypeAdapter[KnowledgeContent] = TypeAdapter(AnyKnowledgeContent)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceReference(_Model):
    """A passage that supports a unit, optionally with the words in it that back the unit."""

    passage_id: uuid.UUID
    role: SourceRole = SourceRole.SUPPORTING
    quote: Quote | None = None


class Uncertainty(_Model):
    """Says out loud when the extractor is not sure, instead of hiding it in a low confidence."""

    is_uncertain: bool = False
    notes: Note | None = None

    @model_validator(mode="after")
    def _notes_need_the_flag(self) -> Uncertainty:
        if self.notes is not None and not self.is_uncertain:
            raise ValueError("Uncertainty notes are only allowed when is_uncertain is true")
        return self


class KnowledgeUnitCreate(_Model):
    """One extracted piece of knowledge: typed content plus the information common to all types."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    book_id: uuid.UUID
    document_id: uuid.UUID
    extraction_run_id: uuid.UUID | None = None

    content: AnyKnowledgeContent

    # Where it comes from. One or more passages; the page range is kept on the unit too.
    sources: Annotated[list[SourceReference], Field(min_length=1, max_length=MAX_SOURCES)]
    chapter_node_id: uuid.UUID | None = None
    section_node_id: uuid.UUID | None = None
    page_start: Page | None = None
    page_end: Page | None = None
    page_status: PageStatus = PageStatus.UNKNOWN
    origin: KnowledgeOrigin = KnowledgeOrigin.AI_EXTRACTED
    # The Phase 1 equation record this unit was built from (equations only).
    equation_id: uuid.UUID | None = None

    confidence: Confidence | None = None
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def knowledge_type(self) -> KnowledgeType:
        return KnowledgeType(self.content.knowledge_type)

    @model_validator(mode="after")
    def _consistent(self) -> KnowledgeUnitCreate:
        passage_ids = [source.passage_id for source in self.sources]
        if len(set(passage_ids)) != len(passage_ids):
            raise ValueError("A passage can only be listed once as a source")
        if sum(source.role == SourceRole.PRIMARY for source in self.sources) > 1:
            raise ValueError("At most one source can be the primary source")

        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError("page_end cannot be before page_start")
        if self.page_status == PageStatus.UNKNOWN and (self.page_start is not None or self.page_end is not None):
            raise ValueError("Pages were given, so page_status cannot be 'unknown'")
        if self.page_status != PageStatus.UNKNOWN and self.page_start is None:
            raise ValueError(f"page_status '{self.page_status}' requires page_start")

        if self.equation_id is not None and self.knowledge_type != KnowledgeType.EQUATION:
            raise ValueError("Only an equation unit can link to an equation record")

        if isinstance(self.content, EquationContent) and self.content.is_incomplete:
            if self.validation_status == ValidationStatus.VALID:
                raise ValueError("An incomplete equation cannot be validated as 'valid'")
            if not self.uncertainty.is_uncertain:
                self.uncertainty = Uncertainty(
                    is_uncertain=True,
                    notes=self.content.status_notes
                    or f"Equation is incomplete (extraction: {self.content.extraction_status}, "
                    f"reconstruction: {self.content.reconstruction_status}).",
                )
        return self


class KnowledgeUnitRead(KnowledgeUnitCreate):
    # A stored unit can have lost its sources: regenerating a document's passages removes the
    # links but keeps the knowledge (and its page range). New units must still cite a source.
    sources: Annotated[list[SourceReference], Field(default_factory=list, max_length=MAX_SOURCES)]
    schema_version: int
    source_status: SourceVerificationStatus = SourceVerificationStatus.UNVERIFIED
    created_at: datetime
    updated_at: datetime

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)


class KnowledgeRelationshipCreate(_Model):
    """`source <relationship_type> target`. Whether the two types fit is checked with
    app.knowledge.relationships.check_relationship_types, which needs the units' types."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    book_id: uuid.UUID
    source_unit_id: uuid.UUID
    target_unit_id: uuid.UUID
    relationship_type: RelationshipType
    explanation: Note | None = None
    confidence: Confidence | None = None
    extraction_run_id: uuid.UUID | None = None
    extraction_execution_id: uuid.UUID | None = None
    # Passages where the book connects the two units, when known.
    source_passage_ids: Annotated[list[uuid.UUID], Field(default_factory=list, max_length=MAX_SOURCES)]
    # "extracted" needs evidence from the book (a quote); "ai_suggested" is a proposal to review.
    origin: RelationshipOrigin = RelationshipOrigin.MANUAL
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    evidence_quote: Quote | None = None

    @model_validator(mode="after")
    def _consistent(self) -> KnowledgeRelationshipCreate:
        if self.source_unit_id == self.target_unit_id:
            raise ValueError("A unit cannot be related to itself")
        if len(set(self.source_passage_ids)) != len(self.source_passage_ids):
            raise ValueError("A passage can only be listed once")
        if self.origin == RelationshipOrigin.EXTRACTED and not (self.source_passage_ids and self.evidence_quote):
            raise ValueError("An extracted relationship needs the passage and the quoted evidence that support it")
        return self


class KnowledgeRelationshipRead(KnowledgeRelationshipCreate):
    created_at: datetime


class ExtractionRunCreate(_Model):
    """What is needed to start an extraction run. The run is over one book, or one document or
    chapter/section of it. `model_name` and `prompt_version` are free text: no provider is assumed."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    book_id: uuid.UUID
    document_id: uuid.UUID | None = None
    structural_node_id: uuid.UUID | None = None
    model_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)] | None = None
    prompt_version: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)] | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    scope_type: Literal["book", "document", "chapter", "section"] | None = None
    model_configuration_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)] | None = None
    provider: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)] | None = None
    model_snapshot: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _node_needs_document(self) -> ExtractionRunCreate:
        if self.structural_node_id is not None and self.document_id is None:
            raise ValueError("A chapter or section belongs to a document, so document_id is required")
        return self


class ExtractionRunRead(ExtractionRunCreate):
    status: ExtractionRunStatus
    passages_processed: Annotated[int, Field(ge=0)]
    units_extracted: Annotated[int, Field(ge=0)]
    failure_count: Annotated[int, Field(ge=0)]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_details: dict[str, JsonValue] | None = None
    total_passages: Annotated[int, Field(ge=0)] = 0
    needs_review_count: Annotated[int, Field(ge=0)] = 0
    skipped_count: Annotated[int, Field(ge=0)] = 0
    duplicate_count: Annotated[int, Field(ge=0)] = 0
    attempt_count: Annotated[int, Field(ge=0)] = 0
    retry_count: Annotated[int, Field(ge=0)] = 0
    validation_failure_count: Annotated[int, Field(ge=0)] = 0
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    estimated_cost_usd: Annotated[float, Field(ge=0)] | None = None
    relationships_extracted: Annotated[int, Field(ge=0)] = 0
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def _status_is_consistent(self) -> ExtractionRunRead:
        if self.status.is_terminal != (self.completed_at is not None):
            raise ValueError("completed_at is set exactly when the run has finished")
        if self.started_at and self.completed_at and self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be before started_at")
        if self.status == ExtractionRunStatus.FAILED and not self.error_message:
            raise ValueError("A failed run must say why (error_message)")
        return self


def content_from_details(knowledge_type: str, details: dict[str, Any]) -> KnowledgeContent:
    """Rebuild typed content from the (type, details) pair stored in the database."""
    return _CONTENT_ADAPTER.validate_python({**details, "knowledge_type": knowledge_type})
