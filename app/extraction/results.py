"""Inputs and outputs of the extraction service.

These describe one extraction request, what came back, how it was validated and exactly how it was
produced (model, prompt, parameters, every attempt). They are plain Pydantic models with no
database ties, so a caller can store them wherever it likes; `ExecutionMetadata` is a frozen
snapshot, so later changes to a model configuration never alter a finished record.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from app.knowledge.content import AnyKnowledgeContent
from app.knowledge.schemas import KnowledgeUnitCreate, SourceReference, Uncertainty
from app.knowledge.types import KnowledgeType, PageStatus, ValidationOutcome, ValidationStatus


def utcnow() -> datetime:
    return datetime.now(UTC)


class _Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# -- Request ---------------------------------------------------------------------------------------


class PassageInput(_Model):
    """A passage as the extractor sees it. Built from a stored Passage by the caller."""

    id: uuid.UUID
    text: str = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    chapter_title: str | None = None
    section_title: str | None = None

    @property
    def has_pages(self) -> bool:
        return self.page_start is not None

    @property
    def last_page(self) -> int | None:
        return self.page_end if self.page_end is not None else self.page_start


class SourceInfo(_Model):
    """The book a passage comes from, so the model knows what it is reading."""

    title: str
    subtitle: str | None = None
    author: str | None = None
    publisher: str | None = None
    year: int | None = None
    language: str | None = None


class ContextInfo(_Model):
    """How the neighbouring passages of one request were handled (kept with the execution)."""

    requested: int = 0  # neighbouring passages asked for
    sent: int = 0  # neighbouring passages actually in the prompt
    dropped: list[uuid.UUID] = Field(default_factory=list)  # left out, furthest first
    reduced: bool = False
    reason: str | None = None
    estimated_input_tokens: int | None = None
    budget_tokens: int | None = None


class ExtractionRequest(_Model):
    """Extract one knowledge category from one passage, optionally with its neighbours as context."""

    knowledge_type: KnowledgeType
    passage: PassageInput
    context_before: list[PassageInput] = Field(default_factory=list)
    context_after: list[PassageInput] = Field(default_factory=list)
    source: SourceInfo | None = None
    # Set by the context preparation service when it already reduced the context.
    context: ContextInfo | None = None

    def passages(self) -> dict[uuid.UUID, PassageInput]:
        """Every passage the model may cite, by id."""
        return {p.id: p for p in [*self.context_before, self.passage, *self.context_after]}


# -- Validation ------------------------------------------------------------------------------------


class ValidationStage(enum.StrEnum):
    STRUCTURAL = "structural"  # does the response match the schema?
    SOURCE = "source"  # do cited passages and pages exist?
    SEMANTIC = "semantic"  # is it supported by the source text?
    QUALITY = "quality"  # completeness, duplicates


class Severity(enum.StrEnum):
    ERROR = "error"  # the response is unusable as it is; triggers a correction attempt
    WARNING = "warning"  # usable, but a reviewer should look
    INFO = "info"


class ValidationIssue(_Model):
    stage: ValidationStage
    severity: Severity
    code: str
    message: str
    path: str | None = None  # e.g. "items[0].content.steps"
    item_index: int | None = None


class QualityAssessment(_Model):
    """Indicators of an extraction's quality. Deliberately no single "reliability score": nothing
    here is derived from the model's name, size or provider, and a clean result still needs review
    when the issues say so."""

    schema_valid: bool
    item_count: int = 0
    completeness: float | None = None  # share of optional content fields that were filled
    source_reference_accuracy: float | None = None  # share of citations (ids, pages, quotes) that checked out
    unsupported_content: list[str] = Field(default_factory=list)
    duplicate_items: list[tuple[int, int]] = Field(default_factory=list)
    semantic_accuracy: float | None = None  # only when evaluation data was available


# -- Extracted knowledge ---------------------------------------------------------------------------


class ExtractedKnowledge(_Model):
    """One extracted unit: typed content plus its provenance, before it is stored."""

    content: AnyKnowledgeContent
    sources: list[SourceReference]
    page_start: int | None = None
    page_end: int | None = None
    page_status: PageStatus = PageStatus.UNKNOWN
    pages_from_model: bool = False  # False when the pages were taken from the cited passages
    basis: Literal["explicit", "inferred", "mixed"] = "explicit"
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def knowledge_type(self) -> KnowledgeType:
        return KnowledgeType(self.content.knowledge_type)

    def to_unit_create(
        self,
        book_id: uuid.UUID,
        document_id: uuid.UUID,
        extraction_run_id: uuid.UUID | None = None,
        metadata: dict[str, JsonValue] | None = None,
        **ties: Any,
    ) -> KnowledgeUnitCreate:
        """The Day 1 schema, ready to be stored. `ties` may carry chapter_node_id, section_node_id, ..."""
        return KnowledgeUnitCreate(
            book_id=book_id,
            document_id=document_id,
            extraction_run_id=extraction_run_id,
            content=self.content,
            sources=self.sources,
            page_start=self.page_start,
            page_end=self.page_end,
            page_status=self.page_status,
            confidence=self.confidence,
            uncertainty=self.uncertainty,
            validation_status=self.validation_status,
            metadata={"basis": self.basis, **(metadata or {})},
            **ties,
        )


# -- Execution metadata ----------------------------------------------------------------------------


class AttemptOutcome(enum.StrEnum):
    VALID = "valid"
    INVALID = "invalid"  # a response arrived but failed validation
    PROVIDER_ERROR = "provider_error"  # no usable response (timeout, rate limit, ...)


class AttemptRecord(_Model):
    """One call to a model. Kept individually so a run with several attempts can be investigated."""

    attempt: int
    kind: Literal["initial", "correction", "retry"]
    model_configuration_id: str
    provider: str
    model_id: str  # as reported by the provider for this attempt
    model_version: str | None = None
    prompt_name: str
    prompt_version: str
    generation_parameters: dict[str, JsonValue]
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    outcome: AttemptOutcome
    # What happened to the call, apart from whether the answer was any good.
    status: Literal["completed", "provider_error"] = "completed"
    # The result of checking the answer: the most serious one, and every one that applied.
    validation_outcome: ValidationOutcome = ValidationOutcome.NOT_VALIDATED
    validation_outcomes: list[ValidationOutcome] = Field(default_factory=list)
    # The configuration as it was for this call (no credentials), whatever it is today.
    model_configuration: dict[str, JsonValue] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    finish_reason: str | None = None
    raw_response: str | None = None  # preserved for debugging, truncated
    provider_error_code: str | None = None
    provider_error: str | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)


class ExecutionMetadata(_Model):
    """Everything needed to investigate or reproduce one extraction."""

    model_configuration_id: str | None
    provider: str | None
    model_id: str | None  # the model that actually answered (last attempt that got a response)
    model_version: str | None = None
    model_configuration: dict[str, JsonValue] | None = None  # snapshot; holds no secrets
    prompt_name: str | None = None
    prompt_version: str | None = None
    generation_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    attempt_count: int = 0
    retry_count: int = 0
    validation_failures: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    context_passages_dropped: list[uuid.UUID] = Field(default_factory=list)
    context: ContextInfo | None = None
    attempts: list[AttemptRecord] = Field(default_factory=list)

    @property
    def prompt_identifier(self) -> str | None:
        return f"{self.prompt_name}@{self.prompt_version}" if self.prompt_name else None

    def run_fields(self) -> dict[str, Any]:
        """Arguments for ExtractionRunCreate (model_name, prompt_version, parameters)."""
        parameters: dict[str, JsonValue] = {
            "model_configuration_id": self.model_configuration_id,
            "provider": self.provider,
            "model_version": self.model_version,
            "prompt_name": self.prompt_name,
            "generation_parameters": self.generation_parameters,
            "attempt_count": self.attempt_count,
            "retry_count": self.retry_count,
            "validation_failures": self.validation_failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }
        return {"model_name": self.model_id, "prompt_version": self.prompt_identifier, "parameters": parameters}


# -- Result ----------------------------------------------------------------------------------------


class ExtractionStatus(enum.StrEnum):
    SUCCEEDED = "succeeded"  # valid, and nothing in it calls for review
    NEEDS_REVIEW = "needs_review"  # usable, but uncertain, partly unsupported, inferred or incomplete
    FAILED = "failed"  # no valid result could be obtained; always needs review


class ExtractionFailure(_Model):
    code: str
    message: str


class ExtractionResult(_Model):
    knowledge_type: KnowledgeType
    status: ExtractionStatus
    items: list[ExtractedKnowledge] = Field(default_factory=list)  # empty and SUCCEEDED = nothing found
    source_references: list[uuid.UUID] = Field(default_factory=list)  # every passage cited by any item
    confidence: float | None = None  # the model's own reported confidence (lowest of the items)
    issues: list[ValidationIssue] = Field(default_factory=list)
    quality: QualityAssessment | None = None
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    failure: ExtractionFailure | None = None
    notes: str | None = None  # remarks the model made about the passage
    raw_response: str | None = None  # the last response received, for debugging
    execution: ExecutionMetadata

    @property
    def is_valid(self) -> bool:
        return self.status != ExtractionStatus.FAILED
