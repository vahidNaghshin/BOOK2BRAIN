"""What an extraction run produced, in a plain form the metrics can read without a database."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ObservedEquation:
    capture_status: str | None = None
    original_expression: str = ""  # what the model reported
    latex: str = ""
    source_text: str | None = None  # what Phase 1 stored (never model output)
    source_latex: str | None = None
    image_reference: str | None = None
    page_start: int | None = None
    surrounding_text: str | None = None


@dataclass
class ObservedUnit:
    id: str
    category: str
    title: str
    text: str  # title, content and every typed field, for lexical checks
    primary_passage: str | None  # dataset passage key of the primary source, if it belongs to the dataset
    source_passages: list[str] = field(default_factory=list)  # every dataset passage cited
    page_start: int | None = None
    page_end: int | None = None
    source_status: str = "unverified"
    validation_status: str = "pending"
    origin: str = "ai_extracted"
    is_uncertain: bool = False
    confidence: float | None = None
    quotes: int = 0  # sources with a quote
    quotes_verified: int = 0
    flags: list[str] = field(default_factory=list)  # codes of open review flags
    run_id: str | None = None
    extracted_from: str | None = None  # dataset key of the passage whose extraction produced the unit
    equation: ObservedEquation | None = None


@dataclass
class ObservedRelationship:
    source_id: str
    target_id: str
    type: str
    origin: str
    review_status: str
    has_evidence: bool


@dataclass
class ObservedExecution:
    passage: str | None  # dataset passage key
    category: str  # a knowledge category, or "relationship"
    status: str
    attempts: int
    retries: int
    validation_failures: int
    duration_ms: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    prompt: str | None


@dataclass
class ObservedAttempt:
    status: str  # "completed" (the call returned an answer) or "provider_error"
    validation_outcome: str
    category: str  # a knowledge category, or "relationship"


@dataclass
class ObservedRun:
    run_id: str
    status: str
    model_configuration_id: str | None
    provider: str | None
    model: str | None
    model_snapshot: dict | None
    generation: dict  # temperature, max output tokens, structured output mode actually used
    parameters: dict
    prompt_versions: list[str]
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    passages: int
    attempts: int
    retries: int
    validation_failures: int
    duplicates: int
    failed: int
    needs_review: int


@dataclass
class Observation:
    runs: list[ObservedRun] = field(default_factory=list)
    units: list[ObservedUnit] = field(default_factory=list)
    relationships: list[ObservedRelationship] = field(default_factory=list)
    executions: list[ObservedExecution] = field(default_factory=list)
    attempts: list[ObservedAttempt] = field(default_factory=list)

    @property
    def covered(self) -> set[tuple[str, str]]:
        """(passage key, category) pairs that were actually extracted, whatever the outcome."""
        return {(e.passage, e.category) for e in self.executions if e.passage}

    @property
    def relationship_passages(self) -> set[str]:
        return {e.passage for e in self.executions if e.passage and e.category == "relationship"}
