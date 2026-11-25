"""Browsing and reviewing extracted knowledge: lists with filters, one unit with its evidence, review decisions."""

from __future__ import annotations

import math
import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.extraction.results import utcnow
from app.knowledge.records import unit_from_record
from app.knowledge.schemas import KnowledgeUnitRead
from app.repositories.knowledge_repository import KnowledgeFilter, KnowledgeRepository
from app.services.equation_service import EquationCapture, EquationService
from app.services.errors import InvalidReviewError, KnowledgeNotFoundError
from app.services.provenance_service import ProvenanceService, VerificationReport
from app.services.relationship_service import RelationshipService, UnitRelationships
from app.services.review_flag_service import FlagRead, ReviewFlagService
from database.models import REVIEW_STATUSES, KnowledgeUnit

__all__ = ["AttemptDetail", "FailedExtraction", "KnowledgeDetail", "KnowledgeFilter", "KnowledgeListItem", "KnowledgeOverview", "KnowledgePage", "KnowledgeService", "KnowledgeSummary", "SourceDetail"]


class KnowledgeListItem(BaseModel):
    id: uuid.UUID
    knowledge_type: str
    title: str
    summary: str
    page_start: int | None
    validation_status: str
    review_status: str
    is_uncertain: bool
    confidence: float | None


class KnowledgePage(BaseModel):
    items: list[KnowledgeListItem]
    total: int
    page: int
    pages: int


class KnowledgeSummary(BaseModel):
    total: int
    by_type: dict[str, int]
    by_review: dict[str, int]
    by_validation: dict[str, int]
    needing_attention: int  # uncertain, or not validated


class FailedExtraction(BaseModel):
    run_id: uuid.UUID
    passage_id: uuid.UUID | None
    page_start: int | None
    category: str
    attempts: int
    failure_code: str | None
    failure_message: str | None
    model: str | None
    prompt: str | None


class KnowledgeOverview(BaseModel):
    """Everything about the extracted knowledge of a book that a reviewer wants at a glance."""

    by_type: dict[str, int]
    equation_status: dict[str, int]  # capture status -> equations
    relationships_by_type: dict[str, int]
    open_flags: dict[str, int]  # review flag code -> open flags
    attempt_outcomes: dict[str, int]  # result of every model call, including provider errors
    failed_extractions: list[FailedExtraction]


class SourceDetail(BaseModel):
    passage_id: uuid.UUID
    role: str
    quote: str | None
    page_start: int | None
    page_end: int | None
    text: str | None  # None when the passage no longer exists (passages were regenerated)
    quote_found: bool | None  # whether the quote is literally in the passage text; None without a quote or a passage
    reference_type: str = "text"
    quote_start: int | None = None
    quote_end: int | None = None
    verification_status: str = "unverified"
    verification_issues: list[dict] = []
    verified_at: datetime | None = None


class AttemptDetail(BaseModel):
    """One model call of the extraction that produced a unit."""

    id: uuid.UUID
    attempt_number: int
    kind: str
    status: str  # the call: completed, or the provider failed
    validation_outcome: str  # the most serious result of checking the answer
    validation_outcomes: list[str]
    model_configuration_id: str
    provider: str
    model_id: str
    model_version: str | None
    prompt: str
    generation_parameters: dict
    model_configuration: dict | None
    duration_ms: int
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    provider_error: str | None
    issues: list[dict]


class ExecutionDetail(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    status: str
    needs_review: bool
    review_reasons: list[str]
    issues: list[dict]
    model_configuration_id: str | None
    provider: str | None
    model_id: str | None
    model_version: str | None
    prompt: str | None
    attempts: int
    retries: int
    validation_failures: int
    duration_ms: int
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: float | None
    context: dict | None


class KnowledgeDetail(BaseModel):
    unit: KnowledgeUnitRead
    chapter_title: str | None
    section_title: str | None
    sources: list[SourceDetail]
    execution: ExecutionDetail | None
    review_note: str | None
    reviewed_at: str | None
    same_title_as: list[uuid.UUID]  # the same title elsewhere in the book: separate occurrences, not merged
    source_status: str = "unverified"
    source_snapshot: list[dict] = []  # the source as it was when the unit was stored
    origin: str = "ai_extracted"
    relationships: UnitRelationships | None = None
    equation: EquationCapture | None = None
    flags: list[FlagRead] = []
    attempts: list[AttemptDetail] = []


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


class KnowledgeService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = KnowledgeRepository(session)
        self.flags = ReviewFlagService(session)

    def list_units(self, f: KnowledgeFilter, *, page: int = 1, page_size: int = 15) -> KnowledgePage:
        total = self.repository.count(f)
        pages = max(1, math.ceil(total / page_size))
        page = min(max(1, page), pages)
        records = self.repository.list(f, limit=page_size, offset=(page - 1) * page_size)
        return KnowledgePage(items=[self._item(r) for r in records], total=total, page=page, pages=pages)

    def items(self, ids: list[uuid.UUID]) -> list[KnowledgeListItem]:
        records = [self.repository.get(i) for i in ids]
        return [self._item(r) for r in records if r is not None]

    def summary(self, f: KnowledgeFilter) -> KnowledgeSummary:
        by_validation = self.repository.counts_by(f, KnowledgeUnit.validation_status)
        return KnowledgeSummary(
            total=self.repository.count(f),
            by_type=self.repository.counts_by(f, KnowledgeUnit.knowledge_type),
            by_review=self.repository.counts_by(f, KnowledgeUnit.review_status),
            by_validation=by_validation,
            needing_attention=self.repository.count(KnowledgeFilter(**{**f.__dict__, "uncertain_only": True})),
        )

    def overview(self, f: KnowledgeFilter, *, failures: int = 20) -> KnowledgeOverview:
        """Counts by category, equation capture status, relationship type and open review flag, plus the extractions that failed."""
        failed = [
            FailedExtraction(
                run_id=execution.extraction_run_id, passage_id=execution.passage_id, page_start=page, category=execution.knowledge_type,
                attempts=execution.attempt_count, failure_code=execution.failure_code, failure_message=execution.failure_message,
                model=execution.model_id,
                prompt=f"{execution.prompt_name}@{execution.prompt_version}" if execution.prompt_name else None,
            )
            for execution, page in self.repository.failed_executions(f.book_id, failures)
        ]
        return KnowledgeOverview(
            by_type=self.repository.counts_by(f, KnowledgeUnit.knowledge_type),
            equation_status=EquationService(self.session).counts_by_status(f.book_id),
            relationships_by_type=self.repository.relationship_counts(f.book_id),
            open_flags=ReviewFlagService(self.session).counts_by_code(f.book_id),
            attempt_outcomes=self.repository.attempt_outcomes(f.book_id),
            failed_extractions=failed,
        )

    def detail(self, unit_id: uuid.UUID) -> KnowledgeDetail:
        record = self.repository.get(unit_id)
        if record is None:
            raise KnowledgeNotFoundError("This knowledge unit no longer exists.")
        unit = unit_from_record(record)
        links = self.repository.sources_of(record)
        passages = self.repository.passages([link.passage_id for link in links])
        sources = []
        for link in links:
            passage = passages.get(link.passage_id)
            found = None
            if link.quote and passage:
                found = _squash(link.quote) in _squash(passage.text)
            sources.append(
                SourceDetail(
                    passage_id=link.passage_id,
                    role=link.role,
                    quote=link.quote,
                    page_start=passage.page_start if passage else None,
                    page_end=passage.page_end if passage else None,
                    text=passage.text if passage else None,
                    quote_found=found,
                    reference_type=link.reference_type,
                    quote_start=link.quote_start,
                    quote_end=link.quote_end,
                    verification_status=link.verification_status,
                    verification_issues=list(link.verification_issues or []),
                    verified_at=link.verified_at,
                )
            )
        titles = self.repository.node_titles([record.chapter_node_id, record.section_node_id])
        review = (record.unit_metadata or {}).get("review") or {}
        return KnowledgeDetail(
            unit=unit,
            chapter_title=titles.get(record.chapter_node_id) if record.chapter_node_id else None,
            section_title=titles.get(record.section_node_id) if record.section_node_id else None,
            sources=sources,
            execution=self._execution(record),
            review_note=review.get("note"),
            reviewed_at=review.get("at"),
            same_title_as=[uuid.UUID(v) for v in (record.unit_metadata or {}).get("same_title_as", [])],
            source_status=record.source_status,
            source_snapshot=list(record.source_snapshot or []),
            origin=record.origin,
            relationships=RelationshipService(self.session).for_unit(unit_id),
            equation=EquationService(self.session).get_capture(unit_id) if record.knowledge_type == "equation" else None,
            flags=self.flags.flags_for_unit(record.book_id, unit_id, status="open"),  # resolved ones are history
            attempts=self._attempts(record),
        )

    def verify_sources(self, unit_id: uuid.UUID) -> VerificationReport:
        """Check the unit's source references again and record the result (and any review flags)."""
        if self.repository.get(unit_id) is None:
            raise KnowledgeNotFoundError("This knowledge unit no longer exists.")
        return ProvenanceService(self.session).verify_unit(unit_id)

    def resolve_flag(self, flag_id: uuid.UUID, note: str | None = None, *, dismiss: bool = False) -> FlagRead:
        try:
            return self.flags.resolve(flag_id, note, dismiss=dismiss)
        except LookupError as exc:
            raise KnowledgeNotFoundError(str(exc)) from exc

    def review(self, unit_id: uuid.UUID, status: str, note: str | None = None) -> KnowledgeDetail:
        """Record a review decision (approved, rejected, needs_changes, or unreviewed to undo one)."""
        if status not in REVIEW_STATUSES:
            raise InvalidReviewError(f"Review status must be one of: {', '.join(REVIEW_STATUSES)}.")
        record = self.repository.get(unit_id)
        if record is None:
            raise KnowledgeNotFoundError("This knowledge unit no longer exists.")
        self.repository.set_review(record, status, (note or "").strip() or None, utcnow())
        return self.detail(unit_id)

    # -- helpers -----------------------------------------------------------------------------------

    @staticmethod
    def _item(record: KnowledgeUnit) -> KnowledgeListItem:
        return KnowledgeListItem(
            id=record.id,
            knowledge_type=record.knowledge_type,
            title=record.title,
            summary=(record.content or "")[:160],
            page_start=record.page_start,
            validation_status=record.validation_status,
            review_status=record.review_status,
            is_uncertain=record.is_uncertain,
            confidence=record.confidence,
        )

    def _attempts(self, record: KnowledgeUnit) -> list[AttemptDetail]:
        execution = self.repository.execution_of(record)
        if execution is None:
            return []
        return [
            AttemptDetail(
                id=a.id, attempt_number=a.attempt_number, kind=a.kind, status=a.status, validation_outcome=a.validation_outcome,
                validation_outcomes=list(a.validation_outcomes or []), model_configuration_id=a.model_configuration_id,
                provider=a.provider, model_id=a.model_id, model_version=a.model_version, prompt=f"{a.prompt_name}@{a.prompt_version}",
                generation_parameters=dict(a.generation_parameters or {}), model_configuration=a.model_configuration,
                duration_ms=a.duration_ms, input_tokens=a.input_tokens, output_tokens=a.output_tokens, finish_reason=a.finish_reason,
                provider_error=a.provider_error, issues=list(a.issues or []),
            )
            for a in execution.attempts
        ]

    def _execution(self, record: KnowledgeUnit) -> ExecutionDetail | None:
        e = self.repository.execution_of(record)
        if e is None:
            return None
        return ExecutionDetail(
            id=e.id,
            run_id=e.extraction_run_id,
            status=e.status,
            needs_review=e.needs_review,
            review_reasons=list(e.review_reasons or []),
            issues=list(e.issues or []),
            model_configuration_id=e.model_configuration_id,
            provider=e.provider,
            model_id=e.model_id,
            model_version=e.model_version,
            prompt=f"{e.prompt_name}@{e.prompt_version}" if e.prompt_name else None,
            attempts=e.attempt_count,
            retries=e.retry_count,
            validation_failures=e.validation_failures,
            duration_ms=e.duration_ms,
            input_tokens=e.input_tokens,
            output_tokens=e.output_tokens,
            estimated_cost_usd=e.estimated_cost_usd,
            context=e.context,
        )
