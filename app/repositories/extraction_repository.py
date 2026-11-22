"""Database access for extraction runs, executions, attempts and the knowledge they produce."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session

from app.extraction.results import AttemptRecord, ExecutionMetadata, ExtractionResult, ExtractionStatus, utcnow
from app.knowledge.fingerprint import (
    NEAR_DUPLICATE_BODY,
    SAME_TITLE_BODY,
    content_fingerprint,
    primary_passage_id,
    similarity,
    title_key,
)
from app.knowledge.records import run_to_record, unit_to_record
from app.knowledge.schemas import ExtractionRunCreate, KnowledgeUnitCreate
from app.knowledge.types import KnowledgeType
from database.models import (
    ExtractionAttempt,
    ExtractionExecution,
    ExtractionPassageRecord,
    ExtractionRun,
    KnowledgeUnit,
    KnowledgeUnitPassage,
    Passage,
)

USABLE_STATUSES = ("succeeded", "needs_review")
ACTIVE_RUN_STATUSES = ("pending", "processing")


@dataclass
class StoreOutcome:
    execution: ExtractionExecution
    stored: int = 0
    duplicates: list[tuple[int, uuid.UUID, str]] = field(default_factory=list)  # (item index, existing unit, why)
    records: list[tuple[int, KnowledgeUnit]] = field(default_factory=list)  # (item index, stored unit)
    possible_duplicates: list[tuple[int, uuid.UUID]] = field(default_factory=list)  # stored, but may repeat a unit


AfterStore = Callable[["StoreOutcome", ExtractionExecution], None]


def _json(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def attempt_to_record(
    attempt: AttemptRecord, run_id: uuid.UUID | None = None, passage_id: uuid.UUID | None = None
) -> ExtractionAttempt:
    return ExtractionAttempt(
        id=uuid.uuid4(),
        extraction_run_id=run_id,
        passage_id=passage_id,
        attempt_number=attempt.attempt,
        kind=attempt.kind,
        outcome=attempt.outcome.value,
        status=attempt.status,
        validation_outcome=attempt.validation_outcome.value,
        validation_outcomes=[o.value for o in attempt.validation_outcomes],
        model_configuration=attempt.model_configuration,
        model_configuration_id=attempt.model_configuration_id,
        provider=attempt.provider,
        model_id=attempt.model_id,
        model_version=attempt.model_version,
        prompt_name=attempt.prompt_name,
        prompt_version=attempt.prompt_version,
        generation_parameters=dict(attempt.generation_parameters),
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        duration_ms=attempt.duration_ms,
        input_tokens=attempt.input_tokens,
        output_tokens=attempt.output_tokens,
        estimated_cost_usd=attempt.estimated_cost_usd,
        finish_reason=attempt.finish_reason,
        raw_response=attempt.raw_response,
        provider_error_code=attempt.provider_error_code,
        provider_error=attempt.provider_error,
        issues=[_json(i) for i in attempt.issues],
    )


def build_execution(
    *,
    kind: str,
    status: str,
    needs_review: bool,
    review_reasons: list[str],
    meta: ExecutionMetadata,
    run_id: uuid.UUID,
    passage_id: uuid.UUID | None,
    items_stored: int,
    quality: dict | None = None,
    issues: list[dict] | None = None,
    failure_code: str | None = None,
    failure_message: str | None = None,
    notes: str | None = None,
    raw_response: str | None = None,
) -> ExtractionExecution:
    """An execution row with one attempt row per model call. Holds no credentials."""
    execution = ExtractionExecution(
        id=uuid.uuid4(),
        extraction_run_id=run_id,
        passage_id=passage_id,
        knowledge_type=kind,
        status=status,
        needs_review=needs_review,
        review_reasons=list(review_reasons),
        items_extracted=items_stored,
        model_configuration_id=meta.model_configuration_id,
        provider=meta.provider,
        model_id=meta.model_id,
        model_version=meta.model_version,
        model_configuration=meta.model_configuration,
        prompt_name=meta.prompt_name,
        prompt_version=meta.prompt_version,
        generation_parameters=dict(meta.generation_parameters),
        started_at=meta.started_at,
        completed_at=meta.completed_at,
        duration_ms=meta.duration_ms,
        attempt_count=meta.attempt_count,
        retry_count=meta.retry_count,
        validation_failures=meta.validation_failures,
        input_tokens=meta.input_tokens,
        output_tokens=meta.output_tokens,
        estimated_cost_usd=meta.estimated_cost_usd,
        context_passages_dropped=[str(p) for p in meta.context_passages_dropped],
        context=_json(meta.context) if meta.context else None,
        quality=quality,
        issues=issues or [],
        failure_code=failure_code,
        failure_message=failure_message,
        notes=notes,
        raw_response=raw_response,
    )
    execution.attempts = [attempt_to_record(a, run_id, passage_id) for a in meta.attempts]
    return execution


def execution_to_record(
    result: ExtractionResult, run_id: uuid.UUID, passage_id: uuid.UUID | None, items_stored: int
) -> ExtractionExecution:
    failure = result.failure
    return build_execution(
        kind=result.knowledge_type.value,
        status=result.status.value,
        needs_review=result.needs_review,
        review_reasons=result.review_reasons,
        meta=result.execution,
        run_id=run_id,
        passage_id=passage_id,
        items_stored=items_stored,
        quality=_json(result.quality) if result.quality else None,
        issues=[_json(i) for i in result.issues],
        failure_code=failure.code if failure else None,
        failure_message=failure.message if failure else None,
        notes=result.notes,
        raw_response=result.raw_response,
    )


class ExtractionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    # -- runs --------------------------------------------------------------------------------------

    def create_run(self, run: ExtractionRunCreate, *, total_passages: int) -> ExtractionRun:
        record = run_to_record(run)
        record.status = "pending"
        record.total_passages = total_passages
        self.session.add(record)
        self.session.commit()
        return record

    def get_run(self, run_id: uuid.UUID) -> ExtractionRun | None:
        return self.session.get(ExtractionRun, run_id)

    def list_runs(self, book_id: uuid.UUID, *, limit: int = 20) -> list[ExtractionRun]:
        return list(
            self.session.scalars(
                select(ExtractionRun)
                .where(ExtractionRun.book_id == book_id)
                .order_by(ExtractionRun.created_at.desc())
                .limit(limit)
            )
        )

    def refresh_status(self, run: ExtractionRun) -> str:
        """The run's status as stored now (another session may have cancelled it)."""
        self.session.refresh(run, ["status"])
        return run.status

    def mark_started(self, run: ExtractionRun) -> None:
        run.status = "processing"
        run.started_at = run.started_at or utcnow()
        run.completed_at = None
        run.error_code = run.error_message = None
        self.session.commit()

    def update_progress(self, run: ExtractionRun, *, passages_processed: int, skipped: int) -> None:
        run.passages_processed = passages_processed
        run.skipped_count = skipped
        self.session.commit()

    def finish(
        self, run: ExtractionRun, status: str, *, error_code: str | None = None, error_message: str | None = None, error_details: dict | None = None
    ) -> None:
        run.status = status
        run.completed_at = utcnow()
        run.error_code, run.error_message, run.error_details = error_code, error_message, error_details
        self.session.commit()

    def cancel(self, run_id: uuid.UUID) -> bool:
        """Ask a pending or processing run to stop; the worker notices before its next extraction."""
        run = self.get_run(run_id)
        if run is None or run.status not in ACTIVE_RUN_STATUSES:
            return False
        run.status = "cancelled"
        run.completed_at = utcnow()
        self.session.commit()
        return True

    def has_active_run(self, book_id: uuid.UUID, stale_after: datetime) -> bool:
        """Is a run of this book pending or processing and still recently updated?"""
        return (
            self.session.scalar(
                select(func.count())
                .select_from(ExtractionRun)
                .where(
                    ExtractionRun.book_id == book_id,
                    ExtractionRun.status.in_(ACTIVE_RUN_STATUSES),
                    ExtractionRun.updated_at >= stale_after,
                )
            )
            or 0
        ) > 0

    # -- what has been done ------------------------------------------------------------------------

    def finished_pairs(
        self, passage_ids: Sequence[uuid.UUID], categories: Iterable[KnowledgeType]
    ) -> set[tuple[uuid.UUID, KnowledgeType]]:
        """Passage/category pairs that already have a usable result in any run."""
        if not passage_ids:
            return set()
        rows = self.session.execute(
            select(ExtractionExecution.passage_id, ExtractionExecution.knowledge_type).where(
                ExtractionExecution.passage_id.in_(list(passage_ids)),
                ExtractionExecution.knowledge_type.in_([c.value for c in categories]),
                ExtractionExecution.status.in_(USABLE_STATUSES),
            )
        )
        return {(passage_id, KnowledgeType(kind)) for passage_id, kind in rows}

    def run_pairs(self, run_id: uuid.UUID) -> set[tuple[uuid.UUID, KnowledgeType]]:
        """Pairs this run already extracted usably, so a resumed run repeats only what failed or was never reached."""
        rows = self.session.execute(
            select(ExtractionExecution.passage_id, ExtractionExecution.knowledge_type).where(
                ExtractionExecution.extraction_run_id == run_id,
                ExtractionExecution.passage_id.is_not(None),
                ExtractionExecution.knowledge_type != "relationship",
                ExtractionExecution.status.in_(USABLE_STATUSES),
            )
        )
        return {(passage_id, KnowledgeType(kind)) for passage_id, kind in rows}

    # -- storing one execution ---------------------------------------------------------------------

    def store_execution(
        self,
        run: ExtractionRun,
        passage: Passage,
        result: ExtractionResult,
        units: list[KnowledgeUnitCreate],
        *,
        passages_processed: int,
        skipped: int,
        after_store: AfterStore | None = None,
    ) -> StoreOutcome:
        """Save one execution, its attempts and its new knowledge units, and update the run's totals, in one transaction.

        Units that duplicate knowledge already stored for the same source passage are not stored again.
        `after_store` runs inside the same transaction once the units exist (provenance, equations, flags).
        """
        execution = execution_to_record(result, run.id, passage.id, 0)
        self.session.add(execution)
        self.session.flush()
        outcome = StoreOutcome(execution)
        for index, unit in enumerate(units):
            duplicate = self.find_duplicate(unit)
            if duplicate is not None:
                outcome.duplicates.append((index, duplicate[0], duplicate[1]))
                execution.issues = [
                    *execution.issues,
                    {
                        "stage": "quality", "severity": "info", "code": "duplicate_skipped", "item_index": index, "path": None,
                        "message": f"Not stored again: {duplicate[1]} (existing knowledge unit {duplicate[0]}).",
                    },
                ]
                continue
            record = unit_to_record(unit)
            record.extraction_execution_id = execution.id
            record.content_fingerprint = content_fingerprint(unit)
            related = self.same_title_elsewhere(unit)
            if related:
                record.unit_metadata = {**record.unit_metadata, "same_title_as": [str(r) for r in related]}
            possible = self.find_possible_duplicate(unit)
            self.session.add(record)
            self.session.flush()
            outcome.stored += 1
            outcome.records.append((index, record))
            if possible is not None:
                outcome.possible_duplicates.append((index, possible))
        execution.items_extracted = outcome.stored
        if outcome.duplicates:
            self._mark_duplicate(execution)
        if after_store is not None:
            after_store(outcome, execution)

        run.passages_processed = passages_processed
        run.skipped_count = skipped
        run.units_extracted += outcome.stored
        run.duplicate_count += len(outcome.duplicates)
        run.attempt_count += execution.attempt_count
        run.retry_count += execution.retry_count
        run.validation_failure_count += execution.validation_failures
        run.needs_review_count += int(execution.needs_review)
        run.failure_count += int(result.status == ExtractionStatus.FAILED)
        if execution.input_tokens is not None:
            run.input_tokens = (run.input_tokens or 0) + execution.input_tokens
        if execution.output_tokens is not None:
            run.output_tokens = (run.output_tokens or 0) + execution.output_tokens
        if execution.estimated_cost_usd is not None:
            run.estimated_cost_usd = (run.estimated_cost_usd or 0.0) + execution.estimated_cost_usd
        self.session.commit()
        return outcome

    # -- duplicate detection -----------------------------------------------------------------------

    def find_duplicate(self, unit: KnowledgeUnitCreate) -> tuple[uuid.UUID, str] | None:
        """An already stored unit that is the same knowledge from the same source passage, with the reason.

        Same passage, same category and identical normalised content is an exact duplicate. Otherwise the
        words are compared: a near-identical text, or the same title with similar text, is a duplicate too.
        Another passage is never a duplicate, however similar: it is a separate occurrence.
        """
        fingerprint = content_fingerprint(unit)
        exact = self.session.scalar(
            select(KnowledgeUnit.id)
            .where(KnowledgeUnit.book_id == unit.book_id, KnowledgeUnit.content_fingerprint == fingerprint)
            .limit(1)
        )
        if exact is not None:
            return exact, "identical content from the same passage"
        primary = primary_passage_id(unit)
        candidates = self.session.execute(
            select(KnowledgeUnit.id, KnowledgeUnit.title, KnowledgeUnit.content)
            .join(KnowledgeUnitPassage, KnowledgeUnitPassage.knowledge_unit_id == KnowledgeUnit.id)
            .where(
                KnowledgeUnit.book_id == unit.book_id,
                KnowledgeUnit.knowledge_type == unit.knowledge_type.value,
                KnowledgeUnitPassage.passage_id == primary,
                KnowledgeUnitPassage.role == "primary",
            )
        ).all()
        body, key = unit.content.body, title_key(unit.content.title)
        for candidate_id, title, content in candidates:
            overlap = similarity(body, content)
            if overlap >= NEAR_DUPLICATE_BODY:
                return candidate_id, f"near-identical wording from the same passage ({overlap:.0%} of words shared)"
            if title_key(title) == key and overlap >= SAME_TITLE_BODY:
                return candidate_id, f"same title and similar wording from the same passage ({overlap:.0%} of words shared)"
        return None

    @staticmethod
    def _mark_duplicate(execution: ExtractionExecution) -> None:
        """The answer repeated stored knowledge: say so in the validation outcomes of the attempt that gave it."""
        valid = [a for a in execution.attempts if a.outcome == "valid"]
        if valid:
            last = valid[-1]
            if "duplicate" not in last.validation_outcomes:
                last.validation_outcomes = [*last.validation_outcomes, "duplicate"]
                if last.validation_outcome == "schema_valid":
                    last.validation_outcome = "duplicate"

    def find_possible_duplicate(self, unit: KnowledgeUnitCreate) -> uuid.UUID | None:
        """A stored unit of the same passage and category with the same title but different content.

        Not confident enough to skip it, so the new unit is stored and flagged for review instead.
        """
        primary = primary_passage_id(unit)
        key = title_key(unit.content.title)
        rows = self.session.execute(
            select(KnowledgeUnit.id, KnowledgeUnit.title)
            .join(KnowledgeUnitPassage, KnowledgeUnitPassage.knowledge_unit_id == KnowledgeUnit.id)
            .where(
                KnowledgeUnit.book_id == unit.book_id,
                KnowledgeUnit.knowledge_type == unit.knowledge_type.value,
                KnowledgeUnitPassage.passage_id == primary,
                KnowledgeUnitPassage.role == "primary",
            )
        ).all()
        for other_id, title in rows:
            if title_key(title) == key:
                return other_id
        return None

    def same_title_elsewhere(self, unit: KnowledgeUnitCreate, limit: int = 5) -> list[uuid.UUID]:
        """Units of the same category and title from other passages: separate occurrences, noted but not merged."""
        title = unit.content.title[:500].casefold()
        return list(
            self.session.scalars(
                select(KnowledgeUnit.id)
                .where(
                    KnowledgeUnit.book_id == unit.book_id,
                    KnowledgeUnit.knowledge_type == unit.knowledge_type.value,
                    func.lower(KnowledgeUnit.title) == title,
                )
                .order_by(KnowledgeUnit.created_at)
                .limit(limit)
            )
        )

    # -- reading -----------------------------------------------------------------------------------

    def run_executions(self, run_id: uuid.UUID) -> list[ExtractionExecution]:
        return list(
            self.session.scalars(
                select(ExtractionExecution)
                .where(ExtractionExecution.extraction_run_id == run_id)
                .order_by(ExtractionExecution.started_at)
            )
        )

    def failed_passage_count(self, run_id: uuid.UUID) -> int:
        """Passages with an extraction that failed and was not repeated successfully in the same run."""
        usable = select(ExtractionExecution.passage_id, ExtractionExecution.knowledge_type).where(
            ExtractionExecution.extraction_run_id == run_id,
            ExtractionExecution.status.in_(USABLE_STATUSES),
            ExtractionExecution.passage_id.is_not(None),  # a NULL would make NOT IN match nothing
        )
        return (
            self.session.scalar(
                select(func.count(func.distinct(ExtractionExecution.passage_id))).where(
                    ExtractionExecution.extraction_run_id == run_id,
                    ExtractionExecution.status == "failed",
                    ExtractionExecution.knowledge_type != "relationship",
                    tuple_(ExtractionExecution.passage_id, ExtractionExecution.knowledge_type).not_in(usable),
                )
            )
            or 0
        )

    def knowledge(
        self,
        book_id: uuid.UUID,
        *,
        knowledge_type: KnowledgeType | None = None,
        run_id: uuid.UUID | None = None,
        review_status: str | None = None,
        validation_status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[KnowledgeUnit]:
        query = select(KnowledgeUnit).where(KnowledgeUnit.book_id == book_id)
        if knowledge_type is not None:
            query = query.where(KnowledgeUnit.knowledge_type == knowledge_type.value)
        if run_id is not None:
            query = query.where(KnowledgeUnit.extraction_run_id == run_id)
        if review_status is not None:
            query = query.where(KnowledgeUnit.review_status == review_status)
        if validation_status is not None:
            query = query.where(KnowledgeUnit.validation_status == validation_status)
        query = query.order_by(KnowledgeUnit.page_start.nulls_last(), KnowledgeUnit.created_at, KnowledgeUnit.id).offset(offset)
        if limit is not None:
            query = query.limit(limit)
        return list(self.session.scalars(query))

    def count_knowledge(self, book_id: uuid.UUID) -> int:
        return self.session.scalar(select(func.count()).select_from(KnowledgeUnit).where(KnowledgeUnit.book_id == book_id)) or 0


    # -- passage records ---------------------------------------------------------------------------

    def create_passage_records(self, run: ExtractionRun, passages: Sequence) -> None:
        """One pending record per selected passage, so a resumed run knows what was never reached."""
        for passage in passages:
            self.session.add(
                ExtractionPassageRecord(
                    id=uuid.uuid4(), extraction_run_id=run.id, passage_id=passage.id, document_sequence=passage.document_sequence
                )
            )
        self.session.commit()

    def passage_records(self, run_id: uuid.UUID) -> list[ExtractionPassageRecord]:
        return list(
            self.session.scalars(
                select(ExtractionPassageRecord)
                .where(ExtractionPassageRecord.extraction_run_id == run_id)
                .order_by(ExtractionPassageRecord.document_sequence, ExtractionPassageRecord.id)
            )
        )

    def passage_record(self, run_id: uuid.UUID, passage_id: uuid.UUID) -> ExtractionPassageRecord | None:
        return self.session.scalars(
            select(ExtractionPassageRecord).where(
                ExtractionPassageRecord.extraction_run_id == run_id, ExtractionPassageRecord.passage_id == passage_id
            )
        ).first()

    def passage_status_counts(self, run_id: uuid.UUID) -> dict[str, int]:
        rows = self.session.execute(
            select(ExtractionPassageRecord.status, func.count())
            .where(ExtractionPassageRecord.extraction_run_id == run_id)
            .group_by(ExtractionPassageRecord.status)
        )
        return {status: total for status, total in rows}

    def errored_passage_count(self, run_id: uuid.UUID) -> int:
        """Passages that finished needing a look because a step failed (such as relationships): a resume retries them."""
        return (
            self.session.scalar(
                select(func.count()).select_from(ExtractionPassageRecord).where(
                    ExtractionPassageRecord.extraction_run_id == run_id,
                    ExtractionPassageRecord.status == "needs_review",
                    ExtractionPassageRecord.error_code.is_not(None),
                )
            )
            or 0
        )

    def start_passage(self, run_id: uuid.UUID, passage_id: uuid.UUID) -> None:
        record = self.passage_record(run_id, passage_id)
        if record is not None:
            record.status = "processing"
            record.started_at = record.started_at or utcnow()
            record.completed_at = None
            record.error_code = record.error_message = None
            self.session.commit()

    def finish_passage(self, run_id: uuid.UUID, passage_id: uuid.UUID) -> str | None:
        """Work out the passage's status from its executions in this run and store it with its counts.

        Per category the latest execution counts: a failure that was later repeated successfully is history.
        """
        record = self.passage_record(run_id, passage_id)
        if record is None:
            return None
        rows = list(
            self.session.scalars(
                select(ExtractionExecution)
                .where(ExtractionExecution.extraction_run_id == run_id, ExtractionExecution.passage_id == passage_id)
                .order_by(ExtractionExecution.started_at)
            )
        )
        latest: dict[str, ExtractionExecution] = {}
        for row in rows:
            latest[row.knowledge_type] = row
        # A failed knowledge extraction fails the passage. A failed relationship step only needs a look: the
        # passage's knowledge is intact.
        failed = [e for e in latest.values() if e.status == "failed" and e.knowledge_type != "relationship"]
        relationship_failed = [e for e in latest.values() if e.status == "failed" and e.knowledge_type == "relationship"]
        if failed:
            status = "failed"
            record.error_code, record.error_message = failed[-1].failure_code, failed[-1].failure_message
        else:
            status = "needs_review" if any(e.needs_review for e in latest.values()) else "completed"
            record.error_code = relationship_failed[-1].failure_code if relationship_failed else None
            record.error_message = relationship_failed[-1].failure_message if relationship_failed else None
        record.status = status
        record.completed_at = utcnow()
        record.execution_count = len(rows)
        record.attempt_count = sum(e.attempt_count for e in rows)
        record.retry_count = sum(e.retry_count for e in rows)
        record.validation_failures = sum(e.validation_failures for e in rows)
        record.units_extracted = sum(e.items_extracted for e in rows if e.knowledge_type != "relationship")
        self.session.commit()
        return status

    def relationships_done(self, passage_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        """Passages whose relationships were already extracted with a usable result (in any run)."""
        if not passage_ids:
            return set()
        return set(
            self.session.scalars(
                select(ExtractionExecution.passage_id).where(
                    ExtractionExecution.passage_id.in_(list(passage_ids)),
                    ExtractionExecution.knowledge_type == "relationship",
                    ExtractionExecution.status.in_(USABLE_STATUSES),
                )
            )
        )

    def store_relationship_execution(
        self, run: ExtractionRun, execution: ExtractionExecution, *, created: int, passages_processed: int | None = None
    ) -> None:
        """Save a relationship execution and update the run's totals. Commits."""
        self.session.add(execution)
        self.session.flush()
        run.relationships_extracted += created
        run.attempt_count += execution.attempt_count
        run.retry_count += execution.retry_count
        run.validation_failure_count += execution.validation_failures
        run.needs_review_count += int(execution.needs_review)
        run.failure_count += int(execution.status == "failed")
        if execution.input_tokens is not None:
            run.input_tokens = (run.input_tokens or 0) + execution.input_tokens
        if execution.output_tokens is not None:
            run.output_tokens = (run.output_tokens or 0) + execution.output_tokens
        if execution.estimated_cost_usd is not None:
            run.estimated_cost_usd = (run.estimated_cost_usd or 0.0) + execution.estimated_cost_usd
        self.session.commit()

    # -- units that no model produced --------------------------------------------------------------

    def store_source_units(
        self,
        run: ExtractionRun,
        units: list[KnowledgeUnitCreate],
        *,
        after_store: Callable[[list[KnowledgeUnit]], None] | None = None,
    ) -> tuple[list[KnowledgeUnit], list[tuple[uuid.UUID, str]]]:
        """Store units created from stored source data (no execution). Duplicates are skipped. Commits."""
        stored: list[KnowledgeUnit] = []
        duplicates: list[tuple[uuid.UUID, str]] = []
        for unit in units:
            duplicate = self.find_duplicate(unit)
            if duplicate is not None:
                duplicates.append(duplicate)
                continue
            record = unit_to_record(unit)
            record.content_fingerprint = content_fingerprint(unit)
            self.session.add(record)
            self.session.flush()
            stored.append(record)
        if after_store is not None and stored:
            after_store(stored)
        run.units_extracted += len(stored)
        run.duplicate_count += len(duplicates)
        self.session.commit()
        return stored, duplicates

    def unit_equation_ids(self, equation_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        """Which of these Phase 1 equations already have a knowledge unit."""
        if not equation_ids:
            return set()
        return set(
            self.session.scalars(select(KnowledgeUnit.equation_id).where(KnowledgeUnit.equation_id.in_(list(equation_ids))))
        )
