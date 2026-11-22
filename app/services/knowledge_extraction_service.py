"""The knowledge extraction pipeline: passages in, validated, verified and stored knowledge out.

It connects Phase 1 (stored passages and structure) with the extraction layer (`app.extraction`):

  1. resolve the scope (book, chapter or section) and the model configuration, and validate both
  2. retrieve the passages in reading order, and record each one as pending in the run
  3. prepare each passage's context for the model's context window, recording any reduction
  4. extract every requested category with the versioned prompts
  5. validate, convert the results into knowledge units, skip duplicates, and persist
  6. in the same transaction: capture and verify the units' sources, record equation captures, raise
     review flags, and keep every attempt with its model, configuration snapshot and validation outcome
  7. optionally extract the relationships between the units of a passage
  8. record the passage's status and the run's progress, so a failed or interrupted run can resume

A run uses one model configuration, chosen explicitly or taken from the application default. Each
model call is committed on its own, so a failed passage never ends the run and an interrupted run keeps
its results; resuming repeats only what failed or was never reached. Phase 1 data is only read.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.extraction.errors import ExtractionError, ModelConfigError
from app.extraction.model_config import LlmModelConfig, ModelRegistry
from app.extraction.prompts import get_prompt
from app.extraction.providers import LlmProvider
from app.extraction.results import (
    ExtractedKnowledge,
    ExtractionFailure,
    ExtractionResult,
    ExtractionStatus,
    utcnow,
)
from app.extraction.service import ExtractionService, ProviderFactory
from app.extraction.settings import ExtractionSettings, load_environment, load_model_registry
from app.extraction.validation import DEFAULT_SEMANTIC_VALIDATORS, SemanticValidator
from app.knowledge.content import EquationContent
from app.knowledge.records import run_from_record, unit_from_record
from app.knowledge.schemas import (
    ExtractionRunCreate,
    ExtractionRunRead,
    KnowledgeUnitCreate,
    KnowledgeUnitRead,
    content_from_details,
)
from app.knowledge.types import EquationCaptureStatus, FlagSeverity, KnowledgeType, ReviewFlagCode
from app.repositories.extraction_repository import ExtractionRepository, StoreOutcome
from app.repositories.passage_repository import ExtractionScope, PassageRepository
from app.services.context_preparation import ContextPreparer, source_info
from app.services.equation_service import EquationService, flag_code_for
from app.services.errors import ExtractionRunError, ExtractionScopeError
from app.services.provenance_service import ProvenanceService
from app.services.relationship_extraction import RelationshipExtractionService
from app.services.review_flag_service import ReviewFlagService
from database.models import ExtractionExecution, ExtractionRun, KnowledgeUnit, Passage

logger = logging.getLogger(__name__)

# A pending or processing run counts as active for this long after its last update.
ACTIVE_RUN_TIMEOUT = timedelta(minutes=15)
_STARTABLE = ("pending", "failed", "cancelled")
_FINISHED = ("completed", "needs_review")
RELATIONSHIP_KIND = "relationship"


class NothingToExtractError(ExtractionRunError):
    """Every passage and category in the scope already has a usable result."""


# -- what callers see ------------------------------------------------------------------------------


class ModelOption(BaseModel):
    id: str
    provider: str
    model: str
    deployment: str
    context_window: int | None
    active: bool
    ready: bool
    reason: str | None = None  # why it cannot be used, when it cannot

    @property
    def label(self) -> str:
        return f"{self.id} ({self.model}, {self.deployment})"


class ExtractionPlan(BaseModel):
    """What a run would do, shown before it starts."""

    scope_kind: str
    total_passages: int
    categories: list[str]
    model_configurations: dict[str, str]  # category -> configuration id
    model_label: str
    model_calls: int  # passage x category pairs that still need extracting
    skipped: int  # pairs that already have a usable result
    relationship_calls: int = 0  # extra calls to relate units, when requested


class ResumeInfo(BaseModel):
    """Whether a run can be continued, and what is left. Nothing completed is repeated."""

    run_id: uuid.UUID
    status: str
    can_resume: bool
    reason: str | None
    total: int
    completed: int  # finished (clean or needing review)
    failed: int
    pending: int  # never reached
    processing: int  # started and not finished: interrupted


class RunStatus(BaseModel):
    """A run's state and progress, for any caller (screen, command line, other services)."""

    run: ExtractionRunRead
    failed_passages: int
    progress: float  # 0..1 of the selected passages visited
    is_active: bool
    model_label: str | None
    duration_seconds: float | None
    passage_counts: dict[str, int] = {}  # status -> number of passages
    can_resume: bool = False

    @property
    def status(self) -> str:
        return self.run.status.value

    @property
    def error(self) -> str | None:
        return self.run.error_message


class PassageStatusView(BaseModel):
    """One passage in a run: where it stands, its work so far, and why it failed if it did."""

    passage_id: uuid.UUID | None
    sequence: int
    page_start: int | None
    status: str
    categories: dict[str, str]  # category (or "relationship") -> status of its latest execution
    executions: int
    attempts: int
    retries: int
    validation_failures: int
    units_extracted: int
    started_at: datetime | None
    completed_at: datetime | None
    error_code: str | None
    error_message: str | None


@dataclass(frozen=True)
class ExtractionProgress:
    """Reported after every extraction (one passage, one category) and every relationship step."""

    passages_done: int
    total_passages: int
    passage_id: uuid.UUID
    page_start: int | None
    category: KnowledgeType | None
    status: str
    items: int
    duplicates: int
    duration_ms: int
    attempts: int
    failure_code: str | None
    step: str = "knowledge"  # "knowledge", or "relationships" (then category is None)


ProgressCallback = Callable[[ExtractionProgress], None]


def _label(run: ExtractionRun) -> str | None:
    if run.model_configuration_id:
        return f"{run.model_configuration_id} ({run.model_name})" if run.model_name else run.model_configuration_id
    return run.model_name


# -- the service -----------------------------------------------------------------------------------


class KnowledgeExtractionService:
    def __init__(
        self,
        session: Session,
        registry: ModelRegistry,
        settings: ExtractionSettings | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        provider_factory: Callable[[LlmModelConfig], LlmProvider] | None = None,
        semantic_validators: Iterable[SemanticValidator] = DEFAULT_SEMANTIC_VALIDATORS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.session = session
        self.registry = registry
        self.settings = settings or ExtractionSettings()
        self._environ = environ
        self._provider_factory: ProviderFactory | None = provider_factory
        self._validators = tuple(semantic_validators)
        self._sleep = sleep
        self.passages = PassageRepository(session)
        self.runs = ExtractionRepository(session)
        self.provenance = ProvenanceService(session)
        self.equations = EquationService(session)
        self.flags = ReviewFlagService(session)

    # -- models ------------------------------------------------------------------------------------

    def available_models(self) -> list[ModelOption]:
        options = []
        for config in self.registry.all():
            reason = None
            try:
                config.check_ready(self._environ)
            except ExtractionError as exc:
                reason = str(exc)
            options.append(
                ModelOption(
                    id=config.id,
                    provider=config.provider.value,
                    model=config.model,
                    deployment=config.deployment.value,
                    context_window=config.context_window,
                    active=config.active,
                    ready=reason is None,
                    reason=reason,
                )
            )
        return options

    def _registry_for(self, model_id: str | None) -> ModelRegistry:
        """The registry a run uses: the selected model for every category, else the application setup."""
        if model_id is None:
            return self.registry
        try:
            self.registry.get(model_id)
            return ModelRegistry(self.registry.all(), default_id=model_id)  # one model per run: no per-category overrides
        except ModelConfigError as exc:
            raise ExtractionRunError(str(exc)) from exc

    def _extractor(self, registry: ModelRegistry) -> ExtractionService:
        return ExtractionService(
            registry,
            self.settings,
            provider_factory=self._provider_factory,
            semantic_validators=self._validators,
            environ=self._environ,
            sleep=self._sleep,
        )

    def _configs(self, extractor: ExtractionService, categories: list[KnowledgeType]) -> dict[KnowledgeType, LlmModelConfig]:
        """The model configuration of each category, checked before anything starts (exists, active, key present)."""
        try:
            return {category: extractor.check_ready(category.value) for category in categories}
        except ExtractionError as exc:
            raise ExtractionRunError(str(exc)) from exc

    def _relationship_config(self, extractor: ExtractionService) -> LlmModelConfig:
        """The model for relationship extraction: its own override if configured, else the run's default."""
        try:
            return extractor.check_ready(RELATIONSHIP_KIND)
        except ExtractionError as exc:
            raise ExtractionRunError(str(exc)) from exc

    # -- scopes ------------------------------------------------------------------------------------

    def book_scope(self, book_id: uuid.UUID, document_id: uuid.UUID | None = None) -> ExtractionScope:
        return self.passages.book_scope(book_id, document_id)

    def chapter_scope(self, node_id: uuid.UUID) -> ExtractionScope:
        return self.passages.chapter_scope(node_id)

    def section_scope(self, node_id: uuid.UUID) -> ExtractionScope:
        return self.passages.section_scope(node_id)

    # -- planning and starting ---------------------------------------------------------------------

    def plan(
        self,
        scope: ExtractionScope,
        categories: Iterable[KnowledgeType | str],
        *,
        model_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        force: bool = False,
        extract_relationships: bool = False,
    ) -> ExtractionPlan:
        kinds = self._categories(categories)
        extractor = self._extractor(self._registry_for(model_id))
        configs = self._configs(extractor, kinds)
        if extract_relationships:
            self._relationship_config(extractor)
        passages = self.passages.select(scope, limit=limit, offset=offset)
        done = set() if force else self.runs.finished_pairs([p.id for p in passages], kinds)
        calls = sum((p.id, c) not in done for p in passages for c in kinds)
        relationship_calls = 0
        if extract_relationships:
            finished = set() if force else self.runs.relationships_done([p.id for p in passages])
            relationship_calls = sum(p.id not in finished for p in passages)
        return ExtractionPlan(
            scope_kind=scope.kind,
            total_passages=len(passages),
            categories=[c.value for c in kinds],
            model_configurations={c.value: cfg.id for c, cfg in configs.items()},
            model_label=", ".join(dict.fromkeys(f"{cfg.id} ({cfg.model})" for cfg in configs.values())),
            model_calls=calls,
            skipped=len(passages) * len(kinds) - calls,
            relationship_calls=relationship_calls,
        )

    def create_run(
        self,
        scope: ExtractionScope,
        categories: Iterable[KnowledgeType | str],
        *,
        model_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        force: bool = False,
        context_passages: int | None = None,
        extract_relationships: bool = False,
    ) -> uuid.UUID:
        """Validate everything and record a pending run. Raises ExtractionRunError with a clear message otherwise."""
        kinds = self._categories(categories)
        extractor = self._extractor(self._registry_for(model_id))
        configs = self._configs(extractor, kinds)
        plan = self.plan(
            scope, kinds, model_id=model_id, limit=limit, offset=offset, force=force, extract_relationships=extract_relationships
        )
        if plan.total_passages == 0:
            raise ExtractionRunError("No passages were found. Process the document's structure and passages first.")
        if plan.model_calls == 0 and plan.relationship_calls == 0:
            raise NothingToExtractError(
                "Everything in this scope has already been extracted with a usable result. Use force to extract it again."
            )
        if self.runs.has_active_run(scope.book_id, utcnow() - ACTIVE_RUN_TIMEOUT):
            raise ExtractionRunError("An extraction of this book is already running.")

        unique = list({config.id: config for config in configs.values()}.values())
        versions = {get_prompt(c, self.settings.prompt_versions.get(c.value)).version for c in kinds}
        single = unique[0] if len(unique) == 1 else None
        run = self.runs.create_run(
            ExtractionRunCreate(
                book_id=scope.book_id,
                document_id=scope.document_id,
                structural_node_id=scope.node_id,
                scope_type=scope.kind,
                model_name=", ".join(dict.fromkeys(c.model for c in unique))[:200],
                prompt_version=versions.pop() if len(versions) == 1 else "mixed",
                model_configuration_id=single.id if single else None,
                provider=single.provider.value if single else None,
                model_snapshot=single.model_dump(mode="json") if single else None,
                parameters={
                    "categories": [c.value for c in kinds],
                    "requested_model_id": model_id,
                    "model_configurations": plan.model_configurations,
                    "limit": limit,
                    "offset": offset,
                    "force": force,
                    "context_passages": self.settings.context_passages if context_passages is None else context_passages,
                    "extract_relationships": extract_relationships,
                },
            ),
            total_passages=plan.total_passages,
        )
        self.runs.create_passage_records(run, self.passages.select(scope, limit=limit, offset=offset))
        return run.id

    def extract_book(self, book_id: uuid.UUID, categories: Iterable[KnowledgeType | str], *, document_id: uuid.UUID | None = None, progress: ProgressCallback | None = None, **options) -> RunStatus:
        return self._run(self.book_scope(book_id, document_id), categories, progress, options)

    def extract_chapter(self, node_id: uuid.UUID, categories: Iterable[KnowledgeType | str], *, progress: ProgressCallback | None = None, **options) -> RunStatus:
        return self._run(self.chapter_scope(node_id), categories, progress, options)

    def extract_section(self, node_id: uuid.UUID, categories: Iterable[KnowledgeType | str], *, progress: ProgressCallback | None = None, **options) -> RunStatus:
        return self._run(self.section_scope(node_id), categories, progress, options)

    def _run(self, scope: ExtractionScope, categories, progress, options) -> RunStatus:
        run_id = self.create_run(scope, categories, **options)
        return self.execute_run(run_id, progress=progress)

    # -- resuming ----------------------------------------------------------------------------------

    def get_resume_info(self, run_id: uuid.UUID) -> ResumeInfo:
        """What a run has finished, what failed and what was never reached; and whether it can be continued."""
        run = self.runs.get_run(run_id)
        if run is None:
            raise ExtractionRunError("The extraction run does not exist.")
        counts = self.runs.passage_status_counts(run_id)
        # "failed" also counts passages that finished with a failed step (a relationship call), which a resume retries;
        # they are not counted as completed.
        errored = self.runs.errored_passage_count(run_id)
        failed = counts.get("failed", 0) + errored
        completed = counts.get("completed", 0) + counts.get("needs_review", 0) - errored
        pending, processing = counts.get("pending", 0), counts.get("processing", 0)
        can, reason = True, None
        if run.status == "processing" and run.updated_at >= utcnow() - ACTIVE_RUN_TIMEOUT:
            can, reason = False, "The run is still being processed."
        elif run.status == "pending":
            can, reason = False, "The run has not started yet."
        elif run.status in _FINISHED and not (failed or pending or processing):
            can, reason = False, "Every passage has finished; there is nothing to resume."
        elif run.status in _STARTABLE or run.status == "processing":
            reason = "Continues with the failed and unfinished passages; completed ones are not repeated."
        else:
            reason = f"{failed + pending + processing} passage(s) failed or were not reached."
        return ResumeInfo(
            run_id=run_id, status=run.status, can_resume=can, reason=reason, total=run.total_passages or sum(counts.values()),
            completed=completed, failed=failed, pending=pending, processing=processing,
        )

    def resume_run(
        self, run_id: uuid.UUID, progress: ProgressCallback | None = None, *, takeover: bool = False, redo_completed: bool = False
    ) -> RunStatus:
        """Continue a failed, cancelled or interrupted run (or a finished one that still has failed passages).

        Passages that completed are not processed again, unless `redo_completed` asks for it explicitly (the
        results are then compared with what is stored: duplicates are never stored twice). `takeover` resumes a
        run that looks like it is still processing (for example after a crash less than a few minutes ago).
        """
        info = self.get_resume_info(run_id)
        if redo_completed:
            if info.status == "processing" and not takeover and not info.can_resume:
                raise ExtractionRunError(info.reason or "The run is still being processed.")
        elif not info.can_resume and not (takeover and info.status == "processing"):
            raise ExtractionRunError(info.reason or "The run cannot be resumed.")
        return self.execute_run(run_id, progress=progress, takeover=takeover, redo=redo_completed)

    def get_passage_status(self, run_id: uuid.UUID) -> list[PassageStatusView]:
        """Every selected passage of a run with its processing status, in reading order."""
        if self.runs.get_run(run_id) is None:
            raise ExtractionRunError("The extraction run does not exist.")
        latest: dict[tuple[uuid.UUID | None, str], str] = {}
        for execution in self.runs.run_executions(run_id):
            latest[(execution.passage_id, execution.knowledge_type)] = execution.status
        views = []
        for record in self.runs.passage_records(run_id):
            passage = self.session.get(Passage, record.passage_id) if record.passage_id else None
            views.append(
                PassageStatusView(
                    passage_id=record.passage_id,
                    sequence=record.document_sequence,
                    page_start=passage.page_start if passage else None,
                    status=record.status,
                    categories={kind: status for (pid, kind), status in latest.items() if pid == record.passage_id},
                    executions=record.execution_count,
                    attempts=record.attempt_count,
                    retries=record.retry_count,
                    validation_failures=record.validation_failures,
                    units_extracted=record.units_extracted,
                    started_at=record.started_at,
                    completed_at=record.completed_at,
                    error_code=record.error_code,
                    error_message=record.error_message,
                )
            )
        return views

    # -- executing ---------------------------------------------------------------------------------

    def execute_run(
        self, run_id: uuid.UUID, progress: ProgressCallback | None = None, *, takeover: bool = False, redo: bool = False
    ) -> RunStatus:
        """Run a pending run, or continue a failed, cancelled, interrupted or partly failed one, to the end."""
        run = self.runs.get_run(run_id)
        if run is None:
            raise ExtractionRunError("The extraction run does not exist.")
        if run.status == "processing" and not takeover and run.updated_at >= utcnow() - ACTIVE_RUN_TIMEOUT:
            raise ExtractionRunError("The run is still being processed. Wait for it, or resume it with takeover if it was interrupted.")
        if run.status in _FINISHED and not redo:
            info = self.get_resume_info(run_id)
            if not info.can_resume:
                raise ExtractionRunError(f"The run has already finished ({run.status}).")
        elif run.status not in (*_STARTABLE, "processing", *_FINISHED):
            raise ExtractionRunError(f"The run has already finished ({run.status}).")
        params = run.parameters or {}
        kinds = self._categories(params.get("categories", []))
        try:
            scope = self.passages.scope_from_ids(run.book_id, run.document_id, run.structural_node_id, run.scope_type)
            extractor = self._extractor(self._registry_for(params.get("requested_model_id")))
            configs = self._configs(extractor, kinds)
            relationship_config = self._relationship_config(extractor) if params.get("extract_relationships") else None
        except ExtractionRunError as exc:
            code = "scope_invalid" if isinstance(exc, ExtractionScopeError) else "model_configuration_invalid"
            self.runs.finish(run, "failed", error_code=code, error_message=str(exc))
            return self.get_run_status(run_id)

        passages = self.passages.select(scope, limit=params.get("limit"), offset=params.get("offset", 0))
        done = set() if redo else self.runs.run_pairs(run.id)
        relationships_finished: set[uuid.UUID] = set()
        if not params.get("force") and not redo:
            done |= self.runs.finished_pairs([p.id for p in passages], kinds)
            relationships_finished = self.runs.relationships_done([p.id for p in passages])
        preparer = ContextPreparer(self.passages, self.settings, neighbours=params.get("context_passages"))
        book = self.passages.get_book(run.book_id)
        source = source_info(book) if book else None
        relationship_extractor = (
            RelationshipExtractionService(self.session, extractor, context_units=self.settings.relationship_context_units)
            if relationship_config
            else None
        )

        self.runs.mark_started(run)
        state = _LoopState()
        try:
            for passage in passages:
                if self.runs.refresh_status(run) == "cancelled":
                    return self.get_run_status(run_id)
                state.visited += 1
                self.runs.start_passage(run.id, passage.id)
                handled = False
                for category in kinds:
                    if (passage.id, category) in done:
                        state.skipped += 1
                        continue
                    if self.runs.refresh_status(run) == "cancelled":
                        return self.get_run_status(run_id)
                    handled = True
                    self._extract_pair(run, passage, category, configs[category], preparer, extractor, source, state, progress)
                    if self._should_stop(state):
                        break
                if relationship_extractor and not self._should_stop(state) and passage.id not in relationships_finished:
                    if self.runs.refresh_status(run) == "cancelled":
                        return self.get_run_status(run_id)
                    handled |= self._extract_relationships(
                        run, passage, relationship_extractor, relationship_config, preparer, source, state, progress
                    )
                self.runs.finish_passage(run.id, passage.id)
                if not handled:
                    self.runs.update_progress(run, passages_processed=state.visited, skipped=state.skipped)
                if self._should_stop(state):
                    self.runs.finish(
                        run, "failed", error_code="too_many_failures",
                        error_message=f"Stopped after {state.consecutive_failures} consecutive failures. Last error: {state.last_error}",
                    )
                    return self.get_run_status(run_id)
        except KeyboardInterrupt:
            self.session.rollback()
            self.runs.finish(run, "cancelled", error_code="interrupted", error_message="Interrupted.")
            raise
        except Exception as exc:
            self.session.rollback()
            logger.exception("Extraction run %s stopped unexpectedly", run_id)
            self.runs.finish(run, "failed", error_code="run_error", error_message=f"{type(exc).__name__}: {exc}")
            raise

        self._finish(run, state)
        return self.get_run_status(run_id)

    def _should_stop(self, state: _LoopState) -> bool:
        limit = self.settings.max_consecutive_failures
        return bool(limit) and state.consecutive_failures >= limit

    def _finish(self, run: ExtractionRun, state: _LoopState) -> None:
        """The run's status follows where its passages stand now: a failure that was repeated successfully is history."""
        counts = self.runs.passage_status_counts(run.id)
        failed, review = counts.get("failed", 0), counts.get("needs_review", 0)
        succeeded = counts.get("completed", 0) + review
        if failed and not succeeded:
            self.runs.finish(
                run, "failed", error_code="all_extractions_failed",
                error_message=f"Every extraction failed. Last error: {state.last_error or 'see the passage statuses'}",
            )
        elif failed or review or counts.get("pending", 0) or counts.get("processing", 0):
            self.runs.finish(run, "needs_review")
        else:
            self.runs.finish(run, "completed")

    # -- one passage, one category -----------------------------------------------------------------

    def _extract_pair(self, run, passage, category, config, preparer, extractor, source, state, progress) -> None:
        prepared = preparer.prepare(passage, category, config, source)
        try:
            result = extractor.extract(prepared.request)
        except Exception as exc:  # a bug or an unexpected provider failure must not end the run
            logger.exception("Extraction of passage %s (%s) crashed", passage.id, category.value)
            result = extractor.failure_result(prepared.request, "unexpected_error", f"{type(exc).__name__}: {exc}")
        units = self._to_units(result, passage, run)
        hook = self._after_store(run, passage, category, result)
        try:
            outcome = self.runs.store_execution(
                run, passage, result, units, passages_processed=state.visited, skipped=state.skipped, after_store=hook
            )
        except Exception as exc:
            self.session.rollback()
            logger.exception("Storing the result for passage %s (%s) failed", passage.id, category.value)
            result = result.model_copy(
                update={
                    "status": ExtractionStatus.FAILED, "items": [], "needs_review": True,
                    "failure": ExtractionFailure(code="storage_failed", message=f"{type(exc).__name__}: {exc}"),
                }
            )
            outcome = self.runs.store_execution(
                run, passage, result, [], passages_processed=state.visited, skipped=state.skipped, after_store=self._after_store(run, passage, category, result)
            )
        if category == KnowledgeType.EQUATION and result.status != ExtractionStatus.FAILED:
            self._keep_unread_equations(run, passage)
        failed = result.status == ExtractionStatus.FAILED
        state.consecutive_failures = state.consecutive_failures + 1 if failed else 0
        if failed:
            state.last_error = result.failure.message if result.failure else "extraction failed"
        if progress:
            meta = result.execution
            progress(
                ExtractionProgress(
                    passages_done=state.visited,
                    total_passages=run.total_passages,
                    passage_id=passage.id,
                    page_start=passage.page_start,
                    category=category,
                    status=result.status.value,
                    items=outcome.stored,
                    duplicates=len(outcome.duplicates),
                    duration_ms=meta.duration_ms,
                    attempts=meta.attempt_count,
                    failure_code=result.failure.code if result.failure else None,
                )
            )

    def _to_units(self, result: ExtractionResult, passage: Passage, run: ExtractionRun) -> list[KnowledgeUnitCreate]:
        """The Day 1 unit schemas for a result. If one cannot be built the whole execution fails: nothing half stored."""
        if not result.items:
            return []
        ties = self._ties(passage)
        try:
            return [item.to_unit_create(passage.book_id, passage.document_id, run.id, **ties) for item in result.items]
        except ValidationError as exc:
            message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])
            logger.error("Extracted knowledge could not be stored as a knowledge unit: %s", message)
            result.status, result.needs_review, result.items = ExtractionStatus.FAILED, True, []
            result.failure = ExtractionFailure(code="unit_conversion_failed", message=message)
            return []

    def _ties(self, passage: Passage) -> dict[str, uuid.UUID | None]:
        located = self.passages.source_of(passage)
        return {
            "chapter_node_id": located.chapter.id if located.chapter else None,
            "section_node_id": located.section.id if located.section else None,
        }

    # -- provenance, equations and review flags, inside the storing transaction --------------------

    def _after_store(self, run: ExtractionRun, passage: Passage, category: KnowledgeType, result: ExtractionResult):
        def hook(outcome: StoreOutcome, execution: ExtractionExecution) -> None:
            equations = self.equations.equations_in(passage) if category == KnowledgeType.EQUATION else []
            for index, record in outcome.records:
                self._finalize_unit(run, passage, record, result.items[index], equations, execution)
            records = dict(outcome.records)
            for index, other_id in outcome.possible_duplicates:
                self.flags.flag_unit(
                    passage.book_id, records[index].id, ReviewFlagCode.POTENTIAL_DUPLICATE,
                    "A unit with the same title was already extracted from this passage with different content.",
                    details={"other_unit_id": str(other_id)}, dedupe_key=str(other_id), run_id=run.id, commit=False,
                )
            self._flag_attempts(run, passage, execution)

        return hook

    def _finalize_unit(
        self,
        run: ExtractionRun,
        passage: Passage,
        record: KnowledgeUnit,
        item: ExtractedKnowledge | None,
        equations,
        execution: ExtractionExecution | None,
    ) -> None:
        """Capture and verify the unit's source, record its equation capture, and raise the flags that apply."""
        book_id = passage.book_id
        self.provenance.capture(record)
        if record.knowledge_type == KnowledgeType.EQUATION.value:
            content = item.content if item else content_from_details(record.knowledge_type, record.details)
            if isinstance(content, EquationContent):
                capture = self.equations.record_capture(record, passage, content, equations)
                code = flag_code_for(EquationCaptureStatus(capture.capture_status))
                if code:
                    self.flags.flag_unit(
                        book_id, record.id, code, " ".join(capture.capture_notes) or f"Equation capture: {capture.capture_status}.",
                        severity=FlagSeverity.WARNING, details={"capture_status": capture.capture_status}, run_id=run.id, commit=False,
                    )
        self.session.flush()
        self.provenance.verify_unit(record.id, flag=True, commit=False, run_id=run.id)
        if item is None:
            return

        def flag(code: ReviewFlagCode, message: str, *, severity=FlagSeverity.WARNING, key: str = "", raised_by: str = "system") -> None:
            self.flags.flag_unit(book_id, record.id, code, message, severity=severity, dedupe_key=key, raised_by=raised_by, run_id=run.id, commit=False)

        if item.uncertainty.is_uncertain:
            flag(ReviewFlagCode.UNCERTAIN_EXTRACTION, item.uncertainty.notes or "The model reported uncertainty about this item.", raised_by="model")
        if item.basis != "explicit":
            flag(ReviewFlagCode.INFERRED_CONTENT, f"Some of this is {item.basis}, not explicitly stated in the source.", severity=FlagSeverity.INFO, raised_by="model")
        if item.confidence is not None and item.confidence < self.settings.low_confidence_threshold:
            flag(ReviewFlagCode.LOW_CONFIDENCE, f"The model's own confidence is {item.confidence:.0%}.", raised_by="model")
        for issue in item.issues:
            if issue.code in ("quote_not_in_source", "expression_not_in_source"):
                flag(ReviewFlagCode.UNSUPPORTED_CLAIM, issue.message, key=f"{issue.code}:{issue.path or ''}")
        if execution is not None and execution.context and execution.context.get("reduced"):
            flag(ReviewFlagCode.CONTEXT_REDUCED, "Neighbouring passages were left out to fit the model's context window.", severity=FlagSeverity.INFO)

    def _flag_attempts(self, run: ExtractionRun, passage: Passage, execution: ExtractionExecution) -> None:
        """Flag the last attempt of an execution that could not be resolved, or needed repeated corrections."""
        if not execution.attempts:
            return
        last = execution.attempts[-1]
        if execution.status == "failed":
            self.flags.flag_attempt(
                passage.book_id, last.id, ReviewFlagCode.UNRESOLVED_EXTRACTION_ERROR,
                execution.failure_message or "The extraction failed.", severity=FlagSeverity.ERROR,
                details={"code": execution.failure_code, "attempts": execution.attempt_count}, run_id=run.id, commit=False,
            )
        if execution.validation_failures >= 2:
            self.flags.flag_attempt(
                passage.book_id, last.id, ReviewFlagCode.REPEATED_VALIDATION_FAILURE,
                f"The answer failed validation {execution.validation_failures} times.",
                details={"validation_failures": execution.validation_failures}, run_id=run.id, commit=False,
            )

    def _keep_unread_equations(self, run: ExtractionRun, passage: Passage) -> None:
        """Equations Phase 1 found in the passage that no model reading covered are kept, not dropped.

        They become source-derived units (no model involved) whose capture status says what is there: text,
        LaTeX, an image only, or something to review.
        """
        candidates = self.equations.equations_in(passage)
        if not candidates:
            return
        covered = self.runs.unit_equation_ids([e.id for e in candidates])
        ties = self._ties(passage)
        units = []
        for equation in candidates:
            if equation.id in covered:
                continue
            unit = self.equations.source_unit(equation, passage, passage.book_id, run.id)
            units.append(unit.model_copy(update=ties))

        if not units:
            return

        def hook(records: list[KnowledgeUnit]) -> None:
            for record in records:
                self._finalize_unit(run, passage, record, None, candidates, None)

        self.runs.store_source_units(run, units, after_store=hook)

    # -- relationships -----------------------------------------------------------------------------

    def _extract_relationships(self, run, passage, relationship_extractor, config, preparer, source, state, progress) -> bool:
        """Relate the units of a passage. Returns whether a model call was made."""
        try:
            outcome = relationship_extractor.extract_for_passage(run, passage, preparer.passage_input(passage), source, config)
        except Exception as exc:  # the relationship step must not end the run either
            self.session.rollback()
            logger.exception("Relationship extraction for passage %s crashed", passage.id)
            state.consecutive_failures += 1
            state.last_error = f"{type(exc).__name__}: {exc}"
            return True
        if outcome is None:
            return False
        failed = outcome.status == "failed"
        state.consecutive_failures = state.consecutive_failures + 1 if failed else 0
        if failed:
            state.last_error = outcome.failure[1] if outcome.failure else "relationship extraction failed"
        if progress:
            progress(
                ExtractionProgress(
                    passages_done=state.visited, total_passages=run.total_passages, passage_id=passage.id,
                    page_start=passage.page_start, category=None, status=outcome.status, items=outcome.created,
                    duplicates=outcome.existing, duration_ms=0, attempts=0,
                    failure_code=outcome.failure[0] if outcome.failure else None, step="relationships",
                )
            )
        return True

    # -- without storing ---------------------------------------------------------------------------

    def dry_run(
        self,
        scope: ExtractionScope,
        categories: Iterable[KnowledgeType | str],
        *,
        model_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        context_passages: int | None = None,
        progress: Callable[[int, int, Passage, ExtractionResult], None] | None = None,
    ) -> list[tuple[Passage, ExtractionResult]]:
        """Call the model for the selected passages and return the results; nothing is stored."""
        kinds = self._categories(categories)
        extractor = self._extractor(self._registry_for(model_id))
        configs = self._configs(extractor, kinds)
        passages = self.passages.select(scope, limit=limit, offset=offset)
        preparer = ContextPreparer(self.passages, self.settings, neighbours=context_passages)
        book = self.passages.get_book(scope.book_id)
        source = source_info(book) if book else None
        work = [(p, c) for p in passages for c in kinds]
        results = []
        for index, (passage, category) in enumerate(work, start=1):
            result = extractor.extract(preparer.prepare(passage, category, configs[category], source).request)
            results.append((passage, result))
            if progress:
                progress(index, len(work), passage, result)
        return results

    # -- status and results ------------------------------------------------------------------------

    def cancel_run(self, run_id: uuid.UUID) -> bool:
        return self.runs.cancel(run_id)

    def get_run_status(self, run_id: uuid.UUID) -> RunStatus:
        run = self.runs.get_run(run_id)
        if run is None:
            raise ExtractionRunError("The extraction run does not exist.")
        self.session.refresh(run)
        return self._status(run)

    def list_runs(self, book_id: uuid.UUID, *, limit: int = 20) -> list[RunStatus]:
        return [self._status(run) for run in self.runs.list_runs(book_id, limit=limit)]

    def _status(self, run: ExtractionRun) -> RunStatus:
        read = run_from_record(run)
        total = run.total_passages or 0
        end = run.completed_at or utcnow()
        counts = self.runs.passage_status_counts(run.id)
        unfinished = counts.get("failed", 0) + counts.get("pending", 0) + counts.get("processing", 0) + self.runs.errored_passage_count(run.id)
        can_resume = read.status.value in ("failed", "cancelled") or (read.status.value in _FINISHED and unfinished > 0)
        return RunStatus(
            run=read,
            failed_passages=self.runs.failed_passage_count(run.id),
            progress=min(1.0, run.passages_processed / total) if total else (1.0 if read.status.is_terminal else 0.0),
            is_active=not read.status.is_terminal,
            model_label=_label(run),
            duration_seconds=(end - run.started_at).total_seconds() if run.started_at else None,
            passage_counts=counts,
            can_resume=can_resume,
        )

    def get_book_knowledge(
        self,
        book_id: uuid.UUID,
        *,
        knowledge_type: KnowledgeType | str | None = None,
        run_id: uuid.UUID | None = None,
        review_status: str | None = None,
        validation_status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[KnowledgeUnitRead]:
        records = self.runs.knowledge(
            book_id,
            knowledge_type=KnowledgeType(knowledge_type) if knowledge_type else None,
            run_id=run_id,
            review_status=review_status,
            validation_status=validation_status,
            limit=limit,
            offset=offset,
        )
        return [unit_from_record(r) for r in records]

    # -- helpers -----------------------------------------------------------------------------------

    @staticmethod
    def _categories(values: Iterable[KnowledgeType | str]) -> list[KnowledgeType]:
        try:
            kinds = list(dict.fromkeys(KnowledgeType(v) for v in values))
        except ValueError as exc:
            valid = ", ".join(t.value for t in KnowledgeType)
            raise ExtractionRunError(f"Unknown knowledge category ({exc}). Choose from: {valid}.") from exc
        if not kinds:
            raise ExtractionRunError("Choose at least one knowledge category.")
        return kinds


def create_extraction_service(
    session: Session, settings: Settings | None = None, environ: Mapping[str, str] | None = None
) -> KnowledgeExtractionService:
    """The service wired from the application settings: the models file, the default model and .env credentials."""
    settings = settings or get_settings()
    return KnowledgeExtractionService(
        session,
        load_model_registry(settings),
        ExtractionSettings.from_settings(settings),
        environ=load_environment() if environ is None else environ,
    )


@dataclass
class _LoopState:
    visited: int = 0
    skipped: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None


__all__ = [
    "ExtractionPlan",
    "ExtractionProgress",
    "ExtractionRunError",
    "KnowledgeExtractionService",
    "ModelOption",
    "NothingToExtractError",
    "PassageStatusView",
    "ResumeInfo",
    "RunStatus",
    "create_extraction_service",
]
