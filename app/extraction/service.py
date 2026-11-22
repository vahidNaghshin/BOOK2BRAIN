"""The extraction service: one passage in, validated knowledge and full execution metadata out.

It knows prompts, validation and the `LlmProvider` interface, and nothing about persistence or the
Phase 1 document structures: it never reads or writes them, it only receives passages as input.
Retries are bounded by a single budget. A response that fails structural or source validation gets a
correction prompt that includes the problems; a transient provider failure is resent after a back-off.
Semantic and quality findings never cause a retry: they flag the result for review instead.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from app.extraction.context import describe, fit_context
from app.extraction.errors import ModelConfigError, ProviderError
from app.extraction.model_config import LlmModelConfig, ModelRegistry
from app.extraction.parsing import parse_response
from app.extraction.prompts import PromptTemplate, get_prompt, render_correction
from app.extraction.providers import LlmMessage, LlmProvider, LlmRequest, LlmResponse, create_provider
from app.extraction.results import (
    ContextInfo,
    AttemptOutcome,
    AttemptRecord,
    ExecutionMetadata,
    ExtractedKnowledge,
    ExtractionFailure,
    ExtractionRequest,
    ExtractionResult,
    ExtractionStatus,
    Severity,
    ValidationIssue,
    ValidationStage,
    utcnow,
)
from app.extraction.settings import ExtractionSettings
from app.extraction.validation import (
    classify_outcomes,
    DEFAULT_SEMANTIC_VALIDATORS,
    SemanticValidator,
    assess_quality,
    cited_passage_ids,
    run_semantic_validation,
    settle_status,
    validate_sources,
)

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[LlmModelConfig], LlmProvider]


class PromptRef(Protocol):
    """What the attempt records need to know about a prompt: PromptTemplate, or any named, versioned prompt."""

    name: str
    version: str

    @property
    def identifier(self) -> str: ...


@dataclass
class StructuredOutcome:
    """The result of `call_with_retries`: the validated value, or why there is none."""

    value: Any
    issues: list[ValidationIssue]
    attempts: list[AttemptRecord]
    raw: str | None
    failure: tuple[str, str] | None  # (code, message)


class ExtractionService:
    def __init__(
        self,
        registry: ModelRegistry,
        settings: ExtractionSettings | None = None,
        *,
        provider_factory: ProviderFactory | None = None,
        semantic_validators: tuple[SemanticValidator, ...] | list[SemanticValidator] = DEFAULT_SEMANTIC_VALIDATORS,
        environ: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = utcnow,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.registry = registry
        self.settings = settings or ExtractionSettings()
        self._environ = environ
        self._provider_factory = provider_factory or (lambda config: create_provider(config, environ))
        self._providers: dict[str, LlmProvider] = {}
        self._semantic_validators = tuple(semantic_validators)
        self._sleep = sleep
        self._clock = clock
        self._timer = timer

    # -- configuration -----------------------------------------------------------------------------

    def check_ready(self, knowledge_type: str) -> LlmModelConfig:
        """Validate the model selected for a category before any extraction starts."""
        config = self.registry.for_category(knowledge_type)
        config.check_ready(self._environ)
        return config

    def _provider(self, config: LlmModelConfig) -> LlmProvider:
        if config.id not in self._providers:
            self._providers[config.id] = self._provider_factory(config)
        return self._providers[config.id]

    # -- extraction --------------------------------------------------------------------------------

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        started_at, t0 = self._clock(), self._timer()
        category = request.knowledge_type

        try:
            config = self.check_ready(category)
            provider = self._provider(config)
        except (ModelConfigError, ProviderError) as exc:
            return self._failed(request, None, None, [], started_at, t0, "model_configuration_invalid", str(exc))

        try:
            prompt = get_prompt(category, self.settings.prompt_versions.get(category.value))
        except KeyError as exc:
            return self._failed(request, config, None, [], started_at, t0, "prompt_not_found", str(exc.args[0]))

        original = request
        fit = fit_context(
            request, prompt, config, chars_per_token=self.settings.chars_per_token,
            safety_margin=self.settings.context_safety_margin,
        )
        context = describe(original, fit)
        if not fit.fits:
            return self._failed(
                original, config, prompt, [], started_at, t0, "context_window_exceeded",
                f"The passage does not fit the context window of model '{config.id}' "
                f"({config.context_window} tokens, of which {config.max_output_tokens} are reserved for the answer; "
                f"about {fit.estimated_input_tokens} are needed even without context). Nothing was truncated.",
                context,
            )
        request = fit.request

        request_for_sources = request

        def validate(text: str, finish_reason: str | None):
            parsed = parse_response(text, category)
            issues = list(parsed.issues)
            items = parsed.items
            if not parsed.has_errors:
                items, source_issues = validate_sources(items, request_for_sources)
                issues.extend(source_issues)
            if finish_reason == "length":
                issues.append(
                    ValidationIssue(
                        stage=ValidationStage.STRUCTURAL,
                        severity=Severity.ERROR,
                        code="truncated_response",
                        message="The response was cut off at the output limit.",
                    )
                )
            if not any(i.severity == Severity.ERROR for i in issues):
                items, semantic_issues = run_semantic_validation(items, request_for_sources, self._semantic_validators)
                issues.extend(semantic_issues)
            return (items, parsed.notes), issues

        outcome = self.call_with_retries(
            config=config,
            prompt=prompt,
            system=prompt.render_system(),
            user=prompt.render_user(request),
            schema=prompt.output_schema(),
            validate=validate,
            label=category.value,
        )
        if outcome.failure is not None:
            code, message = outcome.failure
            return self._failed(
                request, config, prompt, outcome.attempts, started_at, t0, code, message, context, outcome.raw, outcome.issues
            )
        items, notes = outcome.value
        return self._succeeded(request, config, prompt, items, outcome.issues, notes, outcome.attempts, started_at, t0, context, outcome.raw)

    def call_with_retries(
        self,
        *,
        config: LlmModelConfig,
        prompt: PromptRef,
        system: str,
        user: str,
        schema: dict | None,
        validate: Callable[[str, str | None], tuple[Any, list[ValidationIssue]]],
        label: str = "",
    ) -> StructuredOutcome:
        """Send a prompt, check the answer with `validate`, and retry within the limit.

        `validate(text, finish_reason)` returns (value, issues). Any issue of severity error means the answer is
        not usable: it is sent back with the problems as a correction, until the retry limit. Provider failures
        that can be retried are resent after a back-off. Every call is recorded as an attempt with the model,
        configuration snapshot and prompt used. Nothing here knows what is being extracted.
        """
        provider = self._provider(config)
        max_retries = config.max_retries if config.max_retries is not None else self.settings.max_retries
        messages = [LlmMessage("user", user)]
        attempts: list[AttemptRecord] = []
        last_raw: str | None = None
        last_issues: list[ValidationIssue] = []
        kind = "initial"

        for number in range(1, max_retries + 2):
            attempt_start, a0 = self._clock(), self._timer()
            logger.info(
                "Extraction attempt %d: %s model_configuration=%s model=%s prompt=%s",
                number, label, config.id, config.model, prompt.identifier,
            )
            try:
                response = provider.complete(LlmRequest(system=system, messages=messages, json_schema=schema))
            except ProviderError as exc:
                attempts.append(self._attempt(number, kind, config, prompt, attempt_start, a0, AttemptOutcome.PROVIDER_ERROR, error=exc))
                logger.warning("Extraction attempt %d failed: %s (%s)", number, exc.code, exc)
                if not exc.retryable or number > max_retries:
                    return StructuredOutcome(None, last_issues, attempts, last_raw, (exc.code, str(exc)))
                self._sleep(self._backoff(exc, number))
                kind = "retry"
                continue

            last_raw = response.text
            value, issues = validate(response.text, response.finish_reason)
            last_issues = issues
            errors = [i for i in issues if i.severity == Severity.ERROR]
            if errors:
                attempts.append(self._attempt(number, kind, config, prompt, attempt_start, a0, AttemptOutcome.INVALID, response, issues))
                logger.warning("Extraction attempt %d invalid: %d error(s)", number, len(errors))
                if number > max_retries:
                    failure = (
                        "validation_failed",
                        f"No valid response after {number} attempt(s): " + "; ".join(i.message for i in errors[:3]),
                    )
                    return StructuredOutcome(None, issues, attempts, last_raw, failure)
                messages = [
                    LlmMessage("user", user),
                    LlmMessage("assistant", self._truncate(response.text)),
                    LlmMessage("user", render_correction(issues, truncated=response.finish_reason == "length")),
                ]
                kind = "correction"
                continue

            attempts.append(self._attempt(number, kind, config, prompt, attempt_start, a0, AttemptOutcome.VALID, response, issues))
            return StructuredOutcome(value, issues, attempts, last_raw, None)

        raise AssertionError("unreachable: the retry loop always returns")  # pragma: no cover

    def metadata_for(
        self, config: LlmModelConfig | None, prompt: PromptRef | None, attempts: list[AttemptRecord], started_at: datetime, t0: float
    ) -> ExecutionMetadata:
        """Execution metadata for a call made with `call_with_retries` (the caller builds its own result)."""
        return self._metadata(config, prompt, attempts, started_at, t0, None)

    def start(self) -> tuple[datetime, float]:
        return self._clock(), self._timer()

    # -- helpers -----------------------------------------------------------------------------------

    def _backoff(self, error: ProviderError, attempt: int) -> float:
        wait = error.retry_after if error.retry_after is not None else self.settings.retry_backoff_seconds * 2 ** (attempt - 1)
        return min(wait, self.settings.max_backoff_seconds)

    def _truncate(self, text: str) -> str:
        limit = self.settings.raw_response_max_chars
        return text if len(text) <= limit else text[:limit] + "\n[truncated]"

    def _attempt(
        self,
        number: int,
        kind: str,
        config: LlmModelConfig,
        prompt: PromptRef,
        started: datetime,
        t0: float,
        outcome: AttemptOutcome,
        response: LlmResponse | None = None,
        issues: list[ValidationIssue] | None = None,
        error: ProviderError | None = None,
    ) -> AttemptRecord:
        cost = None
        if response and response.input_tokens is not None and response.output_tokens is not None:
            if config.input_cost_per_million_tokens is not None and config.output_cost_per_million_tokens is not None:
                cost = (
                    response.input_tokens * config.input_cost_per_million_tokens
                    + response.output_tokens * config.output_cost_per_million_tokens
                ) / 1_000_000
        return AttemptRecord(
            attempt=number,
            kind=kind,
            model_configuration_id=config.id,
            provider=config.provider.value,
            model_id=response.model if response else config.model,
            model_version=(response.model_version if response else None) or config.model_version,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            generation_parameters=config.generation_parameters(),
            started_at=started,
            completed_at=self._clock(),
            duration_ms=int((self._timer() - t0) * 1000),
            outcome=outcome,
            status="provider_error" if outcome == AttemptOutcome.PROVIDER_ERROR else "completed",
            validation_outcome=classify_outcomes(issues or [], answered=response is not None)[0],
            validation_outcomes=classify_outcomes(issues or [], answered=response is not None)[1],
            model_configuration=config.model_dump(mode="json"),
            input_tokens=response.input_tokens if response else None,
            output_tokens=response.output_tokens if response else None,
            estimated_cost_usd=cost,
            finish_reason=response.finish_reason if response else None,
            raw_response=self._truncate(response.text) if response and outcome != AttemptOutcome.VALID else None,
            provider_error_code=error.code if error else None,
            provider_error=str(error) if error else None,
            issues=issues or [],
        )

    def _metadata(
        self,
        config: LlmModelConfig | None,
        prompt: PromptRef | None,
        attempts: list[AttemptRecord],
        started_at: datetime,
        t0: float,
        context: ContextInfo | None,
    ) -> ExecutionMetadata:
        answered = [a for a in attempts if a.outcome != AttemptOutcome.PROVIDER_ERROR]
        tokens_in = [a.input_tokens for a in attempts if a.input_tokens is not None]
        tokens_out = [a.output_tokens for a in attempts if a.output_tokens is not None]
        costs = [a.estimated_cost_usd for a in attempts if a.estimated_cost_usd is not None]
        return ExecutionMetadata(
            model_configuration_id=config.id if config else None,
            provider=config.provider.value if config else None,
            model_id=(answered[-1].model_id if answered else (config.model if config else None)),
            model_version=(answered[-1].model_version if answered else (config.model_version if config else None)),
            model_configuration=config.model_dump(mode="json") if config else None,
            prompt_name=prompt.name if prompt else None,
            prompt_version=prompt.version if prompt else None,
            generation_parameters=config.generation_parameters() if config else {},
            started_at=started_at,
            completed_at=self._clock(),
            duration_ms=int((self._timer() - t0) * 1000),
            attempt_count=len(attempts),
            retry_count=max(0, len(attempts) - 1),
            validation_failures=sum(a.outcome == AttemptOutcome.INVALID for a in attempts),
            input_tokens=sum(tokens_in) if tokens_in else None,
            output_tokens=sum(tokens_out) if tokens_out else None,
            estimated_cost_usd=sum(costs) if costs else None,
            context_passages_dropped=list(context.dropped) if context else [],
            context=context,
            attempts=attempts,
        )

    def _succeeded(
        self,
        request: ExtractionRequest,
        config: LlmModelConfig,
        prompt: PromptTemplate,
        items: list[ExtractedKnowledge],
        issues: list[ValidationIssue],
        notes: str | None,
        attempts: list[AttemptRecord],
        started_at: datetime,
        t0: float,
        context: ContextInfo | None,
        raw: str | None,
    ) -> ExtractionResult:
        quality = assess_quality(items, issues)
        for first, second in quality.duplicate_items:
            issues.append(
                ValidationIssue(
                    stage=ValidationStage.QUALITY,
                    severity=Severity.WARNING,
                    code="duplicate_item",
                    message=f"Items {first} and {second} appear to be the same knowledge unit.",
                    item_index=second,
                )
            )
        items = [settle_status(item) for item in items]
        reasons = self._review_reasons(items, issues, context.dropped if context else [])
        if attempts and attempts[-1].outcome == AttemptOutcome.VALID:
            attempts[-1].validation_outcome, attempts[-1].validation_outcomes = classify_outcomes(
                issues, needs_review=bool(reasons), duplicate=bool(quality.duplicate_items)
            )
        confidences = [i.confidence for i in items if i.confidence is not None]
        return ExtractionResult(
            knowledge_type=request.knowledge_type,
            status=ExtractionStatus.NEEDS_REVIEW if reasons else ExtractionStatus.SUCCEEDED,
            items=items,
            source_references=cited_passage_ids(items),
            confidence=min(confidences) if confidences else None,
            issues=issues,
            quality=quality,
            needs_review=bool(reasons),
            review_reasons=reasons,
            notes=notes,
            raw_response=raw,
            execution=self._metadata(config, prompt, attempts, started_at, t0, context),
        )

    @staticmethod
    def _review_reasons(items: list[ExtractedKnowledge], issues: list[ValidationIssue], dropped: list[uuid.UUID]) -> list[str]:
        reasons = []
        if any(i.uncertainty.is_uncertain for i in items):
            reasons.append("The model reported uncertainty or incomplete information.")
        if any(i.basis != "explicit" for i in items):
            reasons.append("Some information is inferred rather than explicitly stated.")
        warnings = {i.code for i in issues if i.severity == Severity.WARNING}
        if warnings:
            reasons.append("Validation warnings: " + ", ".join(sorted(warnings)) + ".")
        if dropped:
            reasons.append("Context passages were left out to fit the model's context window.")
        return reasons

    def failure_result(self, request: ExtractionRequest, code: str, message: str) -> ExtractionResult:
        """A failed result for something that went wrong around the extraction (not inside the model call)."""
        started_at = self._clock()
        try:
            config: LlmModelConfig | None = self.registry.for_category(request.knowledge_type.value)
        except ModelConfigError:
            config = None
        return self._failed(request, config, None, [], started_at, self._timer(), code, message)

    def _failed(
        self,
        request: ExtractionRequest,
        config: LlmModelConfig | None,
        prompt: PromptTemplate | None,
        attempts: list[AttemptRecord],
        started_at: datetime,
        t0: float,
        code: str,
        message: str,
        context: ContextInfo | None = None,
        raw: str | None = None,
        issues: list[ValidationIssue] | None = None,
    ) -> ExtractionResult:
        logger.error("Extraction failed: %s", code)
        return ExtractionResult(
            knowledge_type=request.knowledge_type,
            status=ExtractionStatus.FAILED,
            issues=issues or [],
            needs_review=True,
            review_reasons=[f"Extraction failed ({code}) and needs manual review or a rerun."],
            failure=ExtractionFailure(code=code, message=message),
            raw_response=raw,
            execution=self._metadata(config, prompt, attempts, started_at, t0, context),
        )
