"""Fitting a prompt into a model's context window.

Shared by the context preparation service and the extraction service. The strategy is fixed and
predictable: the target passage is never shortened; neighbouring (context) passages are dropped
first, the furthest ones first, alternating between the passages before and after. If the target
passage alone does not fit, nothing is cut: the request is reported as not fitting.

Token counts are estimates (characters per token); the provider has the final say.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.extraction.model_config import LlmModelConfig
from app.extraction.prompts import PromptTemplate
from app.extraction.results import ContextInfo, ExtractionRequest


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(len(text) / chars_per_token) + 1


def input_budget(config: LlmModelConfig, safety_margin: float) -> int | None:
    """Tokens available for the prompt: the window minus the reserved answer, minus a safety margin."""
    if config.context_window is None:
        return None
    return int((config.context_window - config.max_output_tokens) * (1 - safety_margin))


@dataclass(frozen=True)
class ContextFit:
    request: ExtractionRequest
    fits: bool
    dropped: tuple[uuid.UUID, ...]
    estimated_input_tokens: int
    budget_tokens: int | None


def fit_context(
    request: ExtractionRequest,
    prompt: PromptTemplate,
    config: LlmModelConfig,
    *,
    chars_per_token: float = 4.0,
    safety_margin: float = 0.0,
) -> ContextFit:
    budget = input_budget(config, safety_margin)
    system_tokens = estimate_tokens(prompt.render_system(), chars_per_token)

    def tokens(candidate: ExtractionRequest) -> int:
        return system_tokens + estimate_tokens(prompt.render_user(candidate), chars_per_token)

    current = request
    dropped: list[uuid.UUID] = []
    while budget is not None and tokens(current) > budget:
        before, after = list(current.context_before), list(current.context_after)
        if not before and not after:
            return ContextFit(current, False, tuple(dropped), tokens(current), budget)
        if len(before) >= len(after):
            dropped.append(before.pop(0).id)  # the furthest passage before the target
        else:
            dropped.append(after.pop().id)  # the furthest passage after the target
        current = current.model_copy(update={"context_before": before, "context_after": after})
    return ContextFit(current, True, tuple(dropped), tokens(current), budget)


def describe(original: ExtractionRequest, fit: ContextFit) -> ContextInfo:
    """What happened to the context of one request, including any reduction made before it."""
    earlier = original.context
    dropped = [*(earlier.dropped if earlier else []), *fit.dropped]
    requested = earlier.requested if earlier else len(original.context_before) + len(original.context_after)
    sent = len(fit.request.context_before) + len(fit.request.context_after)
    return ContextInfo(
        requested=requested,
        sent=sent,
        dropped=dropped,
        reduced=bool(dropped),
        reason="The prompt did not fit the model's context window; the furthest neighbouring passages were left out."
        if dropped
        else None,
        estimated_input_tokens=fit.estimated_input_tokens,
        budget_tokens=fit.budget_tokens,
    )
