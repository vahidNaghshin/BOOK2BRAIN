"""Optional LLM help for ambiguous headings.

The LLM only classifies candidates the deterministic stages already found: it cannot add headings
or change text. Each decision is validated before it is applied, and every failure is logged and
reported as a validation issue, leaving the candidate as the deterministic stages classified it.
The classifier is a small protocol so tests (and other providers) can supply their own.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.structure.heading_detection import HeadingCandidate
from app.structure.models import (
    RANK_CHAPTER,
    RANK_PART,
    RANK_SECTION,
    RANK_SUBSECTION,
    RANK_SUBSUBSECTION,
    NodeType,
    Severity,
    SourceBlock,
    ValidationIssue,
)
from app.structure.numbering import normalise_for_match

logger = logging.getLogger(__name__)

NOT_A_HEADING = "not_heading"
_ALLOWED = {t.value for t in NodeType} - {
    NodeType.BOOK_TITLE.value,
    NodeType.FRONT_MATTER.value,
    NodeType.BACK_MATTER.value,
}
_RANK_BY_TYPE = {
    NodeType.PART.value: RANK_PART,
    NodeType.CHAPTER.value: RANK_CHAPTER,
    NodeType.APPENDIX.value: RANK_CHAPTER,
    NodeType.SECTION.value: RANK_SECTION,
    NodeType.SUBSECTION.value: RANK_SUBSECTION,
    NodeType.SUBSUBSECTION.value: RANK_SUBSUBSECTION,
}
_RANK_BY_LEVEL = {1: RANK_CHAPTER, 2: RANK_SECTION, 3: RANK_SUBSECTION, 4: RANK_SUBSUBSECTION}
_SNIPPET = 240

SYSTEM_PROMPT = (
    "You classify candidate headings found in a book that was extracted to text. You are given "
    "candidate texts with their surroundings. Do NOT invent headings and do not change text: only "
    "classify the candidates you are given. For each candidate answer with a JSON object "
    '{"results": [...]}, one result per candidate, each with: candidate_id (integer, as given), '
    "original_text (exactly as given), normalized_title (the heading without numbering labels), "
    f"classification (one of: {', '.join(sorted(_ALLOWED))}, or \"{NOT_A_HEADING}\" when the text is "
    "ordinary body text), level (integer depth in the book body: 1 chapter-level, 2 section, "
    "3 subsection; null if unknown), parent_title (title of the parent heading if you can tell, "
    "else null), confidence (0 to 1) and reasoning (one short sentence)."
)


class LlmUnavailableError(Exception):
    """The LLM is misconfigured or could not be reached. The message is safe to show to users."""


class LlmCandidate(BaseModel):
    candidate_id: int
    text: str
    page: int | None = None
    previous_heading: str | None = None
    text_before: str = ""
    text_after: str = ""
    signals: list[str] = Field(default_factory=list)
    deterministic_score: float


class LlmDecision(BaseModel):
    candidate_id: int
    original_text: str = ""
    normalized_title: str = ""
    classification: str
    level: int | None = None
    parent_title: str | None = None
    confidence: float = Field(ge=0, le=1)
    reasoning: str = ""


class HeadingClassifier(Protocol):
    batch_size: int

    def classify(self, candidates: list[LlmCandidate]) -> list[LlmDecision]: ...


@dataclass
class OpenAICompatibleClassifier:
    """Chat-completions client (OpenAI and compatible servers) using only the standard library."""

    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: int = 60
    batch_size: int = 15

    def classify(self, candidates: list[LlmCandidate]) -> list[LlmDecision]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"candidates": [c.model_dump() for c in candidates]})},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(  # noqa: S310 - scheme is checked in classifier_from_settings
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                body = json.loads(response.read())
            content = body["choices"][0]["message"]["content"]
            data = json.loads(content)
        except urllib.error.HTTPError as exc:
            raise LlmUnavailableError(f"The LLM service answered with HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LlmUnavailableError(f"The LLM service could not be reached ({exc}).") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LlmUnavailableError("The LLM service returned an unexpected response.") from exc
        results = data.get("results") if isinstance(data, dict) else data
        if not isinstance(results, list):
            raise LlmUnavailableError("The LLM response has no list of results.")
        decisions: list[LlmDecision] = []
        for item in results:
            try:
                decisions.append(LlmDecision.model_validate(item))
            except ValidationError:
                logger.warning("Ignoring a malformed LLM result: %r", item)
        return decisions


def classifier_from_settings(settings: Settings) -> HeadingClassifier | None:
    """The configured classifier, None when the LLM is switched off. Raises when misconfigured."""
    if not settings.structure_llm_enabled:
        return None
    if not settings.structure_llm_model:
        raise LlmUnavailableError("STRUCTURE_LLM_ENABLED is set but STRUCTURE_LLM_MODEL is empty.")
    if not settings.structure_llm_base_url.lower().startswith(("https://", "http://")):
        raise LlmUnavailableError("STRUCTURE_LLM_BASE_URL must start with http:// or https://.")
    key = settings.structure_llm_api_key.get_secret_value() if settings.structure_llm_api_key else None
    return OpenAICompatibleClassifier(
        base_url=settings.structure_llm_base_url,
        model=settings.structure_llm_model,
        api_key=key,
        timeout_seconds=settings.structure_llm_timeout_seconds,
        batch_size=settings.structure_llm_batch_size,
    )


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET else text[: _SNIPPET - 1] + "…"


def apply_llm_classification(
    candidates: list[HeadingCandidate],
    blocks: list[SourceBlock],
    classifier: HeadingClassifier,
    *,
    max_candidates: int,
    accept_threshold: float,
) -> tuple[list[HeadingCandidate], list[ValidationIssue]]:
    """Ask the classifier about ambiguous candidates only. Returns (kept candidates, issues)."""
    issues: list[ValidationIssue] = []
    ambiguous = [c for c in candidates if c.status == "ambiguous"]
    if not ambiguous:
        return candidates, issues
    if len(ambiguous) > max_candidates:
        issues.append(
            ValidationIssue(
                Severity.INFO,
                "llm_candidate_limit",
                f"{len(ambiguous)} ambiguous headings found; only the first {max_candidates} were sent to the LLM.",
                "document",
            )
        )
        ambiguous = ambiguous[:max_candidates]

    requests: list[LlmCandidate] = []
    last_heading: str | None = None
    previous_by_index = {}
    for candidate in candidates:
        previous_by_index[candidate.index] = last_heading
        if candidate.status == "accepted":
            last_heading = candidate.title
    for position, candidate in enumerate(ambiguous):
        before = blocks[candidate.index - 1].text if candidate.index > 0 else ""
        after_index = candidate.block_indices[-1] + 1
        after = blocks[after_index].text if after_index < len(blocks) else ""
        requests.append(
            LlmCandidate(
                candidate_id=position,
                text=candidate.original_text,
                page=candidate.page,
                previous_heading=previous_by_index.get(candidate.index),
                text_before=_snippet(before),
                text_after=_snippet(after),
                signals=candidate.signals,
                deterministic_score=round(candidate.score, 2),
            )
        )

    decisions: list[LlmDecision] = []
    size = max(1, getattr(classifier, "batch_size", 15))
    for start in range(0, len(requests), size):
        batch = requests[start : start + size]
        try:
            decisions.extend(classifier.classify(batch))
        except Exception as exc:  # any provider failure must leave the deterministic result intact
            logger.warning("LLM heading classification failed for a batch of %d: %s", len(batch), exc)
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    "llm_classification_failed",
                    f"The LLM could not classify {len(batch)} ambiguous headings ({exc}); "
                    "they keep their deterministic classification.",
                    "document",
                )
            )

    removed: set[int] = set()
    for decision in decisions:
        if not 0 <= decision.candidate_id < len(ambiguous):
            issues.append(
                ValidationIssue(Severity.WARNING, "llm_unknown_candidate", f"The LLM answered about an unknown candidate ({decision.candidate_id}); ignored.", "document")
            )
            continue
        candidate = ambiguous[decision.candidate_id]
        if decision.classification == NOT_A_HEADING:
            candidate.status = "rejected"
            candidate.llm = {"classification": NOT_A_HEADING, "confidence": decision.confidence, "reasoning": decision.reasoning}
            removed.add(candidate.index)
            issues.append(
                ValidationIssue(
                    Severity.INFO,
                    "heading_rejected_by_llm",
                    f"'{candidate.original_text}' was classified as body text; its text is kept as a paragraph.",
                    "document",
                    details={"reasoning": decision.reasoning, "confidence": decision.confidence},
                )
            )
            continue
        if decision.classification not in _ALLOWED:
            issues.append(
                ValidationIssue(Severity.WARNING, "llm_invalid_classification", f"The LLM used an unknown classification '{decision.classification}' for '{candidate.original_text}'; ignored.", "document")
            )
            continue
        rank = _RANK_BY_TYPE.get(decision.classification)
        if rank is None and decision.classification == NodeType.OTHER_HEADING.value and decision.level:
            rank = _RANK_BY_LEVEL.get(decision.level)
        info: dict[str, Any] = {
            "node_type": decision.classification,
            "rank": rank,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "parent_title": decision.parent_title,
        }
        proposed = decision.normalized_title.strip()
        if proposed and normalise_for_match(proposed) and normalise_for_match(proposed) in normalise_for_match(candidate.original_text):
            candidate.title = proposed
            info["normalized_title"] = proposed
        elif proposed:
            logger.info("Ignoring LLM title %r: it is not part of the heading %r", proposed, candidate.original_text)
        candidate.llm = info
        if decision.confidence >= accept_threshold:
            candidate.status = "accepted"

    kept = [c for c in candidates if c.index not in removed]
    return kept, issues
