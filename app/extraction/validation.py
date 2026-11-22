"""Source validation, semantic validation and quality assessment.

Structural validation (does it match the schema?) is in app.extraction.parsing. The three stages here
are kept separate because they answer different questions and have different consequences:

- Source validation: do the cited passages and pages exist? Failures are ERRORS, because an invented
  reference is wrong whatever the content says; they trigger a correction attempt.
- Semantic validation: is the content supported by the source text? The built-in check is a cheap,
  conservative heuristic (quotes and equation expressions must appear in the cited passage). Failures
  are WARNINGS that flag the item for review. Stronger checks plug in through `SemanticValidator`.
- Quality assessment: indicators only (completeness, citation accuracy, duplicates). There is no
  overall reliability score and nothing is inferred from the model's name, size or provider.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Protocol

from app.extraction.results import (
    ExtractedKnowledge,
    ExtractionRequest,
    PassageInput,
    QualityAssessment,
    Severity,
    ValidationIssue,
    ValidationStage,
)
from app.knowledge.content import EquationContent
from app.knowledge.types import PageStatus, ValidationOutcome, ValidationStatus

_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def normalise(text: str) -> str:
    """Case-, whitespace-, quote- and ligature-insensitive form used to compare text with its source."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", normalise(text))  # for formulas, where PDF extraction often drops spaces


def _issue(stage: ValidationStage, severity: Severity, code: str, message: str, path: str, index: int) -> ValidationIssue:
    return ValidationIssue(stage=stage, severity=severity, code=code, message=message, path=path, item_index=index)


# -- Source validation -----------------------------------------------------------------------------


def validate_sources(
    items: list[ExtractedKnowledge], request: ExtractionRequest
) -> tuple[list[ExtractedKnowledge], list[ValidationIssue]]:
    """Check citations against the passages that were actually provided.

    Returns the items with pages resolved (taken from the cited passages when the model gave none,
    never invented) and the issues found. Each item also carries its own issues.
    """
    passages = request.passages()
    all_issues: list[ValidationIssue] = []
    checked: list[ExtractedKnowledge] = []
    for index, item in enumerate(items):
        issues: list[ValidationIssue] = []
        cited: list[PassageInput] = []
        for position, source in enumerate(item.sources):
            path = f"items[{index}].sources[{position}].passage_id"
            passage = passages.get(source.passage_id)
            if passage is None:
                issues.append(
                    _issue(
                        ValidationStage.SOURCE,
                        Severity.ERROR,
                        "unknown_passage_id",
                        f"{path} '{source.passage_id}' is not one of the provided passages. Cite only identifiers from the input.",
                        path,
                        index,
                    )
                )
            else:
                cited.append(passage)
        if cited and not any(p.id == request.passage.id for p in cited):
            issues.append(
                _issue(
                    ValidationStage.SOURCE,
                    Severity.WARNING,
                    "target_passage_not_cited",
                    "The item does not cite the target passage, only context passages.",
                    f"items[{index}].sources",
                    index,
                )
            )
        item = _resolve_pages(item, cited, index, issues)
        all_issues.extend(issues)
        checked.append(item.model_copy(update={"issues": [*item.issues, *issues]}))
    return checked, all_issues


def _resolve_pages(
    item: ExtractedKnowledge, cited: list[PassageInput], index: int, issues: list[ValidationIssue]
) -> ExtractedKnowledge:
    known = [p for p in cited if p.has_pages]
    low = min((p.page_start for p in known), default=None)
    high = max((p.last_page for p in known), default=None)
    given_start, given_end = item.page_start, item.page_end

    if given_start is None and given_end is None:
        if low is None:
            return item
        exact = all(p.has_pages for p in cited)
        return item.model_copy(
            update={
                "page_start": low,
                "page_end": high,
                "page_status": PageStatus.EXACT if exact else PageStatus.PARTIAL,
                "pages_from_model": False,
            }
        )

    start = given_start if given_start is not None else given_end
    end = given_end if given_end is not None else given_start
    path = f"items[{index}].page_start"
    if low is None:
        if cited:  # pages were claimed for passages that have none
            issues.append(
                _issue(
                    ValidationStage.SOURCE,
                    Severity.ERROR,
                    "page_not_available",
                    "Pages were given, but the cited passages carry no page information. Use null for pages.",
                    path,
                    index,
                )
            )
        return item
    if end < start:
        issues.append(
            _issue(ValidationStage.SOURCE, Severity.ERROR, "invalid_page_range", "page_end is before page_start.", path, index)
        )
    elif start < low or end > high:
        issues.append(
            _issue(
                ValidationStage.SOURCE,
                Severity.ERROR,
                "page_outside_passages",
                f"Pages {start}-{end} are outside the pages of the cited passages ({low}-{high}). Do not invent pages.",
                path,
                index,
            )
        )
    return item.model_copy(
        update={
            "page_start": start,
            "page_end": end,
            "page_status": PageStatus.EXACT if all(p.has_pages for p in cited) else PageStatus.PARTIAL,
        }
    )


# -- Semantic validation ---------------------------------------------------------------------------


class SemanticValidator(Protocol):
    """Extension point for stronger checks (a second model, a rule set, evaluation data)."""

    def validate(self, item: ExtractedKnowledge, index: int, request: ExtractionRequest) -> list[ValidationIssue]: ...


class SourceTextSupportValidator:
    """Quotes, and an equation's original expression, must really occur in the cited passages."""

    def validate(self, item: ExtractedKnowledge, index: int, request: ExtractionRequest) -> list[ValidationIssue]:
        passages = request.passages()
        issues: list[ValidationIssue] = []
        for position, source in enumerate(item.sources):
            passage = passages.get(source.passage_id)
            if passage is None or source.quote is None:
                continue
            if normalise(source.quote) not in normalise(passage.text):
                issues.append(
                    _issue(
                        ValidationStage.SEMANTIC,
                        Severity.WARNING,
                        "quote_not_in_source",
                        "The quote does not appear in the cited passage, so the item may not be supported by it.",
                        f"items[{index}].sources[{position}].quote",
                        index,
                    )
                )
        if not any(s.quote for s in item.sources):
            issues.append(
                _issue(
                    ValidationStage.SEMANTIC,
                    Severity.INFO,
                    "no_supporting_quote",
                    "The item has no supporting quote, so its support cannot be checked automatically.",
                    f"items[{index}].sources",
                    index,
                )
            )
        if isinstance(item.content, EquationContent) and item.content.original_expression:
            cited_text = " ".join(passages[s.passage_id].text for s in item.sources if s.passage_id in passages)
            if _squash(item.content.original_expression) not in _squash(cited_text):
                issues.append(
                    _issue(
                        ValidationStage.SEMANTIC,
                        Severity.WARNING,
                        "expression_not_in_source",
                        "The equation's original expression does not appear in the cited passage.",
                        f"items[{index}].content.original_expression",
                        index,
                    )
                )
        return issues


DEFAULT_SEMANTIC_VALIDATORS: tuple[SemanticValidator, ...] = (SourceTextSupportValidator(),)


def run_semantic_validation(
    items: list[ExtractedKnowledge],
    request: ExtractionRequest,
    validators: tuple[SemanticValidator, ...] | list[SemanticValidator] = DEFAULT_SEMANTIC_VALIDATORS,
) -> tuple[list[ExtractedKnowledge], list[ValidationIssue]]:
    all_issues: list[ValidationIssue] = []
    out: list[ExtractedKnowledge] = []
    for index, item in enumerate(items):
        found = [issue for validator in validators for issue in validator.validate(item, index, request)]
        all_issues.extend(found)
        out.append(item.model_copy(update={"issues": [*item.issues, *found]}))
    return out, all_issues


# -- Quality assessment ----------------------------------------------------------------------------


def _filled(value: object) -> bool:
    return value not in (None, "", [], {})


def completeness(item: ExtractedKnowledge) -> float:
    """Share of the content's fields (other than its type tag) that carry a value."""
    values = [getattr(item.content, name) for name in type(item.content).model_fields if name != "knowledge_type"]
    return sum(_filled(v) for v in values) / len(values) if values else 1.0


def find_duplicates(items: list[ExtractedKnowledge]) -> list[tuple[int, int]]:
    seen: dict[tuple[str, str], int] = {}
    duplicates = []
    for index, item in enumerate(items):
        key = (str(item.knowledge_type), normalise(item.content.title))
        if key in seen:
            duplicates.append((seen[key], index))
        else:
            seen[key] = index
    return duplicates


def assess_quality(
    items: list[ExtractedKnowledge],
    issues: list[ValidationIssue],
    semantic_accuracy: float | None = None,
) -> QualityAssessment:
    """Quality indicators for one extraction. `issues` are those of the source and semantic stages."""
    checks = sum(len(i.sources) + (i.page_start is not None) + sum(bool(s.quote) for s in i.sources) for i in items)
    failed = {
        (issue.item_index, issue.code, issue.path)
        for issue in issues
        if issue.code
        in {"unknown_passage_id", "page_outside_passages", "page_not_available", "invalid_page_range", "quote_not_in_source"}
    }
    accuracy = max(0.0, 1 - len(failed) / checks) if checks else None
    return QualityAssessment(
        schema_valid=True,
        item_count=len(items),
        completeness=(sum(completeness(i) for i in items) / len(items)) if items else None,
        source_reference_accuracy=accuracy,
        unsupported_content=[
            i.message for i in issues if i.code in {"quote_not_in_source", "expression_not_in_source"}
        ],
        duplicate_items=find_duplicates(items),
        semantic_accuracy=semantic_accuracy,
    )


def settle_status(item: ExtractedKnowledge) -> ExtractedKnowledge:
    """Final validation status of an item: uncertain when anything calls for a human look."""
    needs_look = (
        item.uncertainty.is_uncertain
        or item.basis != "explicit"
        or any(i.severity == Severity.WARNING for i in item.issues)
        or (isinstance(item.content, EquationContent) and item.content.is_incomplete)
    )
    return item.model_copy(update={"validation_status": ValidationStatus.UNCERTAIN if needs_look else ValidationStatus.VALID})


def cited_passage_ids(items: list[ExtractedKnowledge]) -> list[uuid.UUID]:
    seen: dict[uuid.UUID, None] = {}
    for item in items:
        for source in item.sources:
            seen.setdefault(source.passage_id)
    return list(seen)


# -- Validation outcomes of one attempt -------------------------------------------------------------

_OUTCOME_PRECEDENCE = (
    ValidationOutcome.SCHEMA_INVALID,
    ValidationOutcome.SOURCE_INVALID,
    ValidationOutcome.UNSUPPORTED_CONTENT,
    ValidationOutcome.SEMANTIC_CONCERNS,
    ValidationOutcome.DUPLICATE,
    ValidationOutcome.NEEDS_REVIEW,
    ValidationOutcome.SCHEMA_VALID,
)
_UNSUPPORTED_CODES = {"quote_not_in_source", "expression_not_in_source"}


def classify_outcomes(
    issues: list[ValidationIssue], *, answered: bool = True, needs_review: bool = False, duplicate: bool = False
) -> tuple[ValidationOutcome, list[ValidationOutcome]]:
    """The validation outcomes of one answer: the most serious, and all that applied.

    Kept apart from whether the call worked. Passing the schema is `schema_valid` and nothing more: it
    says nothing about whether the content is right, so semantic concerns and unsupported content are
    reported next to it, never hidden by it.
    """
    if not answered:
        return ValidationOutcome.NOT_VALIDATED, []
    found: set[ValidationOutcome] = set()
    structural_errors = [i for i in issues if i.stage == ValidationStage.STRUCTURAL and i.severity == Severity.ERROR]
    if structural_errors:
        found.add(ValidationOutcome.SCHEMA_INVALID)
    else:
        found.add(ValidationOutcome.SCHEMA_VALID)
    if any(i.stage == ValidationStage.SOURCE and i.severity == Severity.ERROR for i in issues):
        found.add(ValidationOutcome.SOURCE_INVALID)
    if any(i.code in _UNSUPPORTED_CODES for i in issues):
        found.add(ValidationOutcome.UNSUPPORTED_CONTENT)
    if any(
        i.severity == Severity.WARNING and i.code not in _UNSUPPORTED_CODES and i.code != "duplicate_item"
        and i.stage in (ValidationStage.SEMANTIC, ValidationStage.SOURCE)
        for i in issues
    ):
        found.add(ValidationOutcome.SEMANTIC_CONCERNS)
    if duplicate or any(i.code == "duplicate_item" for i in issues):
        found.add(ValidationOutcome.DUPLICATE)
    if needs_review:
        found.add(ValidationOutcome.NEEDS_REVIEW)
    ordered = [o for o in _OUTCOME_PRECEDENCE if o in found]
    return ordered[0], ordered
