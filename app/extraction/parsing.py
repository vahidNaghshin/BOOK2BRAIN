"""Turns a raw LLM response into validated extraction items (structural validation).

Nothing is dropped silently: every problem becomes a ValidationIssue with the path of the offending
field, so it can be shown to a reviewer and fed back to the model in a correction prompt. The content
itself is validated by the Day 1 Pydantic models, unchanged.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.extraction.results import (
    ExtractedKnowledge,
    Severity,
    ValidationIssue,
    ValidationStage,
)
from app.knowledge.content import CONTENT_MODELS, EquationContent
from app.knowledge.schemas import SourceReference, Uncertainty
from app.knowledge.types import KnowledgeType, SourceRole, ValidationStatus

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class _RawSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    passage_id: uuid.UUID
    quote: str | None = None
    role: SourceRole = SourceRole.SUPPORTING


class _RawItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: dict[str, Any]
    sources: list[_RawSource] = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    basis: Literal["explicit", "inferred", "mixed"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    is_uncertain: bool = False
    uncertainty_notes: str | None = None


@dataclass
class ParsedResponse:
    items: list[ExtractedKnowledge] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    notes: str | None = None

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)


def extract_json_text(raw: str) -> str:
    """The JSON object inside a response, tolerating markdown fences and surrounding chatter."""
    text = raw.strip()
    if match := _FENCE.match(text):
        text = match.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _issue(code: str, message: str, path: str | None = None, index: int | None = None) -> ValidationIssue:
    return ValidationIssue(
        stage=ValidationStage.STRUCTURAL,
        severity=Severity.ERROR,
        code=code,
        message=message,
        path=path,
        item_index=index,
    )


def _location(prefix: str, loc: tuple[Any, ...]) -> str:
    path = prefix
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _from_pydantic(error: ValidationError, prefix: str, index: int) -> list[ValidationIssue]:
    issues = []
    for e in error.errors():
        path = _location(prefix, e["loc"])
        if e["type"] == "missing":
            issues.append(_issue("missing_required_field", f"Required field is missing: {path}", path, index))
        elif e["type"] == "extra_forbidden":
            issues.append(_issue("unexpected_field", f"Unexpected field {path}: it is not part of the schema", path, index))
        else:
            message = e["msg"].removeprefix("Value error, ")
            issues.append(_issue("invalid_value", f"Invalid value at {path}: {message}", path, index))
    return issues


def parse_response(raw: str, knowledge_type: KnowledgeType) -> ParsedResponse:
    """Parse and structurally validate one response for the requested category."""
    result = ParsedResponse()
    if not raw or not raw.strip():
        result.issues.append(_issue("empty_response", "The response is empty."))
        return result
    try:
        data = json.loads(extract_json_text(raw))
    except json.JSONDecodeError as exc:
        result.issues.append(
            _issue("invalid_json", f"The response is not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno}).")
        )
        return result
    if not isinstance(data, dict):
        result.issues.append(_issue("not_an_object", "The response must be a JSON object with an \"items\" list."))
        return result
    items = data.get("items")
    if items is None:
        result.issues.append(_issue("missing_required_field", 'Required field is missing: "items"', "items"))
        return result
    if not isinstance(items, list):
        result.issues.append(_issue("invalid_value", '"items" must be a list.', "items"))
        return result
    notes = data.get("notes")
    result.notes = notes if isinstance(notes, str) and notes.strip() else None

    model = CONTENT_MODELS[knowledge_type]
    for index, entry in enumerate(items):
        prefix = f"items[{index}]"
        if not isinstance(entry, dict):
            result.issues.append(_issue("invalid_value", f"{prefix} must be an object.", prefix, index))
            continue
        try:
            raw_item = _RawItem.model_validate(entry)
        except ValidationError as exc:
            result.issues.extend(_from_pydantic(exc, prefix, index))
            continue
        content_data = dict(raw_item.content)
        declared = content_data.setdefault("knowledge_type", knowledge_type.value)
        if declared != knowledge_type.value:
            result.issues.append(
                _issue(
                    "wrong_knowledge_type",
                    f"{prefix}.content.knowledge_type is '{declared}' but this extraction is for '{knowledge_type.value}'.",
                    f"{prefix}.content.knowledge_type",
                    index,
                )
            )
            continue
        try:
            content = model.model_validate(content_data)
        except ValidationError as exc:
            result.issues.extend(_from_pydantic(exc, f"{prefix}.content", index))
            continue
        result.items.append(_build_item(raw_item, content))
    if result.has_errors:
        result.items = []  # all-or-nothing: a partly valid answer is corrected as a whole, not trusted piecemeal
    return result


def _build_item(raw: _RawItem, content: Any) -> ExtractedKnowledge:
    uncertain = raw.is_uncertain or raw.uncertainty_notes is not None
    notes = raw.uncertainty_notes
    if isinstance(content, EquationContent) and content.is_incomplete and not uncertain:
        uncertain = True
        notes = content.status_notes or f"Equation is incomplete (extraction: {content.extraction_status})."
    sources = [SourceReference(passage_id=s.passage_id, role=s.role, quote=s.quote or None) for s in raw.sources]
    primary_seen = False
    for position, source in enumerate(sources):  # the schema allows at most one primary source
        if source.role == SourceRole.PRIMARY:
            if primary_seen:
                sources[position] = source.model_copy(update={"role": SourceRole.SUPPORTING})
            primary_seen = True
    return ExtractedKnowledge(
        content=content,
        sources=_dedupe_sources(sources),
        page_start=raw.page_start,
        page_end=raw.page_end,
        pages_from_model=raw.page_start is not None or raw.page_end is not None,
        basis=raw.basis,
        confidence=raw.confidence,
        uncertainty=Uncertainty(is_uncertain=uncertain, notes=notes.strip() if notes and notes.strip() else None),
        validation_status=ValidationStatus.UNCERTAIN if uncertain else ValidationStatus.VALID,
    )


def _dedupe_sources(sources: list[SourceReference]) -> list[SourceReference]:
    seen: set[uuid.UUID] = set()
    unique = []
    for source in sources:
        if source.passage_id not in seen:
            seen.add(source.passage_id)
            unique.append(source)
    return unique
