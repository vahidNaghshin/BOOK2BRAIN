"""Parsing and checking a relationship extraction answer.

The same stance as for knowledge extraction: structural and referential problems (unknown unit ids,
invented relationship types, combinations of categories that make no sense) are errors that trigger a
correction attempt; nothing is silently dropped. A relationship whose evidence quote cannot be found in
the passage is not an error: it is returned as unverified, to be stored as an AI suggestion for review.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.extraction.parsing import extract_json_text
from app.extraction.relationship_prompts import RelationshipRequest
from app.extraction.results import Severity, ValidationIssue, ValidationStage
from app.knowledge.relationships import RELATIONSHIP_RULES, check_relationship_types
from app.knowledge.types import RelationshipType


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


@dataclass(frozen=True)
class ProposedRelationship:
    source_id: uuid.UUID
    target_id: uuid.UUID
    relationship_type: RelationshipType
    explanation: str | None
    quote: str | None
    confidence: float | None
    evidence_verified: bool  # the quote is word for word (ignoring spacing and case) in the passage


@dataclass
class ParsedRelationships:
    items: list[ProposedRelationship] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    notes: str | None = None

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)


def _issue(code: str, message: str, *, severity: Severity = Severity.ERROR, stage: ValidationStage = ValidationStage.STRUCTURAL, index: int | None = None, path: str | None = None) -> ValidationIssue:
    return ValidationIssue(stage=stage, severity=severity, code=code, message=message, path=path, item_index=index)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_relationships(raw: str, request: RelationshipRequest) -> ParsedRelationships:
    result = ParsedRelationships()
    if not raw or not raw.strip():
        result.issues.append(_issue("empty_response", "The response is empty."))
        return result
    try:
        data = json.loads(extract_json_text(raw))
    except json.JSONDecodeError as exc:
        result.issues.append(_issue("invalid_json", f"The response is not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})."))
        return result
    if not isinstance(data, dict):
        result.issues.append(_issue("not_an_object", 'The response must be a JSON object with a "relationships" list.'))
        return result
    entries = data.get("relationships")
    if entries is None:
        result.issues.append(_issue("missing_required_field", 'Required field is missing: "relationships"', path="relationships"))
        return result
    if not isinstance(entries, list):
        result.issues.append(_issue("invalid_value", '"relationships" must be a list.', path="relationships"))
        return result
    result.notes = _text(data.get("notes"))

    known = request.unit_ids
    seen: set[tuple[uuid.UUID, uuid.UUID, RelationshipType]] = set()
    passage_text = _squash(request.passage.text)
    for index, entry in enumerate(entries):
        path = f"relationships[{index}]"
        if not isinstance(entry, dict):
            result.issues.append(_issue("invalid_value", "Each relationship must be an object.", index=index, path=path))
            continue
        ids = []
        for key in ("source_id", "target_id"):
            raw_id = entry.get(key)
            if raw_id is None:
                result.issues.append(_issue("missing_required_field", f'Required field is missing: "{key}"', index=index, path=f"{path}.{key}"))
                continue
            try:
                unit_id = uuid.UUID(str(raw_id))
            except ValueError:
                result.issues.append(_issue("unknown_unit_id", f'"{raw_id}" is not a unit id. Use only the ids given.', stage=ValidationStage.SOURCE, index=index, path=f"{path}.{key}"))
                continue
            if unit_id not in known:
                result.issues.append(_issue("unknown_unit_id", f"Unit id {unit_id} was not in the list given. Never invent ids.", stage=ValidationStage.SOURCE, index=index, path=f"{path}.{key}"))
                continue
            ids.append(unit_id)
        raw_type = entry.get("type")
        try:
            relationship_type = RelationshipType(raw_type)
        except ValueError:
            allowed = ", ".join(t.value for t in RelationshipType)
            result.issues.append(_issue("invalid_relationship_type", f'"{raw_type}" is not a relationship type. Use one of: {allowed}.', index=index, path=f"{path}.type"))
            continue
        if len(ids) != 2:
            continue
        source_id, target_id = ids
        if source_id == target_id:
            result.issues.append(_issue("self_relationship", "A unit cannot be related to itself.", index=index, path=path))
            continue
        if not (known[source_id].in_passage or known[target_id].in_passage):
            result.issues.append(_issue("no_endpoint_in_passage", "At least one of the two units must be one extracted from this passage.", stage=ValidationStage.SOURCE, index=index, path=path))
            continue
        try:
            check_relationship_types(relationship_type, known[source_id].knowledge_type, known[target_id].knowledge_type)
        except ValueError as exc:
            result.issues.append(_issue("type_not_allowed", f"{exc} ({RELATIONSHIP_RULES[relationship_type].description})", index=index, path=path))
            continue
        quote = _text(entry.get("quote"))
        verified = bool(quote) and _squash(quote) in passage_text
        if quote is None:
            result.issues.append(_issue("no_evidence_quote", "No evidence quote was given; the relationship will be a suggestion to review.", severity=Severity.WARNING, stage=ValidationStage.SEMANTIC, index=index, path=f"{path}.quote"))
        elif not verified:
            result.issues.append(_issue("quote_not_in_source", "The quote is not in the passage; the relationship will be a suggestion to review.", severity=Severity.WARNING, stage=ValidationStage.SEMANTIC, index=index, path=f"{path}.quote"))
        confidence = entry.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1 else None
        key = (source_id, target_id, relationship_type)
        if key in seen:
            result.issues.append(_issue("duplicate_relationship", "The same relationship was reported twice; it is kept once.", severity=Severity.INFO, stage=ValidationStage.QUALITY, index=index, path=path))
            continue
        seen.add(key)
        result.items.append(ProposedRelationship(source_id, target_id, relationship_type, _text(entry.get("explanation")), quote, confidence, verified))
    if result.has_errors:
        result.items = []
    return result
