"""Identity of extracted knowledge, used to avoid storing the same extraction twice.

Duplicate detection looks at three things together: the primary source passage, the category and
the content. Knowledge with the same title in a *different* passage is a different occurrence of an
idea and is never merged here; it is only noted so a later step can relate the occurrences.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.knowledge.schemas import KnowledgeUnitCreate
from app.knowledge.types import SourceRole

# Same passage and category: near-identical wording is the same knowledge...
NEAR_DUPLICATE_BODY = 0.85
# ...and the same title with reasonably similar wording is too.
SAME_TITLE_BODY = 0.5

_SPACE = re.compile(r"\s+")
_EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$")
_WORD = re.compile(r"\w+", re.UNICODE)


def normalise(text: str) -> str:
    """Case, spacing and edge punctuation do not make a different statement."""
    return _EDGE_PUNCTUATION.sub("", _SPACE.sub(" ", text.casefold()).strip())


def _normalise_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalise(value)
    if isinstance(value, dict):
        return {key: _normalise_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalise_value(item) for item in value]
    return value


def primary_passage_id(unit: KnowledgeUnitCreate):
    for source in unit.sources:
        if source.role == SourceRole.PRIMARY:
            return source.passage_id
    return unit.sources[0].passage_id


def content_fingerprint(unit: KnowledgeUnitCreate) -> str:
    content = unit.content.model_dump(mode="json", exclude={"knowledge_type"})
    payload = json.dumps(
        [unit.knowledge_type.value, str(primary_passage_id(unit)), _normalise_value(content)], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def title_key(title: str) -> str:
    return normalise(title)


def word_set(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.casefold()))


def similarity(first: str, second: str) -> float:
    """Share of words the two texts have in common (Jaccard). 1.0 for identical word sets."""
    a, b = word_set(first), word_set(second)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
