"""The evaluation dataset: passages with manually reviewed reference extractions.

A dataset is one JSON file under `evaluation/datasets/`. It is written and reviewed by a person and is
never produced by a model, so it can be used as ground truth. It lives apart from the extraction results:
nothing in this module reads an extraction.

Two kinds of passage source:

* `inline`: the dataset carries the passage text and is loaded into the database as its own small
  book each time it is run. It can be run any number of times, with any model.
* `stored_book`: the dataset refers to passages of a book that is already in the database (located by
  the start of their text) and carries only the reference results. It evaluates extractions that were
  already made; it cannot be run again without disturbing the stored knowledge.

A reference unit says what a careful reader expects, in terms a program can check without a model:
the acceptable names, the facts the unit must carry (each fact is a list of alternative words) and
phrases it must not contain.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import PROJECT_ROOT
from app.knowledge.types import EquationCaptureStatus, KnowledgeType, RelationshipType

DATASETS_DIR = PROJECT_ROOT / "evaluation" / "datasets"

NonEmpty = Annotated[str, Field(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Phase1Equation(_Strict):
    """What Phase 1 stored for an equation of an inline passage (never model output)."""

    latex: str | None = None
    original_latex: str | None = None
    plain_text: str | None = None
    surrounding_text: str | None = None
    validation_status: Literal["pending", "valid", "uncertain", "invalid"] = "valid"
    has_image: bool = True


class EquationReference(_Strict):
    """What must survive for an equation of the passage."""

    tokens: list[NonEmpty] = Field(min_length=1)  # symbols or words of the expression that must be kept
    capture_status: list[EquationCaptureStatus] = Field(min_length=1)  # acceptable statuses of the stored capture
    must_be_flagged: bool = False  # the source is incomplete or ambiguous: a careful system flags it
    has_image: bool = False
    in_phase1: bool = True  # False for inline maths in running text: Phase 1 stored no equation record for it
    phase1: Phase1Equation | None = None  # for inline passages: the Phase 1 equation to store


class ReferenceUnit(_Strict):
    key: NonEmpty
    category: KnowledgeType
    names: list[NonEmpty] = Field(min_length=1)
    accept_categories: list[KnowledgeType] = Field(default_factory=list)  # other categories that are also right
    required: bool = True  # optional units count for precision but not for recall
    facts: list[list[NonEmpty]] = Field(default_factory=list)  # each fact: alternative words, any one is enough
    forbidden: list[NonEmpty] = Field(default_factory=list)  # phrases that would mean the unit is unsupported
    quote: str | None = None  # words of the passage that support the unit
    equation: EquationReference | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _equation_block(self) -> ReferenceUnit:
        if (self.category == KnowledgeType.EQUATION) != (self.equation is not None):
            raise ValueError(f"Reference unit '{self.key}': an equation block is required for equations, and only for them.")
        return self

    @property
    def categories(self) -> set[KnowledgeType]:
        return {self.category, *self.accept_categories}


class ReferenceRelationship(_Strict):
    source: NonEmpty  # unit keys
    target: NonEmpty
    types: list[RelationshipType] = Field(min_length=1)  # any of these is correct
    required: bool = True
    note: str | None = None


class ReferencePassage(_Strict):
    key: NonEmpty
    pages: tuple[int, int]  # first and last page of the passage in the source document
    text: str | None = None  # inline datasets
    anchor: str | None = None  # stored books: the start of the passage text
    document_sequence: int | None = None  # stored books: expected reading-order position
    chapter: str | None = None
    section: str | None = None
    characteristics: list[NonEmpty] = Field(default_factory=list)
    exhaustive: bool = False  # the reference lists every unit a reviewer would accept
    expect_nothing: bool = False  # a good extraction finds no required knowledge here
    units: list[ReferenceUnit] = Field(default_factory=list)
    relationships: list[ReferenceRelationship] = Field(default_factory=list)
    note: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ReferencePassage:
        keys = [u.key for u in self.units]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Passage '{self.key}': duplicate unit keys.")
        known = set(keys)
        for relation in self.relationships:
            if relation.source not in known or relation.target not in known:
                raise ValueError(f"Passage '{self.key}': a relationship names a unit that is not in the passage.")
        if self.pages[0] > self.pages[1]:
            raise ValueError(f"Passage '{self.key}': the pages are reversed.")
        if self.expect_nothing and any(u.required for u in self.units):
            raise ValueError(f"Passage '{self.key}': a passage that expects nothing cannot have required units.")
        return self


class DatasetSource(_Strict):
    kind: Literal["inline", "stored_book"]
    book_title: NonEmpty
    author: str | None = None


class Review(_Strict):
    """Who prepared the reference, when and how far it has been checked."""

    prepared_by: NonEmpty
    prepared_on: NonEmpty
    status: Literal["draft", "reviewed", "approved"]
    notes: str | None = None


class EvaluationDataset(_Strict):
    name: NonEmpty
    version: NonEmpty
    description: NonEmpty
    source: DatasetSource
    review: Review
    categories: list[KnowledgeType]
    passages: list[ReferencePassage] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> EvaluationDataset:
        keys = [p.key for p in self.passages]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate passage keys.")
        for passage in self.passages:
            if self.source.kind == "inline":
                if not passage.text:
                    raise ValueError(f"Passage '{passage.key}': an inline dataset needs the passage text.")
                for unit in passage.units:
                    if unit.quote and _squash(unit.quote) not in _squash(passage.text):
                        raise ValueError(f"Passage '{passage.key}', unit '{unit.key}': the quote is not in the passage text.")
                    if unit.equation and unit.equation.phase1 is None:
                        raise ValueError(f"Passage '{passage.key}', unit '{unit.key}': an inline equation needs its Phase 1 data.")
            elif not passage.anchor:
                raise ValueError(f"Passage '{passage.key}': a stored-book dataset needs an anchor.")
        return self

    # -- convenience ---------------------------------------------------------------------------

    def passage(self, key: str) -> ReferencePassage:
        return next(p for p in self.passages if p.key == key)

    @property
    def reference_units(self) -> list[tuple[ReferencePassage, ReferenceUnit]]:
        return [(p, u) for p in self.passages for u in p.units]

    def counts(self) -> dict[str, int]:
        by_category: dict[str, int] = {}
        for _, unit in self.reference_units:
            by_category[unit.category.value] = by_category.get(unit.category.value, 0) + 1
        return by_category

    @property
    def digest(self) -> str:
        """A stable fingerprint of the content, recorded with every evaluation so changed references are visible."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_dataset(name_or_path: str | Path) -> EvaluationDataset:
    """Load a dataset by name (from `evaluation/datasets/`) or by file path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = DATASETS_DIR / f"{name_or_path}.json"
    if not path.exists():
        available = ", ".join(available_datasets()) or "none"
        raise FileNotFoundError(f"No evaluation dataset at {path}. Available: {available}.")
    return EvaluationDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def available_datasets() -> list[str]:
    return sorted(p.stem for p in DATASETS_DIR.glob("*.json"))
