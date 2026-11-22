"""Creating, validating and reading relationships between knowledge units.

A relationship is only stored when it makes sense: both units exist in the same book, the pair of
categories fits the relationship type, the cited passages exist, and it is not a duplicate. A relationship
marked `extracted` must carry a quote that really is in a cited passage; anything the model merely
proposes is stored as `ai_suggested`, flagged for review. Similar wording alone never creates one.
"""

from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.records import relationship_from_record, relationship_to_record
from app.knowledge.relationships import check_relationship_types
from app.knowledge.schemas import KnowledgeRelationshipCreate, KnowledgeRelationshipRead
from app.knowledge.types import KnowledgeType, RelationshipOrigin, RelationshipType, ReviewFlagCode
from app.repositories.relationship_repository import RelationshipRepository
from app.services.errors import RelationshipError
from app.services.review_flag_service import ReviewFlagService
from database.models import REVIEW_STATUSES, KnowledgeRelationshipPassage, Passage

logger = logging.getLogger(__name__)

# "A related_to B" is the same statement as "B related_to A"; so is contradicts.
SYMMETRIC_TYPES = frozenset({RelationshipType.RELATED_TO, RelationshipType.CONTRADICTS})


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


class RelationshipResult(BaseModel):
    relationship: KnowledgeRelationshipRead
    created: bool  # False when an equivalent relationship already existed
    upgraded: bool = False  # a suggestion that now has verified evidence
    flagged: bool = False


class RelationshipView(BaseModel):
    relationship: KnowledgeRelationshipRead
    direction: str  # "outgoing" (the unit is the source) or "incoming"
    other_unit_id: uuid.UUID
    other_title: str
    other_type: KnowledgeType


class UnitRelationships(BaseModel):
    outgoing: list[RelationshipView]
    incoming: list[RelationshipView]

    @property
    def all(self) -> list[RelationshipView]:
        return [*self.outgoing, *self.incoming]


class RelationshipService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = RelationshipRepository(session)
        self.flags = ReviewFlagService(session)

    # -- creating ----------------------------------------------------------------------------------

    def create(self, data: KnowledgeRelationshipCreate, *, commit: bool = True) -> RelationshipResult:
        """Validate and store a relationship, or return the equivalent one that already exists."""
        units = self.repository.units([data.source_unit_id, data.target_unit_id])
        source, target = units.get(data.source_unit_id), units.get(data.target_unit_id)
        if source is None or target is None:
            missing = "source" if source is None else "target"
            raise RelationshipError(f"The {missing} knowledge unit does not exist.")
        if source.book_id != data.book_id or target.book_id != data.book_id:
            raise RelationshipError("Both knowledge units must belong to the relationship's book.")
        try:
            check_relationship_types(data.relationship_type, KnowledgeType(source.knowledge_type), KnowledgeType(target.knowledge_type))
        except ValueError as exc:
            raise RelationshipError(str(exc)) from exc
        passages = self._passages(data)
        if data.origin == RelationshipOrigin.EXTRACTED:
            self._require_evidence(data, passages)

        existing = self._equivalent(data)
        if existing is not None:
            return self._merge(existing, data, commit)

        record = relationship_to_record(data)
        self.session.add(record)
        self.session.flush()
        flagged = False
        if data.origin == RelationshipOrigin.AI_SUGGESTED:
            flagged = self._flag_suggestion(record, source.title, target.title, data)
        if commit:
            self.session.commit()
        return RelationshipResult(relationship=relationship_from_record(record), created=True, flagged=flagged)

    def _passages(self, data: KnowledgeRelationshipCreate) -> dict[uuid.UUID, Passage]:
        if not data.source_passage_ids:
            return {}
        found = {p.id: p for p in self.session.scalars(select(Passage).where(Passage.id.in_(data.source_passage_ids)))}
        for passage_id in data.source_passage_ids:
            passage = found.get(passage_id)
            if passage is None:
                raise RelationshipError(f"Passage {passage_id} does not exist.")
            if passage.book_id != data.book_id:
                raise RelationshipError(f"Passage {passage_id} belongs to another book.")
        return found

    @staticmethod
    def _require_evidence(data: KnowledgeRelationshipCreate, passages: dict[uuid.UUID, Passage]) -> None:
        quote = _squash(data.evidence_quote or "")
        if not any(quote in _squash(p.text) for p in passages.values()):
            raise RelationshipError("The evidence quote is not in any of the cited passages, so the relationship cannot be 'extracted'.")

    def _equivalent(self, data: KnowledgeRelationshipCreate):
        found = self.repository.find_edge(data.source_unit_id, data.target_unit_id, data.relationship_type.value)
        if found is None and data.relationship_type in SYMMETRIC_TYPES:
            found = self.repository.find_edge(data.target_unit_id, data.source_unit_id, data.relationship_type.value)
        return found

    def _merge(self, existing, data: KnowledgeRelationshipCreate, commit: bool) -> RelationshipResult:
        """The relationship is already stored: add any new evidence to it instead of creating a second one."""
        upgraded = False
        known = {link.passage_id for link in existing.sources}
        for passage_id in data.source_passage_ids:
            if passage_id not in known:
                existing.sources.append(KnowledgeRelationshipPassage(passage_id=passage_id, position=len(existing.sources)))
        if existing.origin == RelationshipOrigin.AI_SUGGESTED.value and data.origin == RelationshipOrigin.EXTRACTED:
            existing.origin = RelationshipOrigin.EXTRACTED.value
            existing.evidence_quote = data.evidence_quote
            existing.explanation = existing.explanation or data.explanation
            existing.confidence = existing.confidence if existing.confidence is not None else data.confidence
            upgraded = True
            for flag in self.flags.repository.list(existing.book_id, relationship_id=existing.id, code=ReviewFlagCode.AI_SUGGESTED_RELATIONSHIP.value):
                self.flags.resolve(flag.id, "Evidence from the book was found for this relationship.", commit=False)
        self.session.flush()
        if commit:
            self.session.commit()
        return RelationshipResult(relationship=relationship_from_record(existing), created=False, upgraded=upgraded)

    def _flag_suggestion(self, record, source_title: str, target_title: str, data: KnowledgeRelationshipCreate) -> bool:
        _, created = self.flags.flag_relationship(
            record.book_id,
            record.id,
            ReviewFlagCode.AI_SUGGESTED_RELATIONSHIP,
            f"A model suggests that '{source_title}' {data.relationship_type.value.replace('_', ' ')} '{target_title}' "
            "without evidence that could be verified in the book.",
            details={"explanation": data.explanation, "quote": data.evidence_quote},
            run_id=data.extraction_run_id,
            raised_by="model",
            commit=False,
        )
        return created

    # -- reading -----------------------------------------------------------------------------------

    def get(self, relationship_id: uuid.UUID) -> KnowledgeRelationshipRead:
        record = self.repository.get(relationship_id)
        if record is None:
            raise RelationshipError("The relationship does not exist.")
        return relationship_from_record(record)

    def for_unit(self, unit_id: uuid.UUID) -> UnitRelationships:
        records = self.repository.for_unit(unit_id)
        units = self.repository.units(list({r.source_unit_id for r in records} | {r.target_unit_id for r in records}))
        outgoing, incoming = [], []
        for record in records:
            is_out = record.source_unit_id == unit_id
            other = units[record.target_unit_id if is_out else record.source_unit_id]
            view = RelationshipView(
                relationship=relationship_from_record(record),
                direction="outgoing" if is_out else "incoming",
                other_unit_id=other.id,
                other_title=other.title,
                other_type=KnowledgeType(other.knowledge_type),
            )
            (outgoing if is_out else incoming).append(view)
        return UnitRelationships(outgoing=outgoing, incoming=incoming)

    def list_for_book(self, book_id: uuid.UUID, **filters) -> list[KnowledgeRelationshipRead]:
        return [relationship_from_record(r) for r in self.repository.list(book_id, **filters)]

    def set_review_status(self, relationship_id: uuid.UUID, status: str, *, commit: bool = True) -> KnowledgeRelationshipRead:
        if status not in REVIEW_STATUSES:
            raise RelationshipError(f"Review status must be one of: {', '.join(REVIEW_STATUSES)}.")
        record = self.repository.get(relationship_id)
        if record is None:
            raise RelationshipError("The relationship does not exist.")
        record.review_status = status
        if commit:
            self.session.commit()
        return relationship_from_record(record)
