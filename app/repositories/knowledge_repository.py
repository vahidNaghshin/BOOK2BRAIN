"""Read and review access to stored knowledge units, for browsing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from database.models import (
    ExtractionExecution,
    KnowledgeUnit,
    KnowledgeUnitPassage,
    Passage,
    StructuralNode,
)


@dataclass(frozen=True)
class KnowledgeFilter:
    book_id: uuid.UUID
    document_id: uuid.UUID | None = None
    knowledge_type: str | None = None
    validation_status: str | None = None
    review_status: str | None = None
    run_id: uuid.UUID | None = None
    uncertain_only: bool = False
    search: str | None = None


class KnowledgeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _where(self, query, f: KnowledgeFilter):
        query = query.where(KnowledgeUnit.book_id == f.book_id)
        if f.document_id is not None:
            query = query.where(KnowledgeUnit.document_id == f.document_id)
        if f.knowledge_type:
            query = query.where(KnowledgeUnit.knowledge_type == f.knowledge_type)
        if f.validation_status:
            query = query.where(KnowledgeUnit.validation_status == f.validation_status)
        if f.review_status:
            query = query.where(KnowledgeUnit.review_status == f.review_status)
        if f.run_id is not None:
            query = query.where(KnowledgeUnit.extraction_run_id == f.run_id)
        if f.uncertain_only:
            query = query.where(or_(KnowledgeUnit.is_uncertain, KnowledgeUnit.validation_status != "valid"))
        if f.search and f.search.strip():
            term = "%" + f.search.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            query = query.where(or_(KnowledgeUnit.title.ilike(term), KnowledgeUnit.content.ilike(term)))
        return query

    def list(self, f: KnowledgeFilter, *, limit: int, offset: int = 0) -> list[KnowledgeUnit]:
        query = self._where(select(KnowledgeUnit), f).order_by(
            KnowledgeUnit.page_start.nulls_last(), KnowledgeUnit.knowledge_type, KnowledgeUnit.title, KnowledgeUnit.id
        )
        return list(self.session.scalars(query.limit(limit).offset(offset)))

    def count(self, f: KnowledgeFilter) -> int:
        return self.session.scalar(self._where(select(func.count()).select_from(KnowledgeUnit), f)) or 0

    def counts_by(self, f: KnowledgeFilter, column) -> dict[str, int]:
        rows = self.session.execute(self._where(select(column, func.count()), f).group_by(column))
        return {key: total for key, total in rows}

    def get(self, unit_id: uuid.UUID) -> KnowledgeUnit | None:
        return self.session.get(KnowledgeUnit, unit_id)

    def execution_of(self, unit: KnowledgeUnit) -> ExtractionExecution | None:
        return self.session.get(ExtractionExecution, unit.extraction_execution_id) if unit.extraction_execution_id else None

    def passages(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, Passage]:
        if not ids:
            return {}
        return {p.id: p for p in self.session.scalars(select(Passage).where(Passage.id.in_(ids)))}

    def node_titles(self, ids: list[uuid.UUID | None]) -> dict[uuid.UUID, str]:
        wanted = [i for i in ids if i is not None]
        if not wanted:
            return {}
        return {n.id: n.title for n in self.session.scalars(select(StructuralNode).where(StructuralNode.id.in_(wanted)))}

    def sources_of(self, unit: KnowledgeUnit) -> list[KnowledgeUnitPassage]:
        return sorted(unit.sources, key=lambda link: link.position)

    def set_review(self, unit: KnowledgeUnit, status: str, note: str | None, at: datetime) -> None:
        unit.review_status = status
        metadata = dict(unit.unit_metadata or {})
        if note or status != "unreviewed":
            metadata["review"] = {"status": status, "note": note or None, "at": at.isoformat()}
        else:
            metadata.pop("review", None)
        unit.unit_metadata = metadata  # a new dict, so the change is saved
        self.session.commit()
