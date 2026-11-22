"""Database access for knowledge relationships."""

from __future__ import annotations

import uuid

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from database.models import KnowledgeRelationship, KnowledgeUnit


class RelationshipRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, relationship_id: uuid.UUID) -> KnowledgeRelationship | None:
        return self.session.get(KnowledgeRelationship, relationship_id)

    def find_edge(self, source_id: uuid.UUID, target_id: uuid.UUID, relationship_type: str) -> KnowledgeRelationship | None:
        return self.session.scalars(
            select(KnowledgeRelationship).where(
                KnowledgeRelationship.source_unit_id == source_id,
                KnowledgeRelationship.target_unit_id == target_id,
                KnowledgeRelationship.relationship_type == relationship_type,
            )
        ).first()

    def units(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, KnowledgeUnit]:
        return {u.id: u for u in self.session.scalars(select(KnowledgeUnit).where(KnowledgeUnit.id.in_(ids)))}

    def for_unit(self, unit_id: uuid.UUID) -> list[KnowledgeRelationship]:
        return list(
            self.session.scalars(
                select(KnowledgeRelationship)
                .where(or_(KnowledgeRelationship.source_unit_id == unit_id, KnowledgeRelationship.target_unit_id == unit_id))
                .order_by(KnowledgeRelationship.created_at, KnowledgeRelationship.id)
            )
        )

    def list(
        self,
        book_id: uuid.UUID,
        *,
        relationship_type: str | None = None,
        origin: str | None = None,
        review_status: str | None = None,
        run_id: uuid.UUID | None = None,
        limit: int | None = None,
    ) -> list[KnowledgeRelationship]:
        query = select(KnowledgeRelationship).where(KnowledgeRelationship.book_id == book_id)
        if relationship_type:
            query = query.where(KnowledgeRelationship.relationship_type == relationship_type)
        if origin:
            query = query.where(KnowledgeRelationship.origin == origin)
        if review_status:
            query = query.where(KnowledgeRelationship.review_status == review_status)
        if run_id is not None:
            query = query.where(KnowledgeRelationship.extraction_run_id == run_id)
        query = query.order_by(KnowledgeRelationship.created_at, KnowledgeRelationship.id)
        if limit is not None:
            query = query.limit(limit)
        return list(self.session.scalars(query))
