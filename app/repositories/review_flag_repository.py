"""Database access for review flags."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import ReviewFlag


class ReviewFlagRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def find_open(
        self,
        code: str,
        dedupe_key: str,
        *,
        unit_id: uuid.UUID | None = None,
        attempt_id: uuid.UUID | None = None,
        relationship_id: uuid.UUID | None = None,
    ) -> ReviewFlag | None:
        query = select(ReviewFlag).where(
            ReviewFlag.status == "open", ReviewFlag.code == code, ReviewFlag.dedupe_key == dedupe_key
        )
        query = self._target(query, unit_id, attempt_id, relationship_id)
        return self.session.scalars(query.limit(1)).first()

    @staticmethod
    def _target(query, unit_id, attempt_id, relationship_id):
        if unit_id is not None:
            return query.where(ReviewFlag.knowledge_unit_id == unit_id)
        if attempt_id is not None:
            return query.where(ReviewFlag.extraction_attempt_id == attempt_id)
        return query.where(ReviewFlag.relationship_id == relationship_id)

    def get(self, flag_id: uuid.UUID) -> ReviewFlag | None:
        return self.session.get(ReviewFlag, flag_id)

    def list(
        self,
        book_id: uuid.UUID,
        *,
        status: str | None = "open",
        code: str | None = None,
        unit_id: uuid.UUID | None = None,
        attempt_id: uuid.UUID | None = None,
        relationship_id: uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
        limit: int | None = None,
    ) -> list[ReviewFlag]:
        query = select(ReviewFlag).where(ReviewFlag.book_id == book_id)
        if status:
            query = query.where(ReviewFlag.status == status)
        if code:
            query = query.where(ReviewFlag.code == code)
        if unit_id is not None:
            query = query.where(ReviewFlag.knowledge_unit_id == unit_id)
        if attempt_id is not None:
            query = query.where(ReviewFlag.extraction_attempt_id == attempt_id)
        if relationship_id is not None:
            query = query.where(ReviewFlag.relationship_id == relationship_id)
        if run_id is not None:
            query = query.where(ReviewFlag.extraction_run_id == run_id)
        query = query.order_by(ReviewFlag.created_at, ReviewFlag.id)
        if limit is not None:
            query = query.limit(limit)
        return list(self.session.scalars(query))

    def open_for_unit(self, unit_id: uuid.UUID, codes: list[str] | None = None) -> list[ReviewFlag]:
        query = select(ReviewFlag).where(ReviewFlag.knowledge_unit_id == unit_id, ReviewFlag.status == "open")
        if codes:
            query = query.where(ReviewFlag.code.in_(codes))
        return list(self.session.scalars(query))

    def counts_by_code(self, book_id: uuid.UUID, status: str = "open") -> dict[str, int]:
        rows = self.session.execute(
            select(ReviewFlag.code, func.count())
            .where(ReviewFlag.book_id == book_id, ReviewFlag.status == status)
            .group_by(ReviewFlag.code)
        )
        return {code: total for code, total in rows}
