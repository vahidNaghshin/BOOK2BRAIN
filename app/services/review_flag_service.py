"""Flagging knowledge units, extraction attempts and relationships for review, with the reason.

Flags are findings, not decisions. Raising the same flag again (same target, code and key) while it is
open does nothing, so verifying, resuming or re-running never piles them up. This is not a review
workflow: a flag can be resolved or dismissed with a note, and that is all.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.extraction.results import utcnow
from app.knowledge.types import FlagSeverity, FlagStatus, ReviewFlagCode
from app.repositories.review_flag_repository import ReviewFlagRepository
from database.models import ReviewFlag

logger = logging.getLogger(__name__)


class FlagRead(BaseModel):
    id: uuid.UUID
    book_id: uuid.UUID
    target_kind: str  # "unit", "attempt" or "relationship"
    target_id: uuid.UUID
    code: ReviewFlagCode
    severity: FlagSeverity
    message: str
    details: dict[str, Any]
    raised_by: str
    status: FlagStatus
    extraction_run_id: uuid.UUID | None
    created_at: datetime
    resolved_at: datetime | None
    resolution_note: str | None


def flag_read(flag: ReviewFlag) -> FlagRead:
    if flag.knowledge_unit_id is not None:
        kind, target = "unit", flag.knowledge_unit_id
    elif flag.extraction_attempt_id is not None:
        kind, target = "attempt", flag.extraction_attempt_id
    else:
        kind, target = "relationship", flag.relationship_id
    return FlagRead(
        id=flag.id,
        book_id=flag.book_id,
        target_kind=kind,
        target_id=target,
        code=flag.code,
        severity=flag.severity,
        message=flag.message,
        details=flag.details or {},
        raised_by=flag.raised_by,
        status=flag.status,
        extraction_run_id=flag.extraction_run_id,
        created_at=flag.created_at,
        resolved_at=flag.resolved_at,
        resolution_note=flag.resolution_note,
    )


class ReviewFlagService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = ReviewFlagRepository(session)

    # -- raising -----------------------------------------------------------------------------------

    def flag_unit(self, book_id: uuid.UUID, unit_id: uuid.UUID, code: ReviewFlagCode | str, message: str, **options) -> tuple[ReviewFlag, bool]:
        return self._flag(book_id, code, message, unit_id=unit_id, **options)

    def flag_attempt(self, book_id: uuid.UUID, attempt_id: uuid.UUID, code: ReviewFlagCode | str, message: str, **options) -> tuple[ReviewFlag, bool]:
        return self._flag(book_id, code, message, attempt_id=attempt_id, **options)

    def flag_relationship(self, book_id: uuid.UUID, relationship_id: uuid.UUID, code: ReviewFlagCode | str, message: str, **options) -> tuple[ReviewFlag, bool]:
        return self._flag(book_id, code, message, relationship_id=relationship_id, **options)

    def _flag(
        self,
        book_id: uuid.UUID,
        code: ReviewFlagCode | str,
        message: str,
        *,
        unit_id: uuid.UUID | None = None,
        attempt_id: uuid.UUID | None = None,
        relationship_id: uuid.UUID | None = None,
        severity: FlagSeverity | str = FlagSeverity.WARNING,
        details: dict[str, Any] | None = None,
        dedupe_key: str = "",
        raised_by: str = "system",
        run_id: uuid.UUID | None = None,
        commit: bool = True,
    ) -> tuple[ReviewFlag, bool]:
        """Returns (flag, created). An open flag with the same target, code and key is returned as it is."""
        code = ReviewFlagCode(code).value
        existing = self.repository.find_open(code, dedupe_key, unit_id=unit_id, attempt_id=attempt_id, relationship_id=relationship_id)
        if existing is not None:
            return existing, False
        flag = ReviewFlag(
            id=uuid.uuid4(),
            book_id=book_id,
            knowledge_unit_id=unit_id,
            extraction_attempt_id=attempt_id,
            relationship_id=relationship_id,
            extraction_run_id=run_id,
            code=code,
            severity=FlagSeverity(severity).value,
            message=message,
            details=details or {},
            dedupe_key=dedupe_key,
            raised_by=raised_by,
        )
        self.session.add(flag)
        self.session.flush()
        if commit:
            self.session.commit()
        logger.info("Review flag %s raised (%s)", code, message)
        return flag, True

    # -- reading and resolving ---------------------------------------------------------------------

    def list_flags(self, book_id: uuid.UUID, **filters) -> list[FlagRead]:
        return [flag_read(f) for f in self.repository.list(book_id, **filters)]

    def flags_for_unit(self, book_id: uuid.UUID, unit_id: uuid.UUID, *, status: str | None = None) -> list[FlagRead]:
        return self.list_flags(book_id, unit_id=unit_id, status=status)

    def counts_by_code(self, book_id: uuid.UUID, status: str = "open") -> dict[str, int]:
        return self.repository.counts_by_code(book_id, status)

    def resolve(self, flag_id: uuid.UUID, note: str | None = None, *, dismiss: bool = False, commit: bool = True) -> FlagRead:
        """Close a flag: resolved (the problem was dealt with) or dismissed (it is not a problem)."""
        flag = self.repository.get(flag_id)
        if flag is None:
            raise LookupError("The review flag does not exist.")
        if flag.status == "open":
            flag.status = FlagStatus.DISMISSED.value if dismiss else FlagStatus.RESOLVED.value
            flag.resolved_at = utcnow()
            flag.resolution_note = (note or "").strip() or None
            if commit:
                self.session.commit()
        return flag_read(flag)

    def resolve_missing(
        self, unit_id: uuid.UUID, codes: Iterable[ReviewFlagCode | str], still_present: set[tuple[str, str]], note: str, *, commit: bool = True
    ) -> int:
        """Close open system flags of these codes on a unit whose (code, key) no longer applies."""
        closed = 0
        for flag in self.repository.open_for_unit(unit_id, [ReviewFlagCode(c).value for c in codes]):
            if flag.raised_by == "system" and (flag.code, flag.dedupe_key) not in still_present:
                flag.status = FlagStatus.RESOLVED.value
                flag.resolved_at = utcnow()
                flag.resolution_note = note
                closed += 1
        if closed and commit:
            self.session.commit()
        return closed
