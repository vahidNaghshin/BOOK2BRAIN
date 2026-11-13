import uuid
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.orm import Session

from database.models import (
    Document,
    Equation,
    ExtractedBlock,
    ProcessingError,
    ProcessingRun,
    ProcessingRunStatus,
)

_ACTIVE_RUN_STATUSES = (ProcessingRunStatus.PENDING.value, ProcessingRunStatus.PROCESSING.value)
_INSERT_BATCH = 1000


class ProcessingRepository:
    """Processing runs and errors, plus the row lock used when a run is started."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def lock_document(self, document_id: uuid.UUID) -> Document | None:
        stmt = select(Document).where(Document.id == document_id).with_for_update()
        return self.session.execute(stmt).scalar_one_or_none()

    def add_run(self, run: ProcessingRun) -> ProcessingRun:
        self.session.add(run)
        self.session.flush()
        return run

    def get_run(self, run_id: uuid.UUID) -> ProcessingRun | None:
        return self.session.get(ProcessingRun, run_id)

    def latest_run(self, document_id: uuid.UUID) -> ProcessingRun | None:
        stmt = (
            select(ProcessingRun)
            .where(ProcessingRun.document_id == document_id)
            .order_by(ProcessingRun.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def latest_active_run(self, document_id: uuid.UUID) -> ProcessingRun | None:
        stmt = (
            select(ProcessingRun)
            .where(
                ProcessingRun.document_id == document_id,
                ProcessingRun.status.in_(_ACTIVE_RUN_STATUSES),
            )
            .order_by(ProcessingRun.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def count_runs(self, document_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(ProcessingRun).where(
            ProcessingRun.document_id == document_id
        )
        return self.session.scalar(stmt) or 0

    def update_progress(
        self, run_id: uuid.UUID, *, stage: str, current: int, total: int
    ) -> None:
        self.session.execute(
            update(ProcessingRun)
            .where(ProcessingRun.id == run_id)
            .values(stage=stage, progress_current=current, progress_total=total)
        )

    def set_retry_count(self, run_id: uuid.UUID, retry_count: int) -> None:
        self.session.execute(update(ProcessingRun).where(ProcessingRun.id == run_id).values(retry_count=retry_count))

    def clear_progress(self, run_id: uuid.UUID) -> None:
        self.session.execute(
            update(ProcessingRun)
            .where(ProcessingRun.id == run_id)
            .values(progress_current=None, progress_total=None)
        )

    def add_error(
        self,
        *,
        document_id: uuid.UUID,
        run_id: uuid.UUID | None,
        error_type: str,
        message: str,
        details: dict[str, Any] | None,
        stage: str | None = None,
        retry_attempt: int = 0,
    ) -> ProcessingError:
        error = ProcessingError(
            document_id=document_id,
            processing_run_id=run_id,
            error_type=error_type,
            stage=stage or (details or {}).get("stage"),
            retry_attempt=retry_attempt,
            error_message=message,
            details=details or None,
        )
        self.session.add(error)
        return error

    def list_errors(self, document_id: uuid.UUID, limit: int = 20) -> list[ProcessingError]:
        stmt = (
            select(ProcessingError)
            .where(ProcessingError.document_id == document_id)
            .order_by(ProcessingError.created_at.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())


class ExtractedBlockRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def replace_for_document(self, document_id: uuid.UUID, rows: list[dict[str, Any]]) -> None:
        """Delete the document's blocks and insert the new ones. The caller commits."""
        self.session.execute(
            ExtractedBlock.__table__.delete().where(ExtractedBlock.document_id == document_id)
        )
        for start in range(0, len(rows), _INSERT_BATCH):
            self.session.execute(
                insert(ExtractedBlock),
                [{"document_id": document_id, **row} for row in rows[start : start + _INSERT_BATCH]],
            )

    def count(self, document_id: uuid.UUID, page_number: int | None = None) -> int:
        stmt = select(func.count()).select_from(ExtractedBlock).where(
            ExtractedBlock.document_id == document_id
        )
        if page_number is not None:
            stmt = stmt.where(ExtractedBlock.page_number == page_number)
        return self.session.scalar(stmt) or 0

    def list(
        self,
        document_id: uuid.UUID,
        *,
        offset: int = 0,
        limit: int = 50,
        page_number: int | None = None,
    ) -> list[ExtractedBlock]:
        stmt = select(ExtractedBlock).where(ExtractedBlock.document_id == document_id)
        if page_number is not None:
            stmt = stmt.where(ExtractedBlock.page_number == page_number)
        stmt = stmt.order_by(ExtractedBlock.sequence_number).offset(offset).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def ids_by_sequence(self, document_id: uuid.UUID) -> dict[int, uuid.UUID]:
        stmt = select(ExtractedBlock.sequence_number, ExtractedBlock.id).where(
            ExtractedBlock.document_id == document_id
        )
        return {sequence: block_id for sequence, block_id in self.session.execute(stmt)}


class EquationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def crop_references(self, document_id: uuid.UUID) -> list[str]:
        stmt = select(Equation.crop_reference).where(
            Equation.document_id == document_id, Equation.crop_reference.is_not(None)
        )
        return [reference for (reference,) in self.session.execute(stmt)]

    def replace_for_document(self, document_id: uuid.UUID, rows: list[dict[str, Any]]) -> None:
        """Delete the document's equations and insert the new ones. The caller commits.

        Replacing the whole set in one transaction is what makes retries idempotent: there is
        never a moment with old and new equations side by side, so nothing can be duplicated.
        """
        self.session.execute(Equation.__table__.delete().where(Equation.document_id == document_id))
        for start in range(0, len(rows), _INSERT_BATCH):
            self.session.execute(
                insert(Equation),
                [{"document_id": document_id, **row} for row in rows[start : start + _INSERT_BATCH]],
            )

    def _filtered(
        self,
        document_id: uuid.UUID | None,
        page_number: int | None,
        equation_type: str | None,
        validation_status: str | None,
    ) -> Any:
        stmt = select(Equation)
        if document_id is not None:
            stmt = stmt.where(Equation.document_id == document_id)
        if page_number is not None:
            stmt = stmt.where(Equation.page_number == page_number)
        if equation_type:
            stmt = stmt.where(Equation.equation_type == equation_type)
        if validation_status:
            stmt = stmt.where(Equation.validation_status == validation_status)
        return stmt

    def count(
        self,
        document_id: uuid.UUID | None = None,
        *,
        page_number: int | None = None,
        equation_type: str | None = None,
        validation_status: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(
            self._filtered(document_id, page_number, equation_type, validation_status).subquery()
        )
        return self.session.scalar(stmt) or 0

    def list(
        self,
        document_id: uuid.UUID | None = None,
        *,
        page_number: int | None = None,
        equation_type: str | None = None,
        validation_status: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> list[Equation]:
        stmt = (
            self._filtered(document_id, page_number, equation_type, validation_status)
            .order_by(Equation.document_id, Equation.sequence_number)
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())
