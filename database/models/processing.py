from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.models.base import Base, UUIDPrimaryKeyMixin, utcnow

if TYPE_CHECKING:
    from database.models.document import Document


class ProcessingRunStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(enum.StrEnum):
    """Where a run currently is (or was when it failed). Status says whether it is running."""

    PARSING = "parsing"
    OCR = "ocr"
    NORMALIZING = "normalizing"


class EquationStatus(enum.StrEnum):
    """Equation extraction progress for a run. Independent of the run and document status."""

    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"  # needed an engine or model that is not installed


class ProcessingRun(UUIDPrimaryKeyMixin, Base):
    """One processing attempt for a document."""

    __tablename__ = "processing_runs"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default=ProcessingRunStatus.PENDING.value,
        server_default=ProcessingRunStatus.PENDING.value,
        index=True,
    )
    stage: Mapped[str | None] = mapped_column(String(20))
    progress_current: Mapped[int | None] = mapped_column(Integer)
    progress_total: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    # Extraction summary (page count, OCR pages, warnings...) for completed runs.
    result_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Equation extraction runs after the text is saved and has its own status, so an equation
    # problem never changes whether the document itself was processed successfully.
    equation_status: Mapped[str] = mapped_column(
        String(30),
        default=EquationStatus.NOT_STARTED.value,
        server_default=EquationStatus.NOT_STARTED.value,
    )
    equation_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="processing_runs")
    errors: Mapped[list[ProcessingError]] = relationship(
        back_populates="processing_run", order_by="ProcessingError.created_at"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ("
            + ", ".join(f"'{member.value}'" for member in ProcessingRunStatus)
            + ")",
            name="status_valid",
        ),
        CheckConstraint(
            "stage IN (" + ", ".join(f"'{member.value}'" for member in ProcessingStage) + ")",
            name="stage_valid",
        ),
        CheckConstraint(
            "equation_status IN (" + ", ".join(f"'{m.value}'" for m in EquationStatus) + ")",
            name="equation_status_valid",
        ),
    )


class ProcessingError(UUIDPrimaryKeyMixin, Base):
    """A processing-related error, optionally tied to a specific run."""

    __tablename__ = "processing_errors"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    processing_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("processing_runs.id", ondelete="SET NULL"), index=True
    )
    error_type: Mapped[str] = mapped_column(String(255))
    error_message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="processing_errors")
    processing_run: Mapped[ProcessingRun | None] = relationship(back_populates="errors")
