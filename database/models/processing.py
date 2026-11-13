from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, func
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
    """Where a run currently is (or was when it failed). Status says whether it is running.

    In pipeline order: document_registration, parsing, ocr (only for scanned pages),
    normalizing, text_extraction (saving the text), equation_extraction, structural_detection,
    paragraph_identification, passage_segmentation, persistence, validation, completed.
    """

    DOCUMENT_REGISTRATION = "document_registration"
    PARSING = "parsing"
    OCR = "ocr"
    NORMALIZING = "normalizing"
    TEXT_EXTRACTION = "text_extraction"
    EQUATION_EXTRACTION = "equation_extraction"
    STRUCTURAL_DETECTION = "structural_detection"
    PARAGRAPH_IDENTIFICATION = "paragraph_identification"
    PASSAGE_SEGMENTATION = "passage_segmentation"
    PERSISTENCE = "persistence"
    VALIDATION = "validation"
    COMPLETED = "completed"


# Stages from which a failed run can resume without parsing the document again: the text is
# already saved by then.
RESUMABLE_STAGES = (
    ProcessingStage.EQUATION_EXTRACTION,
    ProcessingStage.STRUCTURAL_DETECTION,
    ProcessingStage.PARAGRAPH_IDENTIFICATION,
    ProcessingStage.PASSAGE_SEGMENTATION,
    ProcessingStage.PERSISTENCE,
    ProcessingStage.VALIDATION,
)


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
    stage: Mapped[str | None] = mapped_column(String(30))
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
    # How many times this run was resumed after a failure (a failed run resumes from the stage
    # that failed when its text is already saved; otherwise a retry starts a new run).
    retry_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # Set when the extracted text was saved: later failures resume without parsing again.
    text_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Per-stage outcome: {stage: {status, started_at, finished_at, ...stage-specific results}}.
    stage_results: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="processing_runs")
    errors: Mapped[list[ProcessingError]] = relationship(
        back_populates="processing_run", order_by="ProcessingError.created_at"
    )

    @property
    def book_id(self) -> uuid.UUID:
        return self.document.book_id

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
        CheckConstraint("retry_count >= 0", name="retry_count_non_negative"),
        Index("ix_processing_runs_document_created", "document_id", "created_at"),
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
    # The stage that failed and which attempt it was (0 = first attempt of the run).
    stage: Mapped[str | None] = mapped_column(String(30))
    retry_attempt: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    error_message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="processing_errors")
    processing_run: Mapped[ProcessingRun | None] = relationship(back_populates="errors")
