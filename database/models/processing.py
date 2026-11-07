from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, func
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
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
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
