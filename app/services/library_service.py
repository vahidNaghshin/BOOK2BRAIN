"""Read model for the library screen: every book with its documents and their processing state."""

import logging
import uuid
from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.schemas import BookRead, DocumentRead
from database.models import (
    Book,
    Passage,
    ProcessingRun,
    StructuralNode,
)

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("pending", "processing")


class LibraryDocument(BaseModel):
    """A document with the state of its latest processing run."""

    document: DocumentRead
    run_id: uuid.UUID | None = None
    run_status: str | None = None
    stage: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0
    run_count: int = 0
    # True when the text was extracted but some pages could not be read (e.g. failed OCR).
    extraction_incomplete: bool = False
    node_count: int = 0
    passage_count: int = 0

    @property
    def is_active(self) -> bool:
        return self.document.processing_status in ACTIVE_STATUSES

    @property
    def is_processed(self) -> bool:
        return self.document.processing_status == "completed" and self.passage_count > 0


class LibraryBook(BaseModel):
    book: BookRead
    documents: list[LibraryDocument] = Field(default_factory=list)

    @property
    def has_active(self) -> bool:
        return any(d.is_active for d in self.documents)


class LibraryService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_books(self) -> list[LibraryBook]:
        """All books (newest first) with their documents, fresh from the database."""
        books = list(
            self.session.scalars(
                select(Book).options(selectinload(Book.documents)).order_by(Book.created_at.desc(), Book.title)
            )
        )
        document_ids = [d.id for b in books for d in b.documents]
        latest = self._latest_runs(document_ids)
        run_counts = dict(
            self.session.execute(
                select(ProcessingRun.document_id, func.count()).where(ProcessingRun.document_id.in_(document_ids)).group_by(ProcessingRun.document_id)
            ).all()
        ) if document_ids else {}
        node_counts = self._counts(StructuralNode.document_id, document_ids)
        passage_counts = self._counts(Passage.document_id, document_ids)

        result = []
        for book in books:
            documents = []
            for document in book.documents:
                run = latest.get(document.id)
                summary = (run.result_summary if run else None) or {}
                documents.append(
                    LibraryDocument(
                        document=DocumentRead.model_validate(document),
                        run_id=run.id if run else None,
                        run_status=run.status if run else None,
                        stage=run.stage if run else None,
                        progress_current=run.progress_current if run else None,
                        progress_total=run.progress_total if run else None,
                        started_at=run.started_at if run else None,
                        completed_at=run.completed_at if run else None,
                        updated_at=run.updated_at if run else None,
                        error_message=run.error_message if run else None,
                        retry_count=(run.retry_count if run else 0) or 0,
                        run_count=run_counts.get(document.id, 0),
                        extraction_incomplete=summary.get("status") == "partial",
                        node_count=node_counts.get(document.id, 0),
                        passage_count=passage_counts.get(document.id, 0),
                    )
                )
            result.append(LibraryBook(book=BookRead.model_validate(book), documents=documents))
        return result

    def _latest_runs(self, document_ids: list[uuid.UUID]) -> dict[uuid.UUID, ProcessingRun]:
        if not document_ids:
            return {}
        newest = (
            select(ProcessingRun.document_id, func.max(ProcessingRun.created_at).label("created_at"))
            .where(ProcessingRun.document_id.in_(document_ids))
            .group_by(ProcessingRun.document_id)
            .subquery()
        )
        stmt = select(ProcessingRun).join(
            newest,
            (ProcessingRun.document_id == newest.c.document_id) & (ProcessingRun.created_at == newest.c.created_at),
        )
        return {run.document_id: run for run in self.session.scalars(stmt)}

    def _counts(self, column, document_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
        if not document_ids:
            return {}
        return dict(self.session.execute(select(column, func.count()).where(column.in_(document_ids)).group_by(column)).all())
