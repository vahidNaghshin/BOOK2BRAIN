import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from database.models import Document


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, document: Document) -> Document:
        """Add and flush, so unique-constraint violations surface immediately."""
        self.session.add(document)
        self.session.flush()
        return document

    def get(self, document_id: uuid.UUID) -> Document | None:
        return self.session.get(Document, document_id)

    def get_by_hash(self, sha256_hash: str) -> Document | None:
        stmt = (
            select(Document)
            .where(Document.sha256_hash == sha256_hash)
            .options(joinedload(Document.book))
        )
        return self.session.execute(stmt).scalar_one_or_none()
