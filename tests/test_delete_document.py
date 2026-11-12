import io
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.schemas import BookMetadata
from app.services.document_service import DocumentService
from app.services.errors import DocumentNotFoundError, ProcessingNotAllowedError
from app.services.storage_service import LocalStorage
from database.models import Book, Document, Equation, ExtractedBlock, ProcessingStatus
from tests.factories import make_epub, make_pdf


def upload(service: DocumentService, data: bytes, filename: str, metadata: BookMetadata):
    return service.upload(file=io.BytesIO(data), filename=filename, metadata=metadata).document


def count(session: Session, model) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_deleting_the_only_document_removes_the_book_and_the_file(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    document = upload(document_service, make_pdf("d1"), "dune.pdf", metadata)
    assert storage.exists(document.file_path)

    book_deleted = document_service.delete_document(document.id)

    assert book_deleted is True
    assert count(session, Document) == 0 and count(session, Book) == 0
    assert not storage.exists(document.file_path)


def test_deleting_one_of_several_documents_keeps_the_book(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    pdf = upload(document_service, make_pdf("d2"), "dune.pdf", metadata)
    epub = upload(document_service, make_epub("d3"), "dune.epub", metadata)

    book_deleted = document_service.delete_document(pdf.id)

    assert book_deleted is False
    assert count(session, Book) == 1
    assert [d.id for d in session.scalars(select(Document))] == [epub.id]
    assert storage.exists(epub.file_path) and not storage.exists(pdf.file_path)


def test_derived_records_and_equation_crops_are_removed(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    document = upload(document_service, make_pdf("d4"), "dune.pdf", metadata)
    crop_key = f"equations/{document.id}/00000.png"
    storage.save(crop_key, io.BytesIO(b"png"))
    session.add(ExtractedBlock(document_id=document.id, sequence_number=0, block_type="paragraph", text="Hello"))
    session.add(
        Equation(
            document_id=document.id, sequence_number=0, identity_hash="h", latex="x", extraction_method="mathml",
            crop_reference=crop_key,
        )
    )
    session.commit()

    document_service.delete_document(document.id)

    assert count(session, ExtractedBlock) == 0 and count(session, Equation) == 0
    assert not storage.exists(crop_key)


def test_documents_being_processed_cannot_be_deleted(
    document_service: DocumentService, session: Session, storage: LocalStorage, metadata: BookMetadata
) -> None:
    document = upload(document_service, make_pdf("d5"), "dune.pdf", metadata)
    session.get(Document, document.id).processing_status = ProcessingStatus.PROCESSING.value
    session.commit()

    with pytest.raises(ProcessingNotAllowedError):
        document_service.delete_document(document.id)

    assert count(session, Document) == 1 and storage.exists(document.file_path)


def test_deleting_an_unknown_document_is_reported(document_service: DocumentService) -> None:
    with pytest.raises(DocumentNotFoundError):
        document_service.delete_document(uuid.uuid4())


def test_a_deleted_file_can_be_uploaded_again(
    document_service: DocumentService, metadata: BookMetadata
) -> None:
    data = make_pdf("d6")
    document = upload(document_service, data, "dune.pdf", metadata)
    document_service.delete_document(document.id)

    assert upload(document_service, data, "dune.pdf", metadata).id != document.id
