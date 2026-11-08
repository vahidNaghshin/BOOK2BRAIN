import uuid
from datetime import timedelta

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.parsing.models import BlockType
from database.models import (
    BLOCK_TYPES,
    Book,
    Document,
    ExtractedBlock,
    ProcessingError,
    ProcessingRun,
    ProcessingRunStatus,
    ProcessingStage,
    ProcessingStatus,
)


def make_book(title: str = "Dune", author: str | None = "Frank Herbert") -> Book:
    return Book(title=title, author=author)


def make_document(book: Book, **overrides) -> Document:
    values = {
        "book": book,
        "original_filename": "dune.pdf",
        "stored_filename": f"{uuid.uuid4()}.pdf",
        "file_path": f"original/{uuid.uuid4()}/dune.pdf",
        "file_format": "pdf",
        "mime_type": "application/pdf",
        "file_size": 1234,
        "sha256_hash": uuid.uuid4().hex + uuid.uuid4().hex,
    }
    values.update(overrides)
    return Document(**values)


def test_book_is_created_with_uuid_and_utc_timestamps(session: Session) -> None:
    book = make_book()
    session.add(book)
    session.commit()

    saved = session.scalars(select(Book)).one()
    assert isinstance(saved.id, uuid.UUID)
    assert saved.title == "Dune"
    assert saved.subtitle is None and saved.isbn is None
    assert saved.created_at.utcoffset() == timedelta(0)
    assert saved.updated_at.utcoffset() == timedelta(0)


def test_updated_at_changes_on_update(session: Session) -> None:
    book = make_book()
    session.add(book)
    session.commit()
    first = book.updated_at

    book.description = "Desert planet"
    session.commit()

    assert book.updated_at > first


def test_document_defaults_to_uploaded_status(session: Session) -> None:
    document = make_document(make_book())
    session.add(document)
    session.commit()

    assert document.processing_status == ProcessingStatus.UPLOADED.value
    assert document.created_at.utcoffset() == timedelta(0)


def test_book_has_many_documents(session: Session) -> None:
    book = make_book()
    pdf = make_document(book, file_format="pdf")
    epub = make_document(book, file_format="epub", mime_type="application/epub+zip")
    session.add_all([book, pdf, epub])
    session.commit()

    session.expire_all()
    saved = session.scalars(select(Book)).one()
    assert {d.file_format for d in saved.documents} == {"pdf", "epub"}
    assert all(d.book is saved for d in saved.documents)


def test_processing_runs_and_errors_relationships(session: Session) -> None:
    document = make_document(make_book())
    run = ProcessingRun(document=document)
    error = ProcessingError(
        document=document,
        processing_run=run,
        error_type="ParseError",
        error_message="boom",
        details={"page": 3},
    )
    session.add_all([document, run, error])
    session.commit()

    session.expire_all()
    saved = session.scalars(select(Document)).one()
    assert saved.processing_runs[0].status == ProcessingRunStatus.PENDING.value
    assert saved.processing_runs[0].started_at is None
    assert saved.processing_errors[0].details == {"page": 3}
    assert saved.processing_runs[0].errors[0].error_type == "ParseError"


def test_deleting_book_cascades_to_documents_runs_and_errors(session: Session) -> None:
    book = make_book()
    document = make_document(book)
    run = ProcessingRun(document=document)
    error = ProcessingError(
        document=document, processing_run=run, error_type="E", error_message="m"
    )
    session.add_all([book, document, run, error])
    session.commit()

    session.delete(book)
    session.commit()

    for model in (Book, Document, ProcessingRun, ProcessingError):
        assert session.scalars(select(model)).all() == []


def test_deleting_run_keeps_error_with_null_run(session: Session) -> None:
    document = make_document(make_book())
    run = ProcessingRun(document=document)
    error = ProcessingError(
        document=document, processing_run=run, error_type="E", error_message="m"
    )
    session.add_all([document, run, error])
    session.commit()

    session.execute(ProcessingRun.__table__.delete())
    session.commit()
    session.expire_all()

    kept = session.scalars(select(ProcessingError)).one()
    assert kept.processing_run_id is None


def test_sha256_hash_must_be_unique(session: Session) -> None:
    book = make_book()
    session.add_all(
        [make_document(book, sha256_hash="a" * 64), make_document(book, sha256_hash="a" * 64)]
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_book_identity_is_unique_case_insensitively(session: Session) -> None:
    session.add(make_book("Dune", "Frank Herbert"))
    session.commit()

    session.add(make_book("DUNE", "frank herbert"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_same_title_with_different_author_is_allowed(session: Session) -> None:
    session.add_all([make_book("Dune", "Frank Herbert"), make_book("Dune", "Someone Else")])
    session.commit()

    assert len(session.scalars(select(Book)).all()) == 2


def test_isbn_is_not_unique(session: Session) -> None:
    session.add_all(
        [
            Book(title="A", isbn="9780306406157"),
            Book(title="B", isbn="9780306406157"),
        ]
    )
    session.commit()


@pytest.mark.parametrize(
    "overrides",
    [{"processing_status": "bogus"}, {"file_size": 0}],
    ids=["invalid-status", "non-positive-size"],
)
def test_document_check_constraints(session: Session, overrides: dict) -> None:
    session.add(make_document(make_book(), **overrides))
    with pytest.raises(IntegrityError):
        session.commit()


def test_processing_run_status_constraint(session: Session) -> None:
    document = make_document(make_book())
    session.add_all([document, ProcessingRun(document=document, status="bogus")])
    with pytest.raises(IntegrityError):
        session.commit()


def test_document_requires_existing_book(session: Session) -> None:
    orphan = Document(
        book_id=uuid.uuid4(),
        original_filename="x.pdf",
        stored_filename="x.pdf",
        file_path="original/x/x.pdf",
        file_format="pdf",
        mime_type="application/pdf",
        file_size=1,
        sha256_hash="b" * 64,
    )
    session.add(orphan)
    with pytest.raises(IntegrityError):
        session.commit()


# -- Day 3: extracted blocks and processing run progress -------------------------------


def make_block(document: Document, sequence: int = 0, **overrides) -> ExtractedBlock:
    values = {
        "document": document,
        "sequence_number": sequence,
        "block_type": "paragraph",
        "text": f"Block {sequence} text.",
    }
    values.update(overrides)
    return ExtractedBlock(**values)


def test_block_types_in_the_database_match_the_parser_enum() -> None:
    assert set(BLOCK_TYPES) == {member.value for member in BlockType}


def test_extracted_blocks_belong_to_a_document_in_sequence_order(session: Session) -> None:
    document = make_document(make_book())
    session.add_all([document, make_block(document, 2), make_block(document, 0), make_block(document, 1)])
    session.commit()

    session.expire_all()
    saved = session.scalars(select(Document)).one()
    assert [b.sequence_number for b in saved.extracted_blocks] == [0, 1, 2]
    assert all(b.document is saved for b in saved.extracted_blocks)
    assert isinstance(saved.extracted_blocks[0].id, uuid.UUID)
    assert saved.extracted_blocks[0].created_at.utcoffset() == timedelta(0)


def test_extracted_block_stores_json_and_optional_fields(session: Session) -> None:
    document = make_document(make_book())
    session.add(
        make_block(
            document,
            page_number=12,
            section_title="Chapter 1",
            raw_text="Original  text",
            source_location={"page": 12, "bbox": [1.5, 2.5, 3.5, 4.5]},
            block_metadata={"heading_level": 2, "nested": {"ok": True}},
        )
    )
    session.commit()

    session.expire_all()
    saved = session.scalars(select(ExtractedBlock)).one()
    assert saved.page_number == 12 and saved.section_title == "Chapter 1"
    assert saved.raw_text == "Original  text"
    assert saved.source_location == {"page": 12, "bbox": [1.5, 2.5, 3.5, 4.5]}
    assert saved.block_metadata == {"heading_level": 2, "nested": {"ok": True}}


def test_extracted_block_optional_fields_default_to_null(session: Session) -> None:
    session.add(make_block(make_document(make_book())))
    session.commit()

    saved = session.scalars(select(ExtractedBlock)).one()
    assert saved.page_number is None and saved.raw_text is None and saved.section_title is None
    assert saved.source_location is None and saved.block_metadata is None


def test_the_metadata_column_has_the_requested_name(engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("extracted_blocks")}

    assert {"id", "document_id", "sequence_number", "block_type", "text", "raw_text", "page_number",
            "section_title", "source_location", "metadata", "created_at"} <= columns


def test_sequence_numbers_are_unique_per_document(session: Session) -> None:
    first, second = make_document(make_book("A")), make_document(make_book("B"))
    session.add_all([first, second, make_block(first, 0), make_block(second, 0)])
    session.commit()  # the same sequence number in different documents is fine

    session.add(make_block(first, 0))
    with pytest.raises(IntegrityError):
        session.commit()


@pytest.mark.parametrize(
    "overrides",
    [{"block_type": "bogus"}, {"page_number": 0}, {"page_number": -3}, {"sequence_number": -1}],
    ids=["invalid-type", "page-zero", "page-negative", "negative-sequence"],
)
def test_extracted_block_check_constraints(session: Session, overrides: dict) -> None:
    session.add(make_block(make_document(make_book()), **{"sequence_number": 0, **overrides}))
    with pytest.raises(IntegrityError):
        session.commit()


def test_deleting_a_document_deletes_its_blocks(session: Session) -> None:
    book = make_book()
    document = make_document(book)
    session.add_all([book, document, make_block(document, 0), make_block(document, 1)])
    session.commit()

    session.delete(document)
    session.commit()

    assert session.scalars(select(ExtractedBlock)).all() == []


def test_processing_run_progress_columns_default_to_null_and_store_a_summary(session: Session) -> None:
    document = make_document(make_book())
    run = ProcessingRun(document=document)
    session.add_all([document, run])
    session.commit()
    assert (run.stage, run.progress_current, run.progress_total, run.result_summary) == (None, None, None, None)

    run.stage = ProcessingStage.OCR.value
    run.progress_current, run.progress_total = 3, 10
    run.result_summary = {"page_count": 10, "warnings": []}
    session.commit()

    session.expire_all()
    saved = session.scalars(select(ProcessingRun)).one()
    assert (saved.stage, saved.progress_current, saved.progress_total) == ("ocr", 3, 10)
    assert saved.result_summary == {"page_count": 10, "warnings": []}


def test_processing_run_stage_constraint(session: Session) -> None:
    document = make_document(make_book())
    session.add_all([document, ProcessingRun(document=document, stage="bogus")])
    with pytest.raises(IntegrityError):
        session.commit()
