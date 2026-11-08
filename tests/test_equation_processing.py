"""Equation persistence and workflow: relationships, retries, failure isolation, crop cleanup."""

import io
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.parsing.equation.equation_models import (
    EquationBlock,
    EquationExtractionResult,
    EquationExtractionStatus,
    EquationType,
    ExtractionMethod,
    ValidationStatus,
)
from app.parsing.equation.equation_service import EquationOutcome, EquationService
from app.schemas import BookMetadata
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import DocumentService
from app.services.errors import ProcessingNotAllowedError
from app.services.storage_service import LocalStorage
from database.models import Book, Document, Equation, ExtractedBlock, ProcessingError, ProcessingRun
from tests.conftest import stored_files
from tests.factories import make_docx_with_equations, make_epub_book

PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def settings(test_database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        database_url=test_database_url, storage_root=tmp_path / "storage", max_upload_size_mb=20, _env_file=None
    )


class ScriptedEquationService(EquationService):
    """Returns prepared equations (and crops), or raises, instead of analysing the file."""

    def __init__(self, settings: Settings, equations=(), crops: bool = False, error: Exception | None = None) -> None:
        super().__init__(settings)
        self.equations = list(equations)
        self.crops = crops
        self.error = error
        self.calls = 0

    def extract(self, *, path, file_format, document_id, parsing, block_ids=None, on_progress=None) -> EquationOutcome:
        self.calls += 1
        if self.error:
            raise self.error
        built = [
            self._finalize(eq.model_copy(update={"document_id": document_id}), index)
            for index, eq in enumerate(self.equations)
        ]
        result = EquationExtractionResult(status=EquationExtractionStatus.COMPLETED, equations=built)
        return EquationOutcome(result, {e.id: PNG for e in built} if self.crops else {})


def equation(latex: str = "x = 1", **kwargs) -> EquationBlock:
    return EquationBlock(
        document_id=uuid.uuid4(),
        latex=latex,
        extraction_method=ExtractionMethod.MATH_OCR,
        equation_type=EquationType.DISPLAY,
        validation_status=ValidationStatus.VALID,
        **kwargs,
    )


@pytest.fixture
def upload(document_service: DocumentService):
    counter = iter(range(1000))

    def _upload(data: bytes, filename: str):
        return document_service.upload(
            file=io.BytesIO(data), filename=filename, metadata=BookMetadata(title=f"Book {next(counter)}")
        ).document

    return _upload


@pytest.fixture
def make_service(session: Session, storage: LocalStorage, settings: Settings):
    def _make(equation_service: EquationService | None = None) -> DocumentProcessingService:
        return DocumentProcessingService(session, storage, settings, equation_service=equation_service)

    return _make


def equations_of(session: Session, document_id: uuid.UUID) -> list[Equation]:
    session.rollback()
    stmt = select(Equation).where(Equation.document_id == document_id).order_by(Equation.sequence_number)
    return list(session.scalars(stmt))


def latest_run(session: Session, document_id: uuid.UUID) -> ProcessingRun:
    session.rollback()
    return session.scalars(
        select(ProcessingRun).where(ProcessingRun.document_id == document_id).order_by(ProcessingRun.created_at.desc())
    ).first()


DOCX_ITEMS = [
    ("heading", "Mechanics"),
    ("paragraph", "Newton's second law is stated below and used throughout this chapter of the book."),
    ("display", "force", "(1.1)"),
    ("paragraph", "Einstein related mass and energy through a famous relation shown next in the text."),
    ("inline", "The energy is ", "energy", " in joules for a body."),
]


# -- persistence and linking ----------------------------------------------------------------


def test_docx_equations_are_persisted_and_linked_to_text_blocks(upload, make_service, session) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    run = make_service().process(document.id)

    assert run.status == "completed"
    assert run.equation_status == "completed"
    assert run.equation_summary["total"] == 2

    stored = equations_of(session, document.id)
    assert [e.latex for e in stored] == ["F=ma", "E=mc^{2}"]  # a separate "(1.1)" is the number, not LaTeX
    assert [e.sequence_number for e in stored] == [0, 1]
    assert stored[0].equation_number == "1.1"
    assert stored[0].equation_type == "numbered"
    assert stored[0].section_title == "Mechanics"

    block_types = {
        b.id: b.block_type for b in session.scalars(select(ExtractedBlock).where(ExtractedBlock.document_id == document.id))
    }
    linked = [e for e in stored if e.extracted_block_id]
    assert linked, "equations should be linked to their text blocks"
    assert all(block_types[e.extracted_block_id] == "equation" for e in linked)


def test_epub_equations_are_persisted(upload, make_service, session) -> None:
    mml = 'xmlns="http://www.w3.org/1998/Math/MathML"'
    body = f"<h1>Energy</h1><p>The famous relation <math {mml}><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></math> follows from relativity, as the text explains at length.</p>"
    document = upload(make_epub_book([("c1.xhtml", "Energy", body)]), "book.epub")

    run = make_service().process(document.id)

    assert run.equation_status == "completed"
    (stored,) = equations_of(session, document.id)
    assert stored.latex == "E=mc^{2}" and stored.extraction_method == "mathml"
    assert stored.validation_status == "valid"
    assert stored.source_location["chapter_title"] == "Energy"


def test_document_without_equations_completes_with_none(upload, make_service, session) -> None:
    document = upload(make_docx_with_equations([("heading", "Plain"), ("paragraph", "Only prose lives in this document, nothing else at all.")]), "plain.docx")
    run = make_service().process(document.id)

    assert run.status == "completed" and run.equation_status == "completed"
    assert run.equation_summary["total"] == 0
    assert equations_of(session, document.id) == []


def test_equations_can_be_listed_filtered_and_counted(upload, make_service, session) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    service = make_service()
    service.process(document.id)

    assert service.count_equations(document.id) == 2
    assert service.count_equations(document.id, equation_type="numbered") == 1
    assert service.count_equations(document.id, validation_status="invalid") == 0
    listed = service.list_equations(document.id, equation_type="inline")
    assert len(listed) == 1 and listed[0].latex == "E=mc^{2}"
    assert service.list_equations(document.id, limit=1)[0].sequence_number == 0


# -- retries and idempotence ---------------------------------------------------------------


def test_retrying_equations_never_duplicates_them(upload, make_service, session) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    service = make_service()
    service.process(document.id)
    first_ids = {e.id for e in equations_of(session, document.id)}

    service.retry_equations(document.id)
    service.retry_equations(document.id)

    again = equations_of(session, document.id)
    assert len(again) == 2
    assert {e.id for e in again}.isdisjoint(first_ids)  # replaced, not appended


def test_retry_does_not_touch_the_text(upload, make_service, session) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    service = make_service()
    service.process(document.id)
    session.rollback()
    before = [(b.id, b.text) for b in session.scalars(select(ExtractedBlock).where(ExtractedBlock.document_id == document.id).order_by(ExtractedBlock.sequence_number))]

    service.retry_equations(document.id)

    session.rollback()
    after = [(b.id, b.text) for b in session.scalars(select(ExtractedBlock).where(ExtractedBlock.document_id == document.id).order_by(ExtractedBlock.sequence_number))]
    assert before == after


def test_identical_equations_in_one_document_are_both_kept(session, upload, make_service) -> None:
    document = upload(make_docx_with_equations([("paragraph", "Intro text that is long enough to count here."), ("display", "force"), ("display", "force")]), "twice.docx")
    make_service().process(document.id)

    stored = equations_of(session, document.id)
    assert [e.latex for e in stored] == ["F=ma", "F=ma"]
    assert len({e.identity_hash for e in stored}) == 2


def test_retry_requires_processed_text(upload, make_service) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    with pytest.raises(ProcessingNotAllowedError):
        make_service().retry_equations(document.id)


# -- failure isolation -----------------------------------------------------------------------


def test_equation_failure_never_fails_the_document(upload, make_service, session, settings) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    failing = ScriptedEquationService(settings, error=RuntimeError("model exploded"))

    run = make_service(failing).process(document.id)

    assert run.status == "completed"  # the text is intact
    assert run.equation_status == "failed"
    session.rollback()
    assert session.get(Document, document.id).processing_status == "completed"
    assert session.scalar(select(func.count()).select_from(ExtractedBlock).where(ExtractedBlock.document_id == document.id)) > 0
    errors = list(session.scalars(select(ProcessingError).where(ProcessingError.document_id == document.id)))
    assert any(e.error_type == "equation_extraction_failed" and "model exploded" in e.error_message for e in errors)
    assert equations_of(session, document.id) == []


def test_failed_retry_keeps_the_previous_equations(upload, make_service, session, settings) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    make_service().process(document.id)
    before = {e.id for e in equations_of(session, document.id)}

    make_service(ScriptedEquationService(settings, error=RuntimeError("boom"))).retry_equations(document.id)

    assert {e.id for e in equations_of(session, document.id)} == before
    assert latest_run(session, document.id).equation_status == "failed"


def test_equations_can_be_disabled(upload, make_service, session, settings) -> None:
    disabled = settings.model_copy(update={"equation_extraction_enabled": False})
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    service = DocumentProcessingService(session, LocalStorage(disabled.storage_root), disabled, equation_service=EquationService(disabled))

    run = service.process(document.id)

    assert run.status == "completed" and run.equation_status == "not_started"
    assert equations_of(session, document.id) == []


# -- crops: stored through the storage layer and never orphaned -------------------------------


def crop_files(storage_root: Path) -> list[Path]:
    return [p for p in stored_files(storage_root) if "equations" in p.parts]


def test_crops_are_stored_and_readable(upload, make_service, session, settings) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    service = make_service(ScriptedEquationService(settings, [equation("a = b")], crops=True))
    service.process(document.id)

    (stored,) = equations_of(session, document.id)
    assert stored.crop_reference and stored.crop_reference.startswith(f"equations/{document.id}/")
    assert service.read_crop(stored.crop_reference) == PNG
    assert len(crop_files(settings.storage_root)) == 1


def test_retry_replaces_old_crops(upload, make_service, session, settings) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    service = make_service(ScriptedEquationService(settings, [equation("a = b")], crops=True))
    service.process(document.id)
    service.retry_equations(document.id)

    assert len(crop_files(settings.storage_root)) == 1  # the first extraction's crop was removed


def test_database_failure_leaves_no_orphaned_crops(upload, make_service, session, settings, monkeypatch) -> None:
    document = upload(make_docx_with_equations(DOCX_ITEMS), "physics.docx")
    service = make_service(ScriptedEquationService(settings, [equation("a = b")], crops=True))

    def fail(*args, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr(service, "_record_equations", fail)
    run = service.process(document.id)

    assert run.status == "completed" and run.equation_status == "failed"
    assert crop_files(settings.storage_root) == []
    assert equations_of(session, document.id) == []


def test_missing_crop_reads_as_none(make_service) -> None:
    assert make_service().read_crop("equations/none/none.png") is None


# -- relationships and constraints -----------------------------------------------------------


def make_row(session: Session, **overrides) -> Equation:
    book = Book(title=f"B{uuid.uuid4().hex[:6]}")
    session.add(book)
    session.flush()
    document = Document(
        book_id=book.id, original_filename="a.pdf", stored_filename="a.pdf", file_path=f"x/{uuid.uuid4()}.pdf",
        file_format="pdf", mime_type="application/pdf", file_size=1, sha256_hash=uuid.uuid4().hex + uuid.uuid4().hex,
        processing_status="completed",
    )
    session.add(document)
    session.flush()
    values = dict(
        document_id=document.id, sequence_number=0, identity_hash=uuid.uuid4().hex, latex="x", equation_type="display",
        extraction_method="math_ocr", validation_status="valid",
    )
    values.update(overrides)
    row = Equation(**values)
    session.add(row)
    session.commit()
    return row


def test_equation_relationships_and_cascade(session: Session) -> None:
    row = make_row(session)
    document = session.get(Document, row.document_id)
    assert [e.id for e in document.equations] == [row.id]
    assert row.document.id == document.id

    session.delete(document)
    session.commit()
    assert session.get(Equation, row.id) is None


def test_deleting_a_block_clears_the_link_but_keeps_the_equation(session: Session) -> None:
    row = make_row(session)
    block = ExtractedBlock(
        document_id=row.document_id, sequence_number=0, block_type="equation", text="E=mc2",
    )
    session.add(block)
    session.commit()
    row.extracted_block_id = block.id
    session.commit()

    session.delete(block)
    session.commit()
    session.refresh(row)
    assert row.extracted_block_id is None and row.latex == "x"


@pytest.mark.parametrize(
    "overrides",
    [
        {"equation_type": "bogus"},
        {"validation_status": "maybe"},
        {"confidence": 1.5},
        {"page_number": 0},
    ],
)
def test_database_rejects_invalid_values(session: Session, overrides) -> None:
    with pytest.raises(IntegrityError):
        make_row(session, **overrides)
    session.rollback()


def test_sequence_numbers_are_unique_per_document(session: Session) -> None:
    row = make_row(session)
    session.add(Equation(
        document_id=row.document_id, sequence_number=row.sequence_number, identity_hash="other", latex="y",
        equation_type="display", extraction_method="math_ocr", validation_status="valid",
    ))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_json_fields_round_trip(session: Session) -> None:
    row = make_row(
        session, source_location={"page": 2, "bbox": [1, 2, 3, 4]}, bounding_box={"l": 1}, validation_warnings=["w"],
        equation_metadata={"k": "v"}, confidence=0.5,
    )
    session.expire_all()
    loaded = session.get(Equation, row.id)
    assert loaded.source_location["page"] == 2 and loaded.validation_warnings == ["w"]
    assert loaded.equation_metadata == {"k": "v"} and loaded.confidence == 0.5


def test_equations_table_is_empty_after_truncate_helper(session: Session, engine) -> None:
    make_row(session)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE equations, extracted_blocks, processing_errors, processing_runs, documents, books CASCADE"))
    assert session.scalar(select(func.count()).select_from(Equation)) == 0
