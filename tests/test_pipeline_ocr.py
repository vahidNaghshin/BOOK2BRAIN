"""OCR inside the pipeline: scanned pages are read, or the limitation is reported, never hidden."""

import io
import shutil
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.parsing.ocr_service import TesseractOcrEngine
from app.parsing.parser_factory import ParserFactory
from app.schemas import BookMetadata
from app.services.pipeline_service import ProcessingPipeline
from database.models import Passage, ProcessingError, ProcessingRun
from tests.factories import make_mixed_pdf, make_scanned_pdf
from tests.fakes import FakeOcrEngine, ocr_result, page_error

LONG = "This is a sentence that is long enough to count as meaningful extracted content."


@pytest.fixture
def settings(test_database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        database_url=test_database_url, storage_root=tmp_path / "storage", max_upload_size_mb=100,
        equation_extraction_enabled=False, _env_file=None,
    )


def register(document_service, data: bytes, name: str):
    return document_service.upload(
        file=io.BytesIO(data), filename=name, metadata=BookMetadata(title=f"Scan {name}")
    ).document


def pipeline(session, storage, settings, engine) -> ProcessingPipeline:
    return ProcessingPipeline(session, storage, settings, ParserFactory(settings, ocr_engine=engine))


def test_unavailable_ocr_fails_the_ocr_stage_and_a_retry_after_fixing_it_succeeds(
    docling_ready, session: Session, document_service, storage, settings
) -> None:
    document = register(document_service, make_mixed_pdf([("text", [LONG]), ("scan", ["x"])]), "needs-ocr.pdf")
    missing = FakeOcrEngine(unavailable="Tesseract OCR is not installed. Run `brew install tesseract`.")

    failed = pipeline(session, storage, settings, missing).process_document(document.id)

    assert failed.status == "failed" and failed.stage == "ocr"
    assert "brew install tesseract" in failed.error_message  # the limitation is reported, not hidden
    session.rollback()
    error = session.scalar(select(ProcessingError).where(ProcessingError.document_id == document.id))
    assert (error.stage, error.error_type) == ("ocr", "ocr_unavailable")
    assert session.scalars(select(Passage)).all() == []  # a scanned page is never silently skipped

    working = FakeOcrEngine([ocr_result("Recovered text of the scanned second page of this book.")])
    done = pipeline(session, storage, settings, working).process_document(document.id)

    assert done.status == "completed" and done.id != failed.id  # parsing failed, so a new run starts
    session.rollback()
    results = session.get(ProcessingRun, done.id).stage_results
    assert results["ocr"]["status"] == "completed" and results["ocr"]["pages_processed"] == 1
    passages = list(session.scalars(select(Passage)))
    assert any("Recovered text" in p.text for p in passages)
    assert {p.page_status for p in passages} == {"exact"}
    assert max(p.page_end for p in passages) == 2  # the OCR text keeps its page


def test_a_page_whose_ocr_failed_completes_with_a_warning_not_as_a_clean_success(
    docling_ready, session: Session, document_service, storage, settings
) -> None:
    document = register(document_service, make_mixed_pdf([("text", [LONG]), ("scan", ["a"]), ("scan", ["b"])]), "partial.pdf")
    engine = FakeOcrEngine([ocr_result("Page two was read by OCR without any trouble at all."), page_error("page 3 is unreadable")])

    run = pipeline(session, storage, settings, engine).process_document(document.id)

    assert run.status == "completed"
    assert run.summary.status.value == "partial" and run.summary.ocr.pages_failed == [3]
    session.rollback()
    results = session.get(ProcessingRun, run.id).stage_results
    assert results["ocr"]["status"] == "partial" and results["ocr"]["pages_failed"] == [3]
    assert "incomplete_extraction" in [i["code"] for i in results["validation"]["issues"]]


@pytest.mark.ocr
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract is not installed")
def test_real_tesseract_reads_a_scanned_book_end_to_end(
    docling_ready, session: Session, document_service, storage, settings
) -> None:
    pages = [
        ["Chapter One: The Scan", "", "Tesseract reads this page of the scanned book and the pipeline stores it."],
        ["Second page of the scanned book keeps going with more words for the passage."],
    ]
    document = register(document_service, make_scanned_pdf(pages), "scan.pdf")

    run = pipeline(session, storage, settings, TesseractOcrEngine()).process_document(document.id)

    assert run.status == "completed"
    session.rollback()
    results = session.get(ProcessingRun, run.id).stage_results
    assert results["ocr"]["pages_processed"] == 2
    text = " ".join(p.text for p in session.scalars(select(Passage)))
    assert "Tesseract reads this page" in text and "Second page" in text
