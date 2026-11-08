import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.parsing.models import BlockType, ContentBlock, ExtractionStatus
from app.parsing.ocr_service import OcrService
from app.parsing.pdf_parser import PdfParser
from tests.factories import (
    make_blank_pdf,
    make_corrupt_pdf,
    make_encrypted_pdf,
    make_mixed_pdf,
    make_scanned_pdf,
    make_text_pdf,
    make_zero_page_pdf,
)
from tests.fakes import FakeOcrEngine, ocr_result, page_error

CHAPTER = ["Chapter One: The Beginning", "It was a bright cold day in April, and the clocks were striking thirteen."]
LONG_LINE = "This page has plenty of perfectly readable extractable text content."


@pytest.fixture
def write(tmp_path: Path):
    def _write(data: bytes, name: str = "book.pdf") -> Path:
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return _write


def make_parser(settings: Settings, engine: FakeOcrEngine | None = None, **converter) -> PdfParser:
    service = (
        OcrService(engine, language=settings.ocr_language, timeout_seconds=30)
        if engine is not None and settings.ocr_enabled
        else None
    )
    return PdfParser(settings, service, **converter)


def parse(parser: PdfParser, path: Path, **kwargs):
    return parser.parse(path, document_id=uuid.uuid4(), original_filename=path.name, **kwargs)


# -- text-based PDFs (real Docling) -------------------------------------------------


@pytest.mark.usefixtures("docling_ready")
def test_text_pdf_extraction_preserves_structure_and_page_numbers(write, parsing_settings) -> None:
    engine = FakeOcrEngine()
    pdf = make_text_pdf([CHAPTER, ["Second page paragraph with plenty of text to extract properly."]])

    result = parse(make_parser(parsing_settings, engine), write(pdf))

    assert result.status is ExtractionStatus.SUCCESS
    assert result.page_count == 2 and result.file_format == "pdf" and result.extraction_method == "docling"
    heading, para1, para2 = result.blocks
    assert (heading.block_type, heading.text, heading.page_number) == (BlockType.HEADING, CHAPTER[0], 1)
    assert (para1.block_type, para1.page_number) == (BlockType.PARAGRAPH, 1)
    assert para2.page_number == 2
    assert para1.section_title == CHAPTER[0] and para2.section_title == CHAPTER[0]
    assert result.ocr is None and engine.calls == 0
    assert result.errors == []


@pytest.mark.usefixtures("docling_ready")
def test_source_location_has_page_and_bounding_box(write, parsing_settings) -> None:
    result = parse(make_parser(parsing_settings), write(make_text_pdf([CHAPTER])))

    location = result.blocks[1].source_location
    assert location["page"] == 1 and len(location["bbox"]) == 4
    assert location["docling_ref"].startswith("#/texts/")


@pytest.mark.usefixtures("docling_ready")
def test_page_numbers_match_the_original_page_for_every_page(write, parsing_settings) -> None:
    pages = [[f"Marker text for original page number {n}, with enough words to count."] for n in range(1, 6)]

    result = parse(make_parser(parsing_settings), write(make_text_pdf(pages, headings=False)))

    assert result.page_count == 5
    for block in result.blocks:
        marker = int(block.text.split("page number ")[1].split(",")[0])
        assert block.page_number == marker
    assert [b.page_number for b in result.blocks] == sorted(b.page_number for b in result.blocks)


@pytest.mark.usefixtures("docling_ready")
def test_progress_events_for_a_digital_pdf(write, parsing_settings) -> None:
    events: list[tuple[str, int, int]] = []

    parse(
        make_parser(parsing_settings),
        write(make_text_pdf([CHAPTER])),
        on_progress=lambda s, c, t: events.append((s.value, c, t)),
    )

    assert events == [("parsing", 0, 1), ("parsing", 1, 1)]


# -- scanned and mixed PDFs ---------------------------------------------------------


@pytest.mark.usefixtures("docling_ready")
def test_scanned_pdf_is_processed_with_ocr(write, parsing_settings) -> None:
    engine = FakeOcrEngine([ocr_result("Chapter Two: The Scan", "A paragraph read by OCR.", confidence=93.5)])

    result = parse(make_parser(parsing_settings, engine), write(make_scanned_pdf([["scan content"]])))

    assert result.status is ExtractionStatus.SUCCESS
    assert [b.text for b in result.blocks] == ["Chapter Two: The Scan", "A paragraph read by OCR."]
    assert all(b.page_number == 1 for b in result.blocks)
    assert result.blocks[0].metadata == {"ocr": True, "ocr_confidence": 93.5, "ocr_reason": "no_text_layer"}
    assert result.blocks[0].source_location["method"] == "ocr"
    assert result.ocr.pages_processed == [1] and result.ocr.pages_failed == []
    assert result.ocr.mean_confidence == 93.5
    assert (result.ocr.engine, result.ocr.language, result.ocr.dpi) == ("fake", "eng", 300)


@pytest.mark.usefixtures("docling_ready")
def test_mixed_pdf_only_ocrs_the_scanned_pages_and_keeps_page_numbers(write, parsing_settings) -> None:
    pdf = make_mixed_pdf(
        [("text", [LONG_LINE]), ("scan", ["scanned"]), ("text", [LONG_LINE + " again"]), ("blank", []), ("scan", ["scanned"])]
    )
    engine = FakeOcrEngine([ocr_result("OCR text of page two."), ocr_result("OCR text of page five.")])

    result = parse(make_parser(parsing_settings, engine), write(pdf))

    assert result.page_count == 5
    assert engine.calls == 2  # pages 2 and 5 only: not the digital pages, not the blank one
    assert [(b.page_number, b.text) for b in result.blocks] == [
        (1, LONG_LINE),
        (2, "OCR text of page two."),
        (3, LONG_LINE + " again"),
        (5, "OCR text of page five."),
    ]
    assert result.ocr.pages_processed == [2, 5]
    assert result.metadata["pages_without_content"] == [4]
    assert [b.metadata for b in result.blocks if b.page_number in (1, 3)] == [None, None]


@pytest.mark.usefixtures("docling_ready")
def test_ocr_blocks_inherit_the_running_section_title(write, parsing_settings) -> None:
    pdf = make_mixed_pdf([("titled", ["Chapter Heading Here", LONG_LINE]), ("scan", ["scanned"])])
    engine = FakeOcrEngine([ocr_result("Scanned continuation text.")])

    result = parse(make_parser(parsing_settings, engine), write(pdf))

    assert result.blocks[0].block_type is BlockType.HEADING
    ocr_block = result.blocks[-1]
    assert ocr_block.page_number == 2 and ocr_block.text == "Scanned continuation text."
    assert ocr_block.section_title == "Chapter Heading Here"


@pytest.mark.usefixtures("docling_ready")
def test_ocr_dpi_is_configurable(write, tmp_path: Path) -> None:
    settings = Settings(database_url="postgresql+psycopg://u:p@h/d", storage_root=tmp_path, ocr_dpi=150, _env_file=None)
    engine = FakeOcrEngine([ocr_result("text")])

    result = parse(make_parser(settings, engine), write(make_scanned_pdf([["x"]])))

    width, height = engine.images[0].size  # US-Letter/A4 at 150 DPI is about 1240 x 1754
    assert 1100 < width < 1400 and 1500 < height < 1900
    assert result.ocr.dpi == 150


@pytest.mark.usefixtures("docling_ready")
def test_ocr_language_is_passed_from_settings(write, tmp_path: Path) -> None:
    settings = Settings(database_url="postgresql+psycopg://u:p@h/d", storage_root=tmp_path, ocr_language="deu", _env_file=None)

    result = parse(make_parser(settings, FakeOcrEngine([ocr_result("Hallo Welt")])), write(make_scanned_pdf([["x"]])))

    assert result.ocr.language == "deu"


@pytest.mark.usefixtures("docling_ready")
def test_progress_events_include_ocr_stage(write, parsing_settings) -> None:
    events: list[tuple[str, int, int]] = []
    engine = FakeOcrEngine([ocr_result("one"), ocr_result("two")])

    parse(
        make_parser(parsing_settings, engine),
        write(make_scanned_pdf([["a"], ["b"]])),
        on_progress=lambda s, c, t: events.append((s.value, c, t)),
    )

    assert events[:2] == [("parsing", 0, 1), ("parsing", 1, 1)]
    assert events[2:] == [("ocr", 0, 2), ("ocr", 1, 2), ("ocr", 2, 2)]


# -- OCR quality rules --------------------------------------------------------------


@pytest.mark.usefixtures("docling_ready")
def test_ocr_never_replaces_better_extracted_text(write, parsing_settings) -> None:
    pdf = make_mixed_pdf([("both", ["Short digital text here"])])  # 19 letters: below the 25 threshold
    engine = FakeOcrEngine([ocr_result("tiny")])

    result = parse(make_parser(parsing_settings, engine), write(pdf))

    assert [b.text for b in result.blocks] == ["Short digital text here"]
    assert result.ocr.pages_kept_digital == [1] and result.ocr.pages_processed == [1]
    assert [w.code for w in result.warnings] == ["ocr_not_better"]
    assert result.blocks[0].metadata is None  # still the digital block


@pytest.mark.usefixtures("docling_ready")
def test_better_ocr_text_replaces_sparse_extracted_text(write, parsing_settings) -> None:
    pdf = make_mixed_pdf([("both", ["Short digital text here"])])
    engine = FakeOcrEngine([ocr_result("A much longer paragraph recovered by the OCR engine.")])

    result = parse(make_parser(parsing_settings, engine), write(pdf))

    assert [b.text for b in result.blocks] == ["A much longer paragraph recovered by the OCR engine."]
    assert result.ocr.pages_kept_digital == []
    assert result.blocks[0].metadata["ocr_reason"] == "insufficient_text"


@pytest.mark.usefixtures("docling_ready")
def test_low_confidence_ocr_does_not_replace_existing_text(write, parsing_settings) -> None:
    pdf = make_mixed_pdf([("both", ["Short digital text here"])])
    engine = FakeOcrEngine([ocr_result("Plenty of garbled words from a poor scan here.", confidence=12.0)])

    result = parse(make_parser(parsing_settings, engine), write(pdf))

    assert [b.text for b in result.blocks] == ["Short digital text here"]
    assert result.ocr.pages_kept_digital == [1]


@pytest.mark.usefixtures("docling_ready")
def test_low_confidence_ocr_is_used_when_nothing_else_exists_but_flagged(write, parsing_settings) -> None:
    engine = FakeOcrEngine([ocr_result("Poor quality but only text available.", confidence=15.0)])

    result = parse(make_parser(parsing_settings, engine), write(make_scanned_pdf([["x"]])))

    assert [b.text for b in result.blocks] == ["Poor quality but only text available."]
    assert [w.code for w in result.warnings] == ["ocr_low_confidence"]
    assert result.warnings[0].page_number == 1


@pytest.mark.usefixtures("docling_ready")
def test_ocr_failure_on_one_page_is_recorded_and_other_pages_continue(write, parsing_settings) -> None:
    pdf = make_mixed_pdf([("scan", ["a"]), ("text", [LONG_LINE]), ("scan", ["c"])])
    engine = FakeOcrEngine([page_error("tesseract crashed"), ocr_result("Page three recovered.")])

    result = parse(make_parser(parsing_settings, engine), write(pdf))

    assert result.status is ExtractionStatus.PARTIAL
    assert result.ocr.pages_failed == [1] and result.ocr.pages_processed == [3]
    assert [(b.page_number, b.text) for b in result.blocks] == [(2, LONG_LINE), (3, "Page three recovered.")]
    failure = next(w for w in result.warnings if w.code == "ocr_failed")
    assert failure.page_number == 1 and "tesseract crashed" in failure.message


@pytest.mark.usefixtures("docling_ready")
def test_ocr_finding_no_text_is_a_warning(write, parsing_settings) -> None:
    pdf = make_mixed_pdf([("text", [LONG_LINE]), ("scan", ["x"])])

    result = parse(make_parser(parsing_settings, FakeOcrEngine([ocr_result()])), write(pdf))

    assert [b.page_number for b in result.blocks] == [1]
    assert [w.code for w in result.warnings] == ["ocr_no_text"]


@pytest.mark.usefixtures("docling_ready")
def test_unavailable_ocr_engine_fails_with_an_actionable_message(write, parsing_settings) -> None:
    engine = FakeOcrEngine(unavailable="Tesseract OCR is not installed. Install it with `brew install tesseract`.")
    events: list[tuple[str, int, int]] = []

    result = parse(
        make_parser(parsing_settings, engine),
        write(make_scanned_pdf([["x"]])),
        on_progress=lambda s, c, t: events.append((s.value, c, t)),
    )

    assert result.status is ExtractionStatus.FAILED
    assert result.errors[0].code == "ocr_unavailable"
    assert "brew install tesseract" in result.errors[0].message
    assert events[-1] == ("ocr", 0, 1)  # the failure is attributed to the OCR stage


@pytest.mark.usefixtures("docling_ready")
def test_digital_pdf_works_even_when_ocr_is_unavailable(write, parsing_settings) -> None:
    engine = FakeOcrEngine(unavailable="no tesseract")

    result = parse(make_parser(parsing_settings, engine), write(make_text_pdf([CHAPTER])))

    assert result.status is ExtractionStatus.SUCCESS and result.ocr is None


@pytest.mark.usefixtures("docling_ready")
def test_ocr_can_be_disabled_by_configuration(write, tmp_path: Path) -> None:
    settings = Settings(database_url="postgresql+psycopg://u:p@h/d", storage_root=tmp_path, ocr_enabled=False, _env_file=None)
    engine = FakeOcrEngine([ocr_result("never used")])
    pdf = make_mixed_pdf([("text", [LONG_LINE]), ("scan", ["x"])])

    result = parse(make_parser(settings, engine), write(pdf))

    assert engine.calls == 0
    assert result.status is ExtractionStatus.PARTIAL
    skipped = result.warnings[0]
    assert skipped.code == "ocr_skipped" and skipped.details == {"pages": [2]}


@pytest.mark.usefixtures("docling_ready")
def test_ocr_thresholds_are_configurable(write, tmp_path: Path) -> None:
    pdf = make_mixed_pdf([("both", ["Short digital text here"])])
    engine = FakeOcrEngine([ocr_result("never used")])
    relaxed = Settings(database_url="postgresql+psycopg://u:p@h/d", storage_root=tmp_path, ocr_min_text_chars=10, _env_file=None)

    result = parse(make_parser(relaxed, engine), write(pdf))

    assert engine.calls == 0  # 19 letters now counts as enough text
    assert [b.text for b in result.blocks] == ["Short digital text here"]


# -- bad PDFs -----------------------------------------------------------------------


def test_corrupted_pdf_fails_gracefully(write, parsing_settings) -> None:
    result = parse(make_parser(parsing_settings), write(make_corrupt_pdf()))

    assert result.status is ExtractionStatus.FAILED
    assert [e.code for e in result.errors] == ["corrupt_document"]


def test_zero_byte_pdf_fails_gracefully(write, parsing_settings) -> None:
    result = parse(make_parser(parsing_settings), write(b""))

    assert [e.code for e in result.errors] == ["corrupt_document"]


@pytest.mark.usefixtures("docling_ready")
def test_truncated_pdf_fails_gracefully_without_raising(write, parsing_settings) -> None:
    data = make_text_pdf([CHAPTER])

    result = parse(make_parser(parsing_settings), write(data[: len(data) // 3]))

    assert result.status is ExtractionStatus.FAILED
    assert result.errors[0].code == "corrupt_document"


def test_encrypted_pdf_is_reported_clearly(write, parsing_settings) -> None:
    result = parse(make_parser(parsing_settings), write(make_encrypted_pdf()))

    assert result.status is ExtractionStatus.FAILED
    assert [e.code for e in result.errors] == ["encrypted_document"]
    assert "password" in result.errors[0].message


def test_pdf_with_no_pages_is_empty(write, parsing_settings) -> None:
    result = parse(make_parser(parsing_settings), write(make_zero_page_pdf()))

    assert [e.code for e in result.errors] == ["empty_document"]
    assert "no pages" in result.errors[0].message


@pytest.mark.usefixtures("docling_ready")
def test_blank_pdf_without_any_text_is_empty_and_never_sent_to_ocr(write, parsing_settings) -> None:
    engine = FakeOcrEngine()

    result = parse(make_parser(parsing_settings, engine), write(make_blank_pdf()))

    assert result.status is ExtractionStatus.FAILED
    assert [e.code for e in result.errors] == ["empty_document"]
    assert engine.calls == 0


# -- Docling failure handling (fake converters) --------------------------------------


def test_docling_conversion_error_becomes_a_failed_result(write, parsing_settings) -> None:
    from docling.exceptions import ConversionError

    class BrokenConverter:
        def convert(self, source):  # type: ignore[no-untyped-def]
            raise ConversionError("parser could not understand the file")

    parser = make_parser(parsing_settings, converter_factory=lambda: BrokenConverter())

    result = parse(parser, write(make_text_pdf([CHAPTER])))

    assert result.status is ExtractionStatus.FAILED
    assert result.errors[0].code == "corrupt_document"
    assert "could not be read" in result.errors[0].message


def test_docling_partial_success_is_reported_as_partial(write, parsing_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    class PartialConverter:
        def convert(self, source):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                document=None,
                errors=[SimpleNamespace(error_message="page 2 could not be processed")],
                status=SimpleNamespace(value="partial_success"),
            )

    monkeypatch.setattr(
        "app.parsing.docling_mapping.blocks_from_docling",
        lambda document, with_pages: [ContentBlock(text=LONG_LINE, page_number=1)],
    )
    parser = make_parser(parsing_settings, converter_factory=lambda: PartialConverter())

    result = parse(parser, write(make_text_pdf([CHAPTER])))

    assert result.status is ExtractionStatus.PARTIAL
    assert {w.code for w in result.warnings} == {"docling_warning", "docling_partial"}


def test_unexpected_errors_are_recorded_not_raised(write, parsing_settings) -> None:
    class ExplodingConverter:
        def convert(self, source):  # type: ignore[no-untyped-def]
            raise MemoryError("out of memory")

    parser = make_parser(parsing_settings, converter_factory=lambda: ExplodingConverter())

    result = parse(parser, write(make_text_pdf([CHAPTER])))

    assert result.status is ExtractionStatus.FAILED
    assert result.errors[0].code == "unexpected_error"
    assert "MemoryError" in result.errors[0].message
    assert "traceback" in result.errors[0].details


# -- optional: real Tesseract end to end --------------------------------------------


@pytest.mark.ocr
@pytest.mark.usefixtures("docling_ready")
def test_real_tesseract_on_a_mixed_pdf(write, parsing_settings) -> None:
    import shutil

    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract is not installed")
    from app.parsing.parser_factory import ParserFactory

    pdf = make_mixed_pdf(
        [("text", [LONG_LINE]), ("scan", ["Chapter Two: The Scan", "", "Tesseract reads this sentence."])]
    )

    result = parse(ParserFactory(parsing_settings).get_parser("pdf"), write(pdf))

    assert result.status is ExtractionStatus.SUCCESS
    page_two = " ".join(b.text for b in result.blocks if b.page_number == 2)
    assert "Chapter Two" in page_two and "Tesseract reads this sentence" in page_two
    assert result.ocr.engine == "tesseract" and result.ocr.pages_processed == [2]
    assert result.ocr.mean_confidence and result.ocr.mean_confidence > 60
