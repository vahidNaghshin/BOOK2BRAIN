"""PDF equation extraction: enrichment routing, math OCR fallback, picture candidates, crops."""

import uuid
from pathlib import Path

import pytest

from app.config import Settings
from app.parsing.equation.equation_models import ExtractionMethod, ValidationStatus
from app.parsing.equation.equation_ocr import MathOcrError
from app.parsing.equation.pdf_equations import PdfEquationExtractor
from app.parsing.models import FormulaRegion
from tests.factories import MATH_CORPUS, make_math_pdf, make_text_pdf
from tests.fakes import FakeFormulaConverter, formula_item, math_service

BOX = (100.0, 700.0, 300.0, 660.0)  # l, t, r, b (BOTTOMLEFT origin)


def region(page: int = 1, box: tuple[float, float, float, float] = BOX, *, candidate: bool = False, text: str | None = None):
    left, top, right, bottom = box
    return FormulaRegion(
        page_number=page,
        bbox={"l": left, "t": top, "r": right, "b": bottom, "coord_origin": "BOTTOMLEFT"},
        docling_ref=f"#/texts/{page}",
        original_text=text,
        candidate=candidate,
    )


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "book.pdf"
    path.write_bytes(make_text_pdf([["Page one text."], ["Page two text."], ["Page three text."]]))
    return path


def extractor(settings: Settings, *, converter: FakeFormulaConverter | None = None, math=None) -> PdfEquationExtractor:
    return PdfEquationExtractor(settings, math, converter.factory if converter else None)


def run(ext: PdfEquationExtractor, path: Path, regions):
    return ext.extract(path, uuid.uuid4(), regions)


def test_enrichment_runs_only_on_pages_with_formulas(parsing_settings: Settings, pdf_path: Path) -> None:
    converter = FakeFormulaConverter([formula_item(1, BOX, r"E = m c^{2}", "E = mc 2"), formula_item(3, BOX, r"a + b", "a+b")])
    out = run(extractor(parsing_settings, converter=converter), pdf_path, [region(1), region(3)])

    assert converter.ranges == [(1, 1), (3, 3)]
    assert out.enrichment.pages == [1, 3]
    assert [e.latex for e in out.equations] == [r"E = m c^{2}", r"a + b"]
    assert all(e.extraction_method is ExtractionMethod.DOCLING_ENRICHMENT for e in out.equations)
    assert out.enrichment.formulas_enriched == 2


def test_adjacent_pages_are_converted_in_one_range(parsing_settings: Settings, pdf_path: Path) -> None:
    converter = FakeFormulaConverter([formula_item(1, BOX, "a", "b"), formula_item(2, BOX, "c", "d")])
    run(extractor(parsing_settings, converter=converter), pdf_path, [region(1), region(2)])
    assert converter.ranges == [(1, 2)]


def test_no_formulas_means_no_work(parsing_settings: Settings, pdf_path: Path) -> None:
    converter = FakeFormulaConverter()
    out = run(extractor(parsing_settings, converter=converter), pdf_path, [])
    assert out.equations == [] and converter.ranges == []


def test_location_and_crop_are_recorded(parsing_settings: Settings, pdf_path: Path) -> None:
    converter = FakeFormulaConverter([formula_item(2, BOX, "x^{2}", "x2")])
    out = run(extractor(parsing_settings, converter=converter), pdf_path, [region(2)])

    equation = out.equations[0]
    assert equation.page_number == 2
    assert equation.bounding_box["l"] == BOX[0]
    assert equation.source_location["docling_ref"] == "#/texts/2"
    assert out.crops[equation.id].startswith(b"\x89PNG")


def test_crops_can_be_disabled(tmp_path: Path, pdf_path: Path) -> None:
    settings = Settings(database_url="postgresql+psycopg://u:p@h/d", storage_root=tmp_path, equation_store_crops=False, _env_file=None)
    converter = FakeFormulaConverter([formula_item(1, BOX, "x", "y")])
    assert run(extractor(settings, converter=converter), pdf_path, [region(1)]).crops == {}


def test_unchanged_text_does_not_count_as_enriched(parsing_settings: Settings, pdf_path: Path) -> None:
    """Docling silently keeps the original text when its formula model fails for an item."""
    converter = FakeFormulaConverter([formula_item(1, BOX, "E = mc 2", "E = mc 2")])
    math = math_service(r"E = m c^{2}")
    out = run(extractor(parsing_settings, converter=converter, math=math), pdf_path, [region(1)])

    assert out.enrichment.formulas_enriched == 0
    assert out.equations[0].extraction_method is ExtractionMethod.MATH_OCR
    assert out.equations[0].latex == r"E = m c^{2}"


def test_enrichment_failure_falls_back_to_math_ocr_and_is_reported(parsing_settings: Settings, pdf_path: Path) -> None:
    converter = FakeFormulaConverter(error=RuntimeError("model missing"))
    out = run(extractor(parsing_settings, converter=converter, math=math_service(r"y = x")), pdf_path, [region(1)])

    assert out.enrichment.executed is False
    assert "model missing" in (out.enrichment.error or "")
    assert any(issue.code == "formula_enrichment_failed" for issue in out.issues)
    assert out.equations[0].latex == "y = x"


def test_unrecoverable_region_is_recorded_as_failed_not_invented(tmp_path: Path, pdf_path: Path) -> None:
    settings = Settings(
        database_url="postgresql+psycopg://u:p@h/d", storage_root=tmp_path, formula_enrichment_enabled=False, math_ocr_enabled=False, _env_file=None
    )
    out = run(extractor(settings), pdf_path, [region(1, text="E = mc 2")])

    equation = out.equations[0]
    assert equation.latex == ""
    assert equation.validation_status is ValidationStatus.INVALID
    assert equation.plain_text == "E = mc 2"  # what the PDF says is preserved
    assert out.engine_unavailable is True


def test_unavailable_engine_is_reported_with_its_reason(parsing_settings: Settings, pdf_path: Path) -> None:
    settings = parsing_settings.model_copy(update={"formula_enrichment_enabled": False})
    out = run(extractor(settings, math=math_service(unavailable="model download blocked")), pdf_path, [region(1)])

    assert out.engine_unavailable is True
    assert "model download blocked" in out.math_ocr.unavailable_reason
    assert out.equations[0].validation_status is ValidationStatus.INVALID


def test_one_failed_recognition_does_not_stop_the_others(parsing_settings: Settings, pdf_path: Path) -> None:
    settings = parsing_settings.model_copy(update={"formula_enrichment_enabled": False})
    math = math_service(MathOcrError("bad crop"), "a = b")
    out = run(extractor(settings, math=math), pdf_path, [region(1), region(2)])

    assert out.equations[0].validation_status is ValidationStatus.INVALID
    assert "bad crop" in out.equations[0].validation_warnings[0]
    assert out.equations[1].latex == "a = b"
    assert (out.math_ocr.attempted, out.math_ocr.succeeded, out.math_ocr.failed) == (2, 1, 1)


def test_math_ocr_confidence_is_passed_through_not_invented(parsing_settings: Settings, pdf_path: Path) -> None:
    from app.parsing.equation.equation_ocr import MathOcrResult

    settings = parsing_settings.model_copy(update={"formula_enrichment_enabled": False})
    math = math_service(MathOcrResult("a = b", confidence=0.91), "c = d")
    out = run(extractor(settings, math=math), pdf_path, [region(1), region(2)])
    assert [e.confidence for e in out.equations] == [0.91, None]


# -- picture candidates -------------------------------------------------------------------


def test_picture_candidate_that_reads_as_maths_becomes_an_equation(parsing_settings: Settings, pdf_path: Path) -> None:
    settings = parsing_settings.model_copy(update={"formula_enrichment_enabled": False})
    out = run(extractor(settings, math=math_service(r"\frac{d y}{d x} = 0")), pdf_path, [region(1, candidate=True)])

    equation = out.equations[0]
    assert equation.metadata["from_picture_region"] is True
    assert any("picture" in warning for warning in equation.validation_warnings)
    assert (out.math_ocr.pictures_accepted, out.math_ocr.pictures_rejected) == (1, 0)


def test_picture_candidate_that_is_not_maths_is_discarded(parsing_settings: Settings, pdf_path: Path) -> None:
    settings = parsing_settings.model_copy(update={"formula_enrichment_enabled": False})
    out = run(extractor(settings, math=math_service("Figure caption")), pdf_path, [region(1, candidate=True)])

    assert out.equations == []
    assert out.math_ocr.pictures_rejected == 1
    assert out.engine_unavailable is False


def test_picture_candidates_are_skipped_without_math_ocr(parsing_settings: Settings, pdf_path: Path) -> None:
    converter = FakeFormulaConverter()
    out = run(extractor(parsing_settings, converter=converter), pdf_path, [region(1, candidate=True)])
    assert out.equations == [] and converter.ranges == []


def test_picture_candidates_can_be_switched_off(parsing_settings: Settings, pdf_path: Path) -> None:
    settings = parsing_settings.model_copy(update={"equation_scan_pictures": False, "formula_enrichment_enabled": False})
    math = math_service("a = b")
    assert run(extractor(settings, math=math), pdf_path, [region(1, candidate=True)]).equations == []
    assert math.engine.calls == 0


def test_picture_candidate_scanning_is_bounded_and_says_so(parsing_settings: Settings, pdf_path: Path) -> None:
    settings = parsing_settings.model_copy(update={"equation_scan_pictures_max": 1, "formula_enrichment_enabled": False})
    math = math_service("a = b", "c = d")
    out = run(extractor(settings, math=math), pdf_path, [region(1, candidate=True), region(2, candidate=True)])

    assert len(out.equations) == 1
    assert any(issue.code == "picture_candidates_limited" for issue in out.issues)


def test_picture_candidate_does_not_trigger_enrichment_on_its_page(parsing_settings: Settings, pdf_path: Path) -> None:
    converter = FakeFormulaConverter()
    run(extractor(parsing_settings, converter=converter, math=math_service("a = b")), pdf_path, [region(1, candidate=True)])
    assert converter.ranges == []


# -- real Docling on generated mathematical PDFs -------------------------------------------

REAL_EQUATIONS = [latex for _, latex in MATH_CORPUS if latex in (r"E = mc^{2}", r"F = ma", r"x^{2} + 2x + 1 = 0")]


@pytest.fixture(scope="module")
def formula_model_ready(docling_ready) -> None:
    """Skip the real-model tests when Docling's formula model can't be loaded (e.g. offline)."""
    from app.parsing.equation.equation_ocr import DoclingFormulaEngine, MathOcrUnavailableError

    try:
        DoclingFormulaEngine().check_available()
    except MathOcrUnavailableError as exc:
        pytest.skip(f"Docling's formula model is not available: {exc}")


@pytest.mark.parametrize("scanned", [False, True], ids=["digital", "scanned"])
def test_real_docling_finds_and_recognises_equations(formula_model_ready, tmp_path: Path, scanned: bool) -> None:
    """End to end with the real layout and formula models (skipped when they are unavailable)."""
    from app.parsing.equation.equation_service import EquationService
    from app.parsing.parser_factory import ParserFactory

    settings = Settings(database_url="postgresql+psycopg://u:p@h/d", storage_root=tmp_path, _env_file=None)
    path = tmp_path / "math.pdf"
    path.write_bytes(make_math_pdf(REAL_EQUATIONS, scanned=scanned))
    document_id = uuid.uuid4()

    parsing = ParserFactory(settings).get_parser("pdf").parse(path, document_id=document_id, original_filename="math.pdf")
    outcome = EquationService(settings).extract(path=path, file_format="pdf", document_id=document_id, parsing=parsing)

    found = {"".join(e.latex.split()) for e in outcome.result.equations}
    assert "E=mc^{2}" in found, found
    assert len(outcome.result.equations) >= 2
    assert all(e.page_number == 1 and e.bounding_box for e in outcome.result.equations)
    assert set(outcome.crops) == {e.id for e in outcome.result.equations}
    if scanned:  # no text layer to confirm anything, so nothing may claim to be fully valid
        assert all(e.validation_status is not ValidationStatus.VALID for e in outcome.result.equations)
