"""EPUB and DOCX equation extraction, and the DOCX text-pass regression for malformed equations."""

import uuid
from pathlib import Path

import pytest

from app.parsing.docx_parser import DocxParser
from app.parsing.equation.equation_extractor import DocxEquationExtractor, EpubEquationExtractor
from app.parsing.equation.equation_models import EquationType, ExtractionMethod, ValidationStatus
from app.parsing.models import BlockType, ExtractionStatus
from tests.factories import make_docx_with_equations, make_epub_book

MML = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def epub_equations(tmp_path: Path, body: str):
    path = tmp_path / "book.epub"
    path.write_bytes(make_epub_book([("c1.xhtml", "Chapter One", body)]))
    return EpubEquationExtractor().extract(path, uuid.uuid4())


def docx_equations(tmp_path: Path, items):
    path = tmp_path / "doc.docx"
    path.write_bytes(make_docx_with_equations(items))
    return path, DocxEquationExtractor().extract(path, uuid.uuid4())


# -- EPUB ---------------------------------------------------------------------------------


def test_epub_mathml_inline_equation(tmp_path: Path) -> None:
    body = f"<h1>Energy</h1><p>Energy is <math {MML}><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></math> always.</p>"
    (equation,) = epub_equations(tmp_path, body).equations

    assert equation.latex == "E=mc^{2}"
    assert equation.equation_type is EquationType.INLINE
    assert equation.extraction_method is ExtractionMethod.MATHML
    assert equation.surrounding_text == "Energy is $E=mc^{2}$ always."
    assert equation.source_location["chapter_title"] == "Chapter One"  # from the table of contents


def test_epub_display_mathml(tmp_path: Path) -> None:
    body = f'<p><math display="block" {MML}><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>+</mo><msqrt><mi>x</mi></msqrt></math></p>'
    (equation,) = epub_equations(tmp_path, body).equations
    assert equation.latex == r"\frac{1}{2}+\sqrt{x}"
    assert equation.equation_type is EquationType.DISPLAY


def test_epub_latex_delimiters_and_equation_number(tmp_path: Path) -> None:
    body = r"<p>Let $a^2+b^2=c^2$ hold.</p><p>$$\int_0^1 x\,dx = \frac12 \tag{2.1}$$</p>"
    inline, display = epub_equations(tmp_path, body).equations

    assert inline.equation_type is EquationType.INLINE and inline.latex == "a^2+b^2=c^2"
    assert display.equation_type is EquationType.DISPLAY
    assert r"\tag{2.1}" in display.latex


def test_epub_prices_are_not_equations(tmp_path: Path) -> None:
    assert epub_equations(tmp_path, "<p>It costs $5 and then $10 more.</p>").equations == []


def test_epub_image_with_latex_alt_text(tmp_path: Path) -> None:
    (equation,) = epub_equations(tmp_path, r'<p><img src="eq.png" alt="$\alpha+\beta$"/></p>').equations
    assert equation.latex == r"\alpha+\beta"
    assert equation.extraction_method is ExtractionMethod.IMAGE_ALT_TEXT


def test_epub_without_equations_yields_none(tmp_path: Path) -> None:
    result = epub_equations(tmp_path, "<h1>Plain</h1><p>No maths here.</p>")
    assert result.equations == [] and result.issues == []


def test_epub_equations_keep_reading_order(tmp_path: Path) -> None:
    body = f"<p>$a$ then <math {MML}><mi>b</mi></math> then $c$.</p>"
    assert [e.latex for e in epub_equations(tmp_path, body).equations] == ["a", "b", "c"]


# -- DOCX ---------------------------------------------------------------------------------


def test_docx_inline_and_display_equations(tmp_path: Path) -> None:
    _, result = docx_equations(
        tmp_path,
        [("heading", "Mechanics"), ("inline", "Energy ", "energy", " holds."), ("display", "kinetic")],
    )
    inline, display = result.equations

    assert inline.latex == "E=mc^{2}" and inline.equation_type is EquationType.INLINE
    assert display.latex == r"KE=\frac{1}{2}mv^{2}" and display.equation_type is EquationType.DISPLAY
    assert inline.section_title == display.section_title == "Mechanics"
    assert inline.extraction_method is ExtractionMethod.OMML


def test_docx_equation_number_is_read(tmp_path: Path) -> None:
    _, result = docx_equations(tmp_path, [("display", "kinetic", "(1.1)")])
    (equation,) = result.equations
    assert equation.equation_number == "1.1"
    assert equation.equation_type is EquationType.NUMBERED


def test_docx_matrix_sum_and_integral(tmp_path: Path) -> None:
    _, result = docx_equations(tmp_path, [("display", "matrix"), ("display", "sum"), ("display", "integral")])
    matrix, summation, integral = (e.latex for e in result.equations)
    assert r"\begin{matrix}" in matrix and r"a&b\\c&d" in matrix
    assert r"\sum_{i=1}^{n}" in summation
    assert r"\int_{0}^{1}" in integral


def test_docx_original_omml_is_preserved(tmp_path: Path) -> None:
    _, result = docx_equations(tmp_path, [("display", "force")])
    assert "<m:oMath" in result.equations[0].metadata["original_omml"]


def test_docx_malformed_equation_is_recorded_not_dropped(tmp_path: Path) -> None:
    _, result = docx_equations(tmp_path, [("display", "force"), ("display", "malformed"), ("display", "energy")])

    assert len(result.equations) == 3
    broken = result.equations[1]
    assert broken.latex == "" and broken.validation_status is ValidationStatus.INVALID
    assert broken.metadata["original_omml"]
    assert any(issue.code == "unsupported_omml" for issue in result.issues)
    assert result.equations[2].latex == "E=mc^{2}"  # later equations are unaffected


# -- DOCX text pass: a malformed equation must not lose the document ------------------------


def test_malformed_equation_no_longer_destroys_the_document_text(tmp_path: Path) -> None:
    path, _ = docx_equations(
        tmp_path,
        [("heading", "Intro"), ("paragraph", "Important text before."), ("display", "malformed"), ("paragraph", "Important text after.")],
    )
    result = DocxParser().parse(path, document_id=uuid.uuid4(), original_filename="doc.docx")

    texts = " ".join(block.text for block in result.blocks)
    assert "Important text before." in texts and "Important text after." in texts
    assert result.status is not ExtractionStatus.FAILED
    assert any(issue.code == "equation_text_only" for issue in result.warnings)


def test_healthy_docx_equations_are_not_flagged_by_the_text_pass(tmp_path: Path) -> None:
    path, _ = docx_equations(tmp_path, [("paragraph", "Text."), ("display", "energy")])
    result = DocxParser().parse(path, document_id=uuid.uuid4(), original_filename="doc.docx")
    assert not any(issue.code == "equation_text_only" for issue in result.warnings)
    assert any(block.block_type is BlockType.EQUATION for block in result.blocks)


def test_the_uploaded_file_is_never_modified_by_the_sanitiser(tmp_path: Path) -> None:
    path, _ = docx_equations(tmp_path, [("display", "malformed")])
    before = path.read_bytes()
    DocxParser().parse(path, document_id=uuid.uuid4(), original_filename="doc.docx")
    assert path.read_bytes() == before
