"""LaTeX normalisation, validation, delimiter detection and MathML conversion (no database)."""

import pytest
from bs4 import BeautifulSoup
from latex2mathml.converter import convert as latex_to_mathml

from app.parsing.equation.equation_detector import find_delimited_math, image_candidate
from app.parsing.equation.equation_models import (
    EquationExtractionStatus,
    EquationType,
    ExtractionMethod,
    ValidationStatus,
)
from app.parsing.equation.equation_validator import text_layer_agrees, validate_latex
from app.parsing.equation.latex_normalizer import normalize_latex
from app.parsing.equation.mathml_converter import convert_mathml
from database.models.equation import EQUATION_TYPES, VALIDATION_STATUSES
from database.models.processing import EquationStatus


def mathml(markup: str):
    return BeautifulSoup(f'<math xmlns="http://www.w3.org/1998/Math/MathML">{markup}</math>', "html.parser").math


# -- normalisation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        r"\frac { 1 } { 2 }",
        r"x ^ { 2 } + y _ { i }",
        r"$$ a + b = c $$",
        r"\[ \sum_{ i = 1 }^{ n } i \]",
        r"\( \alpha + \beta \)",
        r"\text{ if } x > 0",
        r"\sqrt { x ^ { 2 } + 1 }",
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        r"\left( \frac { a } { b } \right)",
    ],
)
def test_normalisation_preserves_meaning_and_is_idempotent(raw: str) -> None:
    once = normalize_latex(raw)
    assert normalize_latex(once.latex).latex == once.latex
    # Same rendered mathematics before and after tidying (compared as MathML).
    assert latex_to_mathml(once.latex).replace(" ", "") == latex_to_mathml(
        normalize_latex(raw).latex
    ).replace(" ", "")
    stripped = raw.strip().strip("$").removeprefix(r"\[").removesuffix(r"\]").removeprefix(r"\(").removesuffix(r"\)")
    assert latex_to_mathml(once.latex).replace(" ", "") == latex_to_mathml(stripped).replace(" ", "")


def test_delimiters_are_removed_and_reported() -> None:
    result = normalize_latex("$$ E = mc^{2} $$")
    assert result.latex == "E = mc^{2}"
    assert result.delimiter == "$$"


def test_spaced_docling_output_is_tightened() -> None:
    assert normalize_latex(r"\frac { 1 } { 2 }").latex == r"\frac{1}{2}"


def test_text_blocks_keep_their_spaces() -> None:
    assert normalize_latex(r"x \text{ if } y").latex == r"x \text{ if } y"


def test_equation_number_comes_from_tag_and_tag_is_kept() -> None:
    result = normalize_latex(r"a = b \tag{3.2}")
    assert result.equation_number == "3.2"
    assert r"\tag{3.2}" in result.latex


def test_unicode_superscript_artifact_becomes_a_real_superscript() -> None:
    assert r"\texttwosuperior" not in normalize_latex(r"x\text{\texttwosuperior}").latex


def test_empty_input_stays_empty() -> None:
    assert normalize_latex("   ").latex == ""


# -- validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latex",
    [
        r"E = mc^{2}",
        r"\frac{a}{b} + \sqrt{c}",
        r"\int_{0}^{1} x\,dx",
        r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
        r"\left( x \right)",
    ],
)
def test_valid_latex_passes(latex: str) -> None:
    assert validate_latex(latex, method=ExtractionMethod.MATHML).status is ValidationStatus.VALID


@pytest.mark.parametrize(
    ("latex", "why"),
    [
        ("", "empty"),
        (r"\frac{1}{", "unfinished"),
        (r"\left( x", "unbalanced"),
        (r"x +", "dangling operator"),
        (r"\input{/etc/passwd}", "unsafe command"),
        (r"\write18{ls}", "unsafe command"),
    ],
)
def test_invalid_latex_is_rejected(latex: str, why: str) -> None:
    result = validate_latex(latex, method=ExtractionMethod.MATHML)
    assert result.status is ValidationStatus.INVALID, why
    assert result.warnings


def test_model_output_without_evidence_is_uncertain() -> None:
    result = validate_latex(r"x^{2}", method=ExtractionMethod.MATH_OCR, confidence=None)
    assert result.status is ValidationStatus.UNCERTAIN


def test_model_output_with_high_confidence_is_valid() -> None:
    result = validate_latex(r"x^{2}", method=ExtractionMethod.MATH_OCR, confidence=0.95, min_confidence=0.7)
    assert result.status is ValidationStatus.VALID


def test_model_output_with_low_confidence_is_uncertain() -> None:
    result = validate_latex(r"x^{2}", method=ExtractionMethod.MATH_OCR, confidence=0.3, min_confidence=0.7)
    assert result.status is ValidationStatus.UNCERTAIN


def test_model_output_is_valid_when_the_text_layer_agrees() -> None:
    result = validate_latex(r"E = m c^{2}", method=ExtractionMethod.DOCLING_ENRICHMENT, original_text="E = mc 2")
    assert result.status is ValidationStatus.VALID


def test_model_output_is_uncertain_when_the_text_layer_disagrees() -> None:
    result = validate_latex(r"E = m", method=ExtractionMethod.DOCLING_ENRICHMENT, original_text="E = mc 2 + zq")
    assert result.status is ValidationStatus.UNCERTAIN


def test_text_layer_comparison_needs_a_text_layer() -> None:
    assert text_layer_agrees("x", None) is None
    assert text_layer_agrees("x", "  ") is None
    assert text_layer_agrees("x^{2}", "x 2") is True
    assert text_layer_agrees("x", "x 2") is False


# -- delimiter detection ------------------------------------------------------------------


def test_delimited_math_is_found_with_its_kind() -> None:
    found = find_delimited_math(r"Let $a^2$ and $$b^2 = c$$ and \(d\) and \[e\].")
    assert [(m.latex.strip(), m.display) for m in found] == [
        ("a^2", False),
        ("b^2 = c", True),
        ("d", False),
        ("e", True),
    ]


def test_currency_is_not_mistaken_for_math() -> None:
    assert find_delimited_math("It costs $5 and then $10 more, about $15 total.") == []


def test_image_candidates() -> None:
    soup = BeautifulSoup(
        '<img alt="$x^2$" src="a.png"/><img src="images/eq_12.png" alt=""/><img src="photo.jpg" alt="A dog"/>',
        "html.parser",
    )
    images = soup.find_all("img")
    first, second, third = (image_candidate(img) for img in images)
    assert first is not None and first.alt_latex == "x^2"
    assert second is not None and second.alt_latex is None
    assert third is None


# -- MathML conversion --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ("<mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup>", "E=mc^{2}"),
        ("<mfrac><mn>1</mn><mn>2</mn></mfrac>", r"\frac{1}{2}"),
        ("<msqrt><mi>x</mi></msqrt>", r"\sqrt{x}"),
        ("<mroot><mi>x</mi><mn>3</mn></mroot>", r"\sqrt[3]{x}"),
        ("<msub><mi>x</mi><mi>i</mi></msub>", "x_{i}"),
        ("<mi>α</mi><mo>+</mo><mi>β</mi>", r"\alpha+\beta"),
    ],
)
def test_mathml_to_latex(markup: str, expected: str) -> None:
    assert convert_mathml(mathml(markup)).latex == expected


def test_mathml_summation_with_limits() -> None:
    markup = "<munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mi>i</mi>"
    latex = convert_mathml(mathml(markup)).latex
    assert latex.startswith(r"\sum_{i=1}^{n}")
    assert latex.endswith("i")


def test_mathml_table_becomes_a_matrix() -> None:
    markup = "<mtable><mtr><mtd><mi>a</mi></mtd><mtd><mi>b</mi></mtd></mtr><mtr><mtd><mi>c</mi></mtd><mtd><mi>d</mi></mtd></mtr></mtable>"
    latex = convert_mathml(mathml(markup)).latex
    assert r"\begin{matrix}" in latex and r"a&b" in latex.replace(" ", "")


def test_embedded_tex_annotation_is_preferred() -> None:
    markup = '<semantics><mi>x</mi><annotation encoding="application/x-tex">x_{\\text{exact}}</annotation></semantics>'
    result = convert_mathml(mathml(markup))
    assert result.tex_annotation == r"x_{\text{exact}}"


def test_unsupported_elements_are_reported_not_hidden() -> None:
    result = convert_mathml(mathml("<mi>a</mi><mglyph/>"))
    assert result.warnings


# -- the enums mirror the database constants ---------------------------------------------


def test_python_enums_match_database_constraints() -> None:
    assert {t.value for t in EquationType} == set(EQUATION_TYPES)
    assert {s.value for s in ValidationStatus} == set(VALIDATION_STATUSES)
    assert {s.value for s in EquationExtractionStatus} == {s.value for s in EquationStatus}
