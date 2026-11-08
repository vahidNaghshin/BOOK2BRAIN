"""Plausibility checks for extracted LaTeX.

Validation never executes LaTeX and never changes it. It answers "is this syntactically
plausible, and how much should it be trusted?":

- invalid:   empty, unparseable, incomplete or unsafe
- uncertain: parses, but something needs review (unsupported construct, unrecognised
             command, low or missing confidence, disagreement with the text layer)
- valid:     parses, renders, and comes from a trustworthy route
"""

import re
from collections import Counter
from dataclasses import dataclass, field

import latex2mathml.converter
from pylatexenc.latexwalker import (
    LatexEnvironmentNode,
    LatexNode,
    LatexWalker,
    LatexWalkerError,
)

from app.parsing.equation.equation_models import ExtractionMethod, ValidationStatus

SUPPORTED_ENVIRONMENTS = frozenset(
    {
        "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix",
        "array", "cases", "aligned", "align", "align*", "gathered", "gather", "gather*",
        "split", "equation", "equation*", "multline", "multline*", "eqnarray", "eqnarray*",
    }
)  # fmt: skip
# Commands that are never mathematics. They are reported, not executed or rendered.
_UNSAFE_COMMANDS = re.compile(
    r"\\(?:input|include|includegraphics|write|immediate|openin|openout|read|catcode|def|edef|"
    r"gdef|xdef|let|newcommand|renewcommand|providecommand|usepackage|documentclass|href|url|"
    r"special|csname|expandafter|loop)(?![A-Za-z])"
)
_DANGLING_END = re.compile(
    r"(?:[=+\-*/<>^_&,]|\\(?:cdot|times|pm|mp|div|leq|geq|neq|approx|equiv|sim|to|"
    r"rightarrow|Rightarrow|frac|dfrac|sqrt|binom|left|right|over))\s*$"
)
_LEFT = re.compile(r"\\left(?![A-Za-z])")
_RIGHT = re.compile(r"\\right(?![A-Za-z])")
_LEAKED_COMMAND = re.compile(r"\\[A-Za-z]+")
_TAGS = re.compile(r"<[^>]+>")
_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")


@dataclass
class ValidationResult:
    status: ValidationStatus
    warnings: list[str] = field(default_factory=list)


def _environments(nodes: list[LatexNode] | None) -> list[str]:
    found: list[str] = []
    for node in nodes or []:
        if isinstance(node, LatexEnvironmentNode):
            found.append(node.environmentname)
            found.extend(_environments(node.nodelist))
        else:
            children = getattr(node, "nodelist", None)
            if children:
                found.extend(_environments(children))
    return found


def text_layer_agrees(latex: str, original_text: str | None) -> bool | None:
    """Do the letters and digits of the PDF text layer all appear in the LaTeX?

    Returns None when there is no text layer to compare with (e.g. a scanned page).
    """
    if not original_text or not original_text.strip():
        return None
    wanted = Counter(_ASCII_ALNUM.findall(original_text))
    available = Counter(_ASCII_ALNUM.findall(latex))
    return all(available[char] >= count for char, count in wanted.items())


def validate_latex(
    latex: str,
    *,
    method: ExtractionMethod,
    confidence: float | None = None,
    min_confidence: float = 0.70,
    original_text: str | None = None,
) -> ValidationResult:
    """Validate one expression and say how far it can be trusted."""
    if not latex or not latex.strip():
        return ValidationResult(ValidationStatus.INVALID, ["Empty expression: nothing was recognised."])

    invalid: list[str] = []
    uncertain: list[str] = []

    if unsafe := _UNSAFE_COMMANDS.search(latex):
        invalid.append(f"Contains a non-mathematical command ({unsafe.group(0)}); it was not rendered.")
        return ValidationResult(ValidationStatus.INVALID, invalid)

    nodes = None
    try:
        nodes, _, _ = LatexWalker(latex, tolerant_parsing=False).get_latex_nodes()
    except LatexWalkerError as exc:
        invalid.append(f"LaTeX syntax error: {str(exc).splitlines()[0]}")

    if len(_LEFT.findall(latex)) != len(_RIGHT.findall(latex)):
        invalid.append(r"Unbalanced \left / \right delimiters.")
    if _DANGLING_END.search(latex):
        invalid.append("The expression ends with an operator or an unfinished construct (incomplete).")

    if not invalid:
        for name in sorted(set(_environments(nodes))):
            if name not in SUPPORTED_ENVIRONMENTS:
                uncertain.append(f"Unsupported environment '{name}'; it may not render.")
        try:
            mathml = latex2mathml.converter.convert(latex)
        except Exception as exc:  # the converter raises many unrelated error types
            uncertain.append(f"Could not be converted for rendering ({type(exc).__name__}).")
        else:
            leaked = sorted(set(_LEAKED_COMMAND.findall(_TAGS.sub(" ", mathml))))
            if leaked:
                uncertain.append("Unrecognised command(s): " + ", ".join(leaked) + ".")

    if invalid:
        return ValidationResult(ValidationStatus.INVALID, invalid + uncertain)

    if not method.authoritative:
        # Model output is not verified mathematics: it needs evidence or a confidence score.
        if confidence is not None:
            if confidence < min_confidence:
                uncertain.append(
                    f"Recognition confidence {confidence:.2f} is below the threshold {min_confidence:.2f}."
                )
        else:
            agrees = text_layer_agrees(latex, original_text)
            if agrees is None:
                uncertain.append(
                    "Recognised from an image without a confidence score; needs review."
                )
            elif not agrees:
                uncertain.append("Disagrees with the text layer of the PDF; needs review.")

    status = ValidationStatus.UNCERTAIN if uncertain else ValidationStatus.VALID
    return ValidationResult(status, uncertain)
