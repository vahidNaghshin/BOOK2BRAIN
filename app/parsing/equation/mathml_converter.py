"""Presentation MathML to LaTeX, for EPUB equations.

If the book's producer stored the TeX source inside the MathML (an <annotation> element),
that author-supplied TeX is returned as-is and the converter is not used. Otherwise
the MathML structure is converted. Unsupported elements are reported, never guessed.
"""

import re
from dataclasses import dataclass, field

from bs4 import NavigableString, Tag

from app.parsing.equation.math_symbols import (
    ACCENTS,
    BIG_OPERATORS,
    FUNCTION_NAMES,
    SYMBOLS,
    convert_text,
)

_TEX_ENCODINGS = {"application/x-tex", "application/x-latex", "tex", "latex", "application/tex"}
_ENDS_WITH_CONTROL_WORD = re.compile(r"\\[A-Za-z]+$")
_FENCES = {"(": "pmatrix", "[": "bmatrix", "{": "Bmatrix", "|": "vmatrix", "‖": "Vmatrix"}
_LATEX_FENCE = {"{": r"\{", "}": r"\}", "‖": r"\|", "⌈": r"\lceil", "⌉": r"\rceil",
                "⌊": r"\lfloor", "⌋": r"\rfloor", "⟨": r"\langle", "⟩": r"\rangle"}  # fmt: skip
_TEXT_ESCAPES = str.maketrans({"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "$": r"\$"})


@dataclass
class MathmlConversion:
    latex: str
    tex_annotation: str | None = None
    alttext: str | None = None
    display: bool = False
    warnings: list[str] = field(default_factory=list)


def _join(parts: list[str]) -> str:
    """Concatenate LaTeX fragments, keeping a space where a control word meets a letter."""
    result = ""
    for part in parts:
        if not part:
            continue
        if result and _ENDS_WITH_CONTROL_WORD.search(result) and part[0].isalnum():
            result += " "
        result += part
    return result


def _group(latex: str) -> str:
    """Wrap in braces unless it is already a single token (so scripts attach to all of it)."""
    if len(latex) == 1 or re.fullmatch(r"\\[A-Za-z]+", latex) or (
        latex.startswith("{") and _balanced_group(latex)
    ):
        return latex
    return "{" + latex + "}"


def _balanced_group(latex: str) -> bool:
    depth = 0
    for index, char in enumerate(latex):
        depth += (char == "{") - (char == "}")
        if depth == 0 and index < len(latex) - 1:
            return False
    return depth == 0


class _Converter:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    @staticmethod
    def _kids(tag: Tag) -> list[Tag]:
        return [child for child in tag.children if isinstance(child, Tag)]

    def row(self, tag: Tag) -> str:
        return _join([self.node(child) for child in self._kids(tag)])

    def arg(self, tag: Tag | None) -> str:
        return self.node(tag) if tag is not None else ""

    def node(self, tag: Tag | NavigableString) -> str:
        if not isinstance(tag, Tag):
            return ""
        handler = getattr(self, f"_{tag.name.replace('-', '_')}", None)
        if handler is None:
            self.warnings.append(f"Unsupported MathML element <{tag.name}>; its content was kept.")
            return self.row(tag)
        return handler(tag)

    # -- token elements --------------------------------------------------------
    def _mi(self, tag: Tag) -> str:
        text = tag.get_text().strip()
        if text in FUNCTION_NAMES:
            return "\\" + text
        if len(text) > 1 and not all(c in SYMBOLS for c in text):
            return r"\mathrm{" + text + "}"
        return convert_text(text)

    def _mn(self, tag: Tag) -> str:
        return tag.get_text().strip()

    def _mo(self, tag: Tag) -> str:
        text = tag.get_text().strip()
        if text in BIG_OPERATORS:
            return BIG_OPERATORS[text]
        return _LATEX_FENCE.get(text) or convert_text(text)

    def _mtext(self, tag: Tag) -> str:
        text = tag.get_text()
        return r"\text{" + text.translate(_TEXT_ESCAPES) + "}" if text.strip() else r"\ "

    def _mspace(self, tag: Tag) -> str:
        return r"\quad" if re.match(r"[1-9]", str(tag.get("width", ""))) else r"\,"

    # -- structure ---------------------------------------------------------------
    def _mphantom(self, tag: Tag) -> str:
        return ""

    def _semantics(self, tag: Tag) -> str:
        kids = [k for k in self._kids(tag) if k.name not in ("annotation", "annotation-xml")]
        return self.node(kids[0]) if kids else ""

    def _annotation(self, tag: Tag) -> str:
        return ""

    _annotation_xml = _annotation

    def _mfrac(self, tag: Tag) -> str:
        parts = self._kids(tag)
        if len(parts) != 2:
            self.warnings.append("<mfrac> without exactly two children.")
            return self.row(tag)
        numerator, denominator = self.node(parts[0]), self.node(parts[1])
        if str(tag.get("linethickness", "")).strip() in ("0", "0px", "0em"):
            return rf"\binom{{{numerator}}}{{{denominator}}}"
        return rf"\frac{{{numerator}}}{{{denominator}}}"

    def _msqrt(self, tag: Tag) -> str:
        return r"\sqrt{" + self.row(tag) + "}"

    def _mroot(self, tag: Tag) -> str:
        parts = self._kids(tag)
        if len(parts) != 2:
            self.warnings.append("<mroot> without exactly two children.")
            return self.row(tag)
        return rf"\sqrt[{self.node(parts[1])}]{{{self.node(parts[0])}}}"

    def _scripts(self, tag: Tag, sub: bool, sup: bool) -> str:
        parts = self._kids(tag)
        needed = 1 + sub + sup
        if len(parts) != needed:
            self.warnings.append(f"<{tag.name}> without exactly {needed} children.")
            return self.row(tag)
        base = _group(self.node(parts[0]))
        result = base
        index = 1
        if sub:
            result += "_{" + self.node(parts[index]) + "}"
            index += 1
        if sup:
            result += "^{" + self.node(parts[index]) + "}"
        return result

    def _msub(self, tag: Tag) -> str:
        return self._scripts(tag, True, False)

    def _msup(self, tag: Tag) -> str:
        return self._scripts(tag, False, True)

    def _msubsup(self, tag: Tag) -> str:
        return self._scripts(tag, True, True)

    def _limits(self, tag: Tag, under: bool, over: bool) -> str:
        parts = self._kids(tag)
        needed = 1 + under + over
        if len(parts) != needed:
            self.warnings.append(f"<{tag.name}> without exactly {needed} children.")
            return self.row(tag)
        base_tag, rest = parts[0], parts[1:]
        base_text = base_tag.get_text().strip()
        base = self.node(base_tag)
        if base_text in BIG_OPERATORS or base_text in FUNCTION_NAMES:
            result = base
            if under:
                result += "_{" + self.node(rest[0]) + "}"
            if over:
                result += "^{" + self.node(rest[-1]) + "}"
            return result
        if over and not under and rest[0].name == "mo" and rest[0].get_text().strip() in ACCENTS:
            return ACCENTS[rest[0].get_text().strip()] + "{" + base + "}"
        result = base
        if under:
            result = rf"\underset{{{self.node(rest[0])}}}{{{result}}}"
        if over:
            result = rf"\overset{{{self.node(rest[-1])}}}{{{result}}}"
        return result

    def _munder(self, tag: Tag) -> str:
        return self._limits(tag, True, False)

    def _mover(self, tag: Tag) -> str:
        return self._limits(tag, False, True)

    def _munderover(self, tag: Tag) -> str:
        return self._limits(tag, True, True)

    def _mtable(self, tag: Tag, environment: str = "matrix") -> str:
        rows = []
        for row in self._kids(tag):
            cells = [self.row(cell) for cell in self._kids(row)] if row.name in ("mtr", "mlabeledtr") else [self.node(row)]
            rows.append(" & ".join(cells))
        return rf"\begin{{{environment}}} " + r" \\ ".join(rows) + rf" \end{{{environment}}}"

    def _mfenced(self, tag: Tag) -> str:
        opening, closing = tag.get("open", "("), tag.get("close", ")")
        separators = str(tag.get("separators", ","))
        parts = [self.node(child) for child in self._kids(tag)]
        inner = ""
        for index, part in enumerate(parts):
            if index:
                inner += (separators[min(index - 1, len(separators) - 1)] if separators else "") + " "
            inner += part
        fence = lambda c: _LATEX_FENCE.get(c, c) if c else "."  # noqa: E731
        return rf"\left{fence(opening)} {inner} \right{fence(closing)}"

    # An mrow like ( [table] ) is a matrix with delimiters, not three unrelated pieces.
    def _mrow(self, tag: Tag) -> str:
        kids = self._kids(tag)
        if (
            len(kids) == 3
            and kids[0].name == "mo"
            and kids[1].name == "mtable"
            and kids[2].name == "mo"
            and kids[0].get_text().strip() in _FENCES
        ):
            return self._mtable(kids[1], _FENCES[kids[0].get_text().strip()])
        return self.row(tag)

    _mstyle = _mpadded = _math = _mrow


def convert_mathml(math: Tag) -> MathmlConversion:
    """Convert one <math> element. Prefers TeX the producer embedded in the MathML."""
    display = str(math.get("display", "")).lower() == "block"
    alttext = math.get("alttext") or None
    for annotation in math.find_all("annotation"):
        if str(annotation.get("encoding", "")).lower() in _TEX_ENCODINGS:
            tex = annotation.get_text().strip()
            if tex:
                return MathmlConversion(
                    latex=tex, tex_annotation=tex, alttext=alttext, display=display
                )
    converter = _Converter()
    latex = converter.node(math)
    return MathmlConversion(
        latex=latex.strip(), alttext=alttext, display=display, warnings=converter.warnings
    )
