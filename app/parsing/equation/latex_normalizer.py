"""Deterministic LaTeX normalization.

Normalization only touches *formatting*: delimiters, whitespace and known extraction
artifacts. It never simplifies, reorders or "corrects" mathematics, never substitutes
variables and never infers missing symbols. It is idempotent.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Commands whose braced argument is text: whitespace inside it is meaningful.
_TEXT_COMMANDS = ("text", "textrm", "textbf", "textit", "textsf", "texttt", "mbox", "hbox")
_TEXT_ARGUMENT = re.compile(r"\\(?:" + "|".join(_TEXT_COMMANDS) + r")\s*\{")

# Artifacts of PDF/Office conversion that stand for a plain superscript digit.
_SUPERSCRIPT_ARTIFACTS = {
    r"\texttwosuperior": "^{2}",
    r"\textthreesuperior": "^{3}",
    r"\textonesuperior": "^{1}",
}
_SUPERSCRIPT_TEXT = re.compile(
    r"\\text\s*\{\s*(" + "|".join(re.escape(k) for k in _SUPERSCRIPT_ARTIFACTS) + r")\s*\}"
)

_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))
_UNICODE_SPACES = dict.fromkeys(
    map(ord, "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"),
    " ",
)

_DELIMITERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
_TAG = re.compile(r"\\tag\*?\s*\{([^{}]*)\}")
# A number such as (1.1), (2), (3a) at the very end, optionally after \quad / \qquad.
_TRAILING_NUMBER = re.compile(r"(?:\\q?quad\s*)*\(\s*(\d+(?:\s*[.\-]\s*\d+)*\s*[a-z]?)\s*\)\s*$")


@dataclass(frozen=True)
class NormalizedLatex:
    latex: str
    delimiter: str | None = None  # the math delimiters that were removed, if any
    equation_number: str | None = None
    changes: list[str] = field(default_factory=list)


def _strip_delimiters(text: str) -> tuple[str, str | None]:
    for opening, closing in _DELIMITERS:
        if (
            len(text) >= len(opening) + len(closing)
            and text.startswith(opening)
            and text.endswith(closing)
            # "$a$ and $b$" is two expressions, not one wrapped in dollars.
            and not (opening == "$" and "$" in text[1:-1])
            and not (opening == "$$" and "$$" in text[2:-2])
        ):
            return text[len(opening) : len(text) - len(closing)].strip(), opening
    return text, None


def _split_protected(text: str) -> list[tuple[str, bool]]:
    """Split into (segment, is_text_argument) so text-mode arguments are left alone."""
    parts: list[tuple[str, bool]] = []
    position = 0
    while match := _TEXT_ARGUMENT.search(text, position):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            if text[end] == "{" and text[end - 1] != "\\":
                depth += 1
            elif text[end] == "}" and text[end - 1] != "\\":
                depth -= 1
            end += 1
        if depth:  # unbalanced: leave the rest to the validator
            break
        parts.append((text[position : match.start()], False))
        parts.append((text[match.start() : end], True))
        position = end
    parts.append((text[position:], False))
    return parts


def _tidy_math_spacing(segment: str) -> str:
    segment = re.sub(r"\s+", " ", segment)
    # Space after a control word before a brace is ignored by TeX (but not before a letter).
    segment = re.sub(r"(\\[A-Za-z]+) \{", r"\1{", segment)
    segment = re.sub(r"\{ +", "{", segment)
    segment = re.sub(r" +\}", "}", segment)
    segment = re.sub(r"\} +\{", "}{", segment)
    segment = re.sub(r" *([\^_]) *", r"\1", segment)
    # Space between the closing brace of an argument and a following ^ or _ is also ignored.
    return segment


def _tidy(text: str) -> str:
    pieces = []
    for segment, is_text in _split_protected(text):
        pieces.append(re.sub(r"\s+", " ", segment) if is_text else _tidy_math_spacing(segment))
    return re.sub(r" {2,}", " ", "".join(pieces)).strip()


def _clean_number(number: str) -> str:
    return re.sub(r"\s+", "", number)


def normalize_latex(raw: str) -> NormalizedLatex:
    """Normalize one LaTeX expression. The input is never modified in meaning."""
    changes: list[str] = []
    text = unicodedata.normalize("NFC", raw)
    text = text.translate(_INVISIBLE).translate(_UNICODE_SPACES)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    text, delimiter = _strip_delimiters(text)
    if delimiter:
        changes.append(f"removed_delimiters:{delimiter}")

    replaced = _SUPERSCRIPT_TEXT.sub(lambda m: _SUPERSCRIPT_ARTIFACTS[m.group(1)], text)
    if replaced != text:
        changes.append("superscript_artifacts")
        text = replaced

    tidy = _tidy(text)
    if tidy != text:
        changes.append("whitespace")
    text = tidy

    equation_number = None
    if tag := _TAG.search(text):
        equation_number = _clean_number(tag.group(1)) or None
        # Keep the tag (it renders the number) but with its content tidied.
        text = _TAG.sub(lambda m: m.group(0).replace(m.group(1), _clean_number(m.group(1))), text)
    elif trailing := _TRAILING_NUMBER.search(text):
        equation_number = _clean_number(trailing.group(1))

    return NormalizedLatex(
        latex=text, delimiter=delimiter, equation_number=equation_number, changes=changes
    )
