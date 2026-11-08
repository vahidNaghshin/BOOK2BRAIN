"""Detection of mathematical content in text and in EPUB markup.

Heuristics are deliberately conservative: a missed equation can be recovered later, but
treating prices, dates or references as equations corrupts the knowledge base.
"""

import re
from dataclasses import dataclass

from bs4 import Tag

_DISPLAY_PATTERNS = (
    (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), "$$"),
    (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), r"\["),
)
_INLINE_PAREN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
# $...$: no space just inside the dollars, single line, and not a price ("$5 and $10").
_INLINE_DOLLAR = re.compile(r"(?<![\w$\\])\$(?!\s|\$)([^$\n]{1,300}?)(?<!\s)\$(?![\w$])")
_ONLY_NUMBER = re.compile(r"^[\d.,\s]+$")
_IMAGE_HINT = re.compile(r"(?:^|[-_\s/])(?:math|equation|equ?|formula|latex|tex|katex|mathjax)(?:[-_\s.\d]|$)", re.I)


@dataclass(frozen=True)
class DelimitedMath:
    start: int
    end: int
    latex: str
    delimiter: str
    display: bool


def find_delimited_math(text: str) -> list[DelimitedMath]:
    """Find LaTeX written with math delimiters inside running text, in reading order."""
    found: list[DelimitedMath] = []
    taken: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in taken)

    def add(match: re.Match[str], delimiter: str, display: bool) -> None:
        latex = match.group(1).strip()
        if not latex or overlaps(match.start(), match.end()):
            return
        taken.append((match.start(), match.end()))
        found.append(DelimitedMath(match.start(), match.end(), latex, delimiter, display))

    for pattern, delimiter in _DISPLAY_PATTERNS:
        for match in pattern.finditer(text):
            add(match, delimiter, True)
    for match in _INLINE_PAREN.finditer(text):
        add(match, r"\(", False)
    for match in _INLINE_DOLLAR.finditer(text):
        if not _ONLY_NUMBER.match(match.group(1)):
            add(match, "$", False)
    return sorted(found, key=lambda item: item.start)


@dataclass(frozen=True)
class ImageCandidate:
    reason: str
    alt_latex: str | None = None  # the alt text, when it is itself LaTeX


def image_candidate(img: Tag) -> ImageCandidate | None:
    """Is this EPUB <img> probably an equation? Returns why, or None for ordinary images."""
    alt = str(img.get("alt", "")).strip()
    if alt:
        delimited = find_delimited_math(alt)
        if len(delimited) == 1 and delimited[0].start == 0 and delimited[0].end == len(alt):
            return ImageCandidate("alt text is LaTeX", delimited[0].latex)
    classes = " ".join(img.get("class", [])) if isinstance(img.get("class"), list) else str(img.get("class", ""))
    for label, value in (("class", classes), ("file name", str(img.get("src", "")).rsplit("/", 1)[-1])):
        if _IMAGE_HINT.search(value):
            return ImageCandidate(f"{label} suggests a formula")
    if re.search(r"\b(?:equation|formula)\b", alt, re.I):
        return ImageCandidate("alt text mentions an equation")
    return None
