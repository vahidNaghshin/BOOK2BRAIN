"""Recognise heading numbering: "Chapter 1", "PART II — Title", "1.2.3 Title", "Appendix A", "Preface"."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.structure.models import (
    RANK_CHAPTER,
    RANK_PART,
    RANK_SECTION,
    NodeType,
)

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}  # fmt: skip
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_SEPARATORS = " \t:.-–—·|)"

_WORD_NUMBER = (
    r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[-\s]+(?:one|two|three|four"
    r"|five|six|seven|eight|nine))?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven"
    r"|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)"
)
_LABELS = {
    "chapter": NodeType.CHAPTER,
    "unit": NodeType.CHAPTER,
    "lesson": NodeType.CHAPTER,
    "module": NodeType.CHAPTER,
    "part": NodeType.PART,
    "book": NodeType.PART,
    "appendix": NodeType.APPENDIX,
    "section": NodeType.SECTION,
}
_LABELLED = re.compile(
    r"^(?P<label>chapter|unit|lesson|module|part|book|appendix|section)\s+"
    r"(?P<num>\d+(?:\.\d+)*|" + _WORD_NUMBER + r"|[IVXLCDM]{1,7}|[ivxlcdm]{1,7}|[A-Z])"
    r"(?=$|[\s:.\-–—·|)])\s*[:.\-–—·|)]*\s*(?P<title>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_DECIMAL = re.compile(r"^(?P<num>\d{1,3}(?:\.\d{1,3})+)\.?\s+(?P<title>\S.*)$", re.DOTALL)
_NUMERIC = re.compile(r"^(?P<num>\d{1,3})(?P<dot>[.)]?)\s+(?P<title>\S.*)$", re.DOTALL)
_ROMAN_DOT = re.compile(r"^(?P<num>[IVX]{1,5})[.)]\s+(?P<title>\S.*)$", re.DOTALL)


def roman_to_int(text: str) -> int | None:
    """Value of a well-formed roman numeral (all upper or all lower case), else None."""
    if not text or not (text.isupper() or text.islower()):
        return None
    upper = text.upper()
    if any(ch not in _ROMAN_VALUES for ch in upper):
        return None
    total = 0
    for index, ch in enumerate(upper):
        value = _ROMAN_VALUES[ch]
        nxt = _ROMAN_VALUES[upper[index + 1]] if index + 1 < len(upper) else 0
        total += -value if value < nxt else value
    return total if 0 < total < 4000 and int_to_roman(total) == upper else None


def int_to_roman(value: int) -> str:
    result = ""
    for symbol, amount in (
        ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400), ("C", 100), ("XC", 90),
        ("L", 50), ("XL", 40), ("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1),
    ):  # fmt: skip
        while value >= amount:
            result += symbol
            value -= amount
    return result


def word_to_int(text: str) -> int | None:
    parts = re.split(r"[-\s]+", text.strip().lower())
    if len(parts) == 1:
        return _ONES.get(parts[0], _TENS.get(parts[0]))
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES and 0 < _ONES[parts[1]] < 10:
        return _TENS[parts[0]] + _ONES[parts[1]]
    return None


@dataclass(frozen=True)
class HeadingPattern:
    """The numbering found at the start of a heading, with the remaining title text."""

    form: str  # label, decimal, numeric, roman_dot
    title: str
    node_type: NodeType | None = None  # for labelled headings: part, chapter, appendix, section
    label: str | None = None  # as written, e.g. "CHAPTER"
    number: str | None = None  # canonical: "1", "II", "A", "1.2.3"
    number_style: str | None = None  # arabic, roman, word, letter, decimal
    number_value: int | None = None  # first (or only) numeric component, for ordering checks
    depth: int = 1  # number of components in a decimal number

    @property
    def explicit_rank(self) -> int:
        """The hierarchy rank this numbering implies (deeper numbering means a deeper heading)."""
        if self.node_type is NodeType.PART:
            return RANK_PART
        if self.node_type in (NodeType.CHAPTER, NodeType.APPENDIX):
            return RANK_CHAPTER
        if self.depth > 1:
            return RANK_CHAPTER + 10 * (self.depth - 1)
        return RANK_SECTION if self.node_type is NodeType.SECTION else RANK_CHAPTER

    @property
    def title_missing(self) -> bool:
        return not self.title.strip()


def _canonical(raw: str, *, label_kind: NodeType | None) -> tuple[str, str, int | None, int]:
    """(canonical number, style, numeric value, depth) for the number token of a labelled heading."""
    if re.fullmatch(r"\d+(?:\.\d+)*", raw):
        first = int(raw.split(".")[0])
        depth = raw.count(".") + 1
        return raw, ("decimal" if depth > 1 else "arabic"), first, depth
    value = word_to_int(raw)
    if value is not None:
        return str(value), "word", value, 1
    roman = roman_to_int(raw)
    if roman is not None and (label_kind is NodeType.APPENDIX or roman <= 60) and not (
        len(raw) == 1 and label_kind is NodeType.APPENDIX
    ):
        return raw.upper(), "roman", roman, 1
    return raw.upper(), "letter", ord(raw.upper()[0]) - 64 if len(raw) == 1 else None, 1


def parse_heading(text: str) -> HeadingPattern | None:
    """Split a heading into numbering and title, or None when it has no recognised numbering."""
    text = text.strip()
    if not text:
        return None

    labelled = _LABELLED.match(text)
    if labelled:
        label = labelled.group("label")
        node_type = _LABELS[label.lower()]
        number, style, value, depth = _canonical(labelled.group("num"), label_kind=node_type)
        # "Section" only counts with decimal or arabic numbers; "Chapter A" is not a thing.
        if style == "letter" and node_type is not NodeType.APPENDIX:
            return None
        return HeadingPattern(
            form="label",
            title=labelled.group("title").strip(),
            node_type=node_type,
            label=label,
            number=number,
            number_style=style,
            number_value=value,
            depth=depth,
        )

    decimal = _DECIMAL.match(text)
    if decimal:
        title = decimal.group("title").strip()
        # Non-ASCII starts (Greek letters in a maths title, "ϵ-Greedy") are not lowercase prose.
        if not (title[0].isupper() or title[0] in "(\"'“‘[" or not title[0].isascii()):
            return None
        number = decimal.group("num")
        return HeadingPattern(
            form="decimal",
            title=title,
            number=number,
            number_style="decimal",
            number_value=int(number.split(".")[0]),
            depth=number.count(".") + 1,
        )

    roman = _ROMAN_DOT.match(text)
    if roman:
        value = roman_to_int(roman.group("num"))
        title = roman.group("title").strip()
        if value is not None and title[0].isupper():
            return HeadingPattern(
                form="roman_dot",
                title=title,
                number=roman.group("num"),
                number_style="roman",
                number_value=value,
            )

    numeric = _NUMERIC.match(text)
    if numeric:
        title = numeric.group("title").strip()
        if title[0].isupper():
            return HeadingPattern(
                form="numeric",
                title=title,
                number=str(int(numeric.group("num"))),
                number_style="arabic",
                number_value=int(numeric.group("num")),
            )
    return None


@dataclass(frozen=True)
class KeywordRule:
    node_type: NodeType | None  # None: chapter-like but with no special type (e.g. "Conclusion")
    position: str  # front, back or any


_KEYWORDS: tuple[tuple[re.Pattern[str], KeywordRule], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), rule)
    for pattern, rule in (
        (r"(?:author'?s?\s+|translator'?s?\s+)?preface(?:\s+to\s+the\s+[\w\s]+?edition)?", KeywordRule(NodeType.PREFACE, "front")),
        (r"foreword", KeywordRule(NodeType.FOREWORD, "front")),
        (r"acknowledge?ments?|acknowledgments?", KeywordRule(NodeType.ACKNOWLEDGEMENT, "any")),
        (r"introduction", KeywordRule(NodeType.INTRODUCTION, "any")),
        (r"prologue|prolog", KeywordRule(None, "front")),
        (r"epilogue|epilog|afterword|conclusions?|concluding\s+remarks", KeywordRule(None, "any")),
        (r"(?:table\s+of\s+)?contents|brief\s+contents|detailed\s+contents", KeywordRule(NodeType.TABLE_OF_CONTENTS, "front")),
        (r"dedication", KeywordRule(NodeType.DEDICATION, "front")),
        (r"about\s+the\s+authors?|author\s+biograph(?:y|ies)|biograph(?:y|ies)\s+of\s+the\s+authors?", KeywordRule(NodeType.ABOUT_AUTHOR, "any")),
        (r"copyright(?:\s+page|\s+notice)?", KeywordRule(NodeType.COPYRIGHT, "front")),
        (r"(?:end|foot)?notes", KeywordRule(NodeType.NOTES, "back")),
        (r"glossary(?:\s+of\s+terms)?", KeywordRule(NodeType.GLOSSARY, "back")),
        (r"bibliography|references|works\s+cited|reference\s+list|further\s+reading|sources", KeywordRule(NodeType.BIBLIOGRAPHY, "back")),
        (r"(?:subject\s+|name\s+|author\s+)?index", KeywordRule(NodeType.INDEX, "back")),
        (r"appendix|appendices", KeywordRule(NodeType.APPENDIX, "back")),
    )
)  # fmt: skip


def match_keyword(title: str) -> KeywordRule | None:
    """Recognise headings such as "Preface" or "References" (the whole heading must match)."""
    cleaned = title.strip().strip(_SEPARATORS + " ")
    if not cleaned or len(cleaned) > 60:
        return None
    for pattern, rule in _KEYWORDS:
        if pattern.fullmatch(cleaned):
            return rule
    return None


_CAPTION = re.compile(
    r"^(?:figure|fig\.?|table|tab\.?|listing|algorithm|exhibit|plate|chart|box|example)\s+\d",
    re.IGNORECASE,
)


def looks_like_caption(text: str) -> bool:
    return bool(_CAPTION.match(text.strip()))


def normalise_for_match(text: str) -> str:
    """Case-, whitespace- and punctuation-insensitive form used to compare heading texts."""
    return re.sub(r"[^\w]+", " ", text.lower()).strip()
