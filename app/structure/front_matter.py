"""Book title, subtitle, author and copyright detection from the opening blocks of a document.

The first page is not assumed to be the title page: covers, blank pages, half-titles and
copyright pages may come first. Every opening block is scored as a possible title using the
parser's heading label, text size, the file's own metadata, and position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.structure.models import SourceBlock
from app.structure.numbering import match_keyword, normalise_for_match, parse_heading
from app.structure.typography import height_ratio, is_all_caps, is_title_case

_FRONT_PAGES = 15
_FRONT_BLOCKS = 60
_STRONG_COPYRIGHT = re.compile(
    r"©|\(c\)\s*\d{4}|copyright|all rights reserved|\bisbn\b|library of congress|cataloging[- ]in[- ]publication",
    re.IGNORECASE,
)
_WEAK_BOILERPLATE = re.compile(
    r"published by|printed in|first (?:published|edition|printing)|\bpublisher\b|www\.|https?://|"
    r"\S+@\S+\.\S+|british library|no part of this|\bedition\b|\bpress\b|\bprinting\b|\bltd\.?$",
    re.IGNORECASE,
)
_EDITION_LINE = re.compile(r"^(?:\w+\s+edition|edition\s+\w+|\d+(?:st|nd|rd|th)\s+edition)$", re.IGNORECASE)
_BY_LINE = re.compile(r"^(?:written\s+)?by\s+(?P<name>.+)$", re.IGNORECASE)
_NAME_TOKEN = re.compile(r"^(?:[A-Z][\w'’\-]*\.?|[A-Z]\.|van|von|de|der|la|le|di|da|bin|al)$")


@dataclass
class MetadataGuess:
    title: str | None = None
    subtitle: str | None = None
    author: str | None = None
    title_confidence: float | None = None
    subtitle_confidence: float | None = None
    author_confidence: float | None = None
    title_source: str | None = None
    # Block indices the detection used (None when a value came from file metadata only).
    title_block: int | None = None
    subtitle_block: int | None = None
    author_block: int | None = None
    copyright_blocks: list[int] = field(default_factory=list)
    copyright_page: int | None = None
    # Other blocks with the title's text (cover, half-title): not headings, kept as front text.
    title_repeats: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "author": self.author,
            "title_confidence": self.title_confidence,
            "subtitle_confidence": self.subtitle_confidence,
            "author_confidence": self.author_confidence,
            "title_source": self.title_source,
            "copyright_page": self.copyright_page,
        }


def is_boilerplate(text: str) -> bool:
    return bool(_STRONG_COPYRIGHT.search(text) or _WEAK_BOILERPLATE.search(text))


# Words that make a capitalised phrase a title or subtitle rather than a person's name.
_NOT_NAME_WORDS = frozenset(
    "introduction guide handbook primer textbook approach perspective edition practice theory "
    "principles applications fundamentals essentials basics foundations".split()
)


def _before_boilerplate(text: str) -> tuple[str, bool]:
    """Text up to the first copyright/publisher wording, and whether there was any."""
    match = min(
        (m for m in (_STRONG_COPYRIGHT.search(text), _WEAK_BOILERPLATE.search(text)) if m),
        key=lambda m: m.start(),
        default=None,
    )
    return (text[: match.start()].strip(), True) if match else (text, False)


def _looks_like_name(text: str) -> bool:
    if len(text) > 80 or any(ch.isdigit() for ch in text):
        return False
    words = text.split()
    if words[0].lower() in ("a", "an", "the") or any(w.lower().strip(".,") in _NOT_NAME_WORDS for w in words):
        return False
    parts = re.split(r"\s*(?:,|&|\band\b|;)\s*", text)
    for part in parts:
        tokens = part.split()
        if not 2 <= len(tokens) <= 5:
            return False
        if not all(_NAME_TOKEN.match(token) for token in tokens):
            return False
    return True


def _front_limit(blocks: list[SourceBlock], structural_start: int | None) -> int:
    """Index (exclusive) up to which blocks can belong to the title/front pages."""
    limit = len(blocks) if any(b.page for b in blocks) else min(len(blocks), _FRONT_BLOCKS)
    for index, block in enumerate(blocks[:limit]):
        if block.page is not None and block.page > _FRONT_PAGES:
            limit = index
            break
    if structural_start is not None:
        limit = min(limit, structural_start)
    return limit


def detect_metadata(
    blocks: list[SourceBlock],
    *,
    reference_height: float | None,
    source_title: str | None,
    structural_start: int | None,
) -> MetadataGuess:
    """Find title, subtitle, author and copyright information.

    `structural_start` is the index of the first real structural heading (preface, chapter...),
    which ends the front pages; None when there is none.
    """
    guess = MetadataGuess()
    limit = _front_limit(blocks, structural_start)
    region = blocks[:limit]
    _find_copyright(region, guess)
    _find_title(region, guess, reference_height, source_title)
    if guess.title_block is not None:
        _find_subtitle_and_author(region, guess)
    return guess


def _find_copyright(region: list[SourceBlock], guess: MetadataGuess) -> None:
    strong = [i for i, b in enumerate(region) if _STRONG_COPYRIGHT.search(b.text) and len(b.text) < 1500]
    if not strong:
        return
    first = strong[0]
    page = region[first].page
    low, high = first, first
    if page is not None:
        members = [i for i, b in enumerate(region) if b.page == page]
        boiler = [i for i in members if is_boilerplate(region[i].text)]
        # A page that is mostly something else (e.g. a title page with a © line) is not a
        # copyright page: then only the boilerplate run itself is used.
        if len(boiler) * 2 >= len(members):
            low, high = members[0], members[-1]
        else:
            low, high = boiler[0], boiler[-1]
    else:
        while low - 1 >= 0 and is_boilerplate(region[low - 1].text):
            low -= 1
        while high + 1 < len(region) and is_boilerplate(region[high + 1].text):
            high += 1
    guess.copyright_blocks = list(range(low, high + 1))
    guess.copyright_page = page


def _find_title(
    region: list[SourceBlock],
    guess: MetadataGuess,
    reference_height: float | None,
    source_title: str | None,
) -> None:
    copyright_blocks = set(guess.copyright_blocks)
    source_key = normalise_for_match(source_title) if source_title else ""
    scored: list[tuple[float, int]] = []
    for index, block in enumerate(region):
        text = " ".join(block.text.split())
        if (
            index in copyright_blocks
            or block.block_type not in ("heading", "paragraph", "other")
            or not text
            or len(text) > 150
            or len(text.split()) > 14
            or text.endswith(".")
            or is_boilerplate(text)
            or _EDITION_LINE.match(text)
            or _BY_LINE.match(text)
            or text.isdigit()
        ):
            continue
        keyword = match_keyword(text)
        if keyword is not None and keyword.node_type is not None:
            continue
        pattern = parse_heading(text)
        if pattern is not None and pattern.form == "label":
            continue
        score = 0.0
        if block.block_type == "heading":
            score += 0.6 if block.heading_level == 0 else 0.15
        ratio = height_ratio(block, reference_height)
        if ratio is not None:
            score += 0.3 if ratio >= 1.5 else 0.2 if ratio >= 1.25 else 0.0
        if source_key and source_key in normalise_for_match(text):
            score += 0.5
        if is_all_caps(text) or is_title_case(text):
            score += 0.1
        if index < 10:
            score += 0.05
        if score >= 0.4:
            scored.append((score, index))

    if scored:
        best_score = max(score for score, _ in scored)
        best_index = min(index for score, index in scored if score == best_score)
        key = normalise_for_match(region[best_index].text)
        first_boiler = min((i for i, b in enumerate(region) if is_boilerplate(b.text)), default=len(region))
        same = [i for i, b in enumerate(region) if normalise_for_match(b.text) == key]
        # Half-title then title page: the later copy (before the copyright page) carries the
        # subtitle and author.
        before = [i for i in same if i < first_boiler]
        chosen = max(before) if before else best_index
        guess.title = " ".join(region[chosen].text.split())
        guess.title_block = chosen
        guess.title_repeats = [i for i in same if i != chosen]
        guess.title_confidence = round(min(1.0, best_score), 2)
        guess.title_source = "document_layout"
    elif source_title:
        guess.title = " ".join(source_title.split())
        guess.title_confidence = 0.5
        guess.title_source = "file_metadata"


def _find_subtitle_and_author(region: list[SourceBlock], guess: MetadataGuess) -> None:
    assert guess.title_block is not None
    title_block = region[guess.title_block]
    copyright_blocks = set(guess.copyright_blocks)
    lines: list[tuple[int, str]] = []
    scanned = 0
    for index in range(guess.title_block + 1, len(region)):
        block = region[index]
        if scanned >= 6:
            break
        if title_block.page is not None and block.page is not None and block.page != title_block.page:
            break
        if not block.text.strip() or block.block_type not in ("heading", "paragraph", "other"):
            scanned += 1
            continue
        stop = False
        for raw in block.text.splitlines():
            text = " ".join(raw.split())
            if not text:
                continue
            # Parsers often merge a title-page line with the copyright notice that follows it.
            text, has_boilerplate = _before_boilerplate(text)
            keyword = match_keyword(text) if text else None
            if keyword is not None and keyword.node_type is not None:
                stop = True
            elif text and len(text.split()) <= 20 and not _EDITION_LINE.match(text):
                lines.append((index, text))
            if has_boilerplate:
                stop = True
            if stop:
                break
        if stop:
            break
        scanned += 1

    by_lines = [(i, m.group("name").strip().rstrip(".")) for i, t in lines if (m := _BY_LINE.match(t))]
    others = [(i, t) for i, t in lines if not _BY_LINE.match(t)]
    if by_lines:
        guess.author_block, guess.author = by_lines[0]
        guess.author_confidence = 0.9
    elif others:
        # Title, subtitle, author is the usual order: a name-like line after another line is the
        # author; a lone name-like line is too. A lone title-like line is a subtitle.
        names = [(i, t) for i, t in others[1:] if _looks_like_name(t)]
        if not names and len(others) == 1 and _looks_like_name(others[0][1]):
            names = [others[0]]
        if names:
            guess.author_block, guess.author = names[0]
            guess.author_confidence = 0.6
    if guess.author is None:
        others = _split_off_known_author(region, guess, others)
    subtitles = [(i, t) for i, t in others if t != guess.author and not t.endswith(".")]
    if subtitles:
        guess.subtitle_block, guess.subtitle = subtitles[0]
        guess.subtitle_confidence = 0.6


def _split_off_known_author(
    region: list[SourceBlock], guess: MetadataGuess, lines: list[tuple[int, str]]
) -> list[tuple[int, str]]:
    """Parsers often run the title page's lines together ("The Textbook Jane Smith IBM Research").
    If a name-like block elsewhere in the front pages (e.g. the cover) appears inside such a line,
    that is the author, and what precedes it is the subtitle."""
    title_key = normalise_for_match(guess.title or "")
    names = {
        " ".join(b.text.split())
        for b in region
        if b.block_type in ("heading", "paragraph", "other") and _looks_like_name(" ".join(b.text.split()))
    }
    names = {n for n in names if normalise_for_match(n) != title_key}
    for position, (index, text) in enumerate(lines):
        for name in sorted(names, key=len, reverse=True):
            at = text.find(name)
            if at >= 0 and len(text.split()) > len(name.split()):
                guess.author, guess.author_block, guess.author_confidence = name, index, 0.7
                before = text[:at].strip()
                remaining = lines[:position] + ([(index, before)] if before else []) + lines[position + 1 :]
                return remaining
    return lines
