"""Deterministic heading detection (stages A-C).

Every extracted block is scored as a possible heading from independent pieces of evidence:
the parser's own heading label, numbering patterns, well-known heading words, text shape, size
relative to body text, position on the page, agreement with the table of contents and consistency
with other headings. Candidates are never invented: each one is an existing block (or, for
"CHAPTER 1" followed by a title line, two adjacent blocks).

Scores at or above `accept` are confident headings. Scores between `reject` and `accept` are
ambiguous: they are kept (as `other_heading` unless classified otherwise) and flagged, and may be
sent to the optional LLM classifier. Below `reject` the block stays body text.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

from app.structure.models import DocumentText, NodeType, SourceBlock
from app.structure.numbering import (
    HeadingPattern,
    KeywordRule,
    looks_like_caption,
    match_keyword,
    normalise_for_match,
    parse_heading,
)
from app.structure.toc import TocResult
from app.structure.typography import (
    body_line_height,
    cluster_ratios,
    height_ratio,
    is_all_caps,
    is_title_case,
)

MAX_HEADING_CHARS = 160
_URL = re.compile(r"https?://|www\.|\S+@\S+\.\S+", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"[.!?]\s+[A-Z]")


@dataclass
class HeadingCandidate:
    index: int  # position in the block list
    sequence: int  # block sequence number
    original_text: str  # as written (joined when the heading spans two blocks)
    title: str  # normalised title, without "Chapter 3:"-style labels
    score: float
    signals: list[str]
    block_indices: list[int]
    page: int | None = None
    pattern: HeadingPattern | None = None
    keyword: KeywordRule | None = None
    raw_level: int | None = None
    ratio: float | None = None
    style: str = "P"
    in_toc: bool = False
    from_heading_block: bool = False
    status: str = "ambiguous"  # accepted, ambiguous or rejected
    llm: dict | None = None
    # Filled in during hierarchy construction.
    rank: int | None = None
    node_type: NodeType | None = None
    context: dict = field(default_factory=dict)

    @property
    def explicit_rank(self) -> int | None:
        return self.pattern.explicit_rank if self.pattern else None


@dataclass
class HeadingDetection:
    candidates: list[HeadingCandidate]  # accepted and ambiguous, in document order
    rejected: list[HeadingCandidate]  # parser-labelled headings that did not hold up as headings
    running_headers: list[str]  # texts discarded as repeated page headers


def _single(text: str) -> str:
    return " ".join(text.split())


_EDGE_PAGE_NUMBER = re.compile(r"^\d{1,4}\s+(?=\S)|(?<=\S)\s+\d{1,4}$")


def _running_key(text: str) -> str:
    """Comparable form of a possible page header: a page number at either end is ignored when
    what remains is a phrase ("12  Foundations of Learning"), but not for "Chapter 3"."""
    stripped = _EDGE_PAGE_NUMBER.sub("", text.strip())
    return normalise_for_match(stripped if len(stripped.split()) >= 3 else text)


def _running_header_keys(blocks: list[SourceBlock]) -> set[str]:
    """Texts repeated on three or more nearby pages: running page headers, not headings."""
    pages_by_text: dict[str, set[int]] = defaultdict(set)
    for block in blocks:
        if block.page is None or len(block.text) > MAX_HEADING_CHARS:
            continue
        key = _running_key(block.text)
        if key:
            pages_by_text[key].add(block.page)
    running: set[str] = set()
    for key, pages in pages_by_text.items():
        ordered = sorted(pages)
        run = 1
        for previous, current in zip(ordered, ordered[1:]):
            run = run + 1 if current - previous <= 2 else 1
            if run >= 3:
                running.add(key)
                break
    return running


def _score(
    block: SourceBlock,
    text: str,
    pattern: HeadingPattern | None,
    keyword: KeywordRule | None,
    *,
    ratio: float | None,
    centered: bool,
    first_on_page: bool,
    followed_by_text: bool,
    isolated: bool,
    in_toc: bool,
) -> tuple[float, list[str]]:
    score, signals = 0.0, []

    def add(value: float, name: str) -> None:
        nonlocal score
        score += value
        signals.append(name if value >= 0 else f"-{name}")

    words = text.split()
    title = pattern.title if pattern else text

    if block.block_type == "heading":
        add(0.45 if block.heading_level == 0 else 0.55, "parser_heading")
    if pattern is not None:
        title_words = len(title.split())
        sentence_like = title.rstrip().endswith((".", ";", ",")) or title_words > 14
        if pattern.form == "label":
            if pattern.title_missing or not sentence_like:
                add(0.65, "label_pattern")
            else:
                add(0.15, "label_pattern_in_sentence")
        elif pattern.form == "decimal":
            add(0.35, "decimal_numbering")
        elif pattern.form == "numeric":
            add(0.15, "numeric_numbering")
        else:
            add(0.10, "roman_numbering")
    if keyword is not None and (pattern is None or pattern.form != "label"):
        add(0.50, "heading_keyword")

    # Shape of the text.
    if len(words) <= 8:
        add(0.10, "short")
    elif len(words) <= 14:
        add(0.05, "fairly_short")
    elif len(words) > 20:
        add(-0.30, "long")
    else:
        add(-0.10, "long")
    if is_all_caps(text):
        add(0.10, "all_caps")
    elif is_title_case(title):
        add(0.10, "title_case")
    last = text.rstrip()[-1:]
    if last in ".;,":
        add(-0.25, "ends_like_sentence")
    elif last == ":":
        add(-0.10, "ends_with_colon")
    math_symbols = sum(1 for ch in text if unicodedata.category(ch) == "Sm")
    if math_symbols >= 2:
        add(-0.40, "mathematical_symbols")
    elif math_symbols == 1 and pattern is None:
        add(-0.25, "mathematical_symbols")
    if "(" in text and ")" in text and ":" in text:
        add(-0.20, "looks_like_code")
    first_letter = next((ch for ch in title if ch.isalpha()), "")
    if first_letter.islower() and pattern is None:
        add(-0.30, "starts_lowercase")
    if _SENTENCE_BREAK.search(text):
        add(-0.25, "several_sentences")

    # Typography and position.
    if ratio is not None:
        if ratio >= 1.5:
            add(0.30, "much_larger_than_body")
        elif ratio >= 1.25:
            add(0.22, "larger_than_body")
        elif ratio >= 1.1:
            add(0.10, "slightly_larger_than_body")
        elif ratio <= 0.95 and block.block_type != "heading":
            add(-0.05, "not_larger_than_body")
    if centered:
        add(0.05, "centered")
    if first_on_page:
        add(0.08, "first_on_page")
    if followed_by_text:
        add(0.05, "followed_by_body_text")
    if isolated:
        add(0.10, "isolated_line")
    if in_toc:
        add(0.25, "matches_table_of_contents")
    return max(0.0, min(1.0, score)), signals


def detect_headings(
    document: DocumentText, toc: TocResult, *, accept: float = 0.70, reject: float = 0.45
) -> HeadingDetection:
    blocks = document.blocks
    reference = body_line_height(blocks)
    running = _running_header_keys(blocks)
    toc_titles = toc.titles

    extents: dict[int, tuple[float, float]] = {}
    first_on_page: dict[int, int] = {}
    for index, block in enumerate(blocks):
        if block.page is None:
            continue
        first_on_page.setdefault(block.page, index)
        if block.bbox:
            low, high = extents.get(block.page, (block.bbox[0], block.bbox[2]))
            extents[block.page] = (min(low, block.bbox[0]), max(high, block.bbox[2]))

    def is_centered(block: SourceBlock) -> bool:
        if block.page is None or block.bbox is None or block.page not in extents:
            return False
        low, high = extents[block.page]
        width = high - low
        if width <= 0:
            return False
        middle = (block.bbox[0] + block.bbox[2]) / 2
        return abs(middle - (low + high) / 2) < 0.03 * width and (block.bbox[2] - block.bbox[0]) < 0.7 * width

    found: list[HeadingCandidate] = []
    running_discarded: list[str] = []
    consumed: set[int] = set()

    for index, block in enumerate(blocks):
        if index in consumed or index in toc.region:
            continue
        if block.block_type not in ("heading", "paragraph", "other", "list"):
            continue
        text = _single(block.text)
        if not text or len(text) > MAX_HEADING_CHARS or text.isdigit() or _URL.search(text):
            continue
        if looks_like_caption(text) and block.block_type != "heading":
            continue
        pattern = parse_heading(text)
        if block.block_type == "list" and (pattern is None or pattern.form not in ("label", "decimal")):
            continue
        keyword = match_keyword(pattern.title if pattern and pattern.form != "label" else text)
        if pattern is not None and pattern.form == "label" and not pattern.title_missing:
            keyword = None
        if _running_key(text) in running:
            running_discarded.append(text)
            continue

        indices = [index]
        original = text
        title = pattern.title if pattern and pattern.title else text
        extra_signals: list[str] = []
        if pattern is not None and pattern.form == "label" and pattern.title_missing:
            following = _title_line_after(blocks, index, toc)
            if following is not None:
                nxt, next_text = following
                indices.append(nxt)
                original = f"{text} {next_text}"
                title = next_text
                extra_signals.append("title_on_next_line")

        ratio = height_ratio(block, reference)
        next_block = blocks[index + 1] if index + 1 < len(blocks) else None
        followed = bool(
            next_block
            and next_block.block_type == "paragraph"
            and len(next_block.text) >= 60
            and index + 1 not in toc.region
        )
        previous = blocks[index - 1] if index > 0 else None
        isolated = followed and (
            previous is None
            or previous.page != block.page
            or previous.block_type not in ("paragraph", "list")
            or (len(previous.text) >= 60 and previous.text.rstrip().endswith((".", "!", "?", ":", '"', "”")))
        )
        in_toc = normalise_for_match(title) in toc_titles and bool(normalise_for_match(title))
        score, signals = _score(
            block,
            text,
            pattern,
            keyword,
            ratio=ratio,
            centered=is_centered(block),
            first_on_page=block.page is not None and first_on_page.get(block.page) == index,
            followed_by_text=followed,
            isolated=isolated,
            in_toc=in_toc,
        )
        if extra_signals:
            score = min(1.0, score + 0.05)
            signals.extend(extra_signals)
        candidate = HeadingCandidate(
            index=index,
            sequence=block.sequence,
            original_text=original,
            title=title,
            score=score,
            signals=signals,
            block_indices=indices,
            page=block.page,
            pattern=pattern,
            keyword=keyword,
            raw_level=block.heading_level if block.block_type == "heading" else None,
            ratio=ratio,
            in_toc=in_toc,
            from_heading_block=block.block_type == "heading",
        )
        if len(indices) > 1:
            consumed.update(indices[1:])
        found.append(candidate)

    _adjust_for_consistency(found, accept=accept, reject=reject)

    kept, rejected = [], []
    for candidate in found:
        if candidate.score >= accept:
            candidate.status = "accepted"
        elif candidate.score >= reject:
            candidate.status = "ambiguous"
        else:
            candidate.status = "rejected"
        if candidate.status == "rejected":
            if candidate.from_heading_block:
                rejected.append(candidate)
        else:
            kept.append(candidate)
    return HeadingDetection(candidates=kept, rejected=rejected, running_headers=sorted(set(running_discarded)))


def _title_line_after(
    blocks: list[SourceBlock], index: int, toc: TocResult
) -> tuple[int, str] | None:
    """The title on the line after a bare "CHAPTER 1" label, if there is one."""
    nxt = index + 1
    if nxt >= len(blocks) or nxt in toc.region:
        return None
    current, candidate = blocks[index], blocks[nxt]
    if candidate.block_type not in ("heading", "paragraph", "other"):
        return None
    if current.page is not None and candidate.page is not None and candidate.page != current.page:
        return None
    text = _single(candidate.text)
    if not text or len(text) > 100 or len(text.split()) > 14 or text.endswith("."):
        return None
    other = parse_heading(text)
    if other is not None and other.form == "label":
        return None
    if looks_like_caption(text):
        return None
    if candidate.block_type != "heading" and not (is_title_case(text) or is_all_caps(text)):
        return None
    return nxt, text


def _adjust_for_consistency(
    found: list[HeadingCandidate], *, accept: float, reject: float
) -> None:
    """Second pass: numbering sequences and typographic style shared with confident headings."""
    # Numbering: "2." next to "1." or "3." is a real sequence; an isolated "7. Something" is not.
    numbers: dict[tuple[str, str], set[int]] = defaultdict(set)
    for candidate in found:
        pattern = candidate.pattern
        if pattern and pattern.number_value is not None and pattern.form in ("numeric", "roman_dot", "decimal"):
            prefix = ".".join(pattern.number.split(".")[:-1]) if pattern.form == "decimal" else ""
            last = int(pattern.number.split(".")[-1]) if pattern.form == "decimal" else pattern.number_value
            numbers[(pattern.form, prefix)].add(last)
    for candidate in found:
        pattern = candidate.pattern
        if not (pattern and pattern.number_value is not None and pattern.form in ("numeric", "roman_dot", "decimal")):
            continue
        prefix = ".".join(pattern.number.split(".")[:-1]) if pattern.form == "decimal" else ""
        last = int(pattern.number.split(".")[-1]) if pattern.form == "decimal" else pattern.number_value
        siblings = numbers[(pattern.form, prefix)]
        if last - 1 in siblings or last + 1 in siblings:
            candidate.score = min(1.0, candidate.score + 0.20)
            candidate.signals.append("numbering_sequence")
        elif not candidate.from_heading_block and pattern.form in ("numeric", "roman_dot"):
            candidate.score = max(0.0, candidate.score - 0.10)
            candidate.signals.append("-isolated_number")

    # Style: size clusters first, so equal-looking headings share a style key.
    ratios = {c.ratio for c in found if c.ratio is not None and c.score >= 0.35}
    clusters = cluster_ratios(ratios) if ratios else {}
    for candidate in found:
        size = clusters.get(candidate.ratio) if candidate.ratio is not None else None
        level = candidate.raw_level
        if level is None and size is None:
            candidate.style = "P"
        else:
            candidate.style = f"L{'-' if level is None else level}|S{'-' if size is None else size}"

    # A style shared by several headings (confident, or at least plausible) is a heading style.
    confident_styles: dict[str, int] = defaultdict(int)
    for candidate in found:
        if candidate.style != "P" and candidate.score >= reject:
            confident_styles[candidate.style] += 1
    for candidate in found:
        if reject <= candidate.score < accept and confident_styles.get(candidate.style, 0) >= 2:
            candidate.score = min(1.0, candidate.score + 0.10)
            candidate.signals.append("consistent_with_other_headings")
