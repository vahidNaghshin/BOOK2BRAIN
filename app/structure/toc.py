"""Table-of-contents detection.

The contents pages are body text for Day 4 purposes (they stay in the book), but their lines look
exactly like headings, so they must not be mistaken for them. The entries (titles and pages) also
corroborate real headings found later in the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.structure.models import SourceBlock
from app.structure.numbering import match_keyword, normalise_for_match, parse_heading

_LEADER = r"(?:\.{2,}|·{2,}|…+|_{2,}|(?:\.\s){3,})"
_ENTRY = re.compile(
    r"^(?P<title>\S.*?)(?:\s*" + _LEADER + r"\s*|\s{2,}|\s+)(?P<page>\d{1,4}|[ivxlc]{1,6})\s*$",
    re.IGNORECASE,
)
_LEADER_LINE = re.compile(r"\S.*?" + _LEADER + r"\s*(?:\d{1,4}|[ivxlc]{1,6})\s*$", re.IGNORECASE)
_MIN_ENTRIES = 3
_MAX_BLANK_RUN = 2


@dataclass
class TocEntry:
    title: str
    page: int | None
    depth: int | None
    block_index: int
    # The title without its numbering ("1.2  Goals" -> "Goals"), which is what headings are matched on.
    heading: str = ""

    @property
    def key(self) -> str:
        return normalise_for_match(self.heading or self.title)


@dataclass
class TocResult:
    # Indices of the blocks that are table-of-contents content (excluding the "Contents" heading).
    region: set[int] = field(default_factory=set)
    entries: list[TocEntry] = field(default_factory=list)
    heading_indices: list[int] = field(default_factory=list)

    @property
    def titles(self) -> dict[str, TocEntry]:
        return {entry.key: entry for entry in self.entries if entry.key}


_SPACED_LEADER = re.compile(r"(?:\s*\.){3,}")


def _lines(block: SourceBlock) -> list[str]:
    """Lines of a block, with table cell separators and dot leaders turned into plain spacing."""
    lines = []
    for raw in block.text.splitlines():
        line = _SPACED_LEADER.sub("  ", raw.replace("|", "  ")).strip()
        if line:
            lines.append(line)
    return lines


def _parse_line(line: str) -> tuple[str, int | None] | None:
    match = _ENTRY.match(line)
    if not match:
        return None
    title = match.group("title").strip(" .·…_")
    if not title or len(title) > 200 or title.isdigit():
        return None
    page = match.group("page")
    return title, int(page) if page.isdigit() else None


def _entries_of(block: SourceBlock, index: int) -> list[TocEntry]:
    """Contents entries in a block. A block counts when most of its lines are entries (tables
    often repeat or garble a cell, so the odd line may not parse)."""
    lines = _lines(block)
    entries: list[TocEntry] = []
    for line in lines:
        parsed = _parse_line(line)
        if parsed is None:
            continue
        title, page = parsed
        pattern = parse_heading(title)
        heading = pattern.title if pattern and pattern.title else title
        entries.append(TocEntry(title, page, pattern.depth if pattern else None, index, heading))
    return entries if lines and len(entries) * 2 >= len(lines) else []


def detect_toc(blocks: list[SourceBlock]) -> TocResult:
    """Find contents lists: entries following a "Contents" heading, or lines with dot leaders."""
    result = TocResult()
    search_limit = max(60, int(len(blocks) * 0.35))

    for index, block in enumerate(blocks[:search_limit]):
        if index in result.region:
            continue
        keyword = match_keyword(block.text) if len(block.text) <= 40 else None
        if keyword is None or keyword.node_type is None or keyword.node_type.value != "table_of_contents":
            continue
        entries: list[TocEntry] = []
        region: list[int] = []
        blank_run = 0
        for j in range(index + 1, len(blocks)):
            candidate = blocks[j]
            found = _entries_of(candidate, j)
            if found:
                entries.extend(found)
                region.extend(range(j - blank_run, j + 1))
                blank_run = 0
            else:
                blank_run += 1
                if blank_run > _MAX_BLANK_RUN or candidate.block_type == "heading":
                    break
        if len(entries) >= _MIN_ENTRIES:
            result.heading_indices.append(index)
            result.entries.extend(entries)
            result.region.update(region)

    # Dot-leader lines are contents entries wherever they are.
    for index, block in enumerate(blocks):
        if index in result.region or len(block.text) > 200:
            continue
        raw = [line.strip() for line in block.text.replace("|", "  ").splitlines() if line.strip()]
        if raw and all(_LEADER_LINE.match(line) for line in raw):
            entries = _entries_of(block, index)
            if entries:
                result.region.add(index)
                result.entries.extend(entries)
    return result
