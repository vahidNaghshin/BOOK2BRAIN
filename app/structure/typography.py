"""Typographic helpers: body text size, heading size ratios, capitalisation."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Sequence

from app.structure.models import SourceBlock

_SMALL_WORDS = frozenset(
    "a an the and or but nor for of in on at to by with from as vs via per into over under".split()
)
# A heading this short (in characters) is assumed to fit on one line, so its box height
# approximates the line height (and so the font size).
_SINGLE_LINE_CHARS = 70


def body_line_height(blocks: Sequence[SourceBlock]) -> float | None:
    """Height of one line of ordinary body text, estimated from the shape of long paragraphs.

    A box of n lines has an area of about 0.6 * font_size^2 per character, so the font size
    follows from area and text length, and a line is about 1.2 font sizes tall. This works for
    paragraphs of any number of lines, unlike taking the height of the shortest paragraphs.
    """
    estimates = []
    for block in blocks:
        if block.block_type != "paragraph" or len(block.text) < 150 or not block.bbox:
            continue
        width, height = abs(block.bbox[2] - block.bbox[0]), block.height
        if width > 0 and height:
            estimates.append(1.2 * math.sqrt(width * height / (0.6 * len(block.text))))
    if len(estimates) < 3:
        return None
    return statistics.median(estimates)


def height_ratio(block: SourceBlock, reference: float | None) -> float | None:
    """Box height relative to a body line, only for text short enough to be one line."""
    if not reference or not block.height or "\n" in block.text.strip():
        return None
    if len(block.text.strip()) > _SINGLE_LINE_CHARS:
        return None
    return block.height / reference


def cluster_ratios(ratios: Iterable[float], tolerance: float = 0.08) -> dict[float, int]:
    """Group similar sizes. Cluster 0 is the largest size. Returns ratio -> cluster id."""
    ordered = sorted(set(ratios), reverse=True)
    mapping: dict[float, int] = {}
    cluster, anchor = -1, None
    for value in ordered:
        if anchor is None or value < anchor * (1 - tolerance):
            cluster += 1
            anchor = value
        mapping[value] = cluster
    return mapping


def letters(text: str) -> str:
    return "".join(ch for ch in text if ch.isalpha())


def is_all_caps(text: str) -> bool:
    chars = letters(text)
    return len(chars) >= 3 and chars.isupper()


def is_title_case(text: str) -> bool:
    """Most significant words start with a capital letter."""
    words = [w for w in re.findall(r"[^\W\d_][\w'’\-]*", text) if w.lower() not in _SMALL_WORDS]
    if not words:
        return False
    capitalised = sum(1 for w in words if w[0].isupper())
    return capitalised / len(words) >= 0.7

