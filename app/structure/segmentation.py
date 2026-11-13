"""Semantic passage segmentation.

Passages are derived from paragraphs, one structural node at a time (so unrelated sections are
never combined), and can be regenerated with different size settings without repeating structure
detection. The rules, in priority order:

* boundaries fall between paragraphs; sections are never merged to reach a target size;
* a section shorter than the target becomes one passage;
* equations stay with the paragraph they explain, tables stay whole, figure captions stay with
  the preceding content (table captions with their table);
* a paragraph longer than the maximum is split at sentence boundaries (rows for tables and
  lists), and anything that cannot be split sensibly is allowed to exceed the maximum;
* each passage repeats the last sentences of the previous passage of the same node (10-15% of
  the target by default), recorded so downstream steps can skip the repeated text.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.structure.models import (
    NodeDraft,
    ParagraphDraft,
    PassageDraft,
    PassagePart,
)
from app.structure.tokenizer import Tokenizer, build_tokenizer

logger = logging.getLogger(__name__)

PARAGRAPH_SEPARATOR = "\n\n"
_PROSE_KINDS = ("paragraph", "list", "heading", "other")
_OVERLAP_KINDS = ("paragraph",)
_ABBREVIATIONS = frozenset(
    "e.g i.e etc vs fig figs eq eqs dr mr mrs ms prof sr jr no cf approx ca vol pp p ch sec st inc "
    "ltd al resp viz".split()
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"'”’)\]]*(\s+)(?=[\"'“‘(\[]?[A-Z0-9])")
_LAST_WORD = re.compile(r"(\S+)$")


@dataclass(frozen=True)
class SegmentationConfig:
    target_tokens: int = 750
    min_tokens: int = 500
    max_tokens: int = 1000
    overlap_ratio: float = 0.12
    tokenizer: str = "approx"
    chars_per_token: float = 4.0
    tiktoken_encoding: str = "cl100k_base"
    table_split_factor: float = 2.0

    def __post_init__(self) -> None:
        if not 1 <= self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("Passage sizes must satisfy min <= target <= max (all positive).")
        if not 0 <= self.overlap_ratio <= 0.4:
            raise ValueError("The overlap ratio must be between 0 and 0.4.")

    @property
    def overlap_target_tokens(self) -> int:
        return round(self.target_tokens * self.overlap_ratio)

    @property
    def overlap_max_tokens(self) -> int:
        """The most overlap ever taken: 25% above the target share (12% becomes at most 15%)."""
        return int(self.target_tokens * self.overlap_ratio * 1.25)

    @classmethod
    def from_settings(cls, settings: Settings) -> SegmentationConfig:
        return cls(
            target_tokens=settings.passage_target_tokens,
            min_tokens=settings.passage_min_tokens,
            max_tokens=settings.passage_max_tokens,
            overlap_ratio=settings.passage_overlap_ratio,
            tokenizer=settings.passage_tokenizer,
            chars_per_token=settings.passage_chars_per_token,
            tiktoken_encoding=settings.tiktoken_encoding,
            table_split_factor=settings.passage_table_split_factor,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "overlap_ratio": self.overlap_ratio,
            "overlap_max_tokens": self.overlap_max_tokens,
            "tokenizer": self.tokenizer,
            "chars_per_token": self.chars_per_token,
            "tiktoken_encoding": self.tiktoken_encoding,
            "table_split_factor": self.table_split_factor,
        }

    def build_tokenizer(self) -> Tokenizer:
        return build_tokenizer(
            self.tokenizer, chars_per_token=self.chars_per_token, tiktoken_encoding=self.tiktoken_encoding
        )


# -- text splitting -------------------------------------------------------------------------


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of the sentences of `text`. Abbreviations ("e.g.", "Fig.") do not end one."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        before = text[start : match.start(1)]
        word = _LAST_WORD.search(before.rstrip("\"'”’)]"))
        token = word.group(1).rstrip(".").lower() if word else ""
        if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha() and word and word.group(1).endswith(".")):
            continue
        spans.append((start, match.start(1)))
        start = match.end(1)
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans or ([(0, len(text))] if text.strip() else [])


def _line_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text) if m.group().strip()]


def _hard_split(text: str, start: int, end: int, limit: int, tokenizer: Tokenizer) -> list[tuple[int, int]]:
    """Split one over-long sentence at word boundaries into pieces of about `limit` tokens."""
    pieces: list[tuple[int, int]] = []
    piece_start, tokens = start, 0
    for word in re.finditer(r"\S+\s*", text[start:end]):
        word_tokens = tokenizer.count(word.group())
        if tokens + word_tokens > limit and tokens > 0:
            pieces.append((piece_start, start + word.start()))
            piece_start, tokens = start + word.start(), 0
        tokens += word_tokens
    pieces.append((piece_start, end))
    return [(s, e) for s, e in pieces if text[s:e].strip()]


def chunk_spans(
    text: str,
    spans: list[tuple[int, int]],
    *,
    limit: int,
    maximum: int,
    tokenizer: Tokenizer,
    hard_split: bool,
) -> list[tuple[int, int]]:
    """Group consecutive spans into chunks of at most `limit` tokens (a lone span may be larger).

    A span larger than `maximum` is split at words when `hard_split` is set (prose), otherwise
    kept whole. A small last chunk is folded into the one before it when that stays within `maximum`.
    """
    chunks: list[tuple[int, int, int]] = []  # start, end, tokens
    current: tuple[int, int, int] | None = None
    for start, end in spans:
        tokens = tokenizer.count(text[start:end])
        if tokens > maximum and hard_split:
            if current:
                chunks.append(current)
                current = None
            for piece_start, piece_end in _hard_split(text, start, end, limit, tokenizer):
                chunks.append((piece_start, piece_end, tokenizer.count(text[piece_start:piece_end])))
            continue
        if current is None:
            current = (start, end, tokens)
        elif current[2] + tokens > limit:
            chunks.append(current)
            current = (start, end, tokens)
        else:
            current = (current[0], end, current[2] + tokens)
    if current:
        chunks.append(current)
    if len(chunks) >= 2 and chunks[-1][2] < limit * 0.3:
        merged_tokens = chunks[-2][2] + chunks[-1][2]
        if merged_tokens <= maximum:
            chunks[-2:] = [(chunks[-2][0], chunks[-1][1], merged_tokens)]
    return [(s, e) for s, e, _ in chunks]


# -- pieces and groups ----------------------------------------------------------------------


@dataclass
class _Piece:
    """A paragraph, or a part of a long one, as it enters a passage."""

    paragraph: ParagraphDraft
    text: str
    tokens: int
    text_start: int | None  # None: the whole paragraph
    text_end: int | None
    part: int = 0
    parts: int = 1

    @property
    def kind(self) -> str:
        return self.paragraph.kind

    @property
    def is_last_part(self) -> bool:
        return self.part == self.parts - 1

    @property
    def is_first_part(self) -> bool:
        return self.part == 0


def _pieces_for(paragraph: ParagraphDraft, config: SegmentationConfig, tokenizer: Tokenizer) -> list[_Piece]:
    text = paragraph.text
    tokens = tokenizer.count(text)
    whole = [_Piece(paragraph, text, tokens, None, None)]
    if paragraph.kind == "equation":
        return whole
    if paragraph.kind == "table":
        if tokens <= config.table_split_factor * config.max_tokens:
            return whole
        spans = _line_spans(text)
        chunks = chunk_spans(
            text, spans, limit=config.max_tokens, maximum=config.max_tokens, tokenizer=tokenizer, hard_split=False
        )
    elif tokens <= config.max_tokens:
        return whole
    elif paragraph.kind == "list":
        chunks = chunk_spans(
            text, _line_spans(text), limit=config.target_tokens, maximum=config.max_tokens,
            tokenizer=tokenizer, hard_split=True,
        )
    else:
        chunks = chunk_spans(
            text, split_sentences(text), limit=config.target_tokens, maximum=config.max_tokens,
            tokenizer=tokenizer, hard_split=True,
        )
    if len(chunks) <= 1:
        return whole
    return [
        _Piece(paragraph, text[s:e], tokenizer.count(text[s:e]), s, e, part=i, parts=len(chunks))
        for i, (s, e) in enumerate(chunks)
    ]


def _group(pieces: list[_Piece]) -> list[list[_Piece]]:
    """Keep related pieces together: equations with their context, captions with their content."""
    groups: list[list[_Piece]] = []
    glue_next = False
    for index, piece in enumerate(pieces):
        following = pieces[index + 1] if index + 1 < len(pieces) else None
        is_figure_caption = piece.kind == "caption" and "table_ref" not in piece.paragraph.metadata
        attach = glue_next and bool(groups)
        glue_next = False
        previous_is_last = bool(groups) and groups[-1][-1].is_last_part

        if piece.kind == "equation" and groups and previous_is_last and piece.is_first_part:
            attach = True
        elif is_figure_caption and groups and previous_is_last:
            attach = True
        elif piece.kind == "caption" and not is_figure_caption and groups and groups[-1][-1].kind == "table":
            attach = True

        if attach:
            groups[-1].append(piece)
        else:
            groups.append([piece])

        if piece.is_last_part and following is not None:
            if piece.kind == "equation" and following.kind in ("paragraph", "list", "equation", "caption"):
                glue_next = True
            elif piece.kind == "caption" and not is_figure_caption and following.kind == "table":
                glue_next = True
            elif piece.kind in _PROSE_KINDS and piece.text.rstrip().endswith(":") and following.kind in (
                "equation", "table", "list",
            ):
                glue_next = True
    return groups


def _pack(groups: list[list[_Piece]], config: SegmentationConfig) -> list[list[list[_Piece]]]:
    """Fill passages with whole groups, aiming for the target size and staying under the maximum."""
    passages: list[list[list[_Piece]]] = []
    current: list[list[_Piece]] = []
    total = 0
    for group in groups:
        size = sum(p.tokens for p in group)
        if not current:
            current, total = [group], size
            continue
        merged = total + size
        if merged <= config.target_tokens:
            add = True
        elif merged > config.max_tokens:
            add = False
        elif total < config.min_tokens:
            add = True  # still short of the minimum and the maximum is respected
        else:
            add = abs(merged - config.target_tokens) < abs(total - config.target_tokens)
        if add:
            current.append(group)
            total = merged
        else:
            passages.append(current)
            current, total = [group], size
    if current:
        passages.append(current)

    # A short last passage joins the previous one when the maximum allows.
    if len(passages) >= 2:
        last = sum(p.tokens for g in passages[-1] for p in g)
        before = sum(p.tokens for g in passages[-2] for p in g)
        if last < config.min_tokens and before + last <= config.max_tokens:
            passages[-2:] = [passages[-2] + passages[-1]]
    return passages


# -- passages -------------------------------------------------------------------------------


def _piece_pages(piece: _Piece) -> tuple[int | None, int | None, str]:
    """Page range and status of a piece. Only recorded source pages are used, never text length."""
    paragraph = piece.paragraph
    spans = paragraph.metadata.get("block_spans")
    if piece.text_start is None or not spans:
        return paragraph.page_start, paragraph.page_end, paragraph.page_status
    overlapping = [s for s in spans if s["char_start"] < piece.text_end and s["char_end"] > piece.text_start]
    pages = [s["page"] for s in overlapping if s["page"] is not None]
    pages += [s["last_page"] for s in overlapping if s.get("last_page") is not None]
    if not pages:
        return None, None, "unknown"
    status = "exact" if all(s["page"] is not None for s in overlapping) else "partial"
    return min(pages), max(pages), status


def _piece_offsets(piece: _Piece) -> tuple[int | None, int | None]:
    paragraph = piece.paragraph
    if piece.text_start is None:
        return paragraph.start_offset, paragraph.end_offset
    if paragraph.offsets_exact and paragraph.start_offset is not None:
        return paragraph.start_offset + piece.text_start, paragraph.start_offset + piece.text_end
    return None, None


def _piece_bboxes(piece: _Piece) -> list[dict[str, Any]]:
    boxes = piece.paragraph.metadata.get("bboxes", [])
    spans = piece.paragraph.metadata.get("block_spans")
    if piece.text_start is None or not spans:
        return list(boxes)
    wanted = {s["sequence"] for s in spans if s["char_start"] < piece.text_end and s["char_end"] > piece.text_start}
    return [b for b in boxes if b["sequence"] in wanted]


@dataclass
class _Overlap:
    pieces: list[_Piece] = field(default_factory=list)

    @property
    def text(self) -> str:
        return PARAGRAPH_SEPARATOR.join(p.text for p in self.pieces)


def _overlap_from(previous: list[_Piece], config: SegmentationConfig, tokenizer: Tokenizer) -> _Overlap:
    """The end of the previous passage to repeat: whole trailing paragraphs, or its last sentences."""
    overlap = _Overlap()
    if config.overlap_max_tokens <= 0:
        return overlap
    remaining, taken = config.overlap_max_tokens, 0
    for piece in reversed(previous):
        if piece.kind not in _OVERLAP_KINDS:
            break
        if piece.tokens <= remaining:
            overlap.pieces.insert(0, piece)
            taken += piece.tokens
            remaining -= piece.tokens
            if taken >= config.overlap_target_tokens:
                break
            continue
        sentences = split_sentences(piece.text)
        start_index = len(sentences)
        used = 0
        for index in range(len(sentences) - 1, -1, -1):
            cost = tokenizer.count(piece.text[sentences[index][0] : sentences[index][1]])
            if used + cost > remaining:
                break
            used += cost
            start_index = index
        if start_index < len(sentences):
            local_start, local_end = sentences[start_index][0], sentences[-1][1]
            base = piece.text_start or 0
            text = piece.text[local_start:local_end]
            overlap.pieces.insert(
                0,
                _Piece(piece.paragraph, text, tokenizer.count(text), base + local_start, base + local_end,
                       part=piece.part, parts=piece.parts),
            )
        break
    return overlap


def segment_node(
    node: NodeDraft,
    paragraphs: list[ParagraphDraft],
    config: SegmentationConfig,
    tokenizer: Tokenizer,
    *,
    start_document_sequence: int = 0,
) -> list[PassageDraft]:
    """Passages for one structural node."""
    if not paragraphs:
        return []
    pieces: list[_Piece] = []
    for paragraph in paragraphs:
        pieces.extend(_pieces_for(paragraph, config, tokenizer))
    packed = _pack(_group(pieces), config)

    passages: list[PassageDraft] = []
    previous_body: list[_Piece] = []
    previous_id: uuid.UUID | None = None
    for sequence, groups in enumerate(packed):
        body = [piece for group in groups for piece in group]
        overlap = _overlap_from(previous_body, config, tokenizer) if previous_body else _Overlap()
        sections = [(p, True) for p in overlap.pieces] + [(p, False) for p in body]
        text = PARAGRAPH_SEPARATOR.join(p.text for p, _ in sections)
        overlap_chars = len(overlap.text) + len(PARAGRAPH_SEPARATOR) if overlap.pieces else 0

        page_info = [_piece_pages(p) for p, _ in sections]
        pages = [(s, e) for s, e, _ in page_info if s is not None and e is not None]
        statuses = {status for _, _, status in page_info}
        status = "exact" if statuses == {"exact"} else "unknown" if not pages else "partial"
        offsets = [_piece_offsets(p) for p, _ in sections]
        known_offsets = all(s is not None and e is not None for s, e in offsets)

        equation_ids = [i for p, is_overlap in sections if not is_overlap for i in p.paragraph.metadata.get("equation_ids", [])]
        split_paragraphs = sorted({str(p.paragraph.id) for p in body if p.parts > 1})
        body_tokens = sum(p.tokens for p in body)
        oversized = None
        if body_tokens > config.max_tokens:
            kinds = {p.kind for p in body}
            oversized = (
                "table" if "table" in kinds else "equation_context" if "equation" in kinds
                else "caption" if "caption" in kinds else "unsplittable"
            )
        metadata: dict[str, Any] = {
            "paragraph_ids": list(dict.fromkeys(str(p.paragraph.id) for p, _ in sections)),
            "page_numbers": sorted({page for first, last in pages for page in range(first, last + 1)}),
            "bboxes": [box for p, _ in sections for box in _piece_bboxes(p)],
            "has_equation": any(p.kind == "equation" for p, _ in sections),
            "has_table": any(p.kind == "table" for p, _ in sections),
            "equation_ids": equation_ids,
            "figure_refs": [p.paragraph.metadata["figure_ref"] for p, _ in sections if "figure_ref" in p.paragraph.metadata],
            "table_refs": [p.paragraph.metadata["table_ref"] for p, _ in sections if "table_ref" in p.paragraph.metadata],
            "body_tokens": body_tokens,
            "overlap_chars": overlap_chars,
            "overlap_tokens": sum(p.tokens for p in overlap.pieces),
            "overlap_with_passage_id": str(previous_id) if overlap.pieces and previous_id else None,
            "split_paragraph_ids": split_paragraphs,
            "oversized": oversized,
            "page_status_detail": sorted(statuses),
        }
        passage = PassageDraft(
            id=uuid.uuid4(),
            node_id=node.id,
            sequence=sequence,
            document_sequence=start_document_sequence + sequence,
            text=text,
            token_count=tokenizer.count(text),
            page_start=min(s for s, _ in pages) if pages else None,
            page_end=max(e for _, e in pages) if pages else None,
            page_status=status,
            start_offset=offsets[0][0] if known_offsets else None,
            end_offset=offsets[-1][1] if known_offsets else None,
            metadata=metadata,
            parts=[
                PassagePart(
                    paragraph_id=p.paragraph.id,
                    position=position,
                    is_overlap=is_overlap,
                    text_start=p.text_start,
                    text_end=p.text_end,
                )
                for position, (p, is_overlap) in enumerate(sections)
            ],
        )
        passages.append(passage)
        previous_body, previous_id = body, passage.id
    return passages


def generate_passages(
    nodes: list[NodeDraft],
    paragraphs: list[ParagraphDraft],
    config: SegmentationConfig,
    tokenizer: Tokenizer | None = None,
) -> list[PassageDraft]:
    """Passages for every node, in document order."""
    tokenizer = tokenizer or config.build_tokenizer()
    by_node: dict[uuid.UUID, list[ParagraphDraft]] = {}
    for paragraph in sorted(paragraphs, key=lambda p: p.document_sequence):
        by_node.setdefault(paragraph.node_id, []).append(paragraph)

    result: list[PassageDraft] = []
    for node in sorted(nodes, key=lambda n: n.sequence):
        node_paragraphs = sorted(by_node.get(node.id, []), key=lambda p: p.sequence)
        result.extend(
            segment_node(node, node_paragraphs, config, tokenizer, start_document_sequence=len(result))
        )
    logger.info("Generated %d passages for %d nodes", len(result), len(nodes))
    return result
