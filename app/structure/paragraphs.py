"""Paragraph identification.

Paragraphs are stored independently of passages. A paragraph that continues across a page break
is kept whole (with its full page range) instead of being split at the break, and runs of list
items become one list paragraph. Tables, equations and captions stay as their own units.
"""

from __future__ import annotations

import re
import uuid

from app.structure.models import (
    DocumentText,
    NodeDraft,
    ParagraphDraft,
    SourceBlock,
)

_TERMINAL = re.compile(r"[.!?:;][\"'”’)\]]*\s*$")
_FIGURE_REF = re.compile(r"^(?:figure|fig\.?)\s*(\d+(?:[.\-]\d+)*[a-z]?)", re.IGNORECASE)
_TABLE_REF = re.compile(r"^table\s*(\d+(?:[.\-]\d+)*[a-z]?)", re.IGNORECASE)
_KINDS = ("paragraph", "list", "table", "equation", "caption", "heading")


def continues_paragraph(previous: SourceBlock, following: SourceBlock) -> bool:
    """True when `following` is the rest of the paragraph `previous` ends on a page break."""
    if previous.page is None or following.page is None or following.page != previous.page + 1:
        return False
    before, after = previous.text.rstrip(), following.text.lstrip()
    if not before or not after:
        return False
    if before.endswith(("-", ",")):
        return True
    return not _TERMINAL.search(before) and after[0].islower()


def _join(left: str, right: str) -> str:
    # A trailing hyphen from page-break hyphenation is kept: nothing is dropped from the text.
    return left + right if left.endswith("-") else f"{left} {right}"


def _page_status(blocks: list[SourceBlock]) -> str:
    known = sum(1 for b in blocks if b.page is not None)
    if known == len(blocks):
        return "exact" if known else "unknown"
    return "partial" if known else "unknown"


def _combine(kind: str, members: list[SourceBlock]) -> tuple[str, list[dict]]:
    """Paragraph text and, for each block, the character range it occupies in that text."""
    text = members[0].text.strip() if len(members) > 1 else members[0].text
    spans = [{"sequence": members[0].sequence, "page": members[0].page, "char_start": 0, "char_end": len(text)}]
    for block in members[1:]:
        piece = block.text.strip()
        text = f"{text}\n{piece}" if kind == "list" else _join(text, piece)
        spans.append(
            {"sequence": block.sequence, "page": block.page, "char_start": len(text) - len(piece), "char_end": len(text)}
        )
    return text, spans


def _make_paragraph(
    document: DocumentText,
    node_id: uuid.UUID,
    kind: str,
    members: list[SourceBlock],
    sequence: int,
    document_sequence: int,
) -> ParagraphDraft:
    text, spans = _combine(kind, members)
    metadata: dict = {
        "offsets_exact": len(members) == 1,
        "block_sequences": [b.sequence for b in members],
        "block_ids": [str(b.id) for b in members if b.id is not None],
        "block_types": sorted({b.block_type for b in members}),
        "bboxes": [
            {"sequence": b.sequence, "page": b.page, "bbox": list(b.bbox), "origin": b.coord_origin}
            for b in members
            if b.bbox is not None
        ],
    }
    pages = [b.page for b in members if b.page is not None]
    if len(members) > 1:
        metadata["block_spans"] = spans
        if kind == "paragraph" and len(set(pages)) > 1:
            metadata["continued_across_pages"] = True
    equation_ids = [
        str(document.equation_ids_by_block[b.id])
        for b in members
        if b.id is not None and b.id in document.equation_ids_by_block
    ]
    if equation_ids:
        metadata["equation_ids"] = equation_ids
    if kind == "caption":
        figure, table = _FIGURE_REF.match(text.strip()), _TABLE_REF.match(text.strip())
        if figure:
            metadata["figure_ref"] = f"Figure {figure.group(1)}"
        if table:
            metadata["table_ref"] = f"Table {table.group(1)}"
    if kind == "table":
        for key in ("rows", "columns"):
            if members[0].metadata.get(key) is not None:
                metadata[key] = members[0].metadata[key]
    if kind == "heading":
        metadata["kept_as_text"] = "Heading-like text that was not confident enough to start a section."
    return ParagraphDraft(
        id=uuid.uuid4(),
        node_id=node_id,
        sequence=sequence,
        document_sequence=document_sequence,
        kind=kind,
        text=text,
        block_sequences=[b.sequence for b in members],
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        page_status=_page_status(members),
        start_offset=members[0].start_offset,
        end_offset=members[-1].end_offset,
        offsets_exact=len(members) == 1,
        metadata=metadata,
    )


def build_paragraphs(
    document: DocumentText, nodes: list[NodeDraft], block_nodes: list[uuid.UUID | None]
) -> list[ParagraphDraft]:
    """Paragraphs for every node, in document order. Heading blocks consumed by nodes are skipped."""
    blocks_by_node: dict[uuid.UUID, list[SourceBlock]] = {node.id: [] for node in nodes}
    for block, node_id in zip(document.blocks, block_nodes):
        if node_id is not None and block.text.strip():
            blocks_by_node[node_id].append(block)

    paragraphs: list[ParagraphDraft] = []
    for node in nodes:
        sequence = 0
        run: list[SourceBlock] = []
        run_kind = ""
        groups: list[tuple[str, list[SourceBlock]]] = []
        for block in blocks_by_node[node.id]:
            kind = block.block_type if block.block_type in _KINDS else "other"
            if kind == "paragraph" and run_kind == "paragraph" and continues_paragraph(run[-1], block):
                run.append(block)
                continue
            if kind == "list" and run_kind == "list":
                run.append(block)
                continue
            if run:
                groups.append((run_kind, run))
            run, run_kind = [block], kind
        if run:
            groups.append((run_kind, run))
        for kind, members in groups:
            paragraphs.append(_make_paragraph(document, node.id, kind, members, sequence, len(paragraphs)))
            sequence += 1
    return paragraphs
