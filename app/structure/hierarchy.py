"""Build the structural tree from detected headings.

Steps: give every heading a hierarchy rank (explicit numbering first, typographic style for the
rest), decide its node type, place front matter / body / back matter, resolve parents with a
stack, and compute page ranges and offsets from the blocks each subtree contains.
"""

from __future__ import annotations

import logging
import statistics
import uuid
from collections import Counter
from dataclasses import dataclass, field

from app.structure.front_matter import MetadataGuess
from app.structure.heading_detection import HeadingCandidate
from app.structure.models import (
    BACK_TYPES,
    FRONT_ONLY_TYPES,
    RANK_CHAPTER,
    RANK_CONTAINER,
    RANK_PART,
    DocumentText,
    NodeDraft,
    NodeType,
    Severity,
    ValidationIssue,
    type_for_rank,
)

logger = logging.getLogger(__name__)

@dataclass
class _Marker:
    node: NodeDraft
    start_index: int
    heading_indices: list[int]
    order: int
    candidate: HeadingCandidate | None = None
    synthetic: bool = False


@dataclass
class HierarchyResult:
    nodes: list[NodeDraft]
    # For each block index: the id of the node it belongs to, or None when the block is a
    # heading consumed by a node (headings are never repeated as body text).
    block_nodes: list[uuid.UUID | None]
    issues: list[ValidationIssue] = field(default_factory=list)


def _style_sort_key(style: str) -> tuple[int, int]:
    if style == "P":
        return 999, 999
    level_part, size_part = style.split("|")
    level = level_part[1:]
    size = size_part[1:]
    return (int(level) if level != "-" else 99, int(size) if size != "-" else 99)


def assign_ranks(candidates: list[HeadingCandidate]) -> None:
    """Give each candidate a hierarchy rank (lower is more prominent).

    Numbering decides first ("Part", "Chapter", "1.2.3"). Styles with no numbering are placed
    relative to the styles that have it: between, after or before them by prominence. With no
    numbering anywhere, the most prominent style is the chapter level, the next sections, and so on.
    """
    by_style: dict[str, list[HeadingCandidate]] = {}
    for candidate in candidates:
        by_style.setdefault(candidate.style, []).append(candidate)
    ordered = sorted(by_style, key=_style_sort_key)

    explicit: dict[str, int] = {}
    for style, members in by_style.items():
        ranks = [c.explicit_rank for c in members if c.explicit_rank is not None]
        if ranks:
            rank, count = Counter(ranks).most_common(1)[0]
            if count / len(ranks) >= 0.6:
                explicit[style] = rank

    chapter_ratios = [
        c.ratio
        for c in candidates
        if c.ratio is not None and c.pattern is not None and c.pattern.node_type in (NodeType.CHAPTER, NodeType.PART)
    ]
    chapter_ratio = statistics.median(chapter_ratios) if len(chapter_ratios) >= 2 else None

    assigned: dict[str, int] = {}
    previous: int | None = None
    first_anchor = next((position for position, style in enumerate(ordered) if style in explicit), None)
    for position, style in enumerate(ordered):
        if style in explicit:
            rank = explicit[style]
        elif first_anchor is not None and position < first_anchor:
            # More prominent than the first numbered style: at the level above it (part-like), not
            # nested below it. Book-level headings such as "Bibliography" or "Index" look like this.
            rank = max(RANK_PART, explicit[ordered[first_anchor]] - 10)
        else:
            following = next((explicit[s] for s in ordered[position + 1 :] if s in explicit), None)
            if previous is None:
                rank = RANK_CHAPTER
            elif following is None or previous + 10 < following:
                rank = previous + 10
            else:
                rank = max(previous + 1, (previous + following) // 2)
        assigned[style] = rank
        previous = rank

    for candidate in candidates:
        if candidate.llm and candidate.llm.get("rank") is not None:
            candidate.rank = int(candidate.llm["rank"])
        elif candidate.explicit_rank is not None:
            candidate.rank = candidate.explicit_rank
        else:
            candidate.rank = assigned[candidate.style]
            # "Bibliography", "Index", "Preface"... printed at least as large as a chapter title are
            # book-level headings, however the other styles happen to be ordered. (The size of
            # headings labelled "Chapter"/"Part" is the most trustworthy reference there is.)
            keyword = candidate.keyword
            if (
                chapter_ratio is not None
                and candidate.ratio is not None
                and candidate.ratio >= chapter_ratio * 0.95
                and keyword is not None
                and keyword.node_type is not None
            ):
                candidate.rank = min(candidate.rank, RANK_CHAPTER)


def _node_type(candidate: HeadingCandidate) -> NodeType:
    rank = candidate.rank if candidate.rank is not None else RANK_CHAPTER
    if candidate.llm and candidate.llm.get("node_type"):
        return NodeType(candidate.llm["node_type"])
    pattern, keyword = candidate.pattern, candidate.keyword
    if pattern is not None and pattern.node_type is not None:
        if pattern.node_type is NodeType.SECTION and pattern.depth > 1:
            return type_for_rank(rank)
        return pattern.node_type
    if pattern is not None:
        # "1. Introduction" is the first chapter, not the book's introduction; only back-matter
        # words ("9. References") keep their meaning when numbered.
        if keyword is not None and keyword.node_type in BACK_TYPES and rank <= RANK_CHAPTER:
            return keyword.node_type
    elif keyword is not None and rank <= RANK_CHAPTER:
        if keyword.node_type is not None:
            return keyword.node_type
        return NodeType.CHAPTER  # "Conclusion", "Epilogue": an unnumbered chapter
    if candidate.status == "ambiguous":
        return NodeType.OTHER_HEADING
    inferred = type_for_rank(rank)
    return NodeType.OTHER_HEADING if inferred is NodeType.PART else inferred


_FRONT_KEYWORD_TYPES = FRONT_ONLY_TYPES | {
    NodeType.INTRODUCTION,
    NodeType.ACKNOWLEDGEMENT,
    NodeType.ABOUT_AUTHOR,
}


def _body_start_index(candidates: list[HeadingCandidate]) -> int | None:
    """Block index of the first confident heading that belongs to the main body of the book."""
    for candidate in candidates:
        if candidate.status != "accepted" or candidate.rank is None or candidate.rank > RANK_CHAPTER:
            continue
        pattern = candidate.pattern
        if pattern is not None and pattern.node_type in (NodeType.PART, NodeType.CHAPTER):
            return candidate.index
        if pattern is not None and pattern.form in ("numeric", "roman_dot"):
            return candidate.index
        if pattern is None and (candidate.keyword is None or candidate.keyword.node_type is None):
            return candidate.index
    return None


def _is_body_heading(node_type: NodeType, rank: int) -> bool:
    return rank <= RANK_CHAPTER and node_type in (NodeType.PART, NodeType.CHAPTER, NodeType.OTHER_HEADING)


def _is_back_start(node_type: NodeType, rank: int) -> bool:
    return rank <= RANK_CHAPTER and (
        node_type in BACK_TYPES or node_type is NodeType.ACKNOWLEDGEMENT
    )


def _candidate_metadata(candidate: HeadingCandidate) -> dict:
    metadata: dict = {
        "method": "llm" if candidate.llm else "deterministic",
        "status": candidate.status,
        "signals": candidate.signals,
        "style": candidate.style,
    }
    if candidate.in_toc:
        metadata["in_table_of_contents"] = True
    if candidate.pattern is not None:
        metadata["numbering"] = {
            "form": candidate.pattern.form,
            "style": candidate.pattern.number_style,
            "value": candidate.pattern.number_value,
        }
    if candidate.llm:
        metadata["llm"] = candidate.llm
    return metadata


def build_hierarchy(
    document: DocumentText,
    candidates: list[HeadingCandidate],
    guess: MetadataGuess,
) -> HierarchyResult:
    blocks = document.blocks
    issues: list[ValidationIssue] = []

    assign_ranks(candidates)

    # Preface, contents, acknowledgements... before the body are front matter whatever their size
    # on the page: they are siblings of one another, not subsections of whatever precedes them.
    body_index = _body_start_index(candidates)
    if body_index is not None:
        for candidate in candidates:
            keyword = candidate.keyword
            if (
                candidate.index < body_index
                and candidate.pattern is None
                and keyword is not None
                and keyword.node_type in _FRONT_KEYWORD_TYPES
                and not candidate.llm
            ):
                candidate.rank = RANK_CHAPTER

    markers: list[_Marker] = []
    for candidate in candidates:
        node_type = _node_type(candidate)
        node = NodeDraft(
            id=uuid.uuid4(),
            node_type=node_type.value,
            title=candidate.title or candidate.original_text,
            original_title=candidate.original_text,
            number=candidate.pattern.number if candidate.pattern else None,
            level=0,
            sequence=0,
            confidence=round(candidate.llm["confidence"], 2) if candidate.llm else round(candidate.score, 2),
            metadata=_candidate_metadata(candidate),
            heading_blocks=[blocks[i].sequence for i in candidate.block_indices],
            rank=candidate.rank if candidate.rank is not None else RANK_CHAPTER,
        )
        markers.append(_Marker(node, candidate.index, list(candidate.block_indices), 0, candidate))

    if guess.title_block is not None and guess.title:
        markers.append(
            _Marker(
                NodeDraft(
                    id=uuid.uuid4(),
                    node_type=NodeType.BOOK_TITLE.value,
                    title=guess.title,
                    original_title=None,
                    level=0,
                    sequence=0,
                    confidence=guess.title_confidence,
                    metadata={"method": "deterministic", "synthetic": False, **guess.to_dict()},
                    heading_blocks=[blocks[guess.title_block].sequence],
                    rank=RANK_CHAPTER,
                ),
                guess.title_block,
                [guess.title_block],
                0,
                synthetic=True,
            )
        )
    if guess.copyright_blocks and not any(
        m.node.node_type == NodeType.COPYRIGHT.value for m in markers
    ):
        markers.append(
            _Marker(
                NodeDraft(
                    id=uuid.uuid4(),
                    node_type=NodeType.COPYRIGHT.value,
                    title="Copyright",
                    level=0,
                    sequence=0,
                    confidence=0.8,
                    metadata={"method": "deterministic", "synthetic": True, "signals": ["copyright_notice"]},
                    rank=RANK_CHAPTER,
                ),
                guess.copyright_blocks[0],
                [],
                0,
                synthetic=True,
            )
        )
    markers.sort(key=lambda m: m.start_index)
    for order, marker in enumerate(markers):
        marker.order = order

    if not markers:
        return _single_root(document, issues)

    # Where does the main body start? The first confident body heading (or, failing that, the
    # first ambiguous one). Everything before it is front matter.
    body_marker: _Marker | None = None
    for wanted in ("accepted", "ambiguous"):
        body_marker = next(
            (
                m
                for m in markers
                if m.candidate is not None
                and m.candidate.status == wanted
                and _is_body_heading(NodeType(m.node.node_type), m.node.rank)
            ),
            None,
        )
        if body_marker is not None:
            break

    nodes: list[_Marker] = []  # markers in creation (document) order, including containers
    stack: list[_Marker] = []
    first = markers[0]
    needs_front = first.start_index > 0 or first is not body_marker
    phase = "body"
    if needs_front:
        front = _container(NodeType.FRONT_MATTER, "Front Matter", 0)
        nodes.append(front)
        stack = [front]
        phase = "front"

    parents: dict[uuid.UUID, _Marker | None] = {}
    for marker in markers:
        node_type = NodeType(marker.node.node_type)
        if phase == "front" and marker is body_marker:
            phase = "body"
            stack = []
        elif phase == "body" and _is_back_start(node_type, marker.node.rank):
            container = _container(NodeType.BACK_MATTER, "Back Matter", marker.start_index)
            nodes.append(container)
            stack = [container]
            phase = "back"
        while stack and stack[-1].node.rank >= marker.node.rank:
            stack.pop()
        parents[marker.node.id] = stack[-1] if stack else None
        nodes.append(marker)
        stack.append(marker)

    return _finish(document, nodes, parents, candidates, issues)


def _container(node_type: NodeType, title: str, start_index: int) -> _Marker:
    return _Marker(
        NodeDraft(
            id=uuid.uuid4(),
            node_type=node_type.value,
            title=title,
            level=0,
            sequence=0,
            metadata={"method": "deterministic", "synthetic": True},
            rank=RANK_CONTAINER,
        ),
        start_index,
        [],
        -1,
        synthetic=True,
    )


def _single_root(document: DocumentText, issues: list[ValidationIssue]) -> HierarchyResult:
    """No headings at all: keep all the text under one node rather than losing structure."""
    blocks = document.blocks
    root = NodeDraft(
        id=uuid.uuid4(),
        node_type=NodeType.OTHER_HEADING.value,
        title=document.source_title or "Untitled content",
        level=0,
        sequence=0,
        confidence=None,
        metadata={"method": "fallback", "synthetic": True},
        start_block=0 if blocks else None,
        end_block=len(blocks) - 1 if blocks else None,
    )
    _apply_ranges(root, blocks, range(len(blocks)))
    issues.append(
        ValidationIssue(
            Severity.WARNING,
            "no_headings_detected",
            "No headings were detected, so all text is kept under a single node.",
            "document",
        )
    )
    return HierarchyResult([root], [root.id] * len(blocks), issues)


def _apply_ranges(node: NodeDraft, blocks: list, indices: range | list[int]) -> None:
    indices = list(indices)
    pages = [blocks[i].page for i in indices if blocks[i].page is not None]
    node.page_start = min(pages) if pages else None
    node.page_end = max(pages) if pages else None
    if indices:
        node.start_offset = blocks[min(indices)].start_offset
        node.end_offset = blocks[max(indices)].end_offset


def _finish(
    document: DocumentText,
    markers: list[_Marker],
    parents: dict[uuid.UUID, _Marker | None],
    candidates: list[HeadingCandidate],
    issues: list[ValidationIssue],
) -> HierarchyResult:
    blocks = document.blocks
    markers = sorted(markers, key=lambda m: (m.start_index, m.order))  # containers (-1) first
    by_id = {m.node.id: m for m in markers}

    # Parent links, levels, document order.
    for sequence, marker in enumerate(markers):
        node = marker.node
        node.sequence = sequence
        parent = parents.get(node.id)
        node.parent_id = parent.node.id if parent else None
    for marker in markers:
        parent_id = marker.node.parent_id
        marker.node.level = 0 if parent_id is None else by_id[parent_id].node.level + 1

    # Which block belongs to which node: the most recently started node (outline semantics).
    block_nodes: list[uuid.UUID | None] = [None] * len(blocks)
    own_blocks: dict[uuid.UUID, list[int]] = {m.node.id: [] for m in markers}
    heading_index_owner = {i: m for m in markers for i in m.heading_indices}
    starts = [(m.start_index, m.order, m) for m in markers]
    pointer = -1
    for index in range(len(blocks)):
        while pointer + 1 < len(starts) and starts[pointer + 1][0] <= index:
            pointer += 1
        owner = heading_index_owner.get(index)
        if owner is not None:
            own_blocks[owner.node.id].append(index)
            continue
        if pointer < 0:
            continue  # cannot happen: a front container starts at block 0 when needed
        current = starts[pointer][2]
        block_nodes[index] = current.node.id
        own_blocks[current.node.id].append(index)

    subtree: dict[uuid.UUID, list[int]] = {m.node.id: list(own_blocks[m.node.id]) for m in markers}
    for marker in reversed(markers):
        parent_id = marker.node.parent_id
        if parent_id is not None:
            subtree[parent_id].extend(subtree[marker.node.id])
    for marker in markers:
        indices = sorted(subtree[marker.node.id])
        marker.node.start_block = indices[0] if indices else None
        marker.node.end_block = indices[-1] if indices else None
        _apply_ranges(marker.node, blocks, indices)

    for candidate in candidates:
        if candidate.status == "ambiguous":
            node = next((m.node for m in markers if m.candidate is candidate), None)
            if node is not None and not (candidate.llm and candidate.llm.get("confidence", 0) >= 0.7):
                issues.append(
                    ValidationIssue(
                        Severity.WARNING,
                        "ambiguous_heading",
                        f"'{candidate.original_text}' may or may not be a heading "
                        f"(confidence {node.confidence:.2f}); it was kept as '{node.node_type}'.",
                        "node",
                        node.id,
                        {"signals": candidate.signals, "page": candidate.page},
                    )
                )
    return HierarchyResult([m.node for m in markers], block_nodes, issues)
