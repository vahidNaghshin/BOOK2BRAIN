"""Validation of the reconstructed structure, paragraphs and passages.

Problems are returned as structured issues (errors and warnings) rather than raised or silently
corrected, and nothing is discarded because it could not be classified confidently.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from collections.abc import Sequence

from app.structure.models import (
    RANK_CONTAINER,
    DocumentText,
    NodeDraft,
    ParagraphDraft,
    PassageDraft,
    Severity,
    ValidationIssue,
)
from app.structure.numbering import normalise_for_match
from app.structure.segmentation import PARAGRAPH_SEPARATOR, SegmentationConfig
from app.structure.tokenizer import Tokenizer

_LOW_CONFIDENCE = 0.6
_MAX_LISTED = 20


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _issue(
    severity: Severity,
    code: str,
    message: str,
    entity_type: str | None = None,
    entity_id: uuid.UUID | None = None,
    **details: object,
) -> ValidationIssue:
    return ValidationIssue(severity, code, message, entity_type, entity_id, dict(details))


def count_by_severity(issues: Sequence[ValidationIssue]) -> dict[str, int]:
    counts = {severity.value: 0 for severity in Severity}
    for issue in issues:
        counts[issue.severity.value] += 1
    return counts


# -- structure ------------------------------------------------------------------------------


def validate_structure(
    document: DocumentText, nodes: list[NodeDraft], paragraphs: list[ParagraphDraft]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(_validate_hierarchy(nodes))
    issues.extend(_validate_node_pages(document, nodes))
    issues.extend(_validate_paragraph_pages(document, nodes, paragraphs))
    issues.extend(_validate_content(document, nodes, paragraphs))
    issues.extend(_validate_empty_nodes(nodes, paragraphs))
    return issues


def _validate_hierarchy(nodes: list[NodeDraft]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_id: dict[uuid.UUID, NodeDraft] = {}
    for node in nodes:
        if node.id in by_id:
            issues.append(_issue(Severity.ERROR, "duplicate_node_id", "Two nodes share one id.", "node", node.id))
        by_id[node.id] = node

    previous_sequence = -1
    for node in nodes:
        if node.sequence <= previous_sequence:
            issues.append(
                _issue(
                    Severity.ERROR, "sequence_not_increasing",
                    f"Node '{node.title}' is out of document order.", "node", node.id,
                    sequence=node.sequence,
                )
            )
        previous_sequence = node.sequence

    children: dict[uuid.UUID | None, list[NodeDraft]] = defaultdict(list)
    for node in nodes:
        children[node.parent_id].append(node)
        if node.parent_id is None:
            if node.level != 0:
                issues.append(_issue(Severity.ERROR, "level_inconsistent", f"Top-level node '{node.title}' has level {node.level}.", "node", node.id))
            continue
        parent = by_id.get(node.parent_id)
        if parent is None:
            issues.append(_issue(Severity.ERROR, "parent_missing", f"The parent of '{node.title}' does not exist.", "node", node.id))
            continue
        if node.level != parent.level + 1:
            issues.append(
                _issue(
                    Severity.ERROR, "level_inconsistent",
                    f"'{node.title}' is at level {node.level} but its parent is at level {parent.level}.",
                    "node", node.id,
                )
            )
        if parent.sequence >= node.sequence:
            issues.append(_issue(Severity.ERROR, "parent_after_child", f"The parent of '{node.title}' comes after it.", "node", node.id))
        if parent.rank != RANK_CONTAINER and node.rank - parent.rank > 10:
            issues.append(
                _issue(
                    Severity.INFO, "level_skipped",
                    f"'{node.title}' ({node.node_type}) sits directly under '{parent.title}' "
                    f"({parent.node_type}), skipping a level.",
                    "node", node.id,
                )
            )
        _check_within(node, parent, issues)

    for node in nodes:
        seen: set[uuid.UUID] = set()
        cursor: NodeDraft | None = node
        while cursor is not None and cursor.parent_id is not None:
            if cursor.id in seen:
                issues.append(_issue(Severity.ERROR, "circular_hierarchy", f"'{node.title}' is part of a cycle.", "node", node.id))
                break
            seen.add(cursor.id)
            cursor = by_id.get(cursor.parent_id)

    for parent_id, siblings in children.items():
        signatures: dict[tuple[str, str, str | None], NodeDraft] = {}
        for node in siblings:
            signature = (node.node_type, normalise_for_match(node.title), node.number)
            if signature in signatures and node.metadata.get("synthetic") is not True:
                issues.append(
                    _issue(
                        Severity.WARNING, "duplicate_node",
                        f"Another {node.node_type} titled '{node.title}' exists in the same place.",
                        "node", node.id, first=str(signatures[signature].id),
                    )
                )
            signatures.setdefault(signature, node)
        issues.extend(_check_numbering(siblings))

    for node in nodes:
        if (
            node.confidence is not None
            and node.confidence < _LOW_CONFIDENCE
            and node.metadata.get("status") != "ambiguous"
            and node.metadata.get("method") != "fallback"
        ):
            issues.append(
                _issue(
                    Severity.WARNING, "low_confidence_node",
                    f"'{node.title}' was classified with low confidence ({node.confidence:.2f}).",
                    "node", node.id,
                )
            )
    return issues


def _validate_empty_nodes(nodes: list[NodeDraft], paragraphs: list[ParagraphDraft]) -> list[ValidationIssue]:
    """A heading with no text anywhere below it is suspicious (a misdetected heading, or lost text)."""
    with_text = {p.node_id for p in paragraphs}
    by_parent: dict[uuid.UUID | None, list[NodeDraft]] = defaultdict(list)
    for node in nodes:
        by_parent[node.parent_id].append(node)

    def has_text(node: NodeDraft) -> bool:
        return node.id in with_text or any(has_text(child) for child in by_parent.get(node.id, []))

    return [
        _issue(Severity.WARNING, "empty_node", f"'{node.title}' has no text under it.", "node", node.id)
        for node in nodes
        if not has_text(node) and not node.metadata.get("synthetic") and node.node_type != "book_title"
    ]


def _check_within(node: NodeDraft, parent: NodeDraft, issues: list[ValidationIssue]) -> None:
    """A child lies within its parent's blocks and offsets."""
    if None in (node.start_block, node.end_block, parent.start_block, parent.end_block):
        return
    assert node.start_block is not None and node.end_block is not None
    assert parent.start_block is not None and parent.end_block is not None
    if node.start_block < parent.start_block or node.end_block > parent.end_block:
        issues.append(
            _issue(Severity.ERROR, "child_outside_parent", f"'{node.title}' extends outside '{parent.title}'.", "node", node.id)
        )
    elif None not in (node.start_offset, node.end_offset, parent.start_offset, parent.end_offset):
        if node.start_offset < parent.start_offset or node.end_offset > parent.end_offset:  # type: ignore[operator]
            issues.append(
                _issue(Severity.ERROR, "child_outside_parent", f"'{node.title}' extends outside '{parent.title}' (offsets).", "node", node.id)
            )


def _check_numbering(siblings: list[NodeDraft]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_type: dict[str, list[NodeDraft]] = defaultdict(list)
    for node in siblings:
        value = (node.metadata.get("numbering") or {}).get("value")
        if value is not None and node.number is not None and "." not in node.number:
            by_type[node.node_type].append(node)
    for node_type, group in by_type.items():
        previous: NodeDraft | None = None
        for node in group:
            value = node.metadata["numbering"]["value"]
            if previous is not None:
                before = previous.metadata["numbering"]["value"]
                if value == before:
                    issues.append(_issue(Severity.WARNING, "duplicate_number", f"Two {node_type}s are numbered {node.number}.", "node", node.id))
                elif value < before:
                    issues.append(_issue(Severity.WARNING, "numbering_out_of_order", f"{node_type.capitalize()} {node.number} comes after {previous.number}.", "node", node.id))
                elif value > before + 1:
                    issues.append(_issue(Severity.INFO, "numbering_gap", f"{node_type.capitalize()} numbering jumps from {previous.number} to {node.number}.", "node", node.id))
            previous = node
    return issues


def _validate_node_pages(document: DocumentText, nodes: list[NodeDraft]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_id = {n.id: n for n in nodes}
    missing: list[str] = []
    for node in nodes:
        start, end = node.page_start, node.page_end
        if start is not None and end is not None and start > end:
            issues.append(_issue(Severity.ERROR, "page_range_inverted", f"'{node.title}' starts on page {start} but ends on page {end}.", "node", node.id))
        for page in (start, end):
            if page is not None and (page < 1 or (document.page_count and page > document.page_count)):
                issues.append(
                    _issue(Severity.ERROR, "page_out_of_range", f"'{node.title}' refers to page {page}, outside the document's {document.page_count} pages.", "node", node.id, page=page)
                )
        parent = by_id.get(node.parent_id) if node.parent_id else None
        if parent and None not in (start, end, parent.page_start, parent.page_end):
            if start < parent.page_start or end > parent.page_end:  # type: ignore[operator]
                issues.append(_issue(Severity.WARNING, "child_pages_outside_parent", f"The pages of '{node.title}' fall outside those of '{parent.title}'.", "node", node.id))
        if document.has_pages and start is None and node.end_block is not None:
            missing.append(str(node.id))
    if missing:
        issues.append(
            _issue(Severity.WARNING, "missing_page_reference", f"{len(missing)} structural nodes have no page reference.", "document", None, node_ids=missing[:_MAX_LISTED], count=len(missing))
        )
    return issues


def _validate_paragraph_pages(
    document: DocumentText, nodes: list[NodeDraft], paragraphs: list[ParagraphDraft]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    missing: list[str] = []
    node_ids = {n.id for n in nodes}
    for paragraph in paragraphs:
        if paragraph.node_id not in node_ids:
            issues.append(_issue(Severity.ERROR, "orphan_paragraph", "A paragraph refers to a node that does not exist.", "paragraph", paragraph.id))
        start, end = paragraph.page_start, paragraph.page_end
        if start is not None and end is not None and start > end:
            issues.append(_issue(Severity.ERROR, "page_range_inverted", "A paragraph starts after it ends.", "paragraph", paragraph.id))
        for page in (start, end):
            if page is not None and (page < 1 or (document.page_count and page > document.page_count)):
                issues.append(_issue(Severity.ERROR, "page_out_of_range", f"A paragraph refers to page {page}, outside the document.", "paragraph", paragraph.id, page=page))
        if document.has_pages and start is None:
            missing.append(str(paragraph.id))
    if missing:
        issues.append(
            _issue(Severity.WARNING, "missing_page_reference", f"{len(missing)} paragraphs have no page reference.", "document", None, paragraph_ids=missing[:_MAX_LISTED], count=len(missing))
        )
    return issues


def _validate_content(
    document: DocumentText, nodes: list[NodeDraft], paragraphs: list[ParagraphDraft]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    blocks = {b.sequence: b for b in document.blocks}
    expected = {seq for seq, b in blocks.items() if b.text.strip()}

    covered: dict[int, int] = defaultdict(int)
    for node in nodes:
        for sequence in node.heading_blocks:
            covered[sequence] += 1
    for paragraph in paragraphs:
        for sequence in paragraph.block_sequences:
            covered[sequence] += 1
    missing = sorted(expected - set(covered))
    duplicated = sorted(seq for seq, count in covered.items() if count > 1)
    if missing:
        issues.append(_issue(Severity.ERROR, "content_missing", f"{len(missing)} extracted blocks are in no node or paragraph.", "document", None, block_sequences=missing[:_MAX_LISTED], count=len(missing)))
    if duplicated:
        issues.append(_issue(Severity.ERROR, "content_duplicated", f"{len(duplicated)} extracted blocks appear more than once (for example a heading repeated as body text).", "document", None, block_sequences=duplicated[:_MAX_LISTED], count=len(duplicated)))

    for paragraph in paragraphs:
        source = "".join(blocks[s].text for s in paragraph.block_sequences if s in blocks)
        if _squash(source) != _squash(paragraph.text):
            issues.append(_issue(Severity.ERROR, "paragraph_text_mismatch", "A paragraph's text differs from its source blocks.", "paragraph", paragraph.id))
        if not paragraph.text.strip():
            issues.append(_issue(Severity.ERROR, "empty_paragraph", "A paragraph has no text.", "paragraph", paragraph.id))

    first_paragraph: dict[uuid.UUID, ParagraphDraft] = {}
    for paragraph in paragraphs:
        if paragraph.node_id not in first_paragraph or paragraph.sequence < first_paragraph[paragraph.node_id].sequence:
            first_paragraph[paragraph.node_id] = paragraph
    for node in nodes:
        paragraph = first_paragraph.get(node.id)
        if paragraph is None:
            continue
        heading_keys = {normalise_for_match(t) for t in (node.title, node.original_title) if t}
        if normalise_for_match(paragraph.text) in heading_keys and paragraph.kind in ("paragraph", "heading"):
            issues.append(_issue(Severity.WARNING, "heading_duplicated_in_body", f"The heading '{node.title}' is repeated as the first paragraph of its own section.", "node", node.id))
    return issues


# -- passages -------------------------------------------------------------------------------


def validate_passages(
    document: DocumentText,
    nodes: list[NodeDraft],
    paragraphs: list[ParagraphDraft],
    passages: list[PassageDraft],
    config: SegmentationConfig,
    tokenizer: Tokenizer,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    node_ids = {n.id for n in nodes}
    paragraphs_by_id = {p.id: p for p in paragraphs}
    by_node: dict[uuid.UUID, list[PassageDraft]] = defaultdict(list)
    for passage in passages:
        by_node[passage.node_id].append(passage)

    previous_document_sequence = -1
    missing_pages: list[str] = []
    for passage in sorted(passages, key=lambda p: p.document_sequence):
        if passage.document_sequence <= previous_document_sequence:
            issues.append(_issue(Severity.ERROR, "passage_order", "Passages are not in a consistent document order.", "passage", passage.id))
        previous_document_sequence = passage.document_sequence

        if passage.node_id not in node_ids:
            issues.append(_issue(Severity.ERROR, "passage_node_missing", "A passage refers to a structural node that does not exist.", "passage", passage.id))
        if not passage.text.strip():
            issues.append(_issue(Severity.ERROR, "empty_passage", "A passage has no text.", "passage", passage.id))
        expected_tokens = tokenizer.count(passage.text)
        if passage.token_count != expected_tokens:
            issues.append(_issue(Severity.ERROR, "token_count_mismatch", f"A passage records {passage.token_count} tokens but has {expected_tokens}.", "passage", passage.id, recorded=passage.token_count, actual=expected_tokens))

        start, end = passage.page_start, passage.page_end
        if start is not None and end is not None and start > end:
            issues.append(_issue(Severity.ERROR, "page_range_inverted", "A passage starts after it ends.", "passage", passage.id))
        for page in (start, end):
            if page is not None and (page < 1 or (document.page_count and page > document.page_count)):
                issues.append(_issue(Severity.ERROR, "page_out_of_range", f"A passage refers to page {page}, outside the document.", "passage", passage.id, page=page))
        if document.has_pages and start is None:
            missing_pages.append(str(passage.id))

        parts = passage.parts
        linked = [paragraphs_by_id.get(part.paragraph_id) for part in parts]
        if not parts or any(p is None for p in linked):
            issues.append(_issue(Severity.ERROR, "passage_paragraph_invalid", "A passage refers to a paragraph that does not exist.", "passage", passage.id))
        else:
            if any(p.node_id != passage.node_id for p in linked if p):
                issues.append(_issue(Severity.ERROR, "passage_paragraph_invalid", "A passage includes a paragraph from a different section.", "passage", passage.id))
            ranges = [(p.page_start, p.page_end) for p in linked if p and p.page_start is not None and p.page_end is not None]
            if ranges and start is not None and end is not None:
                if start < min(s for s, _ in ranges) or end > max(e for _, e in ranges):
                    issues.append(_issue(Severity.ERROR, "passage_pages_inconsistent", "A passage's pages do not match its paragraphs' pages.", "passage", passage.id))

    if missing_pages:
        issues.append(_issue(Severity.WARNING, "missing_page_reference", f"{len(missing_pages)} passages have no page reference.", "document", None, passage_ids=missing_pages[:_MAX_LISTED], count=len(missing_pages)))

    for node_id, node_passages in by_node.items():
        node_passages.sort(key=lambda p: p.sequence)
        if [p.sequence for p in node_passages] != list(range(len(node_passages))):
            issues.append(_issue(Severity.ERROR, "passage_sequence_gap", "The passages of a section are not numbered consecutively.", "node", node_id))
        issues.extend(_validate_node_passages(node_id, node_passages, paragraphs, config, tokenizer))

    covered_nodes = set(by_node)
    for node_id in {p.node_id for p in paragraphs} - covered_nodes:
        issues.append(_issue(Severity.ERROR, "node_without_passages", "A section has paragraphs but no passages (text would be omitted).", "node", node_id))
    return issues


def _validate_node_passages(
    node_id: uuid.UUID,
    node_passages: list[PassageDraft],
    paragraphs: list[ParagraphDraft],
    config: SegmentationConfig,
    tokenizer: Tokenizer,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    expected = _squash("".join(p.text for p in sorted((p for p in paragraphs if p.node_id == node_id), key=lambda p: p.sequence)))
    bodies: list[str] = []
    for index, passage in enumerate(node_passages):
        overlap_chars = int(passage.metadata.get("overlap_chars") or 0)
        bodies.append(passage.text[overlap_chars:])
        if overlap_chars == 0:
            continue  # overlap is legitimately empty when, for example, the previous passage ended in a table
        if index == 0:
            issues.append(_issue(Severity.ERROR, "overlap_without_predecessor", "The first passage of a section starts with overlap.", "passage", passage.id))
            continue
        overlap_text = passage.text[: max(0, overlap_chars - len(PARAGRAPH_SEPARATOR))]
        if not _squash(node_passages[index - 1].text).endswith(_squash(overlap_text)):
            issues.append(_issue(Severity.ERROR, "overlap_not_from_previous", "Repeated text is not the end of the previous passage.", "passage", passage.id))
        overlap_tokens = tokenizer.count(overlap_text)
        if overlap_tokens > config.overlap_max_tokens:
            issues.append(_issue(Severity.ERROR, "overlap_exceeds_policy", f"Overlap is {overlap_tokens} tokens; the policy allows {config.overlap_max_tokens}.", "passage", passage.id, overlap_tokens=overlap_tokens))
    actual = _squash("".join(bodies))
    if actual != expected:
        issues.append(
            _issue(
                Severity.ERROR,
                "passage_text_mismatch",
                "The passages of a section do not contain exactly the section's text (something is omitted or duplicated).",
                "node", node_id, expected_characters=len(expected), actual_characters=len(actual),
            )
        )

    for passage in node_passages:
        body_tokens = int(passage.metadata.get("body_tokens") or passage.token_count)
        if passage.token_count > config.max_tokens + config.overlap_max_tokens and not passage.metadata.get("oversized"):
            issues.append(_issue(Severity.WARNING, "passage_exceeds_maximum", f"A passage has {passage.token_count} tokens, above the maximum of {config.max_tokens}.", "passage", passage.id))
        elif passage.metadata.get("oversized"):
            issues.append(_issue(Severity.INFO, "oversized_passage", f"A passage of {passage.token_count} tokens exceeds the maximum to keep a {passage.metadata['oversized']} whole.", "passage", passage.id))
        if len(node_passages) > 1 and body_tokens < config.min_tokens:
            issues.append(_issue(Severity.INFO, "undersized_passage", f"A passage has only {body_tokens} tokens (minimum {config.min_tokens}); its neighbours are too large to merge with.", "passage", passage.id))
    return issues
