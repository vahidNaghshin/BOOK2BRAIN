"""Helpers for presenting the structural tree: visible rows for an expandable outline."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, TypeVar


class _NodeLike(Protocol):
    id: uuid.UUID
    parent_id: uuid.UUID | None
    sequence: int
    level: int
    node_type: str


N = TypeVar("N", bound=_NodeLike)


@dataclass
class TreeRow:
    node: _NodeLike
    depth: int
    has_children: bool
    expanded: bool


def children_by_parent(nodes: list[N]) -> dict[uuid.UUID | None, list[N]]:
    children: dict[uuid.UUID | None, list[N]] = defaultdict(list)
    for node in sorted(nodes, key=lambda n: n.sequence):
        children[node.parent_id].append(node)
    return children


def visible_rows(nodes: list[N], expanded: set[uuid.UUID]) -> list[TreeRow]:
    """Nodes in document order, skipping the descendants of collapsed nodes."""
    children = children_by_parent(nodes)
    rows: list[TreeRow] = []

    def walk(parent_id: uuid.UUID | None, depth: int) -> None:
        for node in children.get(parent_id, []):
            has_children = bool(children.get(node.id))
            is_open = node.id in expanded
            rows.append(TreeRow(node, depth, has_children, is_open))
            if has_children and is_open:
                walk(node.id, depth + 1)

    walk(None, 0)
    return rows


def descendant_ids(nodes: list[N], node_id: uuid.UUID) -> list[uuid.UUID]:
    """The node's id followed by the ids of all its descendants, in document order."""
    children = children_by_parent(nodes)
    result = [node_id]

    def walk(parent_id: uuid.UUID) -> None:
        for child in children.get(parent_id, []):
            result.append(child.id)
            walk(child.id)

    walk(node_id)
    return result


def ancestor_ids(nodes: list[N], node_id: uuid.UUID) -> list[uuid.UUID]:
    by_id = {n.id: n for n in nodes}
    result: list[uuid.UUID] = []
    cursor = by_id.get(node_id)
    while cursor is not None and cursor.parent_id is not None and cursor.parent_id not in result:
        result.append(cursor.parent_id)
        cursor = by_id.get(cursor.parent_id)
    return result


_OPEN_BY_DEFAULT = frozenset({"front_matter", "back_matter", "part"})


def default_expanded(nodes: list[N]) -> set[uuid.UUID]:
    """Front matter, back matter and parts start open, so every chapter is visible but its
    sections stay collapsed until the reader opens them."""
    return {n.id for n in nodes if n.node_type in _OPEN_BY_DEFAULT}
