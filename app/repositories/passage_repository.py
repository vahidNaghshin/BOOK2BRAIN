"""Read access to stored passages and the structure around them, for knowledge extraction.

Passages belong to Phase 1 and are never created, changed or deleted here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.services.errors import ExtractionScopeError
from database.models import Book, Document, Passage, StructuralNode

CHAPTER_TYPES = ("chapter",)
SECTION_TYPES = ("section", "subsection")


@dataclass(frozen=True)
class ExtractionScope:
    """What an extraction run covers. A chapter or section includes everything inside it."""

    kind: Literal["book", "document", "chapter", "section"]
    book_id: uuid.UUID
    document_id: uuid.UUID | None = None
    node_id: uuid.UUID | None = None


@dataclass(frozen=True)
class PassageSource:
    """A passage with where it sits in the book."""

    passage: Passage
    chapter: StructuralNode | None
    section: StructuralNode | None


class PassageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session
        self._nodes: dict[uuid.UUID, StructuralNode | None] = {}

    # -- scopes ------------------------------------------------------------------------------------

    def book_scope(self, book_id: uuid.UUID, document_id: uuid.UUID | None = None) -> ExtractionScope:
        if self.session.get(Book, book_id) is None:
            raise ExtractionScopeError("The book does not exist.")
        if document_id is None:
            return ExtractionScope("book", book_id)
        document = self.session.get(Document, document_id)
        if document is None or document.book_id != book_id:
            raise ExtractionScopeError("The document does not belong to this book.")
        return ExtractionScope("document", book_id, document_id)

    def chapter_scope(self, node_id: uuid.UUID) -> ExtractionScope:
        return self._node_scope(node_id, "chapter", CHAPTER_TYPES)

    def section_scope(self, node_id: uuid.UUID) -> ExtractionScope:
        return self._node_scope(node_id, "section", SECTION_TYPES)

    def _node_scope(self, node_id: uuid.UUID, kind: Literal["chapter", "section"], types: tuple[str, ...]) -> ExtractionScope:
        node = self.session.get(StructuralNode, node_id)
        if node is None:
            raise ExtractionScopeError(f"The {kind} does not exist.")
        if node.node_type not in types:
            raise ExtractionScopeError(f"'{node.title}' is a {node.node_type.replace('_', ' ')}, not a {kind}.")
        return ExtractionScope(kind, node.book_id, node.document_id, node.id)

    def scope_from_ids(
        self, book_id: uuid.UUID, document_id: uuid.UUID | None, node_id: uuid.UUID | None, kind: str | None
    ) -> ExtractionScope:
        """Rebuild a scope that was stored on a run."""
        if node_id is not None:
            return self.section_scope(node_id) if kind == "section" else self.chapter_scope(node_id)
        return self.book_scope(book_id, document_id)

    # -- passages ----------------------------------------------------------------------------------

    def select(self, scope: ExtractionScope, *, limit: int | None = None, offset: int = 0) -> list[Passage]:
        """The passages of a scope in reading order (documents oldest first, then position in the document)."""
        query = (
            select(Passage)
            .join(Document, Document.id == Passage.document_id)
            .where(Passage.book_id == scope.book_id)
            .order_by(Document.created_at, Document.id, Passage.document_sequence)
            .offset(offset)
        )
        query = self._restrict(query, scope)
        if limit is not None:
            query = query.limit(limit)
        return list(self.session.scalars(query))

    def count(self, scope: ExtractionScope) -> int:
        query = select(func.count()).select_from(Passage).where(Passage.book_id == scope.book_id)
        return self.session.scalar(self._restrict(query, scope)) or 0

    def _restrict(self, query, scope: ExtractionScope):
        if scope.document_id is not None:
            query = query.where(Passage.document_id == scope.document_id)
        if scope.node_id is not None:
            query = query.where(Passage.structural_node_id.in_(self.subtree_ids(scope.node_id)))
        return query

    def subtree_ids(self, node_id: uuid.UUID) -> list[uuid.UUID]:
        node = self.session.get(StructuralNode, node_id)
        if node is None:
            return []
        rows = self.session.execute(
            select(StructuralNode.id, StructuralNode.parent_id).where(StructuralNode.document_id == node.document_id)
        )
        children: dict[uuid.UUID | None, list[uuid.UUID]] = {}
        for child_id, parent_id in rows:
            children.setdefault(parent_id, []).append(child_id)
        found, queue = [], [node_id]
        while queue:
            current = queue.pop()
            found.append(current)
            queue.extend(children.get(current, []))
        return found

    def neighbours(self, passage: Passage, count: int) -> tuple[list[Passage], list[Passage]]:
        """Up to `count` passages before and after, in the same document, in reading order."""
        if count <= 0:
            return [], []
        around = self.session.scalars(
            select(Passage)
            .where(
                Passage.document_id == passage.document_id,
                Passage.document_sequence.between(passage.document_sequence - count, passage.document_sequence + count),
                Passage.id != passage.id,
            )
            .order_by(Passage.document_sequence)
        )
        before: list[Passage] = []
        after: list[Passage] = []
        for other in around:
            (before if other.document_sequence < passage.document_sequence else after).append(other)
        return before, after

    # -- structure ---------------------------------------------------------------------------------

    def get_book(self, book_id: uuid.UUID) -> Book | None:
        return self.session.get(Book, book_id)

    def source_of(self, passage: Passage) -> PassageSource:
        """The chapter and section a passage belongs to: its own node and the nearest ones above it."""
        chapter = section = None
        node = self._node(passage.structural_node_id)
        while node is not None:
            if node.node_type in CHAPTER_TYPES and chapter is None:
                chapter = node
            elif node.node_type in SECTION_TYPES and section is None:
                section = node
            node = self._node(node.parent_id) if node.parent_id else None
        return PassageSource(passage, chapter, section)

    def _node(self, node_id: uuid.UUID) -> StructuralNode | None:
        if node_id not in self._nodes:
            self._nodes[node_id] = self.session.get(StructuralNode, node_id)
        return self._nodes[node_id]

    def outline(self, document_id: uuid.UUID) -> list[tuple[StructuralNode, int]]:
        """Chapters and sections of a document in order, each with its number of passages (children included)."""
        nodes = list(
            self.session.scalars(
                select(StructuralNode)
                .where(StructuralNode.document_id == document_id, StructuralNode.node_type.in_([*CHAPTER_TYPES, *SECTION_TYPES]))
                .order_by(StructuralNode.sequence)
            )
        )
        direct = dict(
            self.session.execute(
                select(Passage.structural_node_id, func.count()).where(Passage.document_id == document_id).group_by(Passage.structural_node_id)
            ).all()
        )
        all_nodes = list(self.session.scalars(select(StructuralNode).where(StructuralNode.document_id == document_id)))
        children: dict[uuid.UUID | None, list[uuid.UUID]] = {}
        for node in all_nodes:
            children.setdefault(node.parent_id, []).append(node.id)

        def total(node_id: uuid.UUID) -> int:
            return direct.get(node_id, 0) + sum(total(child) for child in children.get(node_id, []))

        return [(node, total(node.id)) for node in nodes]
