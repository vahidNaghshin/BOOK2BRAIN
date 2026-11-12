import uuid
from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from app.structure.models import NodeDraft, ParagraphDraft, PassageDraft
from database.models import (
    Document,
    Equation,
    ExtractedBlock,
    Paragraph,
    Passage,
    PassageParagraph,
    ProcessingRun,
    ProcessingRunStatus,
    StructuralNode,
    StructureAnalysis,
)
from database.models.base import utcnow

_INSERT_BATCH = 1000


class StructureRepository:
    """Structural nodes, paragraphs, passages and the per-document analysis record."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # -- source data (Day 3) --------------------------------------------------------

    def latest_completed_run(self, document_id: uuid.UUID) -> ProcessingRun | None:
        stmt = (
            select(ProcessingRun)
            .where(
                ProcessingRun.document_id == document_id,
                ProcessingRun.status == ProcessingRunStatus.COMPLETED.value,
            )
            .order_by(ProcessingRun.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def blocks(self, document_id: uuid.UUID) -> list[ExtractedBlock]:
        stmt = (
            select(ExtractedBlock)
            .where(ExtractedBlock.document_id == document_id)
            .order_by(ExtractedBlock.sequence_number)
        )
        return list(self.session.execute(stmt).scalars().all())

    def equation_ids_by_block(self, document_id: uuid.UUID) -> dict[uuid.UUID, uuid.UUID]:
        stmt = select(Equation.extracted_block_id, Equation.id).where(
            Equation.document_id == document_id, Equation.extracted_block_id.is_not(None)
        )
        return {block_id: equation_id for block_id, equation_id in self.session.execute(stmt)}

    # -- writes ----------------------------------------------------------------------

    def replace_structure(
        self,
        document: Document,
        nodes: list[NodeDraft],
        paragraphs: list[ParagraphDraft],
    ) -> None:
        """Replace the document's nodes and paragraphs (and, by cascade, its passages).

        One transaction, caller commits: there is never a moment with old and new records side
        by side, which is what makes repeating the analysis safe.
        """
        self.session.execute(
            StructuralNode.__table__.delete().where(StructuralNode.document_id == document.id)
        )
        node_rows = [
            {
                "id": node.id,
                "book_id": document.book_id,
                "document_id": document.id,
                "parent_id": node.parent_id,
                "node_type": node.node_type,
                "title": node.title[:1000],
                "original_title": node.original_title[:1000] if node.original_title else None,
                "number": node.number,
                "level": node.level,
                "sequence": node.sequence,
                "page_start": node.page_start,
                "page_end": node.page_end,
                "start_offset": node.start_offset,
                "end_offset": node.end_offset,
                "confidence": node.confidence,
                "node_metadata": node.metadata or None,
            }
            for node in sorted(nodes, key=lambda n: n.sequence)
        ]
        # Parents come before children in document order, so one batch never references a later row.
        self._insert(StructuralNode, node_rows)
        self._insert(
            Paragraph,
            [
                {
                    "id": p.id,
                    "book_id": document.book_id,
                    "document_id": document.id,
                    "structural_node_id": p.node_id,
                    "sequence": p.sequence,
                    "document_sequence": p.document_sequence,
                    "kind": p.kind,
                    "text": p.text,
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                    "page_status": p.page_status,
                    "start_offset": p.start_offset,
                    "end_offset": p.end_offset,
                    "source_metadata": p.metadata or None,
                }
                for p in paragraphs
            ],
        )

    def replace_passages(self, document: Document, passages: list[PassageDraft]) -> None:
        """Replace the document's passages. Nodes and paragraphs are not touched. Caller commits."""
        self.session.execute(Passage.__table__.delete().where(Passage.document_id == document.id))
        self._insert(
            Passage,
            [
                {
                    "id": p.id,
                    "book_id": document.book_id,
                    "document_id": document.id,
                    "structural_node_id": p.node_id,
                    "text": p.text,
                    "sequence": p.sequence,
                    "document_sequence": p.document_sequence,
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                    "page_status": p.page_status,
                    "token_count": p.token_count,
                    "start_offset": p.start_offset,
                    "end_offset": p.end_offset,
                    "passage_metadata": p.metadata or None,
                }
                for p in passages
            ],
        )
        self._insert(
            PassageParagraph,
            [
                {
                    "passage_id": p.id,
                    "position": part.position,
                    "paragraph_id": part.paragraph_id,
                    "is_overlap": part.is_overlap,
                    "text_start": part.text_start,
                    "text_end": part.text_end,
                }
                for p in passages
                for part in p.parts
            ],
        )

    def _insert(self, model: type, rows: list[dict[str, Any]]) -> None:
        for start in range(0, len(rows), _INSERT_BATCH):
            self.session.execute(insert(model), rows[start : start + _INSERT_BATCH])

    # -- analysis record ---------------------------------------------------------------

    def get_analysis(self, document_id: uuid.UUID) -> StructureAnalysis | None:
        stmt = select(StructureAnalysis).where(StructureAnalysis.document_id == document_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def update_analysis(self, document_id: uuid.UUID, **fields: Any) -> StructureAnalysis:
        analysis = self.get_analysis(document_id)
        if analysis is None:
            analysis = StructureAnalysis(document_id=document_id)
            self.session.add(analysis)
        for name, value in fields.items():
            setattr(analysis, name, value)
        analysis.updated_at = utcnow()
        self.session.flush()
        return analysis

    # -- reads -----------------------------------------------------------------------

    def nodes(self, document_id: uuid.UUID) -> list[StructuralNode]:
        stmt = (
            select(StructuralNode)
            .where(StructuralNode.document_id == document_id)
            .order_by(StructuralNode.sequence)
        )
        return list(self.session.execute(stmt).scalars().all())

    def paragraphs(
        self,
        document_id: uuid.UUID,
        *,
        node_ids: Iterable[uuid.UUID] | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[Paragraph]:
        stmt = select(Paragraph).where(Paragraph.document_id == document_id)
        if node_ids is not None:
            stmt = stmt.where(Paragraph.structural_node_id.in_(list(node_ids)))
        stmt = stmt.order_by(Paragraph.document_sequence).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def count_paragraphs(self, document_id: uuid.UUID, node_ids: Iterable[uuid.UUID] | None = None) -> int:
        stmt = select(func.count()).select_from(Paragraph).where(Paragraph.document_id == document_id)
        if node_ids is not None:
            stmt = stmt.where(Paragraph.structural_node_id.in_(list(node_ids)))
        return self.session.scalar(stmt) or 0

    def passages(
        self,
        document_id: uuid.UUID,
        *,
        node_ids: Iterable[uuid.UUID] | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[Passage]:
        stmt = select(Passage).where(Passage.document_id == document_id)
        if node_ids is not None:
            stmt = stmt.where(Passage.structural_node_id.in_(list(node_ids)))
        stmt = stmt.order_by(Passage.document_sequence).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def count_passages(self, document_id: uuid.UUID, node_ids: Iterable[uuid.UUID] | None = None) -> int:
        stmt = select(func.count()).select_from(Passage).where(Passage.document_id == document_id)
        if node_ids is not None:
            stmt = stmt.where(Passage.structural_node_id.in_(list(node_ids)))
        return self.session.scalar(stmt) or 0

    def passage_paragraph_ids(self, passage_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[uuid.UUID]]:
        if not passage_ids:
            return {}
        stmt = (
            select(PassageParagraph.passage_id, PassageParagraph.paragraph_id)
            .where(PassageParagraph.passage_id.in_(passage_ids))
            .order_by(PassageParagraph.passage_id, PassageParagraph.position)
        )
        result: dict[uuid.UUID, list[uuid.UUID]] = {}
        for passage_id, paragraph_id in self.session.execute(stmt):
            result.setdefault(passage_id, []).append(paragraph_id)
        return result

    def paragraph_counts_by_node(self, document_id: uuid.UUID) -> dict[uuid.UUID, int]:
        stmt = (
            select(Paragraph.structural_node_id, func.count())
            .where(Paragraph.document_id == document_id)
            .group_by(Paragraph.structural_node_id)
        )
        return {node_id: count for node_id, count in self.session.execute(stmt)}

    def passage_counts_by_node(self, document_id: uuid.UUID) -> dict[uuid.UUID, int]:
        stmt = (
            select(Passage.structural_node_id, func.count())
            .where(Passage.document_id == document_id)
            .group_by(Passage.structural_node_id)
        )
        return {node_id: count for node_id, count in self.session.execute(stmt)}
