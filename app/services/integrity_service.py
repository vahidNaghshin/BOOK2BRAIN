"""Data integrity validation of what the pipeline saved, read back from the database.

It checks relationships, not quantities: pages, paragraphs, nodes and passages are different kinds
of data and are not expected to match in number. What must hold is that every record belongs to the
right document, points at valid parents, is in a consistent order and keeps valid page references.
Findings are structured issues; any `error` means the run must not be reported as completed.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import Select, and_, func, select
from sqlalchemy.orm import Session, aliased

from app.services.storage_service import StorageBackend
from app.structure.models import Severity, ValidationIssue
from database.models import (
    Book,
    Document,
    ExtractedBlock,
    Paragraph,
    Passage,
    PassageParagraph,
    ProcessingRun,
    StructuralNode,
    StructureAnalysis,
)

logger = logging.getLogger(__name__)

_MAX_LISTED = 20


class IntegrityValidator:
    def __init__(self, session: Session, storage: StorageBackend | None = None) -> None:
        self.session = session
        self.storage = storage

    def validate(self, document_id: uuid.UUID, run_id: uuid.UUID | None = None) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        document = self.session.get(Document, document_id)
        if document is None:
            return [self._issue(Severity.ERROR, "document_missing", "The source document does not exist.")]
        run = self.session.get(ProcessingRun, run_id) if run_id else self.session.scalar(
            select(ProcessingRun)
            .where(ProcessingRun.document_id == document_id, ProcessingRun.result_summary.is_not(None))
            .order_by(ProcessingRun.created_at.desc())
            .limit(1)
        )
        summary = (run.result_summary if run else None) or {}
        page_count = summary.get("page_count")

        issues += self._document(document, run, run_id)
        issues += self._extraction(document, summary, page_count)
        issues += self._structure(document, page_count)
        issues += self._paragraphs(document)
        issues += self._passages(document, page_count)
        issues += self._analysis(document)
        logger.info(
            "Integrity validation of document %s: %d errors, %d warnings",
            document_id,
            sum(1 for i in issues if i.severity is Severity.ERROR),
            sum(1 for i in issues if i.severity is Severity.WARNING),
        )
        return issues

    # -- helpers ----------------------------------------------------------------------------

    @staticmethod
    def _issue(severity: Severity, code: str, message: str, **details: Any) -> ValidationIssue:
        return ValidationIssue(severity, code, message, "document", None, dict(details))

    def _count(self, stmt: Select[Any]) -> int:
        return self.session.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    def _flag(self, stmt: Select[Any], code: str, message: str, severity: Severity = Severity.ERROR) -> list[ValidationIssue]:
        found = self._count(stmt)
        return [self._issue(severity, code, message.format(count=found), count=found)] if found else []

    # -- document and run -------------------------------------------------------------------

    def _document(self, document: Document, run: ProcessingRun | None, run_id: uuid.UUID | None) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if self.session.get(Book, document.book_id) is None:
            issues.append(self._issue(Severity.ERROR, "book_missing", "The document's book does not exist."))
        if run_id is not None and (run is None or run.document_id != document.id):
            issues.append(
                self._issue(Severity.ERROR, "run_wrong_document", "The processing run does not belong to this document.")
            )
        if self.storage is not None and not self.storage.exists(document.file_path):
            issues.append(
                self._issue(Severity.ERROR, "source_file_missing", "The original file is no longer in storage.")
            )
        return issues

    # -- extracted text ---------------------------------------------------------------------

    def _extraction(self, document: Document, summary: dict[str, Any], page_count: int | None) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        blocks = select(ExtractedBlock.id).where(ExtractedBlock.document_id == document.id)
        if not self._count(blocks):
            issues.append(self._issue(Severity.ERROR, "no_extracted_text", "No extracted text is stored for this document."))
            return issues
        if not summary:
            issues.append(self._issue(Severity.ERROR, "no_extraction_summary", "The extraction outcome was not recorded."))
        if page_count:
            issues += self._flag(
                blocks.where(
                    ExtractedBlock.page_number.is_not(None),
                    (ExtractedBlock.page_number < 1) | (ExtractedBlock.page_number > page_count),
                ),
                "block_page_out_of_range",
                "{count} text blocks refer to a page outside the document's pages.",
            )
        empty_pages = (summary.get("metadata") or {}).get("pages_without_content") or []
        if empty_pages:
            issues.append(
                self._issue(Severity.INFO, "empty_pages", f"{len(empty_pages)} pages have no extractable content.", pages=empty_pages[:_MAX_LISTED])
            )
        ocr = summary.get("ocr") or {}
        failed = ocr.get("pages_failed") or []
        if failed or summary.get("status") == "partial":
            issues.append(
                self._issue(
                    Severity.WARNING,
                    "incomplete_extraction",
                    "Some pages could not be read (for example scanned pages whose OCR failed), "
                    "so the extracted text is incomplete.",
                    failed_pages=failed[:_MAX_LISTED],
                )
            )
        return issues

    # -- structure --------------------------------------------------------------------------

    def _structure(self, document: Document, page_count: int | None) -> list[ValidationIssue]:
        nodes = select(StructuralNode.id).where(StructuralNode.document_id == document.id)
        if not self._count(nodes):
            return [self._issue(Severity.ERROR, "no_structure", "No structural nodes are stored for this document.")]
        issues: list[ValidationIssue] = []
        issues += self._flag(nodes.where(StructuralNode.book_id != document.book_id), "node_wrong_book", "{count} structural nodes belong to a different book.")

        parent = aliased(StructuralNode)
        joined = select(StructuralNode.id).join(parent, StructuralNode.parent_id == parent.id)
        issues += self._flag(
            joined.where(StructuralNode.document_id == document.id, parent.document_id != document.id),
            "parent_in_other_document",
            "{count} structural nodes have a parent from another document.",
        )
        issues += self._flag(
            joined.where(StructuralNode.document_id == document.id, StructuralNode.level != parent.level + 1),
            "level_inconsistent",
            "{count} structural nodes are not one level below their parent.",
        )
        issues += self._flag(
            joined.where(StructuralNode.document_id == document.id, StructuralNode.sequence <= parent.sequence),
            "parent_after_child",
            "{count} structural nodes come before their parent in document order.",
        )
        issues += self._flag(
            nodes.where(StructuralNode.parent_id.is_(None), StructuralNode.level != 0),
            "root_level_inconsistent",
            "{count} top-level nodes do not have level 0.",
        )
        low, high, total = self.session.execute(
            select(func.min(StructuralNode.sequence), func.max(StructuralNode.sequence), func.count()).where(
                StructuralNode.document_id == document.id
            )
        ).one()
        if low != 0 or high != total - 1:
            issues.append(self._issue(Severity.ERROR, "node_sequence_gap", "Structural nodes are not numbered consecutively in document order."))

        duplicates = (
            select(func.min(StructuralNode.sequence))
            .where(StructuralNode.document_id == document.id, func.coalesce(StructuralNode.node_metadata["synthetic"].astext, "false") != "true")
            .group_by(
                StructuralNode.parent_id,
                StructuralNode.node_type,
                func.lower(StructuralNode.title),
                func.coalesce(StructuralNode.number, ""),
            )
            .having(func.count() > 1)
        )
        issues += self._flag(duplicates, "duplicate_nodes", "{count} groups of structural nodes look duplicated.", Severity.WARNING)

        issues += self._page_ranges(nodes, StructuralNode, "node", page_count)
        return issues

    def _page_ranges(self, stmt: Select[Any], model: Any, label: str, page_count: int | None) -> list[ValidationIssue]:
        issues = self._flag(
            stmt.where(model.page_start.is_not(None), model.page_end.is_not(None), model.page_end < model.page_start),
            f"{label}_pages_inverted",
            "{count} " + label + "s start on a later page than they end.",
        )
        if page_count:
            issues += self._flag(
                stmt.where(
                    (model.page_start.is_not(None) & (model.page_start > page_count))
                    | (model.page_end.is_not(None) & (model.page_end > page_count))
                ),
                f"{label}_page_out_of_range",
                "{count} " + label + "s refer to a page beyond the document's last page.",
            )
        return issues

    # -- paragraphs and passages --------------------------------------------------------------

    def _paragraphs(self, document: Document) -> list[ValidationIssue]:
        paragraphs = select(Paragraph.id).where(Paragraph.document_id == document.id)
        issues = self._flag(paragraphs.where(Paragraph.book_id != document.book_id), "paragraph_wrong_book", "{count} paragraphs belong to a different book.")
        node = aliased(StructuralNode)
        issues += self._flag(
            paragraphs.join(node, Paragraph.structural_node_id == node.id).where(node.document_id != document.id),
            "paragraph_in_other_document",
            "{count} paragraphs belong to a node of another document.",
        )
        issues += self._flag(
            paragraphs.where(func.length(func.trim(Paragraph.text)) == 0),
            "empty_paragraph",
            "{count} paragraphs have no text.",
        )
        return issues

    def _passages(self, document: Document, page_count: int | None) -> list[ValidationIssue]:
        passages = select(Passage.id).where(Passage.document_id == document.id)
        if not self._count(passages):
            return [self._issue(Severity.ERROR, "no_passages", "No passages are stored for this document.")]
        issues = self._flag(passages.where(Passage.book_id != document.book_id), "passage_wrong_book", "{count} passages belong to a different book.")
        node = aliased(StructuralNode)
        issues += self._flag(
            passages.join(node, Passage.structural_node_id == node.id).where(node.document_id != document.id),
            "passage_in_other_document",
            "{count} passages belong to a node of another document.",
        )
        issues += self._flag(passages.where(func.length(func.trim(Passage.text)) == 0), "empty_passage", "{count} passages have no text.")
        issues += self._flag(passages.where(Passage.token_count <= 0), "passage_without_tokens", "{count} passages have no token count.")
        issues += self._page_ranges(passages, Passage, "passage", page_count)

        # Passage order inside each node must be 0..n-1, and document order 0..n-1.
        per_node = (
            select(Passage.structural_node_id)
            .where(Passage.document_id == document.id)
            .group_by(Passage.structural_node_id)
            .having(
                (func.min(Passage.sequence) != 0) | (func.max(Passage.sequence) != func.count() - 1)
            )
        )
        issues += self._flag(per_node, "passage_sequence_gap", "{count} sections have passages that are not numbered consecutively.")
        low, high, total = self.session.execute(
            select(func.min(Passage.document_sequence), func.max(Passage.document_sequence), func.count()).where(
                Passage.document_id == document.id
            )
        ).one()
        if low != 0 or high != total - 1:
            issues.append(self._issue(Severity.ERROR, "passage_order_gap", "Passages are not numbered consecutively over the document."))

        linked = select(PassageParagraph.passage_id).distinct()
        issues += self._flag(
            passages.where(Passage.id.not_in(linked)),
            "passage_without_paragraphs",
            "{count} passages are not linked to any paragraph.",
        )
        paragraph = aliased(Paragraph)
        issues += self._flag(
            select(PassageParagraph.passage_id)
            .join(Passage, Passage.id == PassageParagraph.passage_id)
            .join(paragraph, paragraph.id == PassageParagraph.paragraph_id)
            .where(Passage.document_id == document.id, paragraph.structural_node_id != Passage.structural_node_id),
            "passage_paragraph_mismatch",
            "{count} passage-paragraph links cross sections.",
        )
        with_text = select(Paragraph.structural_node_id).where(Paragraph.document_id == document.id).distinct()
        has_passages = select(Passage.structural_node_id).where(Passage.document_id == document.id).distinct()
        issues += self._flag(
            select(StructuralNode.id).where(StructuralNode.id.in_(with_text), StructuralNode.id.not_in(has_passages)),
            "text_without_passages",
            "{count} sections have text but no passages (the text would be left out).",
        )
        return issues

    # -- results of the structure step's own validation --------------------------------------

    def _analysis(self, document: Document) -> list[ValidationIssue]:
        analysis = self.session.scalar(select(StructureAnalysis).where(StructureAnalysis.document_id == document.id))
        if analysis is None or analysis.structure_status != "completed" or analysis.passage_status != "completed":
            return [self._issue(Severity.ERROR, "structure_not_completed", "Structure detection or passage generation did not complete.")]
        issues: list[ValidationIssue] = []
        failures = [
            item
            for item in [*(analysis.structure_issues or []), *(analysis.passage_issues or [])]
            if item.get("severity") == "error"
        ]
        if failures:
            codes = sorted({item.get("code", "") for item in failures})
            issues.append(
                self._issue(
                    Severity.ERROR,
                    "structure_validation_failed",
                    f"Structure and passage validation reported {len(failures)} errors ({', '.join(codes[:5])}).",
                    codes=codes,
                    count=len(failures),
                )
            )
        return issues
