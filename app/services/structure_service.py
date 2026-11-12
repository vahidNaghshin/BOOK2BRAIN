import logging
import statistics
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.repositories.book_repository import BookRepository
from app.repositories.structure_repository import StructureRepository
from app.schemas import (
    BookRead,
    ParagraphRead,
    PassageRead,
    StructuralNodeRead,
    StructureAnalysisRead,
    StructureIssueRead,
    StructureOverview,
)
from app.services.errors import (
    DocumentNotFoundError,
    StructureNotAllowedError,
    StructureServiceError,
)
from app.structure.detector import StructureDetector
from app.structure.llm import HeadingClassifier, LlmUnavailableError, classifier_from_settings
from app.structure.models import (
    DocumentText,
    NodeDraft,
    ParagraphDraft,
    PassageDraft,
    Severity,
    SourceBlock,
    ValidationIssue,
)
from app.structure.segmentation import SegmentationConfig, generate_passages
from app.structure.tokenizer import TokenizerUnavailableError
from app.structure.validation import count_by_severity, validate_passages
from database.models import (
    Document,
    Paragraph,
    ProcessingStatus,
    StructuralNode,
    StructureStatus,
)
from database.models.base import utcnow

logger = logging.getLogger(__name__)

_BLOCK_SEPARATOR = len("\n\n")  # canonical text joins blocks with a blank line
_MAX_ERROR_MESSAGE = 2000


class StructureService:
    """Reconstructs a book's hierarchy, paragraphs and passages from the stored Day 3 text.

    Nothing is parsed again: it reads the extracted blocks. Structure detection replaces the
    document's nodes, paragraphs and passages in one transaction (so repeating it never creates
    duplicates). Passages can be regenerated on their own from the stored paragraphs.
    """

    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        classifier: HeadingClassifier | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.classifier = classifier
        self.repo = StructureRepository(session)
        self.books = BookRepository(session)

    # -- queries -------------------------------------------------------------------------

    def default_config(self) -> SegmentationConfig:
        return SegmentationConfig.from_settings(self.settings)

    def get_overview(self, document_id: uuid.UUID) -> StructureOverview:
        document = self._load_document(document_id)
        book = self.books.get(document.book_id)
        if book is None:
            raise DocumentNotFoundError("The book of this document no longer exists.")
        analysis = self.repo.get_analysis(document_id)
        run = self.repo.latest_completed_run(document_id)
        outdated = bool(
            analysis
            and analysis.structure_status == StructureStatus.COMPLETED.value
            and run is not None
            and (analysis.structure_summary or {}).get("source_run_id") != str(run.id)
        )
        return StructureOverview(
            document_id=document_id,
            book=BookRead.model_validate(book),
            analysis=self._read_analysis(analysis) if analysis else None,
            nodes=[StructuralNodeRead.model_validate(n) for n in self.repo.nodes(document_id)],
            paragraph_counts=self.repo.paragraph_counts_by_node(document_id),
            passage_counts=self.repo.passage_counts_by_node(document_id),
            is_outdated=outdated,
        )

    def list_paragraphs(
        self, document_id: uuid.UUID, node_ids: list[uuid.UUID], *, offset: int = 0, limit: int = 20
    ) -> list[ParagraphRead]:
        rows = self.repo.paragraphs(document_id, node_ids=node_ids, offset=offset, limit=limit)
        return [ParagraphRead.model_validate(row) for row in rows]

    def count_paragraphs(self, document_id: uuid.UUID, node_ids: list[uuid.UUID]) -> int:
        return self.repo.count_paragraphs(document_id, node_ids)

    def list_passages(
        self,
        document_id: uuid.UUID,
        node_ids: list[uuid.UUID] | None = None,
        *,
        offset: int = 0,
        limit: int | None = 20,
    ) -> list[PassageRead]:
        rows = self.repo.passages(document_id, node_ids=node_ids, offset=offset, limit=limit)
        links = self.repo.passage_paragraph_ids([row.id for row in rows])
        result = []
        for row in rows:
            passage = PassageRead.model_validate(row)
            passage.paragraph_ids = list(dict.fromkeys(links.get(row.id, [])))
            result.append(passage)
        return result

    def count_passages(self, document_id: uuid.UUID, node_ids: list[uuid.UUID] | None = None) -> int:
        return self.repo.count_passages(document_id, node_ids)

    # -- commands ------------------------------------------------------------------------

    def detect_structure(
        self, document_id: uuid.UUID, config: SegmentationConfig | None = None
    ) -> StructureAnalysisRead:
        """(Re)build nodes, paragraphs and passages from the stored text of a processed document."""
        document = self._lock_processed_document(document_id)
        run = self.repo.latest_completed_run(document_id)
        config = config or self.default_config()
        try:
            text = self._load_text(document, run.result_summary if run else None)
            classifier, llm_issues = self._classifier()
            detector = StructureDetector(
                accept_threshold=self.settings.structure_heading_accept_threshold,
                reject_threshold=self.settings.structure_heading_reject_threshold,
                classifier=classifier,
                llm_max_candidates=self.settings.structure_llm_max_candidates,
            )
            result = detector.detect(text)
            result.issues.extend(llm_issues)
            result.summary["issues"] = count_by_severity(result.issues)
            result.summary["source_run_id"] = str(run.id) if run else None

            tokenizer = config.build_tokenizer()
            passages = generate_passages(result.nodes, result.paragraphs, config, tokenizer)
            passage_issues = validate_passages(text, result.nodes, result.paragraphs, passages, config, tokenizer)

            self.repo.replace_structure(document, result.nodes, result.paragraphs)
            self.repo.replace_passages(document, passages)
            result.summary["applied_book_metadata"] = self._apply_metadata(document, result.metadata_guess)
            result.summary["detected_metadata"] = result.metadata_guess
            self.repo.update_analysis(
                document_id,
                structure_status=StructureStatus.COMPLETED.value,
                structure_error=None,
                structure_detected_at=utcnow(),
                structure_summary=result.summary,
                structure_issues=[i.to_dict() for i in result.issues],
                **self._passage_fields(config, passages, passage_issues),
            )
            self.session.commit()
        except TokenizerUnavailableError as exc:
            self.session.rollback()
            raise StructureServiceError(str(exc)) from exc
        except StructureServiceError:
            self.session.rollback()
            raise
        except Exception as exc:
            self.session.rollback()
            self._record_failure(document_id, "structure", exc)
            raise StructureServiceError(
                f"Structure detection failed unexpectedly ({type(exc).__name__}). See the logs for details."
            ) from exc
        logger.info(
            "Structure of document %s: %d nodes, %d paragraphs, %d passages",
            document_id, len(result.nodes), len(result.paragraphs), len(passages),
        )
        return self._read_analysis(self.repo.get_analysis(document_id))

    def regenerate_passages(
        self, document_id: uuid.UUID, config: SegmentationConfig | None = None
    ) -> StructureAnalysisRead:
        """Rebuild only the passages (e.g. after changing sizes). Structure is not detected again."""
        document = self._lock_processed_document(document_id)
        analysis = self.repo.get_analysis(document_id)
        if analysis is None or analysis.structure_status != StructureStatus.COMPLETED.value:
            self.session.rollback()
            raise StructureNotAllowedError("Detect the book structure before generating passages.")
        config = config or self.default_config()
        try:
            nodes = [self._node_draft(n) for n in self.repo.nodes(document_id)]
            paragraphs = [self._paragraph_draft(p) for p in self.repo.paragraphs(document_id)]
            run = self.repo.latest_completed_run(document_id)
            text = DocumentText(
                blocks=[],
                page_count=(run.result_summary or {}).get("page_count") if run else None,
                file_format=document.file_format,
                known_pages=any(p.page_start is not None for p in paragraphs),
            )
            tokenizer = config.build_tokenizer()
            passages = generate_passages(nodes, paragraphs, config, tokenizer)
            issues = validate_passages(text, nodes, paragraphs, passages, config, tokenizer)
            self.repo.replace_passages(document, passages)
            self.repo.update_analysis(document_id, **self._passage_fields(config, passages, issues))
            self.session.commit()
        except TokenizerUnavailableError as exc:
            self.session.rollback()
            raise StructureServiceError(str(exc)) from exc
        except Exception as exc:
            self.session.rollback()
            self._record_failure(document_id, "passage", exc)
            raise StructureServiceError(
                f"Passage generation failed unexpectedly ({type(exc).__name__}). See the logs for details."
            ) from exc
        logger.info("Regenerated %d passages for document %s", len(passages), document_id)
        return self._read_analysis(self.repo.get_analysis(document_id))

    # -- internals -------------------------------------------------------------------------

    def _load_document(self, document_id: uuid.UUID) -> Document:
        document = self.session.get(Document, document_id)
        if document is None:
            raise DocumentNotFoundError("The document no longer exists.")
        return document

    def _lock_processed_document(self, document_id: uuid.UUID) -> Document:
        stmt = select(Document).where(Document.id == document_id).with_for_update()
        document = self.session.execute(stmt).scalar_one_or_none()
        if document is None:
            self.session.rollback()
            raise DocumentNotFoundError("The document no longer exists.")
        if document.processing_status != ProcessingStatus.COMPLETED.value:
            self.session.rollback()
            raise StructureNotAllowedError(
                "The document must be processed (text extracted) before its structure can be detected."
            )
        return document

    def _classifier(self) -> tuple[HeadingClassifier | None, list[ValidationIssue]]:
        if self.classifier is not None:
            return self.classifier, []
        try:
            return classifier_from_settings(self.settings), []
        except LlmUnavailableError as exc:
            logger.warning("LLM heading classification is not available: %s", exc)
            return None, [
                ValidationIssue(
                    Severity.WARNING,
                    "llm_unavailable",
                    f"LLM heading classification is switched on but unavailable ({exc}). "
                    "Ambiguous headings keep their deterministic classification.",
                    "document",
                )
            ]

    def _load_text(self, document: Document, summary: dict[str, Any] | None) -> DocumentText:
        rows = self.repo.blocks(document.id)
        if not rows:
            raise StructureServiceError("This document has no extracted text. Process it first.")
        blocks: list[SourceBlock] = []
        offset = 0
        for row in rows:
            location = row.source_location or {}
            raw_box = location.get("bbox")
            bbox = tuple(float(v) for v in raw_box) if isinstance(raw_box, list) and len(raw_box) == 4 else None
            metadata = row.block_metadata or {}
            blocks.append(
                SourceBlock(
                    sequence=row.sequence_number,
                    text=row.text,
                    block_type=row.block_type,
                    page=row.page_number,
                    bbox=bbox,  # type: ignore[arg-type]
                    coord_origin=location.get("coord_origin"),
                    heading_level=metadata.get("heading_level"),
                    id=row.id,
                    start_offset=offset,
                    end_offset=offset + len(row.text),
                    metadata=metadata,
                )
            )
            offset += len(row.text) + _BLOCK_SEPARATOR
        summary = summary or {}
        title = (summary.get("metadata") or {}).get("title")
        return DocumentText(
            blocks=blocks,
            page_count=summary.get("page_count"),
            file_format=document.file_format,
            source_title=title if isinstance(title, str) else None,
            equation_ids_by_block=self.repo.equation_ids_by_block(document.id),
        )

    def _apply_metadata(self, document: Document, guess: dict[str, Any]) -> dict[str, Any]:
        """Fill *empty* book fields from detected values when enabled. User values are never replaced."""
        if not self.settings.structure_fill_empty_book_metadata:
            return {}
        book = self.books.get(document.book_id)
        if book is None:
            return {}
        minimum = self.settings.structure_metadata_min_confidence
        applied: dict[str, Any] = {}
        if not book.subtitle and guess.get("subtitle") and (guess.get("subtitle_confidence") or 0) >= minimum:
            book.subtitle = str(guess["subtitle"])[:500]
            applied["subtitle"] = book.subtitle
        author = guess.get("author")
        if not book.author and author and (guess.get("author_confidence") or 0) >= minimum:
            if self.books.find_by_identity(book.title, str(author)) is None:  # never create a duplicate book identity
                book.author = str(author)[:500]
                applied["author"] = book.author
        return applied

    def _passage_fields(
        self, config: SegmentationConfig, passages: list[PassageDraft], issues: list[ValidationIssue]
    ) -> dict[str, Any]:
        counts = [p.token_count for p in passages]
        summary = {
            "passage_count": len(passages),
            "total_tokens": sum(counts),
            "min_tokens": min(counts, default=0),
            "max_tokens": max(counts, default=0),
            "mean_tokens": round(statistics.fmean(counts), 1) if counts else 0,
            "oversized": sum(1 for p in passages if p.metadata.get("oversized")),
            "with_overlap": sum(1 for p in passages if p.metadata.get("overlap_chars")),
            "issues": count_by_severity(issues),
        }
        return {
            "passage_status": StructureStatus.COMPLETED.value,
            "passage_error": None,
            "passage_generated_at": utcnow(),
            "passage_config": config.to_dict(),
            "passage_summary": summary,
            "passage_issues": [i.to_dict() for i in issues],
        }

    def _record_failure(self, document_id: uuid.UUID, step: str, exc: Exception) -> None:
        logger.exception("Structure step '%s' failed for document %s", step, document_id)
        message = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_MESSAGE]
        try:
            self.repo.update_analysis(
                document_id,
                **{f"{step}_status": StructureStatus.FAILED.value, f"{step}_error": message},
            )
            self.session.commit()
        except Exception:
            self.session.rollback()
            logger.exception("Could not record the failure of the '%s' step for document %s", step, document_id)

    @staticmethod
    def _read_analysis(analysis: Any) -> StructureAnalysisRead:
        def issues(raw: list[dict[str, Any]] | None) -> list[StructureIssueRead]:
            return [StructureIssueRead.model_validate(item) for item in raw or []]

        return StructureAnalysisRead(
            structure_status=analysis.structure_status,
            structure_error=analysis.structure_error,
            structure_detected_at=analysis.structure_detected_at,
            structure_summary=analysis.structure_summary,
            structure_issues=issues(analysis.structure_issues),
            passage_status=analysis.passage_status,
            passage_error=analysis.passage_error,
            passage_generated_at=analysis.passage_generated_at,
            passage_config=analysis.passage_config,
            passage_summary=analysis.passage_summary,
            passage_issues=issues(analysis.passage_issues),
        )

    @staticmethod
    def _node_draft(row: StructuralNode) -> NodeDraft:
        return NodeDraft(
            id=row.id,
            node_type=row.node_type,
            title=row.title,
            level=row.level,
            sequence=row.sequence,
            parent_id=row.parent_id,
            original_title=row.original_title,
            number=row.number,
            page_start=row.page_start,
            page_end=row.page_end,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            confidence=row.confidence,
            metadata=row.node_metadata or {},
        )

    @staticmethod
    def _paragraph_draft(row: Paragraph) -> ParagraphDraft:
        metadata = row.source_metadata or {}
        return ParagraphDraft(
            id=row.id,
            node_id=row.structural_node_id,
            sequence=row.sequence,
            document_sequence=row.document_sequence,
            kind=row.kind,
            text=row.text,
            block_sequences=list(metadata.get("block_sequences", [])),
            page_start=row.page_start,
            page_end=row.page_end,
            page_status=row.page_status,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            offsets_exact=bool(metadata.get("offsets_exact", False)),
            metadata=metadata,
        )
