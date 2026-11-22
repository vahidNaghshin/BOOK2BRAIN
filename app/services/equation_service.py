"""Preserving equations with their source: what was captured, in what state, and with what context.

The model's structured reading of an equation lives in `KnowledgeUnit.details`. What the *source* gave us
(the stored Phase 1 equation, its LaTeX, its image reference, the surrounding text) is kept separately in
`equation_sources`, copied from Phase 1 data and never written by a model. The capture status says what
was actually captured, not whether the mathematics is right. Equations the model could not read, down to
image-only ones, are kept as source-derived units instead of being dropped. Nothing here reconstructs or
rewrites an equation.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.content import EquationContent
from app.knowledge.schemas import KnowledgeUnitCreate, SourceReference
from app.knowledge.types import (
    EquationCaptureStatus,
    ExpressionStatus,
    KnowledgeOrigin,
    KnowledgeType,
    PageStatus,
    ReviewFlagCode,
    SourceRole,
)
from app.repositories.passage_repository import PassageRepository
from database.models import Equation, EquationSource, KnowledgeUnit, Passage

logger = logging.getLogger(__name__)

CONTEXT_CHARS = 400
_STRIP = re.compile(r"[\s$]+")


def _compact(text: str | None) -> str:
    return _STRIP.sub("", text or "").casefold()


def classify_capture(
    *,
    original_text: str | None,
    source_latex: str | None,
    image_reference: str | None,
    model_status: ExpressionStatus | None,
    matched_to_source: bool,
    source_validation: str | None = None,
) -> tuple[EquationCaptureStatus, list[str]]:
    """What was captured, with the reasons. Order matters: a contradiction beats a gap beats a doubt."""
    has_text, has_latex, has_image = bool(original_text), bool(source_latex), bool(image_reference)
    notes: list[str] = []
    if not matched_to_source:
        notes.append("The model's expression could not be matched to any equation or text in the source.")
    if not (has_text or has_latex or has_image):
        notes.append("Nothing of the equation could be taken from the source.")
    if notes:
        return EquationCaptureStatus.NEEDS_REVIEW, notes
    if not (has_text or has_latex):  # checked first: Phase 1 marks these invalid only because nothing was readable
        return EquationCaptureStatus.IMAGE_ONLY, ["Only an image of the equation is available."]
    if source_validation == "invalid":
        return EquationCaptureStatus.NEEDS_REVIEW, ["The source extraction of this equation was marked invalid."]
    if model_status in (ExpressionStatus.PARTIAL, ExpressionStatus.FAILED):
        return EquationCaptureStatus.INCOMPLETE, ["Part of the expression is missing or cut off."]
    if model_status == ExpressionStatus.AMBIGUOUS or source_validation == "uncertain":
        return EquationCaptureStatus.AMBIGUOUS, ["The expression can be read more than one way."]
    if has_text and not has_latex:
        return EquationCaptureStatus.TEXT_ONLY, ["The expression was captured as text; there is no LaTeX form."]
    return EquationCaptureStatus.COMPLETE, notes


def flag_code_for(status: EquationCaptureStatus) -> ReviewFlagCode | None:
    return {
        EquationCaptureStatus.INCOMPLETE: ReviewFlagCode.INCOMPLETE_EQUATION,
        EquationCaptureStatus.IMAGE_ONLY: ReviewFlagCode.INCOMPLETE_EQUATION,
        EquationCaptureStatus.AMBIGUOUS: ReviewFlagCode.AMBIGUOUS_EQUATION,
        EquationCaptureStatus.NEEDS_REVIEW: ReviewFlagCode.EQUATION_NEEDS_REVIEW,
    }.get(status)


class EquationCapture(BaseModel):
    knowledge_unit_id: uuid.UUID
    equation_id: uuid.UUID | None
    passage_id: uuid.UUID | None
    structural_node_id: uuid.UUID | None
    page_start: int | None
    page_end: int | None
    original_text: str | None
    source_latex: str | None
    image_reference: str | None
    source_location: dict | None
    surrounding_text: str | None
    context_before: str | None
    context_after: str | None
    capture_status: EquationCaptureStatus
    capture_notes: list[str]
    created_at: datetime


def capture_read(source: EquationSource) -> EquationCapture:
    return EquationCapture(
        knowledge_unit_id=source.knowledge_unit_id,
        equation_id=source.equation_id,
        passage_id=source.passage_id,
        structural_node_id=source.structural_node_id,
        page_start=source.page_start,
        page_end=source.page_end,
        original_text=source.original_text,
        source_latex=source.source_latex,
        image_reference=source.image_reference,
        source_location=source.source_location,
        surrounding_text=source.surrounding_text,
        context_before=source.context_before,
        context_after=source.context_after,
        capture_status=source.capture_status,
        capture_notes=list(source.capture_notes or []),
        created_at=source.created_at,
    )


class EquationService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.passages = PassageRepository(session)

    # -- the source's equations --------------------------------------------------------------------

    def equations_in(self, passage: Passage) -> list[Equation]:
        """The Phase 1 equations that Phase 1 placed in this passage."""
        ids = [uuid.UUID(i) for i in (passage.passage_metadata or {}).get("equation_ids", [])]
        if not ids:
            return []
        found = {e.id: e for e in self.session.scalars(select(Equation).where(Equation.id.in_(ids)))}
        return [found[i] for i in ids if i in found]

    @staticmethod
    def match(content: EquationContent, candidates: list[Equation]) -> Equation | None:
        """The stored equation the model's reading belongs to: the same expression, or the only one there."""
        wanted = [_compact(content.latex), _compact(content.original_expression)]
        wanted = [w for w in wanted if len(w) >= 3]
        for equation in candidates:
            for have in (_compact(equation.latex), _compact(equation.original_latex), _compact(equation.plain_text)):
                if have and any(w == have or w in have or have in w for w in wanted):
                    return equation
        # The only stored equation is the likely one, but not when it is just an image: a model that
        # never saw the image cannot have read it.
        if len(candidates) == 1 and (_compact(candidates[0].latex) or _compact(candidates[0].plain_text)):
            return candidates[0]
        return None

    # -- recording a capture -----------------------------------------------------------------------

    def record_capture(
        self, unit: KnowledgeUnit, passage: Passage, content: EquationContent, equations: list[Equation] | None = None
    ) -> EquationSource:
        """Store what the source gave for the equation of `unit`, next to the model's reading. Does not commit."""
        candidates = equations if equations is not None else self.equations_in(passage)
        # A unit made from a stored equation already knows which one it is; a model reading is matched.
        equation = next((e for e in candidates if e.id == unit.equation_id), None) or self.match(content, candidates)
        expression_in_source = self._found_in(passage.text, content.original_expression or content.latex)
        original_text = (equation.plain_text or equation.original_latex) if equation else None
        if original_text is None and content.original_expression and expression_in_source:
            original_text = self._source_slice(passage.text, content.original_expression)
        source_latex = equation.latex if equation and equation.latex.strip() else None
        status, notes = classify_capture(
            original_text=original_text,
            source_latex=source_latex,
            image_reference=equation.crop_reference if equation else None,
            model_status=content.extraction_status,
            matched_to_source=bool(equation) or expression_in_source,
            source_validation=equation.validation_status if equation else None,
        )
        if equation is None and len(candidates) > 1 and not expression_in_source:
            notes.append(f"The passage holds {len(candidates)} stored equations; none matched the model's expression.")
        before, after = self._context(passage.text, original_text)
        source = EquationSource(
            knowledge_unit_id=unit.id,
            equation_id=equation.id if equation else None,
            passage_id=passage.id,
            structural_node_id=passage.structural_node_id,
            page_start=passage.page_start,
            page_end=passage.page_end,
            original_text=original_text,
            source_latex=source_latex,
            image_reference=equation.crop_reference if equation else None,
            source_location=self._location(equation),
            surrounding_text=equation.surrounding_text if equation else None,
            context_before=before,
            context_after=after,
            capture_status=status.value,
            capture_notes=notes,
        )
        if equation:
            unit.equation_id = equation.id
        self.session.add(source)
        return source

    def source_unit(self, equation: Equation, passage: Passage, book_id: uuid.UUID, run_id: uuid.UUID | None) -> KnowledgeUnitCreate:
        """A unit for a stored equation that no model reading covered. Copied from the source; no model involved."""
        text = (equation.plain_text or "").strip()
        latex = equation.latex.strip() if equation.latex else ""
        status_map = {"valid": ExpressionStatus.COMPLETE, "uncertain": ExpressionStatus.AMBIGUOUS, "invalid": ExpressionStatus.PARTIAL}
        status = status_map.get(equation.validation_status, ExpressionStatus.PARTIAL)
        notes = list(equation.validation_warnings or [])
        if not (text or latex):
            status = ExpressionStatus.FAILED
            notes.append("Only an image of the equation is available.")
        label = equation.equation_number or str(equation.sequence_number + 1)
        content = EquationContent(
            name=f"Equation {label}" + (f" (page {equation.page_number})" if equation.page_number else ""),
            original_expression=text or None,
            latex=latex or None,
            extraction_status=status,
            status_notes="; ".join(dict.fromkeys(notes)) or None,
        )
        return KnowledgeUnitCreate(
            book_id=book_id,
            document_id=passage.document_id,
            extraction_run_id=run_id,
            content=content,
            sources=[SourceReference(passage_id=passage.id, role=SourceRole.PRIMARY)],
            equation_id=equation.id,
            page_start=passage.page_start,
            page_end=passage.page_end,
            page_status=PageStatus(passage.page_status),
            origin=KnowledgeOrigin.SOURCE_DERIVED,
            metadata={"source_derived": "Created from the stored equation; no model read it."},
        )

    # -- reading -----------------------------------------------------------------------------------

    def get_capture(self, unit_id: uuid.UUID) -> EquationCapture | None:
        source = self.session.get(EquationSource, unit_id)
        return capture_read(source) if source else None

    def list_captures(self, book_id: uuid.UUID, status: EquationCaptureStatus | str | None = None) -> list[EquationCapture]:
        query = select(EquationSource).join(KnowledgeUnit, KnowledgeUnit.id == EquationSource.knowledge_unit_id).where(KnowledgeUnit.book_id == book_id)
        if status:
            query = query.where(EquationSource.capture_status == EquationCaptureStatus(status).value)
        return [capture_read(s) for s in self.session.scalars(query.order_by(EquationSource.page_start.nulls_last(), EquationSource.created_at))]

    def counts_by_status(self, book_id: uuid.UUID) -> dict[str, int]:
        counts: dict[str, int] = {}
        for capture in self.list_captures(book_id):
            counts[capture.capture_status.value] = counts.get(capture.capture_status.value, 0) + 1
        return counts

    # -- helpers -----------------------------------------------------------------------------------

    @staticmethod
    def _found_in(text: str, expression: str | None) -> bool:
        return bool(expression) and _compact(expression) in _compact(text)

    @staticmethod
    def _source_slice(text: str, expression: str) -> str | None:
        """The expression as it is written in the passage (not as the model repeated it)."""
        start = text.find(expression)
        return text[start : start + len(expression)] if start >= 0 else None

    @staticmethod
    def _context(text: str, original: str | None) -> tuple[str | None, str | None]:
        if not original:
            return None, None
        start = text.find(original.strip())
        if start < 0:
            return None, None
        end = start + len(original.strip())
        return text[max(0, start - CONTEXT_CHARS) : start].strip() or None, text[end : end + CONTEXT_CHARS].strip() or None

    @staticmethod
    def _location(equation: Equation | None) -> dict | None:
        if equation is None:
            return None
        location = {k: v for k, v in {"source": equation.source_location, "bounding_box": equation.bounding_box, "page": equation.page_number}.items() if v}
        return location or None
