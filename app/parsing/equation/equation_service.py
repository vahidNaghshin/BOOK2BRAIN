"""Coordinates equation extraction for one document. No database, no UI.

    detect -> extract (document markup, Docling enrichment) -> math OCR fallback
           -> normalize -> validate -> link to text blocks -> status

Equation extraction is isolated from text extraction: whatever happens here, the text blocks
are already saved, and problems are reported in the result instead of being raised.
"""

import hashlib
import logging
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.parsing.equation.equation_extractor import (
    DocxEquationExtractor,
    EpubEquationExtractor,
    ExtractedEquations,
)
from app.parsing.equation.equation_models import (
    EquationBlock,
    EquationExtractionResult,
    EquationExtractionStatus,
    EquationIssue,
    EquationType,
    ExtractionMethod,
    MathOcrReport,
    ValidationStatus,
)
from app.parsing.equation.equation_ocr import (
    MathOcrError,
    MathOcrService,
    MathOcrUnavailableError,
    build_math_ocr_service,
)
from app.parsing.equation.equation_validator import validate_latex
from app.parsing.equation.latex_normalizer import normalize_latex
from app.parsing.equation.pdf_equations import PdfEquationExtractor
from app.parsing.models import BlockType, ContentBlock, ParsingResult
from app.parsing.text_normalizer import normalize_text

logger = logging.getLogger(__name__)

_CONTEXT_CHARS = 600
_TEXT_BLOCKS = {BlockType.PARAGRAPH, BlockType.LIST, BlockType.CAPTION, BlockType.HEADING}
_DEFINITION_OPENERS = ("where ", "here, ", "here ", "in which ", "with ", "in this equation")

EquationProgress = Callable[[str, int, int], None]


@dataclass
class EquationOutcome:
    result: EquationExtractionResult
    # Equation id -> PNG bytes of the image region used for recognition (PDF).
    crops: dict[uuid.UUID, bytes] = field(default_factory=dict)


class EquationService:
    def __init__(
        self,
        settings: Settings,
        math_ocr: MathOcrService | None = None,
        converter_factory: Callable[[str, str], object] | None = None,
    ) -> None:
        self.settings = settings
        self.math_ocr = math_ocr if math_ocr is not None else build_math_ocr_service(settings)
        self._converter_factory = converter_factory

    # ------------------------------------------------------------------------------

    def extract(
        self,
        *,
        path: Path,
        file_format: str,
        document_id: uuid.UUID,
        parsing: ParsingResult,
        block_ids: Mapping[int, uuid.UUID] | None = None,
        on_progress: EquationProgress | None = None,
    ) -> EquationOutcome:
        """Extract, recognise, normalize and validate the equations of one document."""
        result = EquationExtractionResult()
        if not self.settings.equation_extraction_enabled:
            return EquationOutcome(result)

        started = time.monotonic()
        result.status = EquationExtractionStatus.PROCESSING
        crops: dict[uuid.UUID, bytes] = {}
        unavailable = False
        try:
            if file_format == "pdf":
                pdf = PdfEquationExtractor(self.settings, self.math_ocr, self._converter_factory).extract(
                    path, document_id, parsing.formula_regions, on_progress
                )
                raw, result.issues = pdf.equations, pdf.issues
                result.enrichment, result.math_ocr, crops = pdf.enrichment, pdf.math_ocr, pdf.crops
                unavailable = pdf.engine_unavailable
            elif file_format == "epub":
                extracted = EpubEquationExtractor().extract(path, document_id)
                raw, result.issues = extracted.equations, extracted.issues
                unavailable = self._recognize_images(extracted, result)
            elif file_format == "docx":
                extracted = DocxEquationExtractor().extract(path, document_id)
                raw, result.issues = extracted.equations, extracted.issues
            else:
                raw = []
        except Exception as exc:  # equation extraction must never break the document
            logger.exception("Equation extraction failed for document %s", document_id)
            result.status = EquationExtractionStatus.FAILED
            result.issues.append(
                EquationIssue(
                    code="equation_extraction_failed",
                    message=f"Equation extraction failed ({type(exc).__name__}): {str(exc)[:300]}",
                    details={"exception": type(exc).__name__},
                )
            )
            result.seconds = round(time.monotonic() - started, 2)
            return EquationOutcome(result)

        result.equations = [self._finalize(equation, number) for number, equation in enumerate(raw)]
        self._link(result.equations, parsing.blocks, block_ids or {}, file_format)
        crops = {eq.id: crops[eq.id] for eq in result.equations if eq.id in crops}
        result.status = self._status(result, unavailable)
        result.seconds = round(time.monotonic() - started, 2)
        logger.info(
            "Equations for %s: %d total (%d valid, %d uncertain, %d invalid) status=%s in %.1fs",
            document_id, len(result.equations), result.count(ValidationStatus.VALID),
            result.count(ValidationStatus.UNCERTAIN), result.count(ValidationStatus.INVALID),
            result.status.value, result.seconds,
        )
        return EquationOutcome(result, crops)

    # -- EPUB equations that exist only as images ----------------------------------

    def _recognize_images(self, extracted: ExtractedEquations, result: EquationExtractionResult) -> bool:
        """Run math OCR on image-only equations. Returns True if an engine was needed but unusable."""
        pending = [e for e in extracted.equations if e.metadata.get("needs_math_ocr") and e.id in extracted.images]
        report = result.math_ocr = MathOcrReport(
            enabled=self.math_ocr is not None,
            engine=self.math_ocr.engine_name if self.math_ocr else None,
        )
        if not pending:
            return False
        reason = None
        if self.math_ocr is None:
            reason = "math OCR is disabled (MATH_OCR_ENABLED=false)"
        else:
            try:
                self.math_ocr.ensure_available()
            except MathOcrUnavailableError as exc:
                report.unavailable_reason = reason = str(exc)
        for equation in pending:
            if reason:
                equation.validation_status = ValidationStatus.INVALID
                equation.validation_warnings.append(f"This equation is an image and math OCR is unavailable: {reason}")
                continue
            from io import BytesIO

            from PIL import Image

            report.attempted += 1
            try:
                image = Image.open(BytesIO(extracted.images[equation.id])).convert("RGB")
                recognized = self.math_ocr.recognize(image)  # type: ignore[union-attr]
            except (MathOcrError, OSError) as exc:
                report.failed += 1
                equation.validation_status = ValidationStatus.INVALID
                equation.validation_warnings.append(f"Math OCR failed on this image: {exc}")
                continue
            if not recognized.latex.strip():
                report.failed += 1
                equation.validation_status = ValidationStatus.INVALID
                equation.validation_warnings.append("Math OCR returned no LaTeX for this image.")
                continue
            report.succeeded += 1
            equation.latex = equation.original_latex = recognized.latex
            equation.confidence = recognized.confidence
            equation.metadata["math_ocr_engine"] = self.math_ocr.engine_name
        return bool(reason)

    # -- normalize and validate ------------------------------------------------------

    def _finalize(self, equation: EquationBlock, sequence: int) -> EquationBlock:
        equation.sequence_number = sequence
        if equation.latex.strip():
            normalized = normalize_latex(equation.latex)
            if equation.original_latex is None:
                equation.original_latex = equation.latex
            equation.latex = normalized.latex
            if normalized.changes:
                equation.metadata["normalization"] = normalized.changes
            if normalized.equation_number and not equation.equation_number:
                equation.equation_number = normalized.equation_number
            if equation.equation_number and equation.equation_type in (EquationType.DISPLAY, EquationType.UNKNOWN):
                equation.equation_type = EquationType.NUMBERED

            check = validate_latex(
                equation.latex,
                method=equation.extraction_method,
                confidence=equation.confidence,
                min_confidence=self.settings.equation_min_confidence,
                original_text=equation.plain_text if equation.page_number else None,
            )
            status = check.status
            if status is ValidationStatus.VALID and equation.metadata.get("converter_warnings"):
                status = ValidationStatus.UNCERTAIN  # part of the MathML could not be converted
            equation.validation_status = status
            equation.validation_warnings = list(dict.fromkeys([*equation.validation_warnings, *check.warnings]))
        else:
            equation.validation_status = ValidationStatus.INVALID
            if not equation.validation_warnings:
                equation.validation_warnings = ["Empty expression: nothing was recognised."]
        equation.metadata.pop("_display", None)
        equation.metadata["identity_hash"] = self._identity(equation)
        return equation

    @staticmethod
    def _identity(equation: EquationBlock) -> str:
        """Stable per occurrence: document, source location, position and content."""
        location = equation.source_location or {}
        if location.get("kind") == "pdf_formula":
            box = ",".join(str(round(float(v))) for v in location.get("bbox", []))
            place = f"p{location.get('page')}:{box}"
        elif location.get("kind") == "omml":
            place = f"p{location.get('paragraph_index')}.{location.get('math_index')}"
        else:
            place = f"{location.get('item_id')}:{location.get('kind')}:{location.get('image_src', '')}"
        parts = [str(equation.document_id), place, str(equation.sequence_number), equation.latex]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    # -- link to text blocks and add context ------------------------------------------

    def _link(
        self,
        equations: list[EquationBlock],
        blocks: list[ContentBlock],
        block_ids: Mapping[int, uuid.UUID],
        file_format: str,
    ) -> None:
        equation_blocks = [b for b in blocks if b.block_type is BlockType.EQUATION]
        used: set[int] = set()
        for equation in equations:
            block = self._find_block(equation, blocks, equation_blocks, used, file_format)
            if block is None:
                continue
            equation.extracted_block_id = block_ids.get(block.sequence_number)
            equation.metadata["block_sequence"] = block.sequence_number
            if equation.section_title is None:
                equation.section_title = block.section_title
            if equation.equation_type in (EquationType.DISPLAY, EquationType.NUMBERED):
                self._add_context(equation, block, blocks)

    @staticmethod
    def _find_block(
        equation: EquationBlock,
        blocks: list[ContentBlock],
        equation_blocks: list[ContentBlock],
        used: set[int],
        file_format: str,
    ) -> ContentBlock | None:
        location = equation.source_location or {}
        if file_format == "pdf":
            ref = location.get("docling_ref")
            return next(
                (b for b in equation_blocks if ref and (b.source_location or {}).get("docling_ref") == ref),
                None,
            )
        if file_format == "docx":
            wanted = (equation.link_hint or "").strip()
            for block in equation_blocks:
                if block.sequence_number not in used and wanted and block.text.strip() == wanted:
                    used.add(block.sequence_number)
                    return block
            return None
        if file_format == "epub" and equation.link_hint is not None:
            wanted = normalize_text(equation.link_hint)
            return next(
                (
                    b for b in blocks
                    if (b.source_location or {}).get("item_id") == location.get("item_id")
                    and wanted and b.text == wanted
                ),
                None,
            )
        return None

    def _add_context(self, equation: EquationBlock, block: ContentBlock, blocks: list[ContentBlock]) -> None:
        """Nearest text before and after a display equation, plus an explicit "where ..." clause."""
        index = block.sequence_number
        before = next((b for b in reversed(blocks[:index]) if b.block_type in _TEXT_BLOCKS and b.text.strip()), None)
        after = next((b for b in blocks[index + 1 :] if b.block_type in _TEXT_BLOCKS and b.text.strip()), None)
        parts = []
        if before is not None:
            parts.append(before.text[-_CONTEXT_CHARS:])
        if after is not None:
            parts.append(after.text[:_CONTEXT_CHARS])
            if after.text.lower().startswith(_DEFINITION_OPENERS):
                # Kept verbatim: the author's own statement, not our interpretation.
                equation.metadata["definitions_text"] = after.text[:_CONTEXT_CHARS]
        if parts:
            equation.surrounding_text = "\n\n".join(parts)

    # -- overall status -------------------------------------------------------------

    @staticmethod
    def _status(result: EquationExtractionResult, unavailable: bool) -> EquationExtractionStatus:
        recovered = sum(1 for e in result.equations if e.latex.strip())
        if unavailable and recovered == 0 and result.equations:
            return EquationExtractionStatus.UNAVAILABLE
        problems = (
            result.issues
            or result.count(ValidationStatus.UNCERTAIN)
            or result.count(ValidationStatus.INVALID)
        )
        return (
            EquationExtractionStatus.COMPLETED_WITH_WARNINGS
            if problems
            else EquationExtractionStatus.COMPLETED
        )
