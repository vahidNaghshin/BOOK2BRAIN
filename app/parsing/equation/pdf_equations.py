"""Equation extraction from PDFs.

The fast text pass already finds formula regions (also on scanned pages). Here:
  1. Docling's formula enrichment recognises LaTeX, but only on pages that contain formulas.
  2. Regions it could not recover go to the math OCR adapter (if enabled and available).
  3. Anything still unrecovered is recorded as a failed equation - never invented.
A crop of every region is kept for traceability.
"""

import io
import logging
import re
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from app.config import Settings
from app.parsing.equation.equation_models import (
    EnrichmentReport,
    EquationBlock,
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
)
from app.parsing.models import FormulaRegion

logger = logging.getLogger(__name__)

# Docling's formula model was trained at ~120 dpi with the box expanded by 18%.
_RECOGNITION_DPI = 120
_RECOGNITION_EXPANSION = 0.18
_CROP_EXPANSION = 0.08
_BBOX_TOLERANCE = 3.0  # points


@dataclass
class PdfEquationOutput:
    equations: list[EquationBlock] = field(default_factory=list)
    issues: list[EquationIssue] = field(default_factory=list)
    enrichment: EnrichmentReport = field(default_factory=EnrichmentReport)
    math_ocr: MathOcrReport = field(default_factory=MathOcrReport)
    crops: dict[uuid.UUID, bytes] = field(default_factory=dict)
    # True when some region could not be recognised because no engine could run.
    engine_unavailable: bool = False


def _contiguous_ranges(pages: Sequence[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for page in sorted(set(pages)):
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], page)
        else:
            ranges.append((page, page))
    return ranges


def _box_distance(region: FormulaRegion, item_box: dict[str, float]) -> float:
    a = region.bbox
    return max(abs(a[k] - item_box[k]) for k in ("l", "t", "r", "b"))


_MATH_SIGNS = re.compile(r"[=^_<>]|\\[A-Za-z]+")


def _looks_like_math(latex: str) -> bool:
    """Recognition output that is plausibly mathematics rather than a caption or a diagram label."""
    return bool(latex.strip()) and bool(_MATH_SIGNS.search(latex))


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


class PdfEquationExtractor:
    def __init__(
        self,
        settings: Settings,
        math_ocr: MathOcrService | None,
        converter_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.math_ocr = math_ocr
        self._converter_factory = converter_factory

    # ------------------------------------------------------------------------------

    def extract(
        self,
        path: Path,
        document_id: uuid.UUID,
        regions: Sequence[FormulaRegion],
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> PdfEquationOutput:
        out = PdfEquationOutput()
        out.math_ocr.enabled = self.math_ocr is not None
        out.math_ocr.engine = self.math_ocr.engine_name if self.math_ocr else None
        progress = on_progress or (lambda *_: None)
        regions = self._select_regions(regions, out)
        if not regions:
            return out

        enriched = self._enrich(path, regions, out, progress)

        pdf = pymupdf.open(str(path))
        try:
            for index, region in enumerate(regions):
                progress("recognizing", index, len(regions))
                equation = self._equation_for(pdf, document_id, region, enriched.get(index), out)
                if equation is not None:
                    out.equations.append(equation)
            progress("recognizing", len(regions), len(regions))
        finally:
            pdf.close()
        return out

    def _select_regions(self, regions: Sequence[FormulaRegion], out: PdfEquationOutput) -> list[FormulaRegion]:
        """All formula regions, plus picture candidates when they can be tried (math OCR on)."""
        scan = self.settings.equation_scan_pictures and self.math_ocr is not None
        budget = self.settings.equation_scan_pictures_max
        selected: list[FormulaRegion] = []
        skipped = 0
        for region in regions:
            if region.candidate:
                if not scan:
                    continue
                if budget <= 0:
                    skipped += 1
                    continue
                budget -= 1
            selected.append(region)
        if skipped:
            out.issues.append(
                EquationIssue(
                    code="picture_candidates_limited",
                    message=(
                        f"{skipped} small picture region(s) were not checked for equations "
                        f"(limit EQUATION_SCAN_PICTURES_MAX={self.settings.equation_scan_pictures_max})."
                    ),
                )
            )
        return selected

    # -- enrichment ------------------------------------------------------------------

    def _enrich(
        self,
        path: Path,
        regions: Sequence[FormulaRegion],
        out: PdfEquationOutput,
        progress: Callable[[str, int, int], None],
    ) -> dict[int, str]:
        """Region index -> LaTeX produced by Docling's formula enrichment."""
        report = out.enrichment
        report.enabled = self.settings.formula_enrichment_enabled
        report.device = self.settings.formula_enrichment_device
        report.dtype = self.settings.formula_enrichment_dtype
        if not report.enabled:
            return {}

        pages = sorted({r.page_number for r in regions if not r.candidate})
        if not pages:
            return {}
        report.pages = pages
        factory = self._converter_factory or self._default_factory()
        started = time.monotonic()
        items: list[tuple[int, dict[str, float], str, str]] = []  # page, box, text, orig
        try:
            converter = factory(report.device, report.dtype)
            for done, (first, last) in enumerate(_contiguous_ranges(pages)):
                progress("formula_enrichment", done, len(_contiguous_ranges(pages)))
                result = converter.convert(str(path), page_range=(first, last))
                items.extend(self._formula_items(result.document))
            report.executed = True
        except Exception as exc:  # model missing, no network on first use, runtime failure...
            report.executed = False
            report.error = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning("Formula enrichment failed: %s", report.error)
            out.issues.append(
                EquationIssue(
                    code="formula_enrichment_failed",
                    message=(
                        "Docling's formula enrichment could not run "
                        f"({type(exc).__name__}). Docling downloads its formula model (about 600 MB) "
                        "on first use: check your internet connection, or see the README section "
                        "'Docling models'. Math OCR is used instead where available."
                    ),
                    details={"error": report.error},
                )
            )
            return {}
        finally:
            report.seconds = round(time.monotonic() - started, 2)

        report.formulas_seen = len(items)
        return self._match(regions, items, report)

    @staticmethod
    def _default_factory() -> Callable[[str, str], Any]:
        from app.parsing.docling_support import get_formula_converter

        return get_formula_converter

    @staticmethod
    def _formula_items(document: Any) -> list[tuple[int, dict[str, float], str, str]]:
        found = []
        for item, _depth in document.iterate_items():
            if getattr(getattr(item, "label", None), "value", "") != "formula" or not item.prov:
                continue
            bbox = item.prov[0].bbox
            box = {"l": float(bbox.l), "t": float(bbox.t), "r": float(bbox.r), "b": float(bbox.b)}
            found.append((int(item.prov[0].page_no), box, (item.text or "").strip(), (getattr(item, "orig", "") or "").strip()))
        return found

    def _match(
        self,
        regions: Sequence[FormulaRegion],
        items: list[tuple[int, dict[str, float], str, str]],
        report: EnrichmentReport,
    ) -> dict[int, str]:
        """Pair enrichment output with text-pass regions (same page, nearly the same box).

        Docling silently keeps the original PDF text when the formula model fails for an item,
        so text equal to the original (or empty) does not count as enriched.
        """
        matched: dict[int, str] = {}
        used: set[int] = set()
        for index, region in enumerate(regions):
            if region.candidate:
                continue
            best, best_distance = None, _BBOX_TOLERANCE
            for position, (page, box, text, orig) in enumerate(items):
                if page != region.page_number or position in used:
                    continue
                distance = _box_distance(region, box)
                if distance <= best_distance:
                    best, best_distance = position, distance
            if best is None:
                continue
            used.add(best)
            _, _, text, orig = items[best]
            if text and _normalize_text(text) != _normalize_text(orig):
                matched[index] = text
        report.formulas_enriched = len(matched)
        return matched

    # -- one equation ----------------------------------------------------------------

    def _equation_for(
        self,
        pdf: Any,
        document_id: uuid.UUID,
        region: FormulaRegion,
        enriched: str | None,
        out: PdfEquationOutput,
    ) -> EquationBlock:
        box = region.bbox
        location = {
            "kind": "pdf_formula",
            "page": region.page_number,
            "bbox": [box["l"], box["t"], box["r"], box["b"]],
            "coord_origin": box.get("coord_origin"),
            "docling_ref": region.docling_ref,
        }
        equation = EquationBlock(
            document_id=document_id,
            latex="",
            plain_text=region.original_text,
            equation_type=EquationType.DISPLAY,
            page_number=region.page_number,
            source_location=location,
            bounding_box=dict(box),
            extraction_method=(
                ExtractionMethod.DOCLING_ENRICHMENT if self.settings.formula_enrichment_enabled else ExtractionMethod.MATH_OCR
            ),
        )

        page = pdf[region.page_number - 1]
        if region.candidate:
            return self._candidate(page, region, equation, out)
        if enriched:
            equation.latex = equation.original_latex = enriched
            equation.extraction_method = ExtractionMethod.DOCLING_ENRICHMENT
        else:
            self._recognize_with_ocr(page, region, equation, out)

        self._attach_crop(page, region, equation, out)
        return equation

    def _candidate(
        self, page: Any, region: FormulaRegion, equation: EquationBlock, out: PdfEquationOutput
    ) -> EquationBlock | None:
        """A picture region tried as an equation: kept only if math OCR returns mathematics."""
        equation.plain_text = self._text_layer(page, region)
        equation.extraction_method = ExtractionMethod.MATH_OCR
        unavailable_before = out.engine_unavailable
        self._recognize_with_ocr(page, region, equation, out)
        out.engine_unavailable = unavailable_before  # a skipped picture is not a missed equation
        if equation.metadata.get("recognition_failed") or not _looks_like_math(equation.latex):
            out.math_ocr.pictures_rejected += 1
            return None
        equation.metadata["from_picture_region"] = True
        equation.validation_warnings.append(
            "Layout analysis labelled this region a picture; it was recognised as an equation by math OCR."
        )
        out.math_ocr.pictures_accepted += 1
        self._attach_crop(page, region, equation, out)
        return equation

    def _attach_crop(self, page: Any, region: FormulaRegion, equation: EquationBlock, out: PdfEquationOutput) -> None:
        if not self.settings.equation_store_crops:
            return
        crop = self._render(page, region, self.settings.equation_crop_dpi, _CROP_EXPANSION)
        if crop is not None:
            buffer = io.BytesIO()
            crop.save(buffer, "PNG")
            out.crops[equation.id] = buffer.getvalue()

    @staticmethod
    def _text_layer(page: Any, region: FormulaRegion) -> str | None:
        box = region.bbox
        height = page.rect.height
        top, bottom = (
            (box["t"], box["b"]) if str(box.get("coord_origin", "")).upper() == "TOPLEFT" else (height - box["t"], height - box["b"])
        )
        text = page.get_text("text", clip=pymupdf.Rect(box["l"], top, box["r"], bottom)).strip()
        return " ".join(text.split()) or None

    def _recognize_with_ocr(
        self,
        page: Any,
        region: FormulaRegion,
        equation: EquationBlock,
        out: PdfEquationOutput,
    ) -> None:
        reason = (
            "Docling's formula enrichment produced no LaTeX for this region"
            if self.settings.formula_enrichment_enabled and out.enrichment.executed
            else "Formula enrichment is disabled or did not run"
        )
        if self.math_ocr is None:
            self._fail(equation, f"{reason}, and math OCR is disabled (MATH_OCR_ENABLED=false).", out, unavailable=True)
            return
        report = out.math_ocr
        try:
            self.math_ocr.ensure_available()
        except MathOcrUnavailableError as exc:
            report.unavailable_reason = str(exc)
            self._fail(equation, f"{reason}, and math OCR is unavailable: {exc}", out, unavailable=True)
            return

        image = self._render(page, region, _RECOGNITION_DPI, _RECOGNITION_EXPANSION)
        if image is None:
            self._fail(equation, "The formula region could not be rendered for recognition.", out)
            return
        report.attempted += 1
        try:
            recognized = self.math_ocr.recognize(image)
        except MathOcrError as exc:
            report.failed += 1
            self._fail(equation, f"Math OCR failed: {exc}", out)
            return
        if not recognized.latex.strip():
            report.failed += 1
            self._fail(equation, "Math OCR returned no LaTeX for this region.", out)
            return
        report.succeeded += 1
        equation.latex = equation.original_latex = recognized.latex
        equation.confidence = recognized.confidence
        equation.extraction_method = ExtractionMethod.MATH_OCR
        equation.metadata["math_ocr_engine"] = self.math_ocr.engine_name

    @staticmethod
    def _fail(equation: EquationBlock, message: str, out: PdfEquationOutput, unavailable: bool = False) -> None:
        equation.validation_status = ValidationStatus.INVALID
        equation.validation_warnings.append(message)
        equation.metadata["recognition_failed"] = True
        out.engine_unavailable = out.engine_unavailable or unavailable

    # -- rendering -------------------------------------------------------------------

    @staticmethod
    def _render(page: Any, region: FormulaRegion, dpi: int, expansion: float) -> Image.Image | None:
        """Crop the region from the page, expanded a little, as a PIL image."""
        box = region.bbox
        height = page.rect.height
        if str(box.get("coord_origin", "")).upper() == "TOPLEFT":
            top, bottom = box["t"], box["b"]
        else:  # PDF coordinates: origin at the bottom-left
            top, bottom = height - box["t"], height - box["b"]
        width, tall = box["r"] - box["l"], bottom - top
        pad_x, pad_y = width * expansion, tall * expansion
        clip = pymupdf.Rect(box["l"] - pad_x, top - pad_y, box["r"] + pad_x, bottom + pad_y) & page.rect
        if clip.is_empty:
            return None
        pixmap = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
