import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.parsing.base import DocumentParser, Extraction, ProgressCallback
from app.parsing.errors import (
    CorruptDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    OcrPageError,
)
from app.parsing.models import (
    BlockType,
    ContentBlock,
    OcrSummary,
    ParseStage,
    ParsingIssue,
)
from app.parsing.ocr_service import OcrPageResult, OcrService
from app.parsing.scan_detection import decide_ocr, measure_text

logger = logging.getLogger(__name__)


def _render_page(pdf_page: Any, dpi: int) -> Image.Image:
    pixmap = pdf_page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _alnum_count(text: str) -> int:
    return measure_text(text).alnum_chars


@dataclass
class _Outcome:
    """Whether OCR text may replace the extracted text of a page, with any warnings."""

    use_ocr: bool
    warnings: list[ParsingIssue]


class PdfParser(DocumentParser):
    """PDF via Docling, with page-level OCR fallback for scanned pages.

    Docling extracts structure and text for pages with a text layer. Each page is then
    assessed on its own: only pages with missing or garbled text (and, for missing text,
    page images) go to OCR, so a mixed digital/scanned PDF is handled page by page.
    """

    file_format = "pdf"
    extraction_method = "docling"

    def __init__(
        self,
        settings: Settings,
        ocr_service: OcrService | None,
        converter_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.ocr_service = ocr_service
        self._converter_factory = converter_factory

    # -- extraction ----------------------------------------------------------------

    def _extract(self, path: Path, on_progress: ProgressCallback) -> Extraction:
        pdf = self._open(path)
        try:
            page_count = pdf.page_count
            warnings: list[ParsingIssue] = []
            partial = False

            on_progress(ParseStage.PARSING, 0, 1)
            digital_blocks, convert_warnings, convert_partial = self._convert(path)
            warnings.extend(convert_warnings)
            partial = partial or convert_partial
            on_progress(ParseStage.PARSING, 1, 1)

            blocks_by_page = self._group_by_page(digital_blocks, page_count)
            ocr_pages, empty_pages = self._pages_needing_ocr(pdf, blocks_by_page)
            ocr_summary: OcrSummary | None = None

            if ocr_pages:
                ocr_summary, ocr_warnings, ocr_partial = self._run_ocr(
                    pdf, blocks_by_page, ocr_pages, on_progress
                )
                warnings.extend(ocr_warnings)
                partial = partial or ocr_partial

            blocks = self._assemble(blocks_by_page, page_count)
            return Extraction(
                blocks=blocks,
                page_count=page_count,
                warnings=warnings,
                ocr=ocr_summary,
                partial=partial,
                metadata={"pages_without_content": empty_pages},
            )
        finally:
            pdf.close()

    def _open(self, path: Path) -> Any:
        import pymupdf

        try:
            pdf = pymupdf.open(str(path))
        except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError, RuntimeError) as exc:
            raise CorruptDocumentError(
                "The PDF file is corrupted or is not a valid PDF.",
                details={"reason": str(exc)[:500]},
            ) from exc
        if pdf.needs_pass:
            pdf.close()
            raise EncryptedDocumentError(
                "The PDF is password-protected. Remove the password and upload it again."
            )
        if pdf.page_count == 0:
            pdf.close()
            raise EmptyDocumentError("The PDF has no pages.", details={"page_count": 0})
        return pdf

    def _convert(self, path: Path) -> tuple[list[ContentBlock], list[ParsingIssue], bool]:
        from docling.exceptions import ConversionError

        from app.parsing.docling_mapping import blocks_from_docling
        from app.parsing.docling_support import get_pdf_converter

        converter = (self._converter_factory or get_pdf_converter)()
        try:
            result = converter.convert(str(path))
        except ConversionError as exc:
            raise CorruptDocumentError(
                "The PDF could not be read. It may be damaged or use an unsupported feature.",
                details={"reason": str(exc)[:500]},
            ) from exc

        warnings = [
            ParsingIssue(code="docling_warning", message=str(error.error_message)[:500])
            for error in getattr(result, "errors", [])
        ]
        status = getattr(getattr(result, "status", None), "value", "")
        partial = status == "partial_success"
        if partial:
            warnings.append(
                ParsingIssue(
                    code="docling_partial",
                    message="Docling could only convert part of this PDF; some content may be missing.",
                )
            )
        return blocks_from_docling(result.document, with_pages=True), warnings, partial

    # -- page assessment and OCR ---------------------------------------------------

    @staticmethod
    def _group_by_page(blocks: list[ContentBlock], page_count: int) -> dict[int, list[ContentBlock]]:
        by_page: dict[int, list[ContentBlock]] = defaultdict(list)
        current = 1
        for block in blocks:
            if block.page_number is not None:
                current = min(max(block.page_number, 1), page_count)
            block.page_number = current
            by_page[current].append(block)
        return by_page

    def _pages_needing_ocr(
        self, pdf: Any, blocks_by_page: dict[int, list[ContentBlock]]
    ) -> tuple[dict[int, str], list[int]]:
        """Return ({page_number: reason} for pages to OCR, pages with no content at all)."""
        ocr_pages: dict[int, str] = {}
        empty_pages: list[int] = []
        for page_number in range(1, pdf.page_count + 1):
            text = "\n".join(block.text for block in blocks_by_page.get(page_number, []))
            quality = measure_text(text)
            has_images = bool(pdf[page_number - 1].get_images())
            decision = decide_ocr(
                quality,
                has_images=has_images,
                min_text_chars=self.settings.ocr_min_text_chars,
                max_garbled_ratio=self.settings.ocr_max_garbled_ratio,
            )
            if decision.needs_ocr:
                ocr_pages[page_number] = decision.reason
            elif quality.alnum_chars == 0:
                empty_pages.append(page_number)
        return ocr_pages, empty_pages

    def _run_ocr(
        self,
        pdf: Any,
        blocks_by_page: dict[int, list[ContentBlock]],
        ocr_pages: dict[int, str],
        on_progress: ProgressCallback,
    ) -> tuple[OcrSummary | None, list[ParsingIssue], bool]:
        pages = sorted(ocr_pages)
        settings = self.settings

        if not settings.ocr_enabled or self.ocr_service is None:
            issue = ParsingIssue(
                code="ocr_skipped",
                message=(
                    f"{len(pages)} page(s) look scanned or have unreadable text, but OCR is "
                    "disabled (OCR_ENABLED=false). Those pages have no text."
                ),
                details={"pages": pages},
            )
            return None, [issue], True

        # Announce the stage first, so a failure here is attributed to OCR, not to parsing.
        on_progress(ParseStage.OCR, 0, len(pages))
        # Fail early, with an actionable message, if OCR cannot run at all.
        self.ocr_service.ensure_available()

        warnings: list[ParsingIssue] = []
        summary = OcrSummary(
            engine=self.ocr_service.engine_name,
            language=self.ocr_service.language,
            dpi=settings.ocr_dpi,
        )
        used_confidences: list[float] = []
        partial = False

        for done, page_number in enumerate(pages, start=1):
            if done > 1:  # (0, N) was announced above; report pages completed so far
                on_progress(ParseStage.OCR, done - 1, len(pages))
            digital = blocks_by_page.get(page_number, [])
            digital_chars = _alnum_count("\n".join(block.text for block in digital))
            try:
                result = self.ocr_service.recognize(
                    _render_page(pdf[page_number - 1], settings.ocr_dpi)
                )
            except OcrPageError as exc:
                logger.warning("OCR failed on page %d: %s", page_number, exc)
                summary.pages_failed.append(page_number)
                partial = True
                warnings.append(
                    ParsingIssue(
                        code="ocr_failed",
                        message=f"OCR failed on page {page_number}: {exc}",
                        page_number=page_number,
                    )
                )
                continue

            summary.pages_processed.append(page_number)
            outcome = self._merge_page_ocr(page_number, result, digital_chars)
            if outcome.use_ocr:
                blocks_by_page[page_number] = self._ocr_blocks(page_number, result, ocr_pages[page_number])
                if result.mean_confidence is not None:
                    used_confidences.append(result.mean_confidence)
            else:
                summary.pages_kept_digital.append(page_number)
            warnings.extend(outcome.warnings)

        on_progress(ParseStage.OCR, len(pages), len(pages))
        if used_confidences:
            summary.mean_confidence = round(sum(used_confidences) / len(used_confidences), 1)
        return summary, warnings, partial

    def _merge_page_ocr(self, page_number: int, result: OcrPageResult, digital_chars: int) -> _Outcome:
        """Decide whether OCR text may replace what Docling extracted for this page."""
        warnings: list[ParsingIssue] = []
        ocr_chars = _alnum_count(result.text)
        confidence = result.mean_confidence

        if ocr_chars == 0:
            warnings.append(
                ParsingIssue(
                    code="ocr_no_text",
                    message=f"OCR found no readable text on page {page_number}.",
                    page_number=page_number,
                )
            )
            return _Outcome(False, warnings)

        low_confidence = confidence is not None and confidence < self.settings.ocr_min_confidence
        if digital_chars > 0 and (ocr_chars <= digital_chars or low_confidence):
            # Never replace extracted text with a worse OCR result.
            warnings.append(
                ParsingIssue(
                    code="ocr_not_better",
                    message=(
                        f"Page {page_number}: kept the extracted text ({digital_chars} characters) "
                        f"over the OCR result ({ocr_chars} characters, confidence {confidence})."
                    ),
                    page_number=page_number,
                )
            )
            return _Outcome(False, warnings)

        if low_confidence:
            warnings.append(
                ParsingIssue(
                    code="ocr_low_confidence",
                    message=(
                        f"Page {page_number}: OCR confidence is low ({confidence}). "
                        "The text may contain errors."
                    ),
                    page_number=page_number,
                    details={"confidence": confidence},
                )
            )
        return _Outcome(True, warnings)

    def _ocr_blocks(self, page_number: int, result: OcrPageResult, reason: str) -> list[ContentBlock]:
        assert self.ocr_service is not None
        location = {
            "page": page_number,
            "method": "ocr",
            "engine": self.ocr_service.engine_name,
            "dpi": self.settings.ocr_dpi,
        }
        return [
            ContentBlock(
                text=paragraph.text,
                block_type=BlockType.PARAGRAPH,
                page_number=page_number,
                source_location=dict(location),
                metadata={"ocr": True, "ocr_confidence": paragraph.confidence, "ocr_reason": reason},
            )
            for paragraph in result.paragraphs
        ]

    # -- assembly ------------------------------------------------------------------

    @staticmethod
    def _assemble(blocks_by_page: dict[int, list[ContentBlock]], page_count: int) -> list[ContentBlock]:
        """Concatenate pages in order; OCR blocks inherit the running section title."""
        ordered: list[ContentBlock] = []
        current_section: str | None = None
        for page_number in range(1, page_count + 1):
            for block in blocks_by_page.get(page_number, []):
                if block.section_title:
                    current_section = block.section_title
                else:
                    block.section_title = current_section
                ordered.append(block)
        return ordered
