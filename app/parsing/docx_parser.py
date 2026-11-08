from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.parsing.base import DocumentParser, Extraction, ProgressCallback
from app.parsing.errors import CorruptDocumentError
from app.parsing.models import ParseStage, ParsingIssue


class DocxParser(DocumentParser):
    """DOCX via Docling. Word has no fixed pages, so page numbers are never set."""

    file_format = "docx"
    extraction_method = "docling"

    def __init__(self, converter_factory: Callable[[], Any] | None = None) -> None:
        self._converter_factory = converter_factory

    def _extract(self, path: Path, on_progress: ProgressCallback) -> Extraction:
        from docling.exceptions import ConversionError

        from app.parsing.docling_mapping import blocks_from_docling
        from app.parsing.docling_support import get_docx_converter

        on_progress(ParseStage.PARSING, 0, 1)
        converter = (self._converter_factory or get_docx_converter)()
        try:
            result = converter.convert(str(path))
        except ConversionError as exc:
            raise CorruptDocumentError(
                "The DOCX file is corrupted or not a valid Word document.",
                details={"reason": str(exc)[:500]},
            ) from exc

        warnings = [
            ParsingIssue(code="docling_warning", message=str(error.error_message)[:500])
            for error in getattr(result, "errors", [])
        ]
        blocks = blocks_from_docling(result.document, with_pages=False)
        on_progress(ParseStage.PARSING, 1, 1)
        return Extraction(blocks=blocks, page_count=None, warnings=warnings)
