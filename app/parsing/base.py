import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from app.parsing.errors import EmptyDocumentError, ParserError
from app.parsing.models import (
    ContentBlock,
    ExtractionStatus,
    FormulaRegion,
    OcrSummary,
    ParseStage,
    ParsingIssue,
    ParsingResult,
)

logger = logging.getLogger(__name__)

# (stage, current, total) - e.g. (ParseStage.OCR, 3, 12) means page 3 of 12 OCR'd.
ProgressCallback = Callable[[ParseStage, int, int], None]


def _no_progress(_stage: ParseStage, _current: int, _total: int) -> None:
    return None


@dataclass
class Extraction:
    """What a parser's `_extract` produces; `parse` turns it into a ParsingResult."""

    blocks: list[ContentBlock]
    page_count: int | None = None
    warnings: list[ParsingIssue] = field(default_factory=list)
    ocr: OcrSummary | None = None
    # True when some content is known to be missing (e.g. a page whose OCR failed).
    partial: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    formula_regions: list[FormulaRegion] = field(default_factory=list)


class DocumentParser(ABC):
    """Common interface for all formats. Parsers know nothing about the UI or the database."""

    file_format: ClassVar[str]
    extraction_method: ClassVar[str]

    def parse(
        self,
        path: Path,
        *,
        document_id: uuid.UUID,
        original_filename: str,
        on_progress: ProgressCallback | None = None,
    ) -> ParsingResult:
        """Parse a local file. Never raises for problems with the document itself."""
        identity: dict[str, Any] = {
            "document_id": document_id,
            "original_filename": original_filename,
            "file_format": self.file_format,
            "extraction_method": self.extraction_method,
        }
        try:
            extraction = self._extract(path, on_progress or _no_progress)
            if not extraction.blocks:
                raise EmptyDocumentError(
                    "No text could be extracted from this document.",
                    details={"page_count": extraction.page_count},
                )
        except ParserError as exc:
            logger.warning("Parsing %s failed (%s): %s", original_filename, exc.code, exc.message)
            return ParsingResult(**identity, status=ExtractionStatus.FAILED, errors=[exc.to_issue()])
        except Exception as exc:
            logger.exception("Unexpected error while parsing %s", original_filename)
            issue = ParsingIssue(
                code="unexpected_error",
                message=f"Unexpected error while parsing: {type(exc).__name__}: {exc}"[:1000],
                details={"exception": type(exc).__name__, "traceback": traceback.format_exc()},
            )
            return ParsingResult(**identity, status=ExtractionStatus.FAILED, errors=[issue])

        for number, block in enumerate(extraction.blocks):
            block.sequence_number = number
        return ParsingResult(
            **identity,
            page_count=extraction.page_count,
            blocks=extraction.blocks,
            warnings=extraction.warnings,
            status=ExtractionStatus.PARTIAL if extraction.partial else ExtractionStatus.SUCCESS,
            ocr=extraction.ocr,
            metadata=extraction.metadata,
            formula_regions=extraction.formula_regions,
        )

    @abstractmethod
    def _extract(self, path: Path, on_progress: ProgressCallback) -> Extraction:
        """Extract content blocks in reading order. Raise ParserError for document problems."""
