import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, computed_field


class BlockType(enum.StrEnum):
    # Must match database.models.extracted_block.BLOCK_TYPES (a test enforces this).
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    LIST = "list"
    CAPTION = "caption"
    EQUATION = "equation"
    OTHER = "other"


class ExtractionStatus(enum.StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"  # usable, but some content is known to be missing
    FAILED = "failed"


class ParseStage(enum.StrEnum):
    PARSING = "parsing"
    OCR = "ocr"


class ParsingIssue(BaseModel):
    """A warning or error found while parsing. `code` is stable and machine-readable."""

    code: str
    message: str
    page_number: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ContentBlock(BaseModel):
    """One piece of extracted text, traceable to its place in the source document."""

    text: str
    block_type: BlockType = BlockType.PARAGRAPH
    # Only set when the format has reliable pages (PDF). Never invented.
    page_number: int | None = None
    section_title: str | None = None
    source_location: dict[str, Any] | None = None
    sequence_number: int = 0
    metadata: dict[str, Any] | None = None
    # The text before normalization; None when normalization changed nothing.
    raw_text: str | None = None


class FormulaRegion(BaseModel):
    """A display-formula region found by layout analysis (PDF), with or without text.

    On a scanned page the region has no text at all; its position is what matters.
    """

    page_number: int
    # Position on the page in PDF points: l, t, r, b and coord_origin ("BOTTOMLEFT").
    bbox: dict[str, Any]
    docling_ref: str | None = None
    original_text: str | None = None  # the PDF's own text for the region, if it has a text layer
    # True for a small wide region that layout analysis called a *picture*. Layout analysis often
    # labels real equations that way, so these are tried with math OCR and kept only if the
    # result looks like mathematics.
    candidate: bool = False


class OcrSummary(BaseModel):
    engine: str
    language: str
    dpi: int
    pages_processed: list[int] = Field(default_factory=list)
    pages_failed: list[int] = Field(default_factory=list)
    pages_kept_digital: list[int] = Field(default_factory=list)
    mean_confidence: float | None = None


class ParsingResult(BaseModel):
    document_id: uuid.UUID
    original_filename: str
    file_format: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_method: str
    page_count: int | None = None
    blocks: list[ContentBlock] = Field(default_factory=list)
    warnings: list[ParsingIssue] = Field(default_factory=list)
    errors: list[ParsingIssue] = Field(default_factory=list)
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    ocr: OcrSummary | None = None
    # Format-specific extras (e.g. EPUB title, number of spine items).
    metadata: dict[str, Any] = Field(default_factory=dict)
    # PDF layout analysis: where display formulas are (used by equation extraction).
    formula_regions: list[FormulaRegion] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def character_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)
