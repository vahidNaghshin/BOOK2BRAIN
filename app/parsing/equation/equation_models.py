"""Structured models for extracted equations. Independent of Streamlit and the database."""

import enum
import uuid
from typing import Any

from pydantic import BaseModel, Field, model_validator


class EquationType(enum.StrEnum):
    # Must match database.models.equation.EQUATION_TYPES (a test enforces this).
    INLINE = "inline"
    DISPLAY = "display"
    NUMBERED = "numbered"
    UNKNOWN = "unknown"


class ValidationStatus(enum.StrEnum):
    # Must match database.models.equation.VALIDATION_STATUSES (a test enforces this).
    PENDING = "pending"
    VALID = "valid"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"


class EquationExtractionStatus(enum.StrEnum):
    # Must match database.models.processing.EquationStatus (a test enforces this).
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class ExtractionMethod(enum.StrEnum):
    """How the LaTeX was obtained. Authoritative sources carry the author's own markup."""

    MATHML = "mathml"
    TEX_ANNOTATION = "tex_annotation"  # TeX stored inside MathML by the book's producer
    LATEX_DELIMITERS = "latex_delimiters"  # $...$, \(...\), \[...\] in the text
    IMAGE_ALT_TEXT = "image_alt_text"
    OMML = "omml"
    DOCLING_ENRICHMENT = "docling_formula_enrichment"
    MATH_OCR = "math_ocr"

    @property
    def authoritative(self) -> bool:
        """True when the markup comes from the document itself, not from a recognition model."""
        return self in {
            ExtractionMethod.MATHML,
            ExtractionMethod.TEX_ANNOTATION,
            ExtractionMethod.LATEX_DELIMITERS,
            ExtractionMethod.IMAGE_ALT_TEXT,
            ExtractionMethod.OMML,
        }


class EquationBlock(BaseModel):
    """One extracted equation, traceable to its place in the source document.

    `latex` is the normalized form; `original_latex` is exactly what the extractor produced.
    `confidence` is only ever what a recognition engine reported - never made up.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    document_id: uuid.UUID
    extracted_block_id: uuid.UUID | None = None
    sequence_number: int = Field(default=0, ge=0)
    latex: str
    original_latex: str | None = None
    plain_text: str | None = None
    equation_type: EquationType = EquationType.UNKNOWN
    equation_category: str | None = None  # unknown for now; kept for later classification
    equation_number: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    section_title: str | None = None
    surrounding_text: str | None = None
    source_location: dict[str, Any] | None = None
    bounding_box: dict[str, Any] | None = None
    crop_reference: str | None = None
    extraction_method: ExtractionMethod
    confidence: float | None = Field(default=None, ge=0, le=1)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_warnings: list[str] = Field(default_factory=list)
    # Room for later phases (variables, units, assumptions...); also holds original MathML/OMML.
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Used only to link the equation to its text block; never stored.
    link_hint: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _empty_latex_is_a_failure(self) -> "EquationBlock":
        if not self.latex.strip() and self.validation_status not in (
            ValidationStatus.PENDING,
            ValidationStatus.INVALID,
        ):
            raise ValueError("An equation without LaTeX can only be pending or invalid")
        return self


class EquationIssue(BaseModel):
    """A problem found while extracting equations (separate from document parsing errors)."""

    code: str
    message: str
    page_number: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EnrichmentReport(BaseModel):
    """Whether Docling's formula enrichment was enabled and actually produced LaTeX."""

    enabled: bool = False
    executed: bool = False
    pages: list[int] = Field(default_factory=list)
    formulas_seen: int = 0
    formulas_enriched: int = 0
    device: str | None = None
    dtype: str | None = None
    seconds: float | None = None
    error: str | None = None


class MathOcrReport(BaseModel):
    engine: str | None = None
    enabled: bool = False
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    pictures_accepted: int = 0  # picture regions that turned out to be equations
    pictures_rejected: int = 0  # picture regions tried and discarded (real figures)
    unavailable_reason: str | None = None


class EquationExtractionResult(BaseModel):
    """Outcome of equation extraction for one document (also stored on the processing run)."""

    status: EquationExtractionStatus = EquationExtractionStatus.NOT_STARTED
    equations: list[EquationBlock] = Field(default_factory=list)
    issues: list[EquationIssue] = Field(default_factory=list)
    enrichment: EnrichmentReport = Field(default_factory=EnrichmentReport)
    math_ocr: MathOcrReport = Field(default_factory=MathOcrReport)
    seconds: float | None = None

    def count(self, status: ValidationStatus) -> int:
        return sum(1 for equation in self.equations if equation.validation_status is status)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary for the processing run (equations themselves are stored apart)."""
        return {
            "status": self.status.value,
            "total": len(self.equations),
            "valid": self.count(ValidationStatus.VALID),
            "uncertain": self.count(ValidationStatus.UNCERTAIN),
            "invalid": self.count(ValidationStatus.INVALID),
            "enrichment": self.enrichment.model_dump(mode="json"),
            "math_ocr": self.math_ocr.model_dump(mode="json"),
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
            "seconds": self.seconds,
        }
