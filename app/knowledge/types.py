"""Vocabularies of the knowledge model. Independent of Streamlit, the database and any LLM provider."""

import enum


class KnowledgeType(enum.StrEnum):
    # Must match database.models.knowledge.KNOWLEDGE_TYPES (a test enforces this).
    CONCEPT = "concept"
    PRINCIPLE = "principle"
    METHOD = "method"
    FRAMEWORK = "framework"
    EVIDENCE = "evidence"
    EXAMPLE = "example"
    EQUATION = "equation"


class RelationshipType(enum.StrEnum):
    """Read as "source <type> target". See app.knowledge.relationships for what each may connect."""

    # Must match database.models.knowledge.RELATIONSHIP_TYPES (a test enforces this).
    SUPPORTS = "supports"
    ILLUSTRATES = "illustrates"
    IMPLEMENTS = "implements"
    DESCRIBES = "describes"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    EXAMPLE_OF = "example_of"
    APPLIES_TO = "applies_to"


class ValidationStatus(enum.StrEnum):
    """Whether a unit passed automatic checks. Independent of human review."""

    # Must match database.models.knowledge.KNOWLEDGE_VALIDATION_STATUSES (a test enforces this).
    PENDING = "pending"
    VALID = "valid"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"


class ReviewStatus(enum.StrEnum):
    # Must match database.models.knowledge.REVIEW_STATUSES (a test enforces this).
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"


class ExtractionRunStatus(enum.StrEnum):
    # Must match database.models.knowledge.EXTRACTION_RUN_STATUSES (a test enforces this).
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"  # finished and nothing needs a human look
    NEEDS_REVIEW = "needs_review"  # finished, but some extractions are uncertain, flagged or failed
    FAILED = "failed"  # nothing usable came out, or the run could not continue
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self not in (ExtractionRunStatus.PENDING, ExtractionRunStatus.PROCESSING)


class SourceRole(enum.StrEnum):
    # Must match database.models.knowledge.SOURCE_ROLES (a test enforces this).
    PRIMARY = "primary"
    SUPPORTING = "supporting"


class PageStatus(enum.StrEnum):
    """How reliable a page range is. Same meaning as Passage.page_status."""

    # Must match database.models.structure.PAGE_STATUSES (a test enforces this).
    EXACT = "exact"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class ExpressionStatus(enum.StrEnum):
    """How completely the equation's expression could be read from the source."""

    COMPLETE = "complete"
    PARTIAL = "partial"  # some of the expression is missing or cut off
    AMBIGUOUS = "ambiguous"  # the expression is there but can be read more than one way
    FAILED = "failed"  # nothing usable could be read


class ReconstructionStatus(enum.StrEnum):
    """Whether a clean LaTeX form was rebuilt from an imperfect original expression."""

    NOT_ATTEMPTED = "not_attempted"
    NOT_NEEDED = "not_needed"  # the original expression was already complete
    RECONSTRUCTED = "reconstructed"
    PARTIAL = "partial"
    FAILED = "failed"


class RelationshipOrigin(enum.StrEnum):
    """Where a relationship comes from. Only `extracted` ones carry evidence from the book."""

    # Must match database.models.knowledge.RELATIONSHIP_ORIGINS (a test enforces this).
    EXTRACTED = "extracted"  # the model found it in the text and quoted the evidence, which was verified
    AI_SUGGESTED = "ai_suggested"  # the model proposes it without verified evidence: needs review
    MANUAL = "manual"  # created by a person or by code that knows better


class KnowledgeOrigin(enum.StrEnum):
    """What a unit's content is. The original source stays authoritative whatever the origin."""

    # Must match database.models.knowledge.KNOWLEDGE_ORIGINS (a test enforces this).
    AI_EXTRACTED = "ai_extracted"  # structured from the source text by a model
    AI_INTERPRETATION = "ai_interpretation"  # a later model interpretation of extracted knowledge
    AI_RECONSTRUCTED = "ai_reconstructed"  # a model's reconstruction of source content (e.g. an equation)
    SOURCE_DERIVED = "source_derived"  # copied from stored source data without a model (e.g. an image-only equation)
    HUMAN = "human"


class SourceVerificationStatus(enum.StrEnum):
    # Must match database.models.knowledge.SOURCE_VERIFICATION_STATUSES (a test enforces this).
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    INVALID = "invalid"


class SourceReferenceType(enum.StrEnum):
    # Must match database.models.knowledge.SOURCE_REFERENCE_TYPES (a test enforces this).
    TEXT = "text"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    OTHER = "other"


class EquationCaptureStatus(enum.StrEnum):
    """What was actually captured of an equation from the source, not whether it is mathematically right."""

    # Must match database.models.knowledge.EQUATION_CAPTURE_STATUSES (a test enforces this).
    COMPLETE = "complete"  # expression and LaTeX both captured, nothing missing
    INCOMPLETE = "incomplete"  # part of the expression is missing or cut off
    AMBIGUOUS = "ambiguous"  # captured, but it can be read more than one way
    IMAGE_ONLY = "image_only"  # only an image of the equation exists
    TEXT_ONLY = "text_only"  # an expression as text, with no LaTeX form
    NEEDS_REVIEW = "needs_review"  # the capture is contradicted by the source or cannot be trusted


class MathValidationStatus(enum.StrEnum):
    # Must match database.models.knowledge.MATH_VALIDATION_STATUSES (a test enforces this).
    NOT_VALIDATED = "not_validated"
    VALID = "valid"
    INVALID = "invalid"
    INCONCLUSIVE = "inconclusive"


class ReviewFlagCode(enum.StrEnum):
    """Why something needs a human look. The message and details carry the specifics."""

    # Must match database.models.knowledge.REVIEW_FLAG_CODES (a test enforces this).
    INVALID_SOURCE_REFERENCE = "invalid_source_reference"
    SOURCE_QUOTE_MISMATCH = "source_quote_mismatch"
    PAGE_INCONSISTENT = "page_inconsistent"
    NO_SOURCE = "no_source"
    INCOMPLETE_EQUATION = "incomplete_equation"
    AMBIGUOUS_EQUATION = "ambiguous_equation"
    EQUATION_NEEDS_REVIEW = "equation_needs_review"
    LOW_CONFIDENCE = "low_confidence"
    UNCERTAIN_EXTRACTION = "uncertain_extraction"
    INFERRED_CONTENT = "inferred_content"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    POTENTIAL_DUPLICATE = "potential_duplicate"
    REPEATED_VALIDATION_FAILURE = "repeated_validation_failure"
    UNRESOLVED_EXTRACTION_ERROR = "unresolved_extraction_error"
    AI_SUGGESTED_RELATIONSHIP = "ai_suggested_relationship"
    CONTEXT_REDUCED = "context_reduced"
    SOURCE_OUTSIDE_TARGET = "source_outside_target"  # the unit's primary source is not the passage it was extracted from


class FlagSeverity(enum.StrEnum):
    # Must match database.models.knowledge.FLAG_SEVERITIES (a test enforces this).
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FlagStatus(enum.StrEnum):
    # Must match database.models.knowledge.FLAG_STATUSES (a test enforces this).
    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class PassageStatus(enum.StrEnum):
    """Processing state of one passage within an extraction run."""

    # Must match database.models.knowledge.PASSAGE_STATUSES (a test enforces this).
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class ValidationOutcome(enum.StrEnum):
    """The result of checking one attempt's answer. Separate from whether the call itself worked."""

    # Must match database.models.knowledge.VALIDATION_OUTCOMES (a test enforces this).
    SCHEMA_VALID = "schema_valid"
    SCHEMA_INVALID = "schema_invalid"
    SOURCE_INVALID = "source_invalid"
    SEMANTIC_CONCERNS = "semantic_concerns"
    UNSUPPORTED_CONTENT = "unsupported_content"
    DUPLICATE = "duplicate"
    NEEDS_REVIEW = "needs_review"
    NOT_VALIDATED = "not_validated"  # no answer to validate (the provider failed)
