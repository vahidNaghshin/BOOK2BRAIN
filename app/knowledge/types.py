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
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (ExtractionRunStatus.COMPLETED, ExtractionRunStatus.FAILED, ExtractionRunStatus.CANCELLED)


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
