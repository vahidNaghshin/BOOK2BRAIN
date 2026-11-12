"""Data carried between the structure-detection, paragraph and passage stages.

These are plain in-memory records (no database access), so every stage can be tested on its own.
The structure service maps them to and from the SQLAlchemy models.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any


class NodeType(enum.StrEnum):
    """Structural node types. The database does not constrain them, so this list can grow."""

    BOOK_TITLE = "book_title"
    FRONT_MATTER = "front_matter"
    BACK_MATTER = "back_matter"
    PREFACE = "preface"
    FOREWORD = "foreword"
    ACKNOWLEDGEMENT = "acknowledgement"
    INTRODUCTION = "introduction"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    SUBSUBSECTION = "subsubsection"
    APPENDIX = "appendix"
    NOTES = "notes"
    BIBLIOGRAPHY = "bibliography"
    GLOSSARY = "glossary"
    OTHER_HEADING = "other_heading"
    # Additions beyond the required minimum.
    TABLE_OF_CONTENTS = "table_of_contents"
    COPYRIGHT = "copyright"
    DEDICATION = "dedication"
    INDEX = "index"
    ABOUT_AUTHOR = "about_author"


# Types that belong in the front matter / back matter of a book.
FRONT_ONLY_TYPES = frozenset(
    {
        NodeType.BOOK_TITLE,
        NodeType.PREFACE,
        NodeType.FOREWORD,
        NodeType.DEDICATION,
        NodeType.COPYRIGHT,
        NodeType.TABLE_OF_CONTENTS,
    }
)
BACK_TYPES = frozenset(
    {
        NodeType.APPENDIX,
        NodeType.NOTES,
        NodeType.BIBLIOGRAPHY,
        NodeType.GLOSSARY,
        NodeType.INDEX,
        NodeType.ABOUT_AUTHOR,
    }
)

# Hierarchy ranks: a lower rank is a more prominent heading. Containers are rank 0.
RANK_CONTAINER = 0
RANK_PART = 10
RANK_CHAPTER = 20
RANK_SECTION = 30
RANK_SUBSECTION = 40
RANK_SUBSUBSECTION = 50


def type_for_rank(rank: int) -> NodeType:
    if rank <= RANK_PART:
        return NodeType.PART
    if rank <= RANK_CHAPTER:
        return NodeType.CHAPTER
    if rank <= RANK_SECTION:
        return NodeType.SECTION
    if rank <= RANK_SUBSECTION:
        return NodeType.SUBSECTION
    return NodeType.SUBSUBSECTION


class Severity(enum.StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A structured finding. `code` is stable and machine-readable."""

    severity: Severity
    code: str
    message: str
    entity_type: str | None = None  # node, paragraph, passage, document
    entity_id: uuid.UUID | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id) if self.entity_id else None,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationIssue:
        entity = data.get("entity_id")
        return cls(
            severity=Severity(data.get("severity", "warning")),
            code=str(data.get("code", "")),
            message=str(data.get("message", "")),
            entity_type=data.get("entity_type"),
            entity_id=uuid.UUID(entity) if entity else None,
            details=dict(data.get("details") or {}),
        )


@dataclass
class SourceBlock:
    """One extracted block (Day 3), plus its offsets in the document's canonical text."""

    sequence: int
    text: str
    block_type: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None  # l, t, r, b
    coord_origin: str | None = None
    heading_level: int | None = None
    id: uuid.UUID | None = None
    start_offset: int = 0
    end_offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> float | None:
        return abs(self.bbox[3] - self.bbox[1]) if self.bbox else None


@dataclass
class DocumentText:
    """Everything the structure stages need to know about a processed document."""

    blocks: list[SourceBlock]
    page_count: int | None = None
    file_format: str = "pdf"
    source_title: str | None = None  # title from the file's own metadata (e.g. EPUB), if any
    equation_ids_by_block: dict[uuid.UUID, uuid.UUID] = field(default_factory=dict)
    # Set when the blocks are not loaded (passage regeneration) but page support is known.
    known_pages: bool | None = None

    @property
    def has_pages(self) -> bool:
        if self.known_pages is not None:
            return self.known_pages
        return any(block.page is not None for block in self.blocks)


@dataclass
class NodeDraft:
    id: uuid.UUID
    node_type: str
    title: str
    level: int
    sequence: int
    parent_id: uuid.UUID | None = None
    original_title: str | None = None
    number: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Block sequence numbers consumed as this node's heading (never repeated as body text).
    heading_blocks: list[int] = field(default_factory=list)
    # First block that belongs to the node (its heading, or its first content block).
    start_block: int | None = None
    # Last block of the node's subtree.
    end_block: int | None = None
    rank: int = RANK_CHAPTER


@dataclass
class ParagraphDraft:
    id: uuid.UUID
    node_id: uuid.UUID
    sequence: int  # inside the node
    document_sequence: int
    kind: str
    text: str
    block_sequences: list[int]
    page_start: int | None = None
    page_end: int | None = None
    page_status: str = "unknown"
    start_offset: int | None = None
    end_offset: int | None = None
    # True when `text` is exactly the slice of the canonical text at the offsets (one block).
    offsets_exact: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PassagePart:
    paragraph_id: uuid.UUID
    position: int
    is_overlap: bool = False
    text_start: int | None = None
    text_end: int | None = None


@dataclass
class PassageDraft:
    id: uuid.UUID
    node_id: uuid.UUID
    sequence: int
    document_sequence: int
    text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    page_status: str = "unknown"
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parts: list[PassagePart] = field(default_factory=list)


@dataclass
class StructureResult:
    """Output of structure detection (nodes + paragraphs) before it is saved."""

    nodes: list[NodeDraft]
    paragraphs: list[ParagraphDraft]
    metadata_guess: dict[str, Any]
    issues: list[ValidationIssue]
    summary: dict[str, Any]
