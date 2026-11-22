"""Source provenance for knowledge units: capture, retrieval and verification.

A unit is backed by passages (Phase 1), which stay the authority: this module only reads them, never
changes them, and never stores model output as if it were source text. When a unit is stored,
`capture` completes its source links from the passages (location, pages, where the quote sits) and keeps
a snapshot of the source, so provenance survives when passages are later regenerated. `verify_unit`
re-checks every reference, records what it found on the links and the unit, and raises review flags;
nothing invalid is silently dropped.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.extraction.results import utcnow
from app.knowledge.types import FlagSeverity, ReviewFlagCode, SourceVerificationStatus
from app.repositories.passage_repository import PassageRepository
from app.services.review_flag_service import ReviewFlagService
from database.models import Book, Document, KnowledgeUnit, KnowledgeUnitPassage, Passage, StructuralNode

logger = logging.getLogger(__name__)

SNAPSHOT_EXCERPT_CHARS = 600
SOURCE_FLAG_CODES = (
    ReviewFlagCode.INVALID_SOURCE_REFERENCE,
    ReviewFlagCode.SOURCE_QUOTE_MISMATCH,
    ReviewFlagCode.PAGE_INCONSISTENT,
    ReviewFlagCode.NO_SOURCE,
)


def _squash(text: str) -> str:
    return " ".join(text.split()).casefold()


class SourceIssue(BaseModel):
    code: ReviewFlagCode
    severity: FlagSeverity
    message: str
    key: str  # tells apart issues of one code (for example one per passage)
    passage_id: uuid.UUID | None = None


class SourceReferenceDetail(BaseModel):
    passage_id: uuid.UUID
    position: int
    role: str
    reference_type: str
    quote: str | None
    quote_start: int | None
    quote_end: int | None
    page_start: int | None
    page_end: int | None
    structural_node_id: uuid.UUID | None
    chapter_title: str | None
    section_title: str | None
    document_id: uuid.UUID | None
    location: dict[str, Any] | None
    verification_status: SourceVerificationStatus
    verification_issues: list[dict[str, Any]]
    verified_at: datetime | None
    text: str | None = None  # the passage text, when asked for


class VerificationReport(BaseModel):
    unit_id: uuid.UUID
    status: SourceVerificationStatus
    references_checked: int
    issues: list[SourceIssue]
    flags_raised: int
    flags_resolved: int
    checked_at: datetime


class ProvenanceService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.passages = PassageRepository(session)
        self.flags = ReviewFlagService(session)

    # -- capture (when a unit is stored) -----------------------------------------------------------

    def capture(self, unit: KnowledgeUnit) -> None:
        """Complete a new unit's source links from its passages and keep a snapshot. Does not commit."""
        ids = [link.passage_id for link in unit.sources]
        found = {p.id: p for p in self.session.scalars(select(Passage).where(Passage.id.in_(ids)))} if ids else {}
        reference_type = "equation" if unit.knowledge_type == "equation" else "text"
        snapshot: list[dict[str, Any]] = []
        for link in sorted(unit.sources, key=lambda link: link.position):
            passage = found.get(link.passage_id)
            if passage is None:
                continue
            located = self.passages.source_of(passage)
            link.reference_type = reference_type
            link.structural_node_id = passage.structural_node_id
            link.page_start, link.page_end = passage.page_start, passage.page_end
            excerpt, is_quote = passage.text[:SNAPSHOT_EXCERPT_CHARS], False
            if link.quote:
                start = passage.text.find(link.quote)
                if start >= 0:
                    link.quote_start, link.quote_end = start, start + len(link.quote)
                    excerpt, is_quote = passage.text[start : start + len(link.quote)], True
            snapshot.append(
                {
                    "passage_id": str(passage.id),
                    "document_id": str(passage.document_id),
                    "chapter": located.chapter.title if located.chapter else None,
                    "section": located.section.title if located.section else None,
                    "pages": [passage.page_start, passage.page_end],
                    "excerpt": excerpt,
                    "excerpt_is_quote": is_quote,
                    "role": link.role,
                }
            )
        unit.source_snapshot = snapshot

    # -- retrieval ---------------------------------------------------------------------------------

    def sources_for(self, unit_id: uuid.UUID, *, include_text: bool = False) -> list[SourceReferenceDetail]:
        unit = self.session.get(KnowledgeUnit, unit_id)
        if unit is None:
            raise LookupError("The knowledge unit does not exist.")
        links = sorted(unit.sources, key=lambda link: link.position)
        passages = {p.id: p for p in self.session.scalars(select(Passage).where(Passage.id.in_([l.passage_id for l in links])))} if links else {}
        details = []
        for link in links:
            passage = passages.get(link.passage_id)
            located = self.passages.source_of(passage) if passage else None
            details.append(
                SourceReferenceDetail(
                    passage_id=link.passage_id,
                    position=link.position,
                    role=link.role,
                    reference_type=link.reference_type,
                    quote=link.quote,
                    quote_start=link.quote_start,
                    quote_end=link.quote_end,
                    page_start=link.page_start,
                    page_end=link.page_end,
                    structural_node_id=link.structural_node_id,
                    chapter_title=located.chapter.title if located and located.chapter else None,
                    section_title=located.section.title if located and located.section else None,
                    document_id=passage.document_id if passage else None,
                    location=link.location,
                    verification_status=link.verification_status,
                    verification_issues=list(link.verification_issues or []),
                    verified_at=link.verified_at,
                    text=passage.text if (include_text and passage) else None,
                )
            )
        return details

    def snapshot_of(self, unit_id: uuid.UUID) -> list[dict[str, Any]]:
        """The source as it was when the unit was stored (still there if the passages were regenerated)."""
        unit = self.session.get(KnowledgeUnit, unit_id)
        if unit is None:
            raise LookupError("The knowledge unit does not exist.")
        return list(unit.source_snapshot or [])

    # -- verification ------------------------------------------------------------------------------

    def verify_unit(self, unit_id: uuid.UUID, *, flag: bool = True, commit: bool = True, run_id: uuid.UUID | None = None) -> VerificationReport:
        unit = self.session.get(KnowledgeUnit, unit_id)
        if unit is None:
            raise LookupError("The knowledge unit does not exist.")
        self.session.refresh(unit)  # links may have changed since the unit was loaded
        now = utcnow()
        issues: list[SourceIssue] = []
        link_issues: dict[uuid.UUID, list[SourceIssue]] = {}

        def add(code: ReviewFlagCode, severity: FlagSeverity, message: str, key: str, passage_id: uuid.UUID | None = None) -> None:
            issue = SourceIssue(code=code, severity=severity, message=message, key=key, passage_id=passage_id)
            issues.append(issue)
            if passage_id is not None:
                link_issues.setdefault(passage_id, []).append(issue)

        book = self.session.get(Book, unit.book_id)
        document = self.session.get(Document, unit.document_id)
        if book is None:
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, "The unit's book does not exist.", "book")
        if document is None:
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, "The unit's document does not exist.", "document")
        elif document.book_id != unit.book_id:
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, "The unit's document belongs to another book.", "document")

        links = sorted(unit.sources, key=lambda link: link.position)
        snapshot_ids = {s.get("passage_id") for s in (unit.source_snapshot or [])}
        if not links:
            if snapshot_ids:
                add(
                    ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.WARNING,
                    "The source passages no longer exist (the document's passages were regenerated). "
                    "The source snapshot taken when the unit was stored is kept.", "passages_gone",
                )
            else:
                add(ReviewFlagCode.NO_SOURCE, FlagSeverity.ERROR, "The unit cites no source passage.", "no_source")

        passages = {p.id: p for p in self.session.scalars(select(Passage).where(Passage.id.in_([l.passage_id for l in links])))} if links else {}
        for link in links:
            self._check_link(unit, link, passages.get(link.passage_id), add)
        if links and snapshot_ids and not {str(l.passage_id) for l in links} <= snapshot_ids:
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.WARNING, "A source link is not in the unit's stored source snapshot.", "snapshot")

        self._check_structure(unit, add)
        self._check_pages(unit, links, add)

        # record on the links and the unit
        for link in links:
            mine = link_issues.get(link.passage_id, [])
            link.verification_status = (SourceVerificationStatus.INVALID if mine else SourceVerificationStatus.VERIFIED).value
            link.verification_issues = [i.model_dump(mode="json") for i in mine]
            link.verified_at = now
        status = SourceVerificationStatus.INVALID if issues else SourceVerificationStatus.VERIFIED
        unit.source_status = status.value
        if issues and unit.validation_status == "valid":
            unit.validation_status = "uncertain"

        raised = resolved = 0
        if flag:
            for issue in issues:
                _, created = self.flags.flag_unit(
                    unit.book_id, unit.id, issue.code, issue.message, severity=issue.severity, dedupe_key=issue.key,
                    details={"passage_id": str(issue.passage_id) if issue.passage_id else None}, run_id=run_id, commit=False,
                )
                raised += created
            resolved = self.flags.resolve_missing(
                unit.id, SOURCE_FLAG_CODES, {(i.code.value, i.key) for i in issues}, "Verified again: no longer applies.", commit=False
            )
        if commit:
            self.session.commit()
        else:
            self.session.flush()
        return VerificationReport(
            unit_id=unit.id, status=status, references_checked=len(links), issues=issues, flags_raised=raised, flags_resolved=resolved, checked_at=now
        )

    def verify_book(self, book_id: uuid.UUID, *, only_unverified: bool = False) -> dict[str, int]:
        """Verify every unit of a book. Returns counts of verified and invalid units."""
        query = select(KnowledgeUnit.id).where(KnowledgeUnit.book_id == book_id)
        if only_unverified:
            query = query.where(KnowledgeUnit.source_status == "unverified")
        counts = {"verified": 0, "invalid": 0}
        for unit_id in list(self.session.scalars(query)):
            counts[self.verify_unit(unit_id, commit=False).status.value] += 1
        self.session.commit()
        return counts

    # -- the checks --------------------------------------------------------------------------------

    def _check_link(self, unit: KnowledgeUnit, link: KnowledgeUnitPassage, passage: Passage | None, add) -> None:
        pid = link.passage_id
        if passage is None:
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, f"Passage {pid} does not exist.", f"passage:{pid}", pid)
            return
        if passage.book_id != unit.book_id:
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, f"Passage {pid} belongs to another book.", f"passage:{pid}", pid)
        elif passage.document_id != unit.document_id:
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, f"Passage {pid} belongs to another document than the unit.", f"passage:{pid}", pid)
        if link.structural_node_id is not None and link.structural_node_id != passage.structural_node_id:
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.WARNING, f"The stored location of passage {pid} no longer matches the passage.", f"location:{pid}", pid)
        if (link.page_start, link.page_end) != (passage.page_start, passage.page_end) and link.page_start is not None:
            add(ReviewFlagCode.PAGE_INCONSISTENT, FlagSeverity.WARNING, f"The stored pages of passage {pid} no longer match the passage.", f"pages:{pid}", pid)
        if link.quote:
            if _squash(link.quote) not in _squash(passage.text):
                add(ReviewFlagCode.SOURCE_QUOTE_MISMATCH, FlagSeverity.WARNING, "The quoted words are not in the cited passage.", f"quote:{pid}", pid)
            elif link.quote_start is not None and passage.text[link.quote_start : link.quote_end] != link.quote:
                add(ReviewFlagCode.SOURCE_QUOTE_MISMATCH, FlagSeverity.WARNING, "The stored position of the quote no longer matches the passage text.", f"quote:{pid}", pid)

    def _check_structure(self, unit: KnowledgeUnit, add) -> None:
        nodes: dict[str, StructuralNode | None] = {}
        for label, node_id in (("chapter", unit.chapter_node_id), ("section", unit.section_node_id)):
            if node_id is None:
                continue
            node = self.session.get(StructuralNode, node_id)
            nodes[label] = node
            if node is None:
                add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, f"The {label} {node_id} does not exist.", f"node:{node_id}")
            elif node.book_id != unit.book_id:
                add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, f"The {label} '{node.title}' belongs to another book.", f"node:{node_id}")
            elif node.document_id != unit.document_id:
                add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.ERROR, f"The {label} '{node.title}' belongs to another document.", f"node:{node_id}")
        chapter, section = nodes.get("chapter"), nodes.get("section")
        if chapter and section and chapter.book_id == section.book_id and section.id not in self.passages.subtree_ids(chapter.id):
            add(ReviewFlagCode.INVALID_SOURCE_REFERENCE, FlagSeverity.WARNING, f"The section '{section.title}' is not inside the chapter '{chapter.title}'.", f"node:{section.id}")

    def _check_pages(self, unit: KnowledgeUnit, links: list[KnowledgeUnitPassage], add) -> None:
        if unit.page_start is None:
            return
        if unit.page_end is not None and unit.page_end < unit.page_start:
            add(ReviewFlagCode.PAGE_INCONSISTENT, FlagSeverity.WARNING, "The page range is reversed.", "pages")
            return
        passages = [p for p in (self.session.get(Passage, l.passage_id) for l in links) if p is not None]
        starts = [p.page_start for p in passages if p.page_start is not None]
        ends = [p.page_end or p.page_start for p in passages if p.page_start is not None]
        last = unit.page_end or unit.page_start
        if passages and not starts:
            add(ReviewFlagCode.PAGE_INCONSISTENT, FlagSeverity.WARNING, "The unit has pages but its source passages have none.", "pages")
        elif starts and (unit.page_start < min(starts) or last > max(ends)):
            add(ReviewFlagCode.PAGE_INCONSISTENT, FlagSeverity.WARNING, f"The pages {unit.page_start}-{last} are outside the cited passages' pages {min(starts)}-{max(ends)}.", "pages")
        document_last = self.session.scalar(select(func.max(Passage.page_end)).where(Passage.document_id == unit.document_id))
        if document_last is not None and last > document_last:
            add(ReviewFlagCode.PAGE_INCONSISTENT, FlagSeverity.WARNING, f"Page {last} is beyond the document's last known page ({document_last}).", "pages_document")
