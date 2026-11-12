"""Structure detection pipeline: headings -> metadata -> hierarchy -> paragraphs -> validation.

Pure computation on a `DocumentText`; no database access. The deterministic stages always run;
the LLM stage runs only when a classifier is supplied and ambiguous headings remain.
"""

from __future__ import annotations

import logging
from collections import Counter

from app.structure.front_matter import detect_metadata
from app.structure.heading_detection import HeadingCandidate, detect_headings
from app.structure.hierarchy import build_hierarchy
from app.structure.llm import HeadingClassifier, apply_llm_classification
from app.structure.models import (
    DocumentText,
    NodeType,
    Severity,
    StructureResult,
    ValidationIssue,
)
from app.structure.numbering import normalise_for_match
from app.structure.paragraphs import build_paragraphs
from app.structure.toc import detect_toc
from app.structure.typography import body_line_height
from app.structure.validation import count_by_severity, validate_structure

logger = logging.getLogger(__name__)

_STRUCTURAL_KEYWORD_TYPES = {
    NodeType.PREFACE, NodeType.FOREWORD, NodeType.TABLE_OF_CONTENTS, NodeType.DEDICATION,
    NodeType.ACKNOWLEDGEMENT, NodeType.INTRODUCTION,
}  # fmt: skip


def _first_structural_index(candidates: list[HeadingCandidate]) -> int | None:
    """Index of the first heading that clearly belongs to the book proper (not a title line)."""
    for candidate in candidates:
        if candidate.status != "accepted":
            continue
        pattern, keyword = candidate.pattern, candidate.keyword
        if pattern is not None and pattern.form == "label" and pattern.node_type in (NodeType.PART, NodeType.CHAPTER):
            return candidate.index
        if keyword is not None and keyword.node_type in _STRUCTURAL_KEYWORD_TYPES:
            return candidate.index
    return None


class StructureDetector:
    def __init__(
        self,
        *,
        accept_threshold: float = 0.70,
        reject_threshold: float = 0.45,
        classifier: HeadingClassifier | None = None,
        llm_max_candidates: int = 60,
    ) -> None:
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.classifier = classifier
        self.llm_max_candidates = llm_max_candidates

    def detect(self, document: DocumentText) -> StructureResult:
        blocks = document.blocks
        issues: list[ValidationIssue] = []

        toc = detect_toc(blocks)
        detection = detect_headings(
            document, toc, accept=self.accept_threshold, reject=self.reject_threshold
        )
        candidates = detection.candidates

        guess = detect_metadata(
            blocks,
            reference_height=body_line_height(blocks),
            source_title=document.source_title,
            structural_start=_first_structural_index(candidates),
        )

        # Title-page and copyright lines are not headings, so they are not worth an LLM question.
        front_lines = {
            guess.title_block, guess.subtitle_block, guess.author_block, *guess.title_repeats, *guess.copyright_blocks,
        }
        candidates = [
            c
            for c in candidates
            if not (set(c.block_indices) & front_lines)
            or (c.keyword is not None and c.keyword.node_type is NodeType.COPYRIGHT)
        ]

        llm_used = 0
        if self.classifier is not None:
            before = sum(1 for c in candidates if c.status == "ambiguous")
            candidates, llm_issues = apply_llm_classification(
                candidates,
                blocks,
                self.classifier,
                max_candidates=self.llm_max_candidates,
                accept_threshold=self.accept_threshold,
            )
            issues.extend(llm_issues)
            llm_used = sum(1 for c in candidates if c.llm)
            logger.info("LLM looked at %d ambiguous headings, classified %d", before, llm_used)

        hierarchy = build_hierarchy(document, candidates, guess)
        issues.extend(hierarchy.issues)
        paragraphs = build_paragraphs(document, hierarchy.nodes, hierarchy.block_nodes)

        for rejected in detection.rejected:
            issues.append(
                ValidationIssue(
                    Severity.INFO,
                    "heading_rejected",
                    f"The text '{rejected.original_text[:80]}' was labelled a heading by the parser but "
                    "did not look like one; it is kept as body text.",
                    "document",
                    details={"sequence": rejected.sequence, "page": rejected.page, "score": round(rejected.score, 2)},
                )
            )
        if detection.running_headers:
            issues.append(
                ValidationIssue(
                    Severity.INFO,
                    "running_headers_ignored",
                    f"{len(detection.running_headers)} repeated page headers were not treated as headings.",
                    "document",
                    details={"texts": detection.running_headers[:10]},
                )
            )
        issues.extend(self._toc_issues(toc.entries, hierarchy.nodes))
        if not document.has_pages:
            issues.append(
                ValidationIssue(
                    Severity.INFO,
                    "no_page_information",
                    f"This {document.file_format.upper()} has no reliable page numbers; page references are unknown.",
                    "document",
                )
            )

        issues.extend(validate_structure(document, hierarchy.nodes, paragraphs))

        counts = Counter(node.node_type for node in hierarchy.nodes)
        summary = {
            "node_count": len(hierarchy.nodes),
            "paragraph_count": len(paragraphs),
            "node_types": dict(counts),
            "max_level": max((n.level for n in hierarchy.nodes), default=0),
            "ambiguous_headings": sum(1 for n in hierarchy.nodes if n.metadata.get("status") == "ambiguous"),
            "headings_rejected": len(detection.rejected),
            "llm_classified": llm_used,
            "toc_entries": len(toc.entries),
            "block_count": len(blocks),
            "page_count": document.page_count,
            "issues": count_by_severity(issues),
        }
        logger.info(
            "Detected %d structural nodes and %d paragraphs from %d blocks (%d ambiguous headings, %d issues)",
            len(hierarchy.nodes), len(paragraphs), len(blocks), summary["ambiguous_headings"], len(issues),
        )
        return StructureResult(
            nodes=hierarchy.nodes,
            paragraphs=paragraphs,
            metadata_guess=guess.to_dict(),
            issues=issues,
            summary=summary,
        )

    @staticmethod
    def _toc_issues(entries: list, nodes: list) -> list[ValidationIssue]:
        if len(entries) < 3:
            return []
        titles = {normalise_for_match(n.title) for n in nodes} | {
            normalise_for_match(n.original_title) for n in nodes if n.original_title
        }
        matched = sum(1 for entry in entries if entry.key in titles or any(entry.key in t for t in titles if t))
        if matched / len(entries) < 0.5:
            return [
                ValidationIssue(
                    Severity.WARNING,
                    "toc_mismatch",
                    f"Only {matched} of {len(entries)} table-of-contents entries match a detected heading; "
                    "the detected structure may be incomplete.",
                    "document",
                    details={"toc_entries": len(entries), "matched": matched},
                )
            ]
        return []
