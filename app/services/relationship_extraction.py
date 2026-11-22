"""Extracting relationships between the knowledge units of a passage, and storing them.

One model call per passage, after its knowledge units exist. The model sees the passage, the units
extracted from it and a few earlier units of the same chapter as possible targets. A relationship it can
quote evidence for (verified word for word in the passage) is stored as `extracted`; one without
verifiable evidence is stored as an `ai_suggested` proposal and flagged for review. Similar names never
create a relationship by themselves. Duplicates are handled by RelationshipService, so repeating or
resuming the step does not create them twice.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.extraction.model_config import LlmModelConfig
from app.extraction.relationship_parsing import ParsedRelationships, parse_relationships
from app.extraction.relationship_prompts import RELATIONSHIP_PROMPT, RelationshipRequest, UnitSummary
from app.extraction.results import ExecutionMetadata, PassageInput, Severity, SourceInfo, ValidationIssue, ValidationStage
from app.extraction.service import ExtractionService
from app.knowledge.schemas import KnowledgeRelationshipCreate
from app.knowledge.types import KnowledgeType, RelationshipOrigin, ReviewFlagCode
from app.repositories.extraction_repository import ExtractionRepository, build_execution
from app.services.errors import RelationshipError
from app.services.relationship_service import RelationshipService
from app.services.review_flag_service import ReviewFlagService
from database.models import ExtractionRun, KnowledgeUnit, KnowledgeUnitPassage, Passage

logger = logging.getLogger(__name__)


@dataclass
class RelationshipOutcome:
    status: str  # succeeded, needs_review, failed
    created: int = 0
    extracted: int = 0  # created with verified evidence
    suggested: int = 0  # created as AI suggestions
    existing: int = 0  # already stored
    upgraded: int = 0
    rejected: list[str] = field(default_factory=list)
    failure: tuple[str, str] | None = None
    execution_id: uuid.UUID | None = None


class RelationshipExtractionService:
    def __init__(self, session: Session, extractor: ExtractionService, *, context_units: int = 10) -> None:
        self.session = session
        self.extractor = extractor
        self.context_units = context_units
        self.relationships = RelationshipService(session)
        self.flags = ReviewFlagService(session)
        self.runs = ExtractionRepository(session)

    # -- which units the model is shown ------------------------------------------------------------

    def candidates(self, passage: Passage) -> list[UnitSummary]:
        """The units extracted from this passage, then a few earlier units of the same chapter."""
        own = self._units_of(passage)
        if not own:
            return []
        summaries = [self._summary(u, True) for u in own]
        chapter_ids = {u.chapter_node_id for u in own if u.chapter_node_id}
        if self.context_units and chapter_ids:
            earlier = self.session.scalars(
                select(KnowledgeUnit)
                .join(KnowledgeUnitPassage, KnowledgeUnitPassage.knowledge_unit_id == KnowledgeUnit.id)
                .join(Passage, Passage.id == KnowledgeUnitPassage.passage_id)
                .where(
                    KnowledgeUnit.book_id == passage.book_id,
                    KnowledgeUnit.chapter_node_id.in_(chapter_ids),
                    KnowledgeUnitPassage.role == "primary",
                    Passage.id != passage.id,
                    Passage.document_id == passage.document_id,
                    Passage.document_sequence < passage.document_sequence,
                )
                .order_by(Passage.document_sequence.desc(), KnowledgeUnit.created_at)
                .limit(self.context_units)
            )
            summaries += [self._summary(u, False) for u in earlier]
        return summaries

    def _units_of(self, passage: Passage) -> list[KnowledgeUnit]:
        return list(
            self.session.scalars(
                select(KnowledgeUnit)
                .join(KnowledgeUnitPassage, KnowledgeUnitPassage.knowledge_unit_id == KnowledgeUnit.id)
                .where(
                    KnowledgeUnit.book_id == passage.book_id,
                    KnowledgeUnitPassage.passage_id == passage.id,
                    KnowledgeUnitPassage.role == "primary",
                )
                .order_by(KnowledgeUnit.created_at, KnowledgeUnit.id)
            )
        )

    @staticmethod
    def _summary(unit: KnowledgeUnit, in_passage: bool) -> UnitSummary:
        return UnitSummary(unit.id, KnowledgeType(unit.knowledge_type), unit.title, (unit.summary or unit.content or "")[:300], in_passage)

    # -- the extraction ----------------------------------------------------------------------------

    def extract_for_passage(
        self, run: ExtractionRun, passage: Passage, passage_input: PassageInput, source: SourceInfo | None, config: LlmModelConfig
    ) -> RelationshipOutcome | None:
        """Ask the model for the relationships in one passage and store them. None when there is nothing to relate."""
        units = self.candidates(passage)
        if len(units) < 2:
            return None
        request = RelationshipRequest(passage_input, units, source)
        started_at, t0 = self.extractor.start()

        def validate(text: str, finish_reason: str | None) -> tuple[ParsedRelationships, list[ValidationIssue]]:
            parsed = parse_relationships(text, request)
            issues = list(parsed.issues)
            if finish_reason == "length":
                issues.append(ValidationIssue(stage=ValidationStage.STRUCTURAL, severity=Severity.ERROR, code="truncated_response", message="The response was cut off at the output limit."))
            return parsed, issues

        outcome = self.extractor.call_with_retries(
            config=config,
            prompt=RELATIONSHIP_PROMPT,
            system=RELATIONSHIP_PROMPT.render_system(),
            user=RELATIONSHIP_PROMPT.render_user(request),
            schema=RELATIONSHIP_PROMPT.output_schema(),
            validate=validate,
            label="relationships",
        )
        meta = self.extractor.metadata_for(config, RELATIONSHIP_PROMPT, outcome.attempts, started_at, t0)
        if outcome.failure is not None:
            return self._store_failure(run, passage, meta, outcome)

        parsed: ParsedRelationships = outcome.value
        execution = self._new_execution(run, passage, meta, outcome, status="succeeded", needs_review=False, reasons=[], count=0, parsed_notes=parsed.notes)
        self.session.add(execution)
        self.session.flush()
        result = self._store_relationships(run, passage, parsed, execution.id)
        execution.items_extracted = result.created
        reasons = []
        if result.suggested:
            reasons.append(f"{result.suggested} relationship(s) were suggested by the model without evidence that could be verified.")
        if result.rejected:
            reasons.append(f"{len(result.rejected)} relationship(s) could not be stored: " + "; ".join(result.rejected[:3]))
        execution.needs_review, execution.review_reasons = bool(reasons), reasons
        execution.status = "needs_review" if reasons else "succeeded"
        result.status, result.execution_id = execution.status, execution.id
        self.runs.store_relationship_execution(run, execution, created=result.created)
        return result

    def _store_relationships(self, run: ExtractionRun, passage: Passage, parsed: ParsedRelationships, execution_id: uuid.UUID) -> RelationshipOutcome:
        outcome = RelationshipOutcome(status="succeeded")
        for item in parsed.items:
            origin = RelationshipOrigin.EXTRACTED if item.evidence_verified else RelationshipOrigin.AI_SUGGESTED
            try:
                data = KnowledgeRelationshipCreate(
                    book_id=passage.book_id,
                    source_unit_id=item.source_id,
                    target_unit_id=item.target_id,
                    relationship_type=item.relationship_type,
                    explanation=item.explanation,
                    confidence=item.confidence,
                    extraction_run_id=run.id,
                    extraction_execution_id=execution_id,
                    source_passage_ids=[passage.id],
                    origin=origin,
                    evidence_quote=item.quote,
                )
                result = self.relationships.create(data, commit=False)
            except (RelationshipError, ValueError) as exc:
                logger.warning("A relationship could not be stored: %s", exc)
                outcome.rejected.append(str(exc))
                continue
            if result.created:
                outcome.created += 1
                outcome.extracted += origin == RelationshipOrigin.EXTRACTED
                outcome.suggested += origin == RelationshipOrigin.AI_SUGGESTED
            else:
                outcome.existing += 1
                outcome.upgraded += result.upgraded
        return outcome

    # -- records -----------------------------------------------------------------------------------

    def _new_execution(self, run, passage, meta: ExecutionMetadata, outcome, *, status, needs_review, reasons, count, parsed_notes=None):
        return build_execution(
            kind="relationship",
            status=status,
            needs_review=needs_review,
            review_reasons=reasons,
            meta=meta,
            run_id=run.id,
            passage_id=passage.id,
            items_stored=count,
            issues=[i.model_dump(mode="json") for i in outcome.issues],
            notes=parsed_notes,
            raw_response=outcome.raw,
        )

    def _store_failure(self, run: ExtractionRun, passage: Passage, meta: ExecutionMetadata, outcome) -> RelationshipOutcome:
        code, message = outcome.failure
        execution = self._new_execution(
            run, passage, meta, outcome, status="failed", needs_review=True,
            reasons=[f"Relationship extraction failed ({code}) and needs a rerun."], count=0,
        )
        execution.failure_code, execution.failure_message = code, message
        self.runs.store_relationship_execution(run, execution, created=0)
        if execution.attempts:
            self.flags.flag_attempt(
                passage.book_id, execution.attempts[-1].id, ReviewFlagCode.UNRESOLVED_EXTRACTION_ERROR,
                f"Relationship extraction for a passage failed: {message}", severity="error", details={"code": code},
                run_id=run.id, commit=True,
            )
        return RelationshipOutcome(status="failed", failure=(code, message), execution_id=execution.id)
