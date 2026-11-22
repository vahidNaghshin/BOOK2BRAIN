"""Conversion between the knowledge schemas and the database records.

Kept apart so the schemas stay free of the database and the models free of Pydantic. Nothing
here touches a session: callers add the returned records and commit.
"""

from __future__ import annotations

from app.knowledge.schemas import (
    ExtractionRunCreate,
    ExtractionRunRead,
    KnowledgeRelationshipCreate,
    KnowledgeRelationshipRead,
    KnowledgeUnitCreate,
    KnowledgeUnitRead,
    SourceReference,
    Uncertainty,
    content_from_details,
)
from database.models import (
    CONTENT_SCHEMA_VERSION,
    ExtractionRun,
    KnowledgeRelationship,
    KnowledgeRelationshipPassage,
    KnowledgeUnit,
    KnowledgeUnitPassage,
)

_TITLE_LIMIT = 500


def unit_to_record(unit: KnowledgeUnitCreate) -> KnowledgeUnit:
    """A KnowledgeUnit (with its passage links) ready to be added to a session.

    The full typed content goes to `details`; title, content and summary are projections of it.
    """
    content = unit.content
    return KnowledgeUnit(
        id=unit.id,
        book_id=unit.book_id,
        document_id=unit.document_id,
        extraction_run_id=unit.extraction_run_id,
        chapter_node_id=unit.chapter_node_id,
        section_node_id=unit.section_node_id,
        equation_id=unit.equation_id,
        knowledge_type=unit.knowledge_type.value,
        title=content.title[:_TITLE_LIMIT],
        content=content.body,
        summary=content.summary,
        details=content.model_dump(mode="json", exclude={"knowledge_type"}),
        schema_version=CONTENT_SCHEMA_VERSION,
        page_start=unit.page_start,
        page_end=unit.page_end,
        page_status=unit.page_status.value,
        origin=unit.origin.value,
        confidence=unit.confidence,
        is_uncertain=unit.uncertainty.is_uncertain,
        uncertainty_notes=unit.uncertainty.notes,
        validation_status=unit.validation_status.value,
        review_status=unit.review_status.value,
        unit_metadata=dict(unit.metadata),
        sources=[
            KnowledgeUnitPassage(passage_id=source.passage_id, position=position, role=source.role.value, quote=source.quote)
            for position, source in enumerate(unit.sources)
        ],
    )


def unit_from_record(record: KnowledgeUnit) -> KnowledgeUnitRead:
    return KnowledgeUnitRead(
        id=record.id,
        book_id=record.book_id,
        document_id=record.document_id,
        extraction_run_id=record.extraction_run_id,
        content=content_from_details(record.knowledge_type, record.details),
        sources=[
            SourceReference(passage_id=link.passage_id, role=link.role, quote=link.quote) for link in record.sources
        ],
        chapter_node_id=record.chapter_node_id,
        section_node_id=record.section_node_id,
        page_start=record.page_start,
        page_end=record.page_end,
        page_status=record.page_status,
        equation_id=record.equation_id,
        confidence=record.confidence,
        uncertainty=Uncertainty(is_uncertain=record.is_uncertain, notes=record.uncertainty_notes),
        validation_status=record.validation_status,
        review_status=record.review_status,
        metadata=record.unit_metadata or {},
        schema_version=record.schema_version,
        origin=record.origin,
        source_status=record.source_status,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def relationship_to_record(relationship: KnowledgeRelationshipCreate) -> KnowledgeRelationship:
    return KnowledgeRelationship(
        id=relationship.id,
        book_id=relationship.book_id,
        source_unit_id=relationship.source_unit_id,
        target_unit_id=relationship.target_unit_id,
        relationship_type=relationship.relationship_type.value,
        explanation=relationship.explanation,
        confidence=relationship.confidence,
        extraction_run_id=relationship.extraction_run_id,
        extraction_execution_id=relationship.extraction_execution_id,
        origin=relationship.origin.value,
        review_status=relationship.review_status.value,
        evidence_quote=relationship.evidence_quote,
        sources=[
            KnowledgeRelationshipPassage(passage_id=passage_id, position=position)
            for position, passage_id in enumerate(relationship.source_passage_ids)
        ],
    )


def relationship_from_record(record: KnowledgeRelationship) -> KnowledgeRelationshipRead:
    return KnowledgeRelationshipRead(
        id=record.id,
        book_id=record.book_id,
        source_unit_id=record.source_unit_id,
        target_unit_id=record.target_unit_id,
        relationship_type=record.relationship_type,
        explanation=record.explanation,
        confidence=record.confidence,
        extraction_run_id=record.extraction_run_id,
        extraction_execution_id=record.extraction_execution_id,
        origin=record.origin,
        review_status=record.review_status,
        evidence_quote=record.evidence_quote,
        source_passage_ids=[link.passage_id for link in record.sources],
        created_at=record.created_at,
    )


def run_to_record(run: ExtractionRunCreate) -> ExtractionRun:
    return ExtractionRun(
        id=run.id,
        book_id=run.book_id,
        document_id=run.document_id,
        structural_node_id=run.structural_node_id,
        model_name=run.model_name,
        prompt_version=run.prompt_version,
        parameters=dict(run.parameters),
        scope_type=run.scope_type,
        model_configuration_id=run.model_configuration_id,
        provider=run.provider,
        model_snapshot=dict(run.model_snapshot) if run.model_snapshot is not None else None,
    )


def run_from_record(record: ExtractionRun) -> ExtractionRunRead:
    return ExtractionRunRead(
        id=record.id,
        book_id=record.book_id,
        document_id=record.document_id,
        structural_node_id=record.structural_node_id,
        model_name=record.model_name,
        prompt_version=record.prompt_version,
        parameters=record.parameters or {},
        status=record.status,
        passages_processed=record.passages_processed,
        units_extracted=record.units_extracted,
        failure_count=record.failure_count,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error_code=record.error_code,
        error_message=record.error_message,
        error_details=record.error_details,
        scope_type=record.scope_type,
        model_configuration_id=record.model_configuration_id,
        provider=record.provider,
        model_snapshot=record.model_snapshot,
        total_passages=record.total_passages,
        needs_review_count=record.needs_review_count,
        skipped_count=record.skipped_count,
        duplicate_count=record.duplicate_count,
        attempt_count=record.attempt_count,
        retry_count=record.retry_count,
        validation_failure_count=record.validation_failure_count,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        estimated_cost_usd=record.estimated_cost_usd,
        relationships_extracted=record.relationships_extracted,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
