"""Database rules added on Day 4: vocabularies, constraints and what happens when parents are removed."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.knowledge import types as t
from database import models as m
from tests.extraction_db_helpers import Library, make_service, store_library

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


@pytest.fixture
def extracted(session: Session, library: Library):
    """A run over one passage with a concept unit, so there is an execution, attempt, link and unit to work with."""
    service, _ = make_service(session)
    status = service.extract_section(library.nodes["ss1"].id, ["concept"])
    unit = session.scalars(select(m.KnowledgeUnit)).one()
    return status.run.id, unit


def values(enum):
    return {member.value for member in enum}


def test_the_database_vocabularies_match_the_code_vocabularies():
    assert set(m.RELATIONSHIP_TYPES) == values(t.RelationshipType)
    assert set(m.RELATIONSHIP_ORIGINS) == values(t.RelationshipOrigin)
    assert set(m.KNOWLEDGE_ORIGINS) == values(t.KnowledgeOrigin)
    assert set(m.SOURCE_VERIFICATION_STATUSES) == values(t.SourceVerificationStatus)
    assert set(m.SOURCE_REFERENCE_TYPES) == values(t.SourceReferenceType)
    assert set(m.EQUATION_CAPTURE_STATUSES) == values(t.EquationCaptureStatus)
    assert set(m.MATH_VALIDATION_STATUSES) == values(t.MathValidationStatus)
    assert set(m.REVIEW_FLAG_CODES) == values(t.ReviewFlagCode)
    assert set(m.FLAG_SEVERITIES) == values(t.FlagSeverity)
    assert set(m.FLAG_STATUSES) == values(t.FlagStatus)
    assert set(m.PASSAGE_STATUSES) == values(t.PassageStatus)
    assert set(m.VALIDATION_OUTCOMES) == values(t.ValidationOutcome)
    assert set(m.EXECUTION_KINDS) == values(t.KnowledgeType) | {"relationship"}
    assert set(m.ATTEMPT_STATUSES) == {"completed", "provider_error"}


def update(session, record, **fields):
    for key, value in fields.items():
        setattr(record, key, value)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_a_unit_has_a_known_origin_and_source_status(session, extracted):
    _, unit = extracted
    assert (unit.origin, unit.source_status) == ("ai_extracted", "verified")
    update(session, unit, origin="invented")
    update(session, unit, source_status="checked")


def test_source_links_are_validated(session, extracted):
    _, unit = extracted
    link = unit.sources[0]
    assert (link.reference_type, link.verification_status) == ("text", "verified")
    update(session, link, reference_type="audio")
    update(session, link, verification_status="maybe")
    update(session, link, quote_start=5, quote_end=5)  # an empty range
    update(session, link, quote_start=5, quote_end=None)  # half a range
    update(session, link, quote_start=-1, quote_end=3)
    update(session, link, page_start=0)
    update(session, link, page_start=5, page_end=3)


def test_relationships_have_a_known_origin_and_review_status(session, extracted):
    _, unit = extracted
    other = m.KnowledgeUnit(
        book_id=unit.book_id, document_id=unit.document_id, knowledge_type="principle", title="P", content="p", details={"statement": "p"},
    )
    session.add(other)
    session.commit()
    relationship = m.KnowledgeRelationship(book_id=unit.book_id, source_unit_id=other.id, target_unit_id=unit.id, relationship_type="derived_from")
    session.add(relationship)
    session.commit()
    assert (relationship.origin, relationship.review_status) == ("manual", "unreviewed")
    update(session, relationship, origin="guessed")
    update(session, relationship, review_status="maybe")
    update(session, relationship, relationship_type="resembles")


def test_attempts_have_a_known_status_and_validation_outcome(session, extracted):
    run_id, _ = extracted
    attempt = session.scalars(select(m.ExtractionAttempt)).one()
    assert (attempt.status, attempt.validation_outcome, attempt.extraction_run_id) == ("completed", "schema_valid", run_id)
    update(session, attempt, status="running")
    update(session, attempt, validation_outcome="looks_fine")


def test_an_execution_can_be_about_relationships_but_not_about_anything(session, extracted):
    run_id, _ = extracted
    execution = session.scalars(select(m.ExtractionExecution)).one()
    execution.knowledge_type = "relationship"
    session.commit()
    update(session, execution, knowledge_type="poem")


def test_a_passage_appears_once_per_run_and_has_a_known_status(session, library, extracted):
    run_id, _ = extracted
    record = session.scalars(select(m.ExtractionPassageRecord)).one()
    assert (record.status, record.passage_id) == ("completed", library.passages["ss1a"].id)
    session.add(m.ExtractionPassageRecord(extraction_run_id=run_id, passage_id=library.passages["ss1a"].id))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    update(session, record, status="done")
    update(session, record, attempt_count=-1)
    update(session, record, started_at=NOW, completed_at=datetime(2025, 1, 1, tzinfo=UTC))


def test_a_run_counts_relationships_and_rejects_negative_numbers(session, extracted):
    run_id, _ = extracted
    run = session.get(m.ExtractionRun, run_id)
    assert run.relationships_extracted == 0
    update(session, run, relationships_extracted=-1)


def test_open_flags_are_unique_per_target_code_and_key_but_resolved_ones_are_not(session, library, extracted):
    _, unit = extracted

    def flag(**fields):
        return m.ReviewFlag(book_id=library.book.id, knowledge_unit_id=unit.id, code="low_confidence", message="x", **fields)

    session.add(flag())
    session.commit()
    session.add(flag())
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.add(flag(dedupe_key="another"))
    session.add(flag(status="resolved", resolved_at=NOW))  # history may repeat
    session.add(flag(status="resolved", resolved_at=NOW))
    session.commit()
    for bad in (flag(status="resolved"), flag(status="open", resolved_at=NOW), flag(severity="fatal"), flag(raised_by="nobody"), m.ReviewFlag(book_id=library.book.id, knowledge_unit_id=unit.id, code="made_up", message="x")):
        session.add(bad)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_equation_sources_and_reconstructions_need_known_values(session, extracted):
    _, unit = extracted
    session.add(m.EquationSource(knowledge_unit_id=unit.id, capture_status="image_only", image_reference="equations/x.png"))
    session.commit()
    source = session.get(m.EquationSource, unit.id)
    update(session, source, capture_status="pretty_good")
    session.add(m.EquationReconstruction(knowledge_unit_id=unit.id, reconstructed_latex="x"))
    session.commit()


def test_removing_parents_cleans_up_what_depends_on_them(session, library, extracted):
    run_id, unit = extracted
    session.add(m.EquationSource(knowledge_unit_id=unit.id, passage_id=library.passages["ss1a"].id, capture_status="complete"))
    session.add(m.EquationReconstruction(knowledge_unit_id=unit.id, reconstructed_latex="x"))
    attempt = session.scalars(select(m.ExtractionAttempt)).one()
    session.add(m.ReviewFlag(book_id=library.book.id, extraction_attempt_id=attempt.id, code="unresolved_extraction_error", message="x"))
    session.add(m.ReviewFlag(book_id=library.book.id, knowledge_unit_id=unit.id, code="low_confidence", message="y"))
    session.commit()

    # passages are regenerated: the record, the capture and the unit stay, pointing at nothing
    session.delete(library.passages["ss1a"])
    session.commit()
    session.expire_all()
    assert session.scalars(select(m.ExtractionPassageRecord)).one().passage_id is None
    assert session.get(m.EquationSource, unit.id).passage_id is None and session.get(m.KnowledgeUnit, unit.id) is not None
    assert session.scalars(select(m.ExtractionAttempt)).one().passage_id is None

    # the run is deleted: its history goes, the knowledge stays
    session.delete(session.get(m.ExtractionRun, run_id))
    session.commit()
    session.expire_all()
    assert session.scalars(select(m.ExtractionPassageRecord)).all() == [] and session.scalars(select(m.ExtractionAttempt)).all() == []
    assert session.get(m.KnowledgeUnit, unit.id) is not None
    assert [f.code for f in session.scalars(select(m.ReviewFlag))] == ["low_confidence"]  # the attempt's flag went with the attempt

    # the unit is deleted: everything about it goes
    session.delete(session.get(m.KnowledgeUnit, unit.id))
    session.commit()
    assert all(session.scalars(select(model)).all() == [] for model in (m.EquationSource, m.EquationReconstruction, m.ReviewFlag))


def test_an_id_generated_for_flags_and_reconstructions_is_unique(session, library, extracted):
    _, unit = extracted
    a, b = (m.EquationReconstruction(knowledge_unit_id=unit.id, reconstructed_latex="x") for _ in range(2))
    session.add_all([a, b])
    session.commit()
    assert isinstance(a.id, uuid.UUID) and a.id != b.id
