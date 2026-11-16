"""Phase 2 knowledge models: persistence, relationships, source passages, runs, constraints, cascades."""

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import delete, insert, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.knowledge.records import (
    relationship_from_record,
    relationship_to_record,
    run_from_record,
    run_to_record,
    unit_from_record,
    unit_to_record,
)
from app.knowledge.schemas import ExtractionRunCreate, KnowledgeRelationshipCreate
from app.knowledge.types import KnowledgeType
from database.models import (
    Document,
    ExtractionRun,
    KnowledgeRelationship,
    KnowledgeRelationshipPassage,
    KnowledgeUnit,
    KnowledgeUnitPassage,
    Passage,
    StructuralNode,
)
from database.models.base import utcnow
from tests.knowledge_helpers import (
    BUILDERS,
    StoredBook,
    concept_data,
    equation_data,
    evidence_data,
    example_data,
    framework_data,
    method_data,
    principle_data,
    store_book,
    store_equation,
)


@pytest.fixture
def stored(session: Session) -> StoredBook:
    return store_book(session)


def save(session: Session, unit) -> KnowledgeUnit:
    record = unit_to_record(unit)
    session.add(record)
    session.commit()
    return record


def count(session: Session, model) -> int:
    return len(session.scalars(select(model)).all())


# -- every category is stored and read back unchanged -----------------------------------------------


@pytest.mark.parametrize("knowledge_type", list(KnowledgeType))
def test_each_category_is_persisted_and_read_back(session: Session, stored: StoredBook, knowledge_type) -> None:
    unit = stored.unit(BUILDERS[knowledge_type](), confidence=0.82, metadata={"tags": ["thermo"]})
    save(session, unit)
    session.expire_all()

    record = session.get(KnowledgeUnit, unit.id)
    assert record.knowledge_type == knowledge_type.value
    assert record.title == unit.content.title and record.content == unit.content.body
    assert record.summary == unit.content.summary
    assert "knowledge_type" not in record.details
    assert record.schema_version == 1
    assert (record.created_at, record.updated_at) != (None, None)

    again = unit_from_record(record)
    assert again.content == unit.content
    assert again.sources == unit.sources
    assert again.confidence == 0.82 and again.metadata == {"tags": ["thermo"]}
    assert (again.page_start, again.page_end, again.page_status) == (3, 4, "exact")
    assert again.chapter_node_id == stored.chapter.id and again.section_node_id == stored.section.id


def test_type_specific_data_is_stored_as_typed_details(session: Session, stored: StoredBook) -> None:
    save(session, stored.unit(method_data()))
    save(session, stored.unit(framework_data()))
    session.expire_all()

    method = session.scalar(select(KnowledgeUnit).where(KnowledgeUnit.knowledge_type == "method"))
    framework = session.scalar(select(KnowledgeUnit).where(KnowledgeUnit.knowledge_type == "framework"))
    assert method.details["steps"] == ["Weigh the sample", "Heat it", "Record the temperature change"]
    assert framework.details["components"][0] == {"name": "Hot reservoir", "description": "Supplies heat"}
    assert framework.title == "Thermodynamic cycle" and framework.summary == "Designing engines and refrigerators."


def test_details_can_be_queried_in_the_database(session: Session, stored: StoredBook) -> None:
    save(session, stored.unit(method_data(inputs=["sample mass"])))
    save(session, stored.unit(method_data(name="Other", inputs=["voltage"])))

    found = session.scalars(
        select(KnowledgeUnit.title).where(KnowledgeUnit.details["inputs"].contains(["voltage"]))
    ).all()

    assert found == ["Other"]


def test_an_incomplete_equation_is_stored_without_pretending(session: Session, stored: StoredBook) -> None:
    unit = stored.unit(
        equation_data(
            original_expression="E = m c", latex=None, variables=[], parameters=[],
            extraction_status="partial", reconstruction_status="not_attempted", status_notes="Exponent cut off.",
        ),
        validation_status="uncertain",
    )
    save(session, unit)
    session.expire_all()

    record = session.get(KnowledgeUnit, unit.id)
    assert record.content == "E = m c"
    assert record.details["latex"] is None and record.details["extraction_status"] == "partial"
    assert record.is_uncertain and record.uncertainty_notes == "Exponent cut off."
    assert record.validation_status == "uncertain"
    again = unit_from_record(record)
    assert again.content.is_incomplete and again.content.original_expression == "E = m c"


def test_an_equation_unit_can_link_to_the_phase_1_equation(session: Session, stored: StoredBook) -> None:
    equation = store_equation(session, stored)
    unit = save(session, stored.unit(equation_data(), equation_id=equation.id))

    assert unit_from_record(unit).equation_id == equation.id
    session.delete(equation)
    session.commit()
    session.refresh(unit)
    assert unit.equation_id is None  # the knowledge outlives the extracted record


def test_defaults_are_applied_by_the_database(session: Session, stored: StoredBook) -> None:
    session.execute(
        text(
            "INSERT INTO knowledge_units (id, book_id, document_id, knowledge_type, title, content, details) "
            "VALUES (:id, :book, :doc, 'concept', 'T', 'C', '{}'::jsonb)"
        ),
        {"id": uuid.uuid4(), "book": stored.book.id, "doc": stored.document.id},
    )
    session.commit()

    record = session.scalar(select(KnowledgeUnit))
    assert (record.validation_status, record.review_status, record.page_status) == ("pending", "unreviewed", "unknown")
    assert record.is_uncertain is False and record.unit_metadata == {} and record.schema_version == 1


# -- several source passages ------------------------------------------------------------------------


def test_one_unit_can_be_supported_by_several_passages(session: Session, stored: StoredBook) -> None:
    unit = stored.unit(
        principle_data(), passages=3,
        sources=[
            {"passage_id": stored.passages[0].id, "role": "primary", "quote": "never decreases"},
            {"passage_id": stored.passages[2].id},
            {"passage_id": stored.passages[4].id, "quote": "in summary"},
        ],
    )
    record = save(session, unit)
    session.expire_all()

    links = session.get(KnowledgeUnit, record.id).sources
    assert [l.passage_id for l in links] == [stored.passages[i].id for i in (0, 2, 4)]
    assert [l.position for l in links] == [0, 1, 2]
    assert [l.role for l in links] == ["primary", "supporting", "supporting"]
    assert links[0].quote == "never decreases" and links[1].quote is None
    assert unit_from_record(session.get(KnowledgeUnit, record.id)).sources == unit.sources


def test_one_passage_can_support_several_units(session: Session, stored: StoredBook) -> None:
    first = save(session, stored.unit(concept_data()))
    second = save(session, stored.unit(principle_data()))

    units_of_passage = session.scalars(
        select(KnowledgeUnitPassage.knowledge_unit_id).where(KnowledgeUnitPassage.passage_id == stored.passages[0].id)
    ).all()

    assert set(units_of_passage) == {first.id, second.id}


def test_a_passage_is_linked_to_a_unit_only_once(session: Session, stored: StoredBook) -> None:
    record = save(session, stored.unit(concept_data()))

    with pytest.raises(IntegrityError):
        session.execute(
            insert(KnowledgeUnitPassage).values(
                knowledge_unit_id=record.id, passage_id=stored.passages[0].id, position=5
            )
        )


def test_source_positions_are_unique_and_never_negative(session: Session, stored: StoredBook) -> None:
    record = save(session, stored.unit(concept_data()))
    session.add(KnowledgeUnitPassage(knowledge_unit_id=record.id, passage_id=stored.passages[1].id, position=0))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.add(KnowledgeUnitPassage(knowledge_unit_id=record.id, passage_id=stored.passages[1].id, position=-1))
    with pytest.raises(IntegrityError):
        session.commit()


def test_a_unit_cannot_have_two_primary_sources(session: Session, stored: StoredBook) -> None:
    record = save(session, stored.unit(concept_data(), sources=[{"passage_id": stored.passages[0].id, "role": "primary"}]))
    session.add(
        KnowledgeUnitPassage(knowledge_unit_id=record.id, passage_id=stored.passages[1].id, position=1, role="primary")
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_a_source_passage_must_exist(session: Session, stored: StoredBook) -> None:
    session.add(unit_to_record(stored.unit(concept_data(), sources=[{"passage_id": uuid.uuid4()}])))

    with pytest.raises(IntegrityError):
        session.commit()


# -- relationships ----------------------------------------------------------------------------------


def make_pair(session: Session, stored: StoredBook):
    evidence = save(session, stored.unit(evidence_data()))
    principle = save(session, stored.unit(principle_data()))
    return evidence, principle


def test_a_relationship_between_two_units_is_stored(session: Session, stored: StoredBook) -> None:
    evidence, principle = make_pair(session, stored)
    created = KnowledgeRelationshipCreate(
        book_id=stored.book.id, source_unit_id=evidence.id, target_unit_id=principle.id,
        relationship_type="supports", explanation="The study backs the law.", confidence=0.9,
        source_passage_ids=[stored.passages[0].id, stored.passages[1].id],
    )
    session.add(relationship_to_record(created))
    session.commit()
    session.expire_all()

    record = session.get(KnowledgeRelationship, created.id)
    assert record.source.title == evidence.title and record.target.title == principle.title
    assert record.created_at is not None
    assert relationship_from_record(record).model_dump(exclude={"created_at"}) == created.model_dump()
    assert [l.passage_id for l in record.sources] == created.source_passage_ids


def test_a_unit_knows_its_outgoing_and_incoming_relationships(session: Session, stored: StoredBook) -> None:
    evidence, principle = make_pair(session, stored)
    session.add(
        KnowledgeRelationship(
            book_id=stored.book.id, source_unit_id=evidence.id, target_unit_id=principle.id, relationship_type="supports"
        )
    )
    session.commit()
    session.expire_all()

    assert [r.relationship_type for r in session.get(KnowledgeUnit, evidence.id).outgoing_relationships] == ["supports"]
    assert session.get(KnowledgeUnit, evidence.id).incoming_relationships == []
    assert len(session.get(KnowledgeUnit, principle.id).incoming_relationships) == 1


@pytest.mark.parametrize(
    ("relationship_type", "source", "target"),
    [
        ("supports", evidence_data(), principle_data()),
        ("illustrates", example_data(), method_data()),
        ("implements", method_data(), framework_data()),
        ("describes", equation_data(), concept_data()),
    ],
)
def test_the_spec_examples_of_relationships_are_storable(
    session: Session, stored: StoredBook, relationship_type, source, target
) -> None:
    left, right = save(session, stored.unit(source)), save(session, stored.unit(target))
    session.add(
        KnowledgeRelationship(
            book_id=stored.book.id, source_unit_id=left.id, target_unit_id=right.id, relationship_type=relationship_type
        )
    )
    session.commit()

    assert count(session, KnowledgeRelationship) == 1


def test_the_same_relationship_cannot_be_stored_twice_but_other_types_can(session: Session, stored: StoredBook) -> None:
    evidence, principle = make_pair(session, stored)

    def add(kind: str) -> None:
        session.add(
            KnowledgeRelationship(
                book_id=stored.book.id, source_unit_id=evidence.id, target_unit_id=principle.id, relationship_type=kind
            )
        )
        session.commit()

    add("supports")
    add("related_to")
    with pytest.raises(IntegrityError):
        add("supports")


def test_the_reverse_direction_is_a_different_relationship(session: Session, stored: StoredBook) -> None:
    left, right = make_pair(session, stored)
    for source, target in ((left, right), (right, left)):
        session.add(
            KnowledgeRelationship(
                book_id=stored.book.id, source_unit_id=source.id, target_unit_id=target.id, relationship_type="related_to"
            )
        )
    session.commit()

    assert count(session, KnowledgeRelationship) == 2


def test_a_unit_cannot_be_related_to_itself_in_the_database(session: Session, stored: StoredBook) -> None:
    unit = save(session, stored.unit(concept_data()))
    session.add(
        KnowledgeRelationship(
            book_id=stored.book.id, source_unit_id=unit.id, target_unit_id=unit.id, relationship_type="related_to"
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_both_ends_of_a_relationship_must_belong_to_the_book(session: Session, stored: StoredBook) -> None:
    other = store_book(session, "Another book")
    mine = save(session, stored.unit(evidence_data()))
    theirs = save(session, other.unit(principle_data()))
    session.add(
        KnowledgeRelationship(
            book_id=stored.book.id, source_unit_id=mine.id, target_unit_id=theirs.id, relationship_type="supports"
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_a_relationship_needs_existing_units_and_a_known_type(session: Session, stored: StoredBook) -> None:
    unit = save(session, stored.unit(concept_data()))
    for target, kind in ((uuid.uuid4(), "related_to"), (unit.id, "likes")):
        session.rollback()
        session.add(
            KnowledgeRelationship(
                book_id=stored.book.id, source_unit_id=unit.id, target_unit_id=target, relationship_type=kind
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_relationship_confidence_is_checked(session: Session, stored: StoredBook) -> None:
    left, right = make_pair(session, stored)
    session.add(
        KnowledgeRelationship(
            book_id=stored.book.id, source_unit_id=left.id, target_unit_id=right.id, relationship_type="supports",
            confidence=1.5,
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()


def test_deleting_a_unit_deletes_its_relationships_and_their_passage_links(session: Session, stored: StoredBook) -> None:
    evidence, principle = make_pair(session, stored)
    relationship = relationship_to_record(
        KnowledgeRelationshipCreate(
            book_id=stored.book.id, source_unit_id=evidence.id, target_unit_id=principle.id,
            relationship_type="supports", source_passage_ids=[stored.passages[0].id],
        )
    )
    session.add(relationship)
    session.commit()

    session.delete(session.get(KnowledgeUnit, principle.id))
    session.commit()

    assert count(session, KnowledgeRelationship) == 0 and count(session, KnowledgeRelationshipPassage) == 0
    assert count(session, KnowledgeUnit) == 1  # the other end stays


# -- unit constraints -------------------------------------------------------------------------------


def raw_unit(stored: StoredBook, **overrides) -> KnowledgeUnit:
    values = {
        "book_id": stored.book.id, "document_id": stored.document.id, "knowledge_type": "concept", "title": "T",
        "content": "C", "details": {}, "unit_metadata": {},
    }
    return KnowledgeUnit(**(values | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"knowledge_type": "theorem"},
        {"validation_status": "great"},
        {"review_status": "done"},
        {"page_status": "sure"},
        {"confidence": 1.1},
        {"confidence": -0.1},
        {"page_start": 0},
        {"page_start": 5, "page_end": 4},
        {"title": "   "},
        {"content": ""},
        {"details": []},
        {"schema_version": 0},
        {"uncertainty_notes": "doubt", "is_uncertain": False},
        {"equation_id": uuid.uuid4(), "knowledge_type": "concept"},
    ],
    ids=lambda o: next(iter(o)),
)
def test_the_database_rejects_invalid_unit_values(session: Session, stored: StoredBook, overrides) -> None:
    session.add(raw_unit(stored, **overrides))

    with pytest.raises(IntegrityError):
        session.commit()


def test_a_unit_needs_an_existing_book_and_document(session: Session, stored: StoredBook) -> None:
    for overrides in ({"book_id": uuid.uuid4()}, {"document_id": uuid.uuid4()}):
        session.rollback()
        session.add(raw_unit(stored, **overrides))
        with pytest.raises(IntegrityError):
            session.commit()


# -- cascades and Phase 1 integration ---------------------------------------------------------------


def test_deleting_a_document_deletes_its_knowledge(session: Session, stored: StoredBook) -> None:
    evidence, principle = make_pair(session, stored)
    session.add(
        KnowledgeRelationship(
            book_id=stored.book.id, source_unit_id=evidence.id, target_unit_id=principle.id, relationship_type="supports"
        )
    )
    session.add(run_to_record(ExtractionRunCreate(book_id=stored.book.id, document_id=stored.document.id)))
    session.commit()

    session.delete(session.get(Document, stored.document.id))
    session.commit()

    for model in (KnowledgeUnit, KnowledgeUnitPassage, KnowledgeRelationship, ExtractionRun):
        assert count(session, model) == 0, model


def test_deleting_a_book_deletes_its_knowledge_only(session: Session, stored: StoredBook) -> None:
    other = store_book(session, "Another book")
    save(session, stored.unit(concept_data()))
    keep = save(session, other.unit(concept_data()))

    session.delete(stored.book)
    session.commit()

    assert [u.id for u in session.scalars(select(KnowledgeUnit))] == [keep.id]


def test_regenerating_passages_keeps_the_knowledge_but_drops_the_links(session: Session, stored: StoredBook) -> None:
    unit = save(session, stored.unit(principle_data(), passages=3))

    # What Phase 1 does when passages are regenerated.
    session.execute(delete(Passage).where(Passage.document_id == stored.document.id))
    session.commit()
    session.expire_all()

    record = session.get(KnowledgeUnit, unit.id)
    assert record is not None and record.sources == []
    assert (record.page_start, record.page_end, record.page_status) == (3, 4, "exact")  # provenance survives
    assert unit_from_record(record).sources == []  # readable, but now without sources


def test_regenerating_the_structure_keeps_the_knowledge_and_clears_node_references(
    session: Session, stored: StoredBook
) -> None:
    unit = save(session, stored.unit(concept_data()))
    run = run_to_record(
        ExtractionRunCreate(book_id=stored.book.id, document_id=stored.document.id, structural_node_id=stored.chapter.id)
    )
    session.add(run)
    session.commit()

    session.execute(delete(StructuralNode).where(StructuralNode.document_id == stored.document.id))
    session.commit()
    session.expire_all()

    record = session.get(KnowledgeUnit, unit.id)
    assert record is not None and record.chapter_node_id is None and record.section_node_id is None
    assert session.get(ExtractionRun, run.id).structural_node_id is None


# -- extraction runs --------------------------------------------------------------------------------


def test_an_extraction_run_is_created_with_its_defaults(session: Session, stored: StoredBook) -> None:
    created = ExtractionRunCreate(book_id=stored.book.id, model_name="any-model", prompt_version="v1", parameters={"t": 0})
    session.add(run_to_record(created))
    session.commit()
    session.expire_all()

    record = session.get(ExtractionRun, created.id)
    assert (record.status, record.passages_processed, record.units_extracted, record.failure_count) == ("pending", 0, 0, 0)
    assert record.document_id is None and record.structural_node_id is None
    assert record.started_at is None and record.completed_at is None
    read = run_from_record(record)
    assert read.model_name == "any-model" and read.prompt_version == "v1" and read.parameters == {"t": 0}


def test_a_run_can_be_scoped_to_a_section_and_tracks_progress(session: Session, stored: StoredBook) -> None:
    created = ExtractionRunCreate(book_id=stored.book.id, document_id=stored.document.id, structural_node_id=stored.section.id)
    record = run_to_record(created)
    session.add(record)
    session.commit()

    started = utcnow()
    record.status, record.started_at = "processing", started
    session.commit()
    record.passages_processed, record.units_extracted, record.failure_count = 5, 7, 1
    record.status, record.completed_at = "completed", started + timedelta(minutes=3)
    session.commit()
    session.expire_all()

    read = run_from_record(session.get(ExtractionRun, created.id))
    assert read.structural_node_id == stored.section.id
    assert (read.passages_processed, read.units_extracted, read.failure_count) == (5, 7, 1)
    assert read.completed_at - read.started_at == timedelta(minutes=3)


def test_a_failed_run_keeps_its_error_information(session: Session, stored: StoredBook) -> None:
    record = run_to_record(ExtractionRunCreate(book_id=stored.book.id))
    now = utcnow()
    record.status, record.started_at, record.completed_at = "failed", now, now
    record.error_code, record.error_message = "model_unavailable", "The model did not answer."
    record.error_details = {"passage_id": str(stored.passages[0].id), "attempts": 3}
    session.add(record)
    session.commit()
    session.expire_all()

    read = run_from_record(session.get(ExtractionRun, record.id))
    assert read.status == "failed" and read.error_code == "model_unavailable"
    assert read.error_details == {"passage_id": str(stored.passages[0].id), "attempts": 3}


def test_units_remember_their_run_and_outlive_it(session: Session, stored: StoredBook) -> None:
    run = run_to_record(ExtractionRunCreate(book_id=stored.book.id, document_id=stored.document.id))
    session.add(run)
    session.commit()
    unit = save(session, stored.unit(concept_data(), extraction_run_id=run.id))
    assert [u.id for u in session.get(ExtractionRun, run.id).knowledge_units] == [unit.id]

    session.delete(session.get(ExtractionRun, run.id))
    session.commit()
    session.refresh(unit)

    assert unit.extraction_run_id is None and count(session, KnowledgeUnit) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "stuck"},
        {"passages_processed": -1},
        {"units_extracted": -1},
        {"failure_count": -1},
        {"started_at": utcnow(), "completed_at": utcnow() - timedelta(hours=1)},
    ],
    ids=lambda o: next(iter(o)),
)
def test_the_database_rejects_invalid_run_values(session: Session, stored: StoredBook, overrides) -> None:
    session.add(ExtractionRun(book_id=stored.book.id, **overrides))

    with pytest.raises(IntegrityError):
        session.commit()
