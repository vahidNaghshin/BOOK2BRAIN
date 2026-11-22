"""Relationships between knowledge units: creation, validation, duplicates, review flags and reading."""

import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.knowledge.relationships import RELATIONSHIP_RULES
from app.knowledge.schemas import KnowledgeRelationshipCreate
from app.knowledge.types import KnowledgeType, RelationshipOrigin, RelationshipType, ReviewFlagCode
from app.services.errors import RelationshipError
from app.services.relationship_service import RelationshipService
from app.services.review_flag_service import ReviewFlagService
from database.models import KnowledgeRelationship, KnowledgeUnit
from tests.extraction_db_helpers import Library, make_service, store_library

QUOTE = "Entropy is a measure of the disorder of a system."
ALL_TYPES = ["concept", "principle", "method", "framework", "evidence", "example", "equation"]


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


@pytest.fixture
def units(session: Session, library: Library) -> dict[str, KnowledgeUnit]:
    """One unit of every category, all extracted from the subsection passage ss1a."""
    service, _ = make_service(session)
    service.extract_section(library.nodes["ss1"].id, ALL_TYPES)
    return {u.knowledge_type: u for u in session.scalars(select(KnowledgeUnit))}


def data(library, source, target, kind, **fields) -> KnowledgeRelationshipCreate:
    return KnowledgeRelationshipCreate(
        book_id=library.book.id, source_unit_id=source.id, target_unit_id=target.id, relationship_type=kind, **fields
    )


def flags(session, relationship_id):
    return ReviewFlagService(session).list_flags(
        session.get(KnowledgeRelationship, relationship_id).book_id, relationship_id=relationship_id, status=None
    )


# ---------------------------------------------------------------- the supported types


def test_the_supported_relationship_types():
    wanted = {"related_to", "depends_on", "supports", "contradicts", "part_of", "derived_from", "example_of", "applies_to"}
    assert wanted <= {t.value for t in RelationshipType}
    assert set(RELATIONSHIP_RULES) == set(RelationshipType)


@pytest.mark.parametrize(
    "kind, source, target",
    [
        ("supports", "evidence", "principle"), ("illustrates", "example", "concept"), ("implements", "method", "framework"),
        ("describes", "equation", "concept"), ("part_of", "concept", "framework"), ("depends_on", "principle", "concept"),
        ("contradicts", "concept", "principle"), ("related_to", "concept", "principle"), ("derived_from", "equation", "principle"),
        ("example_of", "example", "concept"), ("applies_to", "method", "concept"),
    ],
)
def test_every_relationship_type_can_be_stored(session, library, units, kind, source, target):
    result = RelationshipService(session).create(data(library, units[source], units[target], kind))
    assert result.created and result.relationship.relationship_type == RelationshipType(kind)
    assert result.relationship.origin == RelationshipOrigin.MANUAL


@pytest.mark.parametrize(
    "kind, source, target",
    [("supports", "example", "concept"), ("example_of", "concept", "example"), ("applies_to", "evidence", "concept"), ("derived_from", "example", "concept")],
)
def test_a_type_that_does_not_fit_the_categories_is_rejected(session, library, units, kind, source, target):
    with pytest.raises(RelationshipError, match="cannot be"):
        RelationshipService(session).create(data(library, units[source], units[target], kind))


# ---------------------------------------------------------------- creation and retrieval


def test_a_relationship_keeps_what_was_recorded_about_it(session, library, units):
    run_id = uuid.uuid4()
    created = RelationshipService(session).create(
        data(library, units["principle"], units["concept"], "depends_on", explanation="The law needs the notion of entropy.", confidence=0.7,
             source_passage_ids=[library.passages["ss1a"].id])
    ).relationship
    stored = RelationshipService(session).get(created.id)
    assert (stored.explanation, stored.confidence, stored.source_passage_ids) == ("The law needs the notion of entropy.", 0.7, [library.passages["ss1a"].id])
    assert stored.created_at is not None and stored.extraction_run_id is None and stored.review_status.value == "unreviewed"
    assert run_id  # an extraction run id can be recorded as well (see the extraction tests)
    with pytest.raises(RelationshipError, match="does not exist"):
        RelationshipService(session).get(uuid.uuid4())


def test_a_unit_shows_its_outgoing_and_incoming_relationships(session, library, units):
    service = RelationshipService(session)
    service.create(data(library, units["principle"], units["concept"], "depends_on"))
    service.create(data(library, units["example"], units["principle"], "illustrates"))
    related = service.for_unit(units["principle"].id)
    assert [(v.direction, v.relationship.relationship_type.value, v.other_title, v.other_type) for v in related.outgoing] == [
        ("outgoing", "depends_on", units["concept"].title, KnowledgeType.CONCEPT)
    ]
    assert [(v.direction, v.relationship.relationship_type.value, v.other_title) for v in related.incoming] == [("incoming", "illustrates", units["example"].title)]
    assert len(related.all) == 2 and service.for_unit(units["method"].id).all == []


def test_relationships_of_a_book_can_be_listed_and_filtered(session, library, units):
    service = RelationshipService(session)
    service.create(data(library, units["principle"], units["concept"], "depends_on"))
    service.create(data(library, units["evidence"], units["principle"], "supports"))
    assert len(service.list_for_book(library.book.id)) == 2
    assert [r.relationship_type.value for r in service.list_for_book(library.book.id, relationship_type="supports")] == ["supports"]
    assert service.list_for_book(library.book.id, origin="ai_suggested") == []
    assert len(service.list_for_book(library.book.id, review_status="unreviewed", limit=1)) == 1


def test_review_status_of_a_relationship_can_be_set(session, library, units):
    service = RelationshipService(session)
    relationship = service.create(data(library, units["principle"], units["concept"], "depends_on")).relationship
    assert service.set_review_status(relationship.id, "approved").review_status.value == "approved"
    with pytest.raises(RelationshipError, match="must be one of"):
        service.set_review_status(relationship.id, "maybe")
    with pytest.raises(RelationshipError, match="does not exist"):
        service.set_review_status(uuid.uuid4(), "approved")


# ---------------------------------------------------------------- validation


def test_both_units_must_exist(session, library, units):
    service = RelationshipService(session)
    ghost = uuid.uuid4()
    for source, target, missing in ((ghost, units["concept"].id, "source"), (units["concept"].id, ghost, "target")):
        with pytest.raises(RelationshipError, match=f"The {missing} knowledge unit does not exist"):
            service.create(KnowledgeRelationshipCreate(book_id=library.book.id, source_unit_id=source, target_unit_id=target, relationship_type="related_to"))
    assert session.scalar(select(KnowledgeRelationship.id).limit(1)) is None


def test_both_units_must_belong_to_the_relationships_book(session, library, units):
    other = store_library(session, "Another book")
    other_service, _ = make_service(session)
    other_service.extract_section(other.nodes["ss1"].id, ["concept"])
    foreign = session.scalars(select(KnowledgeUnit).where(KnowledgeUnit.book_id == other.book.id)).one()
    with pytest.raises(RelationshipError, match="belong to the relationship's book"):
        RelationshipService(session).create(data(library, units["principle"], foreign, "depends_on"))


def test_a_unit_cannot_be_related_to_itself(session, library, units):
    with pytest.raises(ValidationError, match="cannot be related to itself"):
        data(library, units["concept"], units["concept"], "related_to")
    session.add(KnowledgeRelationship(book_id=library.book.id, source_unit_id=units["concept"].id, target_unit_id=units["concept"].id, relationship_type="related_to"))
    with pytest.raises(IntegrityError):  # the database refuses it too
        session.commit()


def test_an_unknown_relationship_type_is_refused(session, library, units):
    with pytest.raises(ValidationError):
        data(library, units["concept"], units["principle"], "resembles")
    session.add(KnowledgeRelationship(book_id=library.book.id, source_unit_id=units["concept"].id, target_unit_id=units["principle"].id, relationship_type="resembles"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_cited_passages_must_exist_and_belong_to_the_book(session, library, units):
    service = RelationshipService(session)
    with pytest.raises(RelationshipError, match="does not exist"):
        service.create(data(library, units["principle"], units["concept"], "depends_on", source_passage_ids=[uuid.uuid4()]))
    other = store_library(session, "Another book")
    with pytest.raises(RelationshipError, match="belongs to another book"):
        service.create(data(library, units["principle"], units["concept"], "depends_on", source_passage_ids=[other.passages["c1a"].id]))


# ---------------------------------------------------------------- duplicates


def test_the_same_relationship_is_stored_once(session, library, units):
    service = RelationshipService(session)
    first = service.create(data(library, units["principle"], units["concept"], "depends_on", explanation="first"))
    second = service.create(data(library, units["principle"], units["concept"], "depends_on", explanation="second"))
    assert first.created and not second.created and second.relationship.id == first.relationship.id
    assert second.relationship.explanation == "first"
    assert session.scalars(select(KnowledgeRelationship)).all().__len__() == 1


def test_a_symmetric_relationship_is_the_same_in_both_directions(session, library, units):
    service = RelationshipService(session)
    for kind in ("related_to", "contradicts"):
        forward = service.create(data(library, units["concept"], units["principle"], kind))
        backward = service.create(data(library, units["principle"], units["concept"], kind))
        assert forward.created and not backward.created and backward.relationship.id == forward.relationship.id


def test_directed_relationships_and_other_types_between_the_same_units_are_distinct(session, library, units):
    service = RelationshipService(session)
    results = [
        service.create(data(library, units["principle"], units["concept"], "depends_on")),
        service.create(data(library, units["concept"], units["principle"], "depends_on")),
        service.create(data(library, units["principle"], units["concept"], "related_to")),
    ]
    assert all(r.created for r in results) and len({r.relationship.id for r in results}) == 3


def test_a_repeated_relationship_adds_its_new_evidence_passage(session, library, units):
    service = RelationshipService(session)
    service.create(data(library, units["principle"], units["concept"], "depends_on", source_passage_ids=[library.passages["ss1a"].id]))
    again = service.create(data(library, units["principle"], units["concept"], "depends_on", source_passage_ids=[library.passages["ss1a"].id, library.passages["s1a"].id]))
    assert not again.created and again.relationship.source_passage_ids == [library.passages["ss1a"].id, library.passages["s1a"].id]


# ---------------------------------------------------------------- evidence and AI suggestions


def test_an_extracted_relationship_needs_evidence_that_is_really_in_the_passage(session, library, units):
    service = RelationshipService(session)
    with pytest.raises(ValidationError, match="needs the passage and the quoted evidence"):
        data(library, units["principle"], units["concept"], "depends_on", origin="extracted")
    with pytest.raises(RelationshipError, match="evidence quote is not in any of the cited passages"):
        service.create(data(library, units["principle"], units["concept"], "depends_on", origin="extracted",
                            source_passage_ids=[library.passages["ss1a"].id], evidence_quote="Words that are not in the passage."))
    result = service.create(data(library, units["principle"], units["concept"], "depends_on", origin="extracted",
                                 source_passage_ids=[library.passages["ss1a"].id], evidence_quote="entropy IS a measure   of the disorder of a system."))
    assert result.created and result.relationship.origin == RelationshipOrigin.EXTRACTED and not result.flagged


def test_an_ai_suggestion_is_stored_flagged_for_review(session, library, units):
    result = RelationshipService(session).create(
        data(library, units["principle"], units["concept"], "depends_on", origin="ai_suggested", explanation="Seems related.", extraction_run_id=None)
    )
    assert result.created and result.flagged and result.relationship.origin == RelationshipOrigin.AI_SUGGESTED
    [flag] = flags(session, result.relationship.id)
    assert flag.code == ReviewFlagCode.AI_SUGGESTED_RELATIONSHIP and flag.raised_by == "model" and flag.target_kind == "relationship"
    assert "without evidence that could be verified" in flag.message and flag.details["explanation"] == "Seems related."


def test_a_suggestion_becomes_an_extracted_relationship_when_evidence_is_found(session, library, units):
    service = RelationshipService(session)
    suggested = service.create(data(library, units["principle"], units["concept"], "depends_on", origin="ai_suggested")).relationship
    upgraded = service.create(
        data(library, units["principle"], units["concept"], "depends_on", origin="extracted", explanation="Stated in the text.",
             source_passage_ids=[library.passages["ss1a"].id], evidence_quote=QUOTE)
    )
    assert not upgraded.created and upgraded.upgraded and upgraded.relationship.id == suggested.id
    assert upgraded.relationship.origin == RelationshipOrigin.EXTRACTED and upgraded.relationship.evidence_quote == QUOTE
    assert upgraded.relationship.explanation == "Stated in the text."
    [flag] = flags(session, suggested.id)
    assert flag.status.value == "resolved" and "Evidence from the book" in flag.resolution_note


def test_a_unit_that_is_removed_takes_its_relationships_with_it(session, library, units):
    service = RelationshipService(session)
    service.create(data(library, units["principle"], units["concept"], "depends_on"))
    session.delete(units["concept"])
    session.commit()
    assert session.scalar(select(KnowledgeRelationship.id).limit(1)) is None
