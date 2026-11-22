"""Source provenance: capture when a unit is stored, retrieval, verification and the review flags it raises."""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.types import ReviewFlagCode, SourceVerificationStatus
from app.services.knowledge_service import KnowledgeService
from app.services.provenance_service import ProvenanceService
from app.services.review_flag_service import ReviewFlagService
from database.models import KnowledgeUnit, KnowledgeUnitPassage, Passage, ReviewFlag, StructuralNode
from tests.extraction_db_helpers import Library, add_second_document, item, make_service, one_item, reply, store_library

QUOTE = "Entropy is a measure of the disorder of a system."


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


def extract_one(session, library, passage="c1a", answer=None, category="concept", **options) -> KnowledgeUnit:
    """Extract one passage and return its (first) unit."""
    service, _ = make_service(session, answer or one_item, **options)
    node = {"c1a": "ch1", "c1b": "ch1", "s1a": "s1", "s1b": "s1", "ss1a": "ss1", "c2a": "ch2"}[passage]
    scope = service.chapter_scope(library.nodes[node].id) if node.startswith("ch") else service.section_scope(library.nodes[node].id)
    service.extract_book(library.book.id, [category], limit=1, offset=library.passages[passage].document_sequence, document_id=library.document.id)
    return session.scalars(select(KnowledgeUnit).order_by(KnowledgeUnit.created_at)).first()


def quote_answer(library, quote=QUOTE, supporting=()):
    def answer(call):
        sources = [{"passage_id": call.target_id, "role": "primary", "quote": quote}]
        sources += [{"passage_id": str(library.passages[name].id), "role": "supporting"} for name in supporting]
        return reply(item(call, sources=sources))

    return answer


def flags_of(session, unit, status="open"):
    return [(f.code, f.dedupe_key) for f in session.scalars(select(ReviewFlag).where(ReviewFlag.knowledge_unit_id == unit.id, ReviewFlag.status == status))]


# ---------------------------------------------------------------- capture when a unit is stored


def test_a_stored_unit_has_complete_source_links_and_a_snapshot(session, library):
    unit = extract_one(session, library, "s1a", quote_answer(library))
    [link] = unit.sources
    passage = library.passages["s1a"]
    assert (link.reference_type, link.role, link.structural_node_id) == ("text", "primary", passage.structural_node_id)
    assert (link.page_start, link.page_end) == (passage.page_start, passage.page_end)
    assert passage.text[link.quote_start : link.quote_end] == QUOTE == link.quote
    [snapshot] = unit.source_snapshot
    assert snapshot["passage_id"] == str(passage.id) and snapshot["document_id"] == str(library.document.id)
    assert (snapshot["chapter"], snapshot["section"], snapshot["pages"]) == ("Entropy", "The second law", [4, 4])
    assert snapshot["excerpt"] == QUOTE and snapshot["excerpt_is_quote"] is True and snapshot["role"] == "primary"


def test_without_a_quote_the_snapshot_keeps_the_start_of_the_passage(session, library):
    unit = extract_one(session, library, "c1a")
    [snapshot] = unit.source_snapshot
    assert snapshot["excerpt"].startswith("Passage c1a.") and snapshot["excerpt_is_quote"] is False
    assert unit.sources[0].quote_start is None and unit.sources[0].quote_end is None


def test_a_quote_that_is_not_in_the_passage_is_kept_as_claimed_but_has_no_position(session, library):
    unit = extract_one(session, library, "c1a", quote_answer(library, "Words that are not in the passage."))
    link = unit.sources[0]
    assert link.quote == "Words that are not in the passage." and link.quote_start is None
    assert unit.source_snapshot[0]["excerpt_is_quote"] is False  # the snapshot never carries unverified words as source text


def test_an_equation_unit_marks_its_sources_as_equation_references(session, library):
    unit = extract_one(session, library, "c1a", category="equation")
    assert unit.sources[0].reference_type == "equation"


# ---------------------------------------------------------------- several sources


def test_a_unit_can_be_supported_by_several_passages(session, library):
    unit = extract_one(session, library, "s1a", quote_answer(library, supporting=("c1b", "s1b")))
    details = ProvenanceService(session).sources_for(unit.id)
    assert [d.role for d in details] == ["primary", "supporting", "supporting"]
    assert [d.passage_id for d in details] == [library.passages[n].id for n in ("s1a", "c1b", "s1b")]
    assert [d.page_start for d in details] == [4, 3, 5]
    assert details[0].chapter_title == "Entropy" and details[0].section_title == "The second law"
    assert len(unit.source_snapshot) == 3
    assert (unit.page_start, unit.page_end) == (3, 5)


def test_sources_come_with_the_passage_text_on_request(session, library):
    unit = extract_one(session, library, "c1a", quote_answer(library))
    service = ProvenanceService(session)
    assert service.sources_for(unit.id)[0].text is None
    assert service.sources_for(unit.id, include_text=True)[0].text == library.passages["c1a"].text
    with pytest.raises(LookupError):
        service.sources_for(uuid.uuid4())


# ---------------------------------------------------------------- verification


def test_units_are_verified_when_they_are_stored(session, library):
    unit = extract_one(session, library, "c1a", quote_answer(library))
    assert unit.source_status == "verified" and unit.sources[0].verification_status == "verified"
    assert unit.sources[0].verified_at is not None and unit.sources[0].verification_issues == []
    assert flags_of(session, unit) == []


def test_verifying_again_reports_a_clean_unit_as_verified(session, library):
    unit = extract_one(session, library, "s1a", quote_answer(library, supporting=("c1b",)))
    report = ProvenanceService(session).verify_unit(unit.id)
    assert report.status == SourceVerificationStatus.VERIFIED and report.references_checked == 2 and report.issues == []
    assert report.flags_raised == 0 and report.flags_resolved == 0


def test_a_reference_to_a_passage_of_another_book_is_invalid_and_flagged_not_dropped(session, library):
    unit = extract_one(session, library, "c1a")
    other = store_library(session, "Another book")
    session.add(KnowledgeUnitPassage(knowledge_unit_id=unit.id, passage_id=other.passages["c1a"].id, position=1, role="supporting"))
    session.commit()
    report = ProvenanceService(session).verify_unit(unit.id)
    assert report.status == SourceVerificationStatus.INVALID
    assert {i.code for i in report.issues} == {ReviewFlagCode.INVALID_SOURCE_REFERENCE}
    passage_issue = next(i for i in report.issues if i.passage_id == other.passages["c1a"].id)
    assert "another book" in passage_issue.message and report.flags_raised == 2
    assert any(i.key == "snapshot" for i in report.issues)  # the added link is not in the stored snapshot either
    session.refresh(unit)
    assert unit.source_status == "invalid" and len(unit.sources) == 2  # nothing was silently discarded
    statuses = {link.passage_id: link.verification_status for link in unit.sources}
    assert statuses[library.passages["c1a"].id] == "verified" and statuses[other.passages["c1a"].id] == "invalid"
    assert unit.validation_status == "uncertain"
    assert ("invalid_source_reference", f"passage:{other.passages['c1a'].id}") in flags_of(session, unit)


def test_a_passage_of_another_document_than_the_unit_is_invalid(session, library):
    unit = extract_one(session, library, "c1a")
    [foreign] = add_second_document(session, library, older=False)[:1]
    session.add(KnowledgeUnitPassage(knowledge_unit_id=unit.id, passage_id=foreign.id, position=1, role="supporting"))
    session.commit()
    report = ProvenanceService(session).verify_unit(unit.id)
    assert report.status == SourceVerificationStatus.INVALID and "another document" in report.issues[0].message


def test_structural_references_must_belong_to_the_units_book_and_document(session, library):
    unit = extract_one(session, library, "s1a")
    other = store_library(session, "Another book")
    unit.chapter_node_id = other.nodes["ch1"].id
    session.commit()
    report = ProvenanceService(session).verify_unit(unit.id)
    assert any(i.key == f"node:{other.nodes['ch1'].id}" and "another book" in i.message for i in report.issues)

    unit.chapter_node_id = library.nodes["ch2"].id  # a chapter of this book, but the section is not inside it
    session.commit()
    report = ProvenanceService(session).verify_unit(unit.id)
    assert any("is not inside the chapter" in i.message for i in report.issues)

    unit.chapter_node_id = library.nodes["ch1"].id
    session.commit()
    assert ProvenanceService(session).verify_unit(unit.id).status == SourceVerificationStatus.VERIFIED


def test_a_document_of_another_book_is_reported(session, library):
    unit = extract_one(session, library, "c1a")
    other = store_library(session, "Another book")
    unit.document_id = other.document.id  # the unit's document belongs to another book
    session.commit()
    report = ProvenanceService(session).verify_unit(unit.id)
    assert any(i.key == "document" and "another book" in i.message for i in report.issues)


def test_page_references_are_checked_against_the_source(session, library):
    unit = extract_one(session, library, "c1a")  # passage c1a is on page 2
    service = ProvenanceService(session)

    unit.page_start = unit.page_end = 40
    session.commit()
    report = service.verify_unit(unit.id)
    assert {i.code for i in report.issues} == {ReviewFlagCode.PAGE_INCONSISTENT}
    assert any("outside the cited passages" in i.message for i in report.issues) and any("beyond the document" in i.message for i in report.issues)

    library.passages["c1a"].page_start = library.passages["c1a"].page_end = None
    unit.page_start, unit.page_end = 2, 2
    session.commit()
    assert any("source passages have none" in i.message for i in service.verify_unit(unit.id).issues)


def test_the_stored_location_and_pages_of_a_link_must_still_match_the_passage(session, library):
    unit = extract_one(session, library, "c1a")
    passage = library.passages["c1a"]
    passage.page_start = passage.page_end = 3  # Phase 1 repaginated after the unit was stored
    session.commit()
    keys = {i.key for i in ProvenanceService(session).verify_unit(unit.id).issues}
    assert f"pages:{passage.id}" in keys


def test_a_quote_that_is_not_in_the_passage_is_flagged(session, library):
    unit = extract_one(session, library, "c1a", quote_answer(library, "Words that are not in the passage."))
    assert unit.source_status == "invalid"
    assert ("source_quote_mismatch", f"quote:{library.passages['c1a'].id}") in flags_of(session, unit)
    assert unit.sources[0].verification_issues[0]["code"] == "source_quote_mismatch"


def test_a_quote_position_that_no_longer_matches_the_text_is_flagged(session, library):
    unit = extract_one(session, library, "c1a", quote_answer(library))
    unit.sources[0].quote_start += 3
    unit.sources[0].quote_end += 3
    session.commit()
    report = ProvenanceService(session).verify_unit(unit.id)
    assert any("stored position" in i.message for i in report.issues)


def test_a_unit_without_any_source_is_flagged(session, library):
    unit = extract_one(session, library, "c1a")
    for link in list(unit.sources):
        session.delete(link)
    unit.source_snapshot = []
    session.commit()
    session.refresh(unit)
    report = ProvenanceService(session).verify_unit(unit.id)
    assert report.status == SourceVerificationStatus.INVALID and [i.code for i in report.issues] == [ReviewFlagCode.NO_SOURCE]


def test_regenerated_passages_leave_the_snapshot_and_are_reported(session, library):
    unit = extract_one(session, library, "s1a", quote_answer(library))
    for passage in session.scalars(select(Passage)):
        session.delete(passage)
    session.commit()
    session.expire_all()
    service = ProvenanceService(session)
    assert service.sources_for(unit.id) == []
    [snapshot] = service.snapshot_of(unit.id)
    assert snapshot["excerpt"] == QUOTE and snapshot["pages"] == [4, 4] and snapshot["section"] == "The second law"
    report = service.verify_unit(unit.id)
    assert report.status == SourceVerificationStatus.INVALID and "regenerated" in report.issues[0].message
    assert KnowledgeService(session).detail(unit.id).source_snapshot[0]["excerpt"] == QUOTE


def test_flags_are_not_duplicated_and_are_resolved_when_the_problem_is_fixed(session, library):
    unit = extract_one(session, library, "c1a")
    unit.page_start = unit.page_end = 40
    session.commit()
    service = ProvenanceService(session)
    first = service.verify_unit(unit.id)
    second = service.verify_unit(unit.id)
    assert first.flags_raised == 2 and second.flags_raised == 0
    assert len(flags_of(session, unit)) == 2

    unit.page_start = unit.page_end = 2
    session.commit()
    fixed = service.verify_unit(unit.id)
    assert fixed.status == SourceVerificationStatus.VERIFIED and fixed.flags_resolved == 2
    assert flags_of(session, unit) == [] and len(flags_of(session, unit, "resolved")) == 2
    resolved = session.scalars(select(ReviewFlag).where(ReviewFlag.status == "resolved")).first()
    assert "no longer applies" in resolved.resolution_note and resolved.resolved_at is not None


def test_verification_can_run_without_raising_flags(session, library):
    unit = extract_one(session, library, "c1a")
    unit.page_start = unit.page_end = 40
    session.commit()
    report = ProvenanceService(session).verify_unit(unit.id, flag=False)
    assert report.status == SourceVerificationStatus.INVALID and flags_of(session, unit) == []


def test_a_whole_book_can_be_verified(session, library):
    service, _ = make_service(session)
    service.extract_chapter(library.nodes["ch2"].id, ["concept"])
    broken = session.scalars(select(KnowledgeUnit)).first()
    broken.page_start = broken.page_end = 99
    session.commit()
    assert ProvenanceService(session).verify_book(library.book.id) == {"verified": 1, "invalid": 1}
    assert ProvenanceService(session).verify_book(library.book.id, only_unverified=True) == {"verified": 0, "invalid": 0}
    open_flags = ReviewFlagService(session).counts_by_code(library.book.id)
    assert open_flags["page_inconsistent"] == 2


# ---------------------------------------------------------------- the source stays authoritative


def test_extraction_never_changes_the_source_text(session, library):
    before = {p.id: (p.text, p.page_start, p.structural_node_id, p.passage_metadata) for p in library.passages.values()}
    service, _ = make_service(session, quote_answer(library, "Words the model made up."))
    service.extract_book(library.book.id, ["concept", "equation"])
    ProvenanceService(session).verify_book(library.book.id)
    session.expire_all()
    after = {p.id: (p.text, p.page_start, p.structural_node_id, p.passage_metadata) for p in session.scalars(select(Passage))}
    assert after == before


def test_model_output_is_marked_as_such_and_kept_apart_from_the_source(session, library):
    unit = extract_one(session, library, "c1a", quote_answer(library))
    assert unit.origin == "ai_extracted"
    assert unit.details["name"] == "Entropy" and unit.sources[0].quote == QUOTE  # the unit's content vs the quoted source words
    assert library.passages["c1a"].text.count(QUOTE) == 1  # the passage still holds the original wording
    detail = KnowledgeService(session).detail(unit.id)
    assert detail.origin == "ai_extracted" and detail.source_status == "verified"
