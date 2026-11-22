"""Relationship extraction: the prompt, the answer parser, and the pipeline step that stores the results."""

import json
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.extraction.errors import ProviderTimeoutError
from app.extraction.relationship_parsing import parse_relationships
from app.extraction.relationship_prompts import RELATIONSHIP_PROMPT, RelationshipRequest, UnitSummary
from app.extraction.results import PassageInput, Severity, ValidationStage
from app.knowledge.types import KnowledgeType, ReviewFlagCode
from app.services.relationship_extraction import RelationshipExtractionService
from app.services.review_flag_service import ReviewFlagService
from database.models import ExtractionAttempt, ExtractionExecution, ExtractionPassageRecord, KnowledgeRelationship, KnowledgeUnit
from tests.extraction_db_helpers import (
    Library,
    make_service,
    one_item,
    only_provider,
    relationship,
    relationships_reply,
    store_library,
    with_relationships,
)
from tests.extraction_helpers import hosted_config, make_config

QUOTE = "Entropy is a measure of the disorder of a system."
CATEGORIES = ["concept", "principle"]


# ---------------------------------------------------------------- the answer parser (pure)

PASSAGE = PassageInput(id=uuid.uuid4(), text="Heat flows from hot to cold. This is why entropy increases in an isolated system.", page_start=3, page_end=3)
CONCEPT, PRINCIPLE, EARLIER = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
REQUEST = RelationshipRequest(
    PASSAGE,
    [
        UnitSummary(CONCEPT, KnowledgeType.CONCEPT, "Entropy", "A measure of disorder.", True),
        UnitSummary(PRINCIPLE, KnowledgeType.PRINCIPLE, "Second law", "Entropy of an isolated system never decreases.", True),
        UnitSummary(EARLIER, KnowledgeType.CONCEPT, "Heat", "Energy in transit.", False),
    ],
)


def answer(*items, **extra):
    return json.dumps({"relationships": list(items), "notes": None} | extra)


def entry(source=PRINCIPLE, target=CONCEPT, kind="depends_on", **fields):
    return {"source_id": str(source), "target_id": str(target), "type": kind, "explanation": "Because.", "quote": "entropy increases in an isolated system", "confidence": 0.9} | fields


def codes(parsed):
    return {i.code for i in parsed.issues}


def test_a_valid_answer_is_parsed_with_its_evidence_checked():
    parsed = parse_relationships(answer(entry()), REQUEST)
    assert not parsed.issues and len(parsed.items) == 1
    [item] = parsed.items
    assert (item.source_id, item.target_id, item.relationship_type.value, item.confidence) == (PRINCIPLE, CONCEPT, "depends_on", 0.9)
    assert item.evidence_verified and item.explanation == "Because."


def test_fences_and_chatter_around_the_json_are_tolerated():
    assert len(parse_relationships("Here you go:\n```json\n" + answer(entry()) + "\n```", REQUEST).items) == 1


@pytest.mark.parametrize("raw, code", [("", "empty_response"), ("nope", "invalid_json"), ("[]", "not_an_object"), ("{}", "missing_required_field"), ('{"relationships": 3}', "invalid_value")])
def test_malformed_answers_give_clear_errors(raw, code):
    parsed = parse_relationships(raw, REQUEST)
    assert code in codes(parsed) and parsed.items == [] and parsed.has_errors


def test_an_empty_list_means_nothing_is_related():
    parsed = parse_relationships(answer(notes="Nothing connects them."), REQUEST)
    assert parsed.items == [] and not parsed.issues and parsed.notes == "Nothing connects them."


@pytest.mark.parametrize(
    "bad, code",
    [
        (entry(source=uuid.uuid4()), "unknown_unit_id"),
        (entry(target="not-a-uuid"), "unknown_unit_id"),
        (entry(kind="resembles"), "invalid_relationship_type"),
        (entry(source=CONCEPT, target=CONCEPT), "self_relationship"),
        (entry(kind="supports"), "type_not_allowed"),  # a principle cannot support a concept
        ({"source_id": str(PRINCIPLE), "type": "depends_on"}, "missing_required_field"),
    ],
)
def test_invented_ids_unknown_types_self_links_and_impossible_combinations_are_errors(bad, code):
    parsed = parse_relationships(answer(entry(), bad), REQUEST)
    assert code in codes(parsed) and parsed.items == []  # all or nothing: the whole answer is corrected, nothing is dropped silently
    assert all(i.severity == Severity.ERROR for i in parsed.issues if i.code == code)


def test_a_relationship_between_two_context_units_is_refused():
    other = uuid.uuid4()
    request = RelationshipRequest(PASSAGE, [*REQUEST.units, UnitSummary(other, KnowledgeType.CONCEPT, "Other", "x", False)])
    parsed = parse_relationships(answer(entry(source=other, target=EARLIER, kind="related_to")), request)
    assert "no_endpoint_in_passage" in codes(parsed) and parsed.items == []


def test_evidence_that_cannot_be_verified_is_a_warning_and_marks_the_relationship_unverified():
    parsed = parse_relationships(answer(entry(quote="words that are nowhere in the passage"), entry(target=EARLIER, quote=None)), REQUEST)
    assert not parsed.has_errors and [i.code for i in parsed.issues] == ["quote_not_in_source", "no_evidence_quote"]
    assert all(i.severity == Severity.WARNING and i.stage == ValidationStage.SEMANTIC for i in parsed.issues)
    assert [item.evidence_verified for item in parsed.items] == [False, False]


def test_the_same_relationship_reported_twice_is_kept_once_and_said_so():
    parsed = parse_relationships(answer(entry(), entry()), REQUEST)
    assert len(parsed.items) == 1 and [i.code for i in parsed.issues] == ["duplicate_relationship"]
    assert parsed.issues[0].severity == Severity.INFO


def test_confidence_outside_zero_to_one_is_ignored_not_trusted():
    parsed = parse_relationships(answer(entry(confidence=7), entry(target=EARLIER, confidence=True)), REQUEST)
    assert [i.confidence for i in parsed.items] == [None, None]


# ---------------------------------------------------------------- the prompt


def test_the_prompt_is_versioned_and_tells_the_model_what_it_may_and_may_not_do():
    assert RELATIONSHIP_PROMPT.identifier == "relationship_extraction@1.0.0"
    system = RELATIONSHIP_PROMPT.render_system().lower()
    for phrase in ("only when this passage supports it", "never invent an id", "exact", "similar", "suggestion", "json", "in_passage"):
        assert phrase in system
    for kind in ("related_to", "depends_on", "supports", "contradicts", "part_of", "derived_from", "example_of", "applies_to"):
        assert kind in system
    user = RELATIONSHIP_PROMPT.render_user(REQUEST)
    assert str(PASSAGE.id) in user and str(CONCEPT) in user and '"in_passage": false' in user and "Heat flows" in user
    enum = RELATIONSHIP_PROMPT.output_schema()["properties"]["relationships"]["items"]["properties"]["type"]["enum"]
    assert "derived_from" in enum and len(enum) == 11


# ---------------------------------------------------------------- the pipeline step


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


def related(call):
    """The passage says the second law depends on entropy (with evidence) and that the two are related (without)."""
    return relationships_reply(
        relationship(call, "Second law", "Entropy", "depends_on", quote=QUOTE),
        relationship(call, "Entropy", "Second law", "related_to", quote=None),
    )


def run(session, library, relations=related, knowledge=one_item, scope="ss1", categories=CATEGORIES, **options):
    service, providers = make_service(session, with_relationships(knowledge, relations), **options.pop("service", {}))
    node = library.nodes[scope]
    extract = service.extract_chapter if node.node_type == "chapter" else service.extract_section
    status = extract(node.id, categories, extract_relationships=True, **options)
    return service, providers, status


def relationship_calls(providers):
    return [r for p in providers.values() for r in p.requests if "Knowledge units:" in r.messages[0].content]


def test_relationships_are_extracted_and_stored_with_their_evidence_or_as_suggestions(session, library):
    _, providers, status = run(session, library)
    stored = {r.relationship_type: r for r in session.scalars(select(KnowledgeRelationship))}
    assert set(stored) == {"depends_on", "related_to"}
    depends, related_to = stored["depends_on"], stored["related_to"]
    assert (depends.origin, depends.evidence_quote, depends.review_status) == ("extracted", QUOTE, "unreviewed")
    assert [link.passage_id for link in depends.sources] == [library.passages["ss1a"].id]
    assert (related_to.origin, related_to.evidence_quote) == ("ai_suggested", None)
    assert depends.source.knowledge_type == "principle" and depends.target.knowledge_type == "concept"
    assert depends.extraction_run_id == status.run.id and depends.extraction_execution_id is not None
    assert status.run.relationships_extracted == 2 and len(relationship_calls(providers)) == 1


def test_a_passage_counts_its_units_but_not_its_relationships(session, library):
    service, _, status = run(session, library)
    stored = len(session.scalars(select(KnowledgeUnit)).all())
    assert status.run.relationships_extracted == 2
    assert sum(v.units_extracted for v in service.get_passage_status(status.run.id)) == stored == status.run.units_extracted


def test_a_suggestion_is_flagged_but_a_verified_relationship_is_not(session, library):
    run(session, library)
    flags = ReviewFlagService(session).list_flags(library.book.id, status=None)
    [flag] = [f for f in flags if f.code == ReviewFlagCode.AI_SUGGESTED_RELATIONSHIP]
    suggested = session.scalars(select(KnowledgeRelationship).where(KnowledgeRelationship.origin == "ai_suggested")).one()
    assert flag.target_id == suggested.id and flag.raised_by == "model"
    assert not [f for f in flags if f.target_kind == "relationship" and f.target_id != suggested.id]


def test_the_relationship_call_is_recorded_with_model_prompt_and_attempts(session, library):
    _, _, status = run(session, library)
    execution = session.scalars(select(ExtractionExecution).where(ExtractionExecution.knowledge_type == "relationship")).one()
    assert (execution.prompt_name, execution.prompt_version) == ("relationship_extraction", "1.0.0")
    assert (execution.model_configuration_id, execution.status, execution.items_extracted) == ("local-test", "needs_review", 2)
    assert execution.needs_review and "suggested by the model" in execution.review_reasons[0]
    [attempt] = execution.attempts
    # the schema was fine, but one relationship came without evidence: that is reported next to it, not hidden by it
    assert (attempt.status, attempt.validation_outcome, attempt.model_configuration["id"]) == ("completed", "semantic_concerns", "local-test")
    assert attempt.validation_outcomes == ["semantic_concerns", "schema_valid"]
    assert attempt.passage_id == library.passages["ss1a"].id and attempt.extraction_run_id == status.run.id
    assert status.run.attempt_count == 3  # concept, principle and the relationship call
    assert status.status == "needs_review"


def test_the_model_is_shown_the_units_of_the_passage_and_earlier_units_of_the_chapter_as_targets(session, library):
    seen = []

    def relations(call):
        seen.append([(u["title"], u["in_passage"]) for u in call.units])
        return relationships_reply()

    run(session, library, relations, scope="ch1")
    assert len(seen) == 5  # every passage of the chapter has two units of its own
    assert seen[0] == [("Entropy", True), ("Second law", True)]
    assert seen[1][:2] == [("Entropy", True), ("Second law", True)] and seen[1][2:] == [("Entropy", False), ("Second law", False)]
    assert len(seen[4]) == 2 + 8  # earlier units are capped (by default 10)


def test_a_relationship_can_point_at_an_earlier_unit_of_the_chapter(session, library):
    def relations(call):
        if len(call.units) <= 2:
            return relationships_reply()
        return relationships_reply(relationship(call, "Second law", "Entropy", "depends_on", target_in_passage=False, quote=QUOTE))

    run(session, library, relations, scope="ch1", limit=2)
    [stored] = session.scalars(select(KnowledgeRelationship)).all()
    first = session.scalars(select(KnowledgeUnit).where(KnowledgeUnit.page_start == 2, KnowledgeUnit.knowledge_type == "concept")).one()
    assert stored.target_unit_id == first.id and stored.source.page_start == 3  # evidence in this passage, target from the earlier one
    assert [link.passage_id for link in stored.sources] == [library.passages["c1b"].id]


def test_a_passage_with_nothing_to_relate_makes_no_call(session, library):
    _, providers, status = run(session, library, scope="ch1", categories=["concept"])
    assert len(relationship_calls(providers)) == 4  # the first passage has one unit and no earlier ones
    assert status.run.relationships_extracted == 0


def test_unknown_ids_are_corrected_and_the_validation_failure_is_counted(session, library):
    calls = {"n": 0}

    def relations(call):
        calls["n"] += 1
        if calls["n"] == 1:
            return relationships_reply({"source_id": str(uuid.uuid4()), "target_id": call.unit("Entropy"), "type": "depends_on"})
        return related(call)

    _, providers, status = run(session, library, relations)
    execution = session.scalars(select(ExtractionExecution).where(ExtractionExecution.knowledge_type == "relationship")).one()
    assert (execution.attempt_count, execution.validation_failures) == (2, 1)
    first, second = execution.attempts
    assert (first.validation_outcome, first.kind, first.status) == ("source_invalid", "initial", "completed")
    assert first.outcome == "invalid" and second.kind == "correction" and first.issues[0]["code"] == "unknown_unit_id"
    assert status.run.validation_failure_count == 1 and session.scalars(select(KnowledgeRelationship)).all().__len__() == 2
    assert "Never invent ids" in only_provider(providers).requests[-1].messages[-1].content  # the correction names the problem


def test_a_failing_relationship_step_is_flagged_and_does_not_fail_the_passage_or_the_run(session, library):
    def relations(call):
        raise ProviderTimeoutError("slow")

    _, _, status = run(session, library, relations, service={"max_retries": 1})
    assert session.scalars(select(KnowledgeUnit)).all().__len__() == 2  # the knowledge is intact
    execution = session.scalars(select(ExtractionExecution).where(ExtractionExecution.knowledge_type == "relationship")).one()
    assert (execution.status, execution.failure_code, execution.attempt_count) == ("failed", "provider_timeout", 2)
    flags = ReviewFlagService(session).list_flags(library.book.id)
    assert [(f.target_kind, f.code) for f in flags if f.target_kind == "attempt"] == [("attempt", ReviewFlagCode.UNRESOLVED_EXTRACTION_ERROR)]
    record = session.scalars(select(ExtractionPassageRecord)).one()
    assert record.status == "needs_review" and record.error_code == "provider_timeout"
    assert status.status == "needs_review" and status.failed_passages == 0


def test_repeating_the_step_never_creates_a_relationship_twice(session, library):
    service, providers, first = run(session, library)
    before = session.scalars(select(KnowledgeRelationship.id)).all()
    again = service.extract_section(library.nodes["ss1"].id, CATEGORIES, extract_relationships=True, force=True)
    assert again.run.relationships_extracted == 0 and again.run.duplicate_count == 2  # the knowledge repeated too, and was skipped
    assert session.scalars(select(KnowledgeRelationship.id)).all() == before
    assert len(relationship_calls(providers)) == 2  # force asked the model again


def test_finished_relationship_work_is_not_repeated_without_force(session, library):
    service, providers, _ = run(session, library)
    with pytest.raises(Exception, match="already been extracted"):
        service.extract_section(library.nodes["ss1"].id, CATEGORIES, extract_relationships=True)
    assert len(relationship_calls(providers)) == 1


def test_relationships_can_be_added_later_to_knowledge_that_is_already_extracted(session, library):
    service, providers = make_service(session, with_relationships(one_item, related))
    service.extract_section(library.nodes["ss1"].id, CATEGORIES)
    assert session.scalars(select(KnowledgeRelationship)).all() == [] and relationship_calls(providers) == []
    plan = service.plan(service.section_scope(library.nodes["ss1"].id), CATEGORIES, extract_relationships=True)
    assert (plan.model_calls, plan.relationship_calls) == (0, 1)
    later = service.extract_section(library.nodes["ss1"].id, CATEGORIES, extract_relationships=True)
    assert later.run.relationships_extracted == 2 and later.run.attempt_count == 1  # only the relationship call was made
    assert len(relationship_calls(providers)) == 1


def test_a_resumed_run_retries_only_the_relationship_calls_that_failed(session, library):
    state = {"fail": True}
    failing = str(library.passages["s1b"].id)

    def relations(call):
        if state["fail"] and call.target_id == failing:
            return ProviderTimeoutError("slow")
        return relationships_reply(relationship(call, "Second law", "Entropy", "depends_on", quote=QUOTE))

    service, providers = make_service(session, with_relationships(one_item, relations), max_retries=0)
    run_id = service.create_run(service.chapter_scope(library.nodes["ch1"].id), CATEGORIES, extract_relationships=True)
    first = service.execute_run(run_id)
    assert first.status == "needs_review" and first.run.relationships_extracted == 4 and first.failed_passages == 0
    info = service.get_resume_info(run_id)
    assert (info.can_resume, info.failed, info.completed) == (True, 1, 4) and first.can_resume
    calls_before, knowledge_before = len(relationship_calls(providers)), len(only_provider(providers).requests)

    state["fail"] = False
    finished = service.resume_run(run_id)
    assert len(relationship_calls(providers)) - calls_before == 1  # only the failed passage was asked again
    assert len(only_provider(providers).requests) - knowledge_before == 1  # and no knowledge call was repeated
    assert finished.status == "completed" and finished.run.relationships_extracted == 5
    assert session.scalars(select(KnowledgeRelationship)).all().__len__() == 5


def test_the_relationship_step_can_use_its_own_model(session, library):
    local, hosted = make_config(), hosted_config()
    service, providers = make_service(
        session, with_relationships(one_item, related), configs=[local, hosted], default="local-test", overrides={"relationship": "hosted-test"}
    )
    service.extract_section(library.nodes["ss1"].id, CATEGORIES, extract_relationships=True)
    assert len(providers["hosted-test"].requests) == 1 and len(providers["local-test"].requests) == 2
    execution = session.scalars(select(ExtractionExecution).where(ExtractionExecution.knowledge_type == "relationship")).one()
    assert execution.model_configuration_id == "hosted-test"

    service, providers = make_service(session, with_relationships(one_item, related), configs=[local, hosted], default="local-test", overrides={"relationship": "hosted-test"})
    service.extract_section(library.nodes["s1"].id, CATEGORIES, extract_relationships=True, model_id="local-test")
    assert providers["hosted-test"].requests == []  # one model per run when one is chosen


def test_similar_titles_alone_never_create_a_relationship(session, library):
    run(session, library, lambda call: relationships_reply())
    assert session.scalars(select(KnowledgeRelationship)).all() == []
    candidates = RelationshipExtractionService(session, None).candidates(library.passages["ss1a"])
    assert [c.title for c in candidates] == ["Entropy", "Second law"]  # both are shown; nothing is related just because they look alike


def test_a_run_without_the_option_makes_no_relationship_calls(session, library):
    service, providers = make_service(session, with_relationships(one_item, related))
    status = service.extract_section(library.nodes["ss1"].id, CATEGORIES)
    assert relationship_calls(providers) == [] and status.run.relationships_extracted == 0
    assert status.run.parameters["extract_relationships"] is False
