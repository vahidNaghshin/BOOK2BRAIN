"""Extraction attempts, validation outcomes and review flags: what is recorded per model call, and what is flagged."""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.extraction.errors import ProviderAuthError, ProviderTimeoutError
from app.extraction.results import Severity, ValidationIssue, ValidationStage
from app.extraction.validation import classify_outcomes
from app.knowledge.types import FlagSeverity, FlagStatus, ReviewFlagCode, ValidationOutcome
from app.services.knowledge_service import KnowledgeService
from app.services.review_flag_service import ReviewFlagService
from database.models import ExtractionAttempt, ExtractionExecution, KnowledgeUnit, ReviewFlag
from tests.extraction_db_helpers import Library, item, make_service, one_item, reply, store_library
from tests.extraction_helpers import LATEST, hosted_config, make_config

O = ValidationOutcome


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


def attempts(session) -> list[ExtractionAttempt]:
    return list(session.scalars(select(ExtractionAttempt).order_by(ExtractionAttempt.started_at, ExtractionAttempt.attempt_number)))


def run_one(session, library, answer=one_item, categories=("concept",), **options):
    service, providers = make_service(session, answer, **options)
    status = service.extract_section(library.nodes["ss1"].id, list(categories))
    return service, providers, status


def issue(code, stage, severity=Severity.WARNING):
    return ValidationIssue(stage=stage, severity=severity, code=code, message=code)


def flags(session, library, **filters):
    return ReviewFlagService(session).list_flags(library.book.id, status=filters.pop("status", None), **filters)


# ---------------------------------------------------------------- what an attempt records


def test_an_attempt_records_everything_needed_to_investigate_it(session, library):
    config = make_config(temperature=0.2, model_version="2025-01", input_cost_per_million_tokens=2.0, output_cost_per_million_tokens=10.0)
    _, _, status = run_one(session, library, configs=[config])
    [attempt] = attempts(session)
    passage = library.passages["ss1a"]

    assert (attempt.extraction_run_id, attempt.passage_id, attempt.attempt_number, attempt.kind) == (status.run.id, passage.id, 1, "initial")
    assert (attempt.model_configuration_id, attempt.provider, attempt.model_id, attempt.model_version) == ("local-test", "ollama", "test-model:1b", "2025-01")
    assert (attempt.prompt_name, attempt.prompt_version) == ("concept_extraction", LATEST)
    assert attempt.generation_parameters["temperature"] == 0.2 and attempt.generation_parameters["max_output_tokens"] == 1024
    assert attempt.started_at <= attempt.completed_at and attempt.duration_ms >= 0
    assert (attempt.input_tokens, attempt.output_tokens, attempt.finish_reason) == (100, 40, "stop")
    assert attempt.estimated_cost_usd == pytest.approx(100 * 2 / 1e6 + 40 * 10 / 1e6)
    assert (attempt.status, attempt.outcome, attempt.provider_error) == ("completed", "valid", None)
    assert (attempt.validation_outcome, attempt.validation_outcomes) == ("schema_valid", ["schema_valid"])


def test_every_attempt_keeps_a_snapshot_of_the_configuration_it_ran_with(session, library):
    config = make_config(temperature=0.1, extra_params={"seed": 7})
    run_one(session, library, configs=[config])
    [attempt] = attempts(session)
    snapshot = attempt.model_configuration
    assert snapshot["id"] == "local-test" and snapshot["temperature"] == 0.1 and snapshot["extra_params"] == {"seed": 7}
    assert snapshot["model"] == "test-model:1b" and snapshot["deployment"] == "local" and snapshot["context_window"] == 8192


def test_the_snapshot_does_not_change_when_the_configuration_changes_later(session, library):
    run_one(session, library, configs=[make_config(temperature=0.1)])
    changed = make_config(model="other-model:7b", temperature=0.9, max_output_tokens=2048)
    service, _ = make_service(session, one_item, configs=[changed])
    service.extract_section(library.nodes["s1"].id, ["concept"])
    first, *_, last = attempts(session)
    assert (first.model_configuration["model"], first.model_configuration["temperature"], first.generation_parameters["max_output_tokens"]) == ("test-model:1b", 0.1, 1024)
    assert (last.model_configuration["model"], last.model_configuration["temperature"], last.generation_parameters["max_output_tokens"]) == ("other-model:7b", 0.9, 2048)


def test_no_credentials_are_stored_in_a_snapshot(session, library):
    secret = "sai_never_store_me"
    run_one(session, library, configs=[hosted_config()], default="hosted-test", environ={"TEST_LLM_KEY": secret})
    [attempt] = attempts(session)
    assert attempt.model_configuration["api_key_env"] == "TEST_LLM_KEY" and secret not in str(attempt.model_configuration)


def test_the_prompt_version_of_every_attempt_follows_the_pinned_version(session, library):
    run_one(session, library, prompt_versions={"concept": "1.0.0"})
    [attempt] = attempts(session)
    assert (attempt.prompt_name, attempt.prompt_version) == ("concept_extraction", "1.0.0")


# ---------------------------------------------------------------- the call versus the answer


def test_a_provider_failure_is_a_failed_call_with_nothing_to_validate(session, library):
    calls = {"n": 0}

    def answer(call):
        calls["n"] += 1
        return ProviderTimeoutError("slow") if calls["n"] == 1 else one_item(call)

    run_one(session, library, answer)
    first, second = attempts(session)
    assert (first.kind, first.status, first.outcome, first.validation_outcome, first.validation_outcomes) == ("initial", "provider_error", "provider_error", "not_validated", [])
    assert (first.provider_error_code, first.provider_error) == ("provider_timeout", "slow") and first.input_tokens is None
    assert (second.kind, second.status, second.validation_outcome) == ("retry", "completed", "schema_valid")
    assert second.model_configuration["id"] == "local-test" and first.model_configuration["id"] == "local-test"


def test_a_validation_failure_is_kept_apart_from_the_call_and_corrected(session, library):
    calls = {"n": 0}

    def answer(call):
        calls["n"] += 1
        return "this is not json" if calls["n"] == 1 else one_item(call)

    _, providers, status = run_one(session, library, answer)
    first, second = attempts(session)
    assert (first.status, first.validation_outcome, first.validation_outcomes) == ("completed", "schema_invalid", ["schema_invalid"])  # the call worked; the answer did not
    assert first.raw_response == "this is not json" and first.issues[0]["code"] == "invalid_json"
    assert (second.kind, second.validation_outcome) == ("correction", "schema_valid")
    assert "not valid JSON" in providers["local-test"].requests[1].messages[-1].content  # the correction carries the feedback
    assert status.run.retry_count == 1 and status.run.validation_failure_count == 1


def test_an_invented_passage_id_is_a_source_failure_not_a_schema_failure(session, library):
    calls = {"n": 0}

    def answer(call):
        calls["n"] += 1
        sources = [{"passage_id": str(uuid.uuid4()), "role": "primary"}] if calls["n"] == 1 else [{"passage_id": call.target_id, "role": "primary"}]
        return reply(item(call, sources=sources))

    run_one(session, library, answer)
    first, second = attempts(session)
    assert first.validation_outcome == "source_invalid" and first.status == "completed"
    assert second.validation_outcome == "schema_valid"


def test_valid_structure_is_not_treated_as_correct_content(session, library):
    def answer(call):
        return reply(item(call, sources=[{"passage_id": call.target_id, "role": "primary", "quote": "Words that are not in the passage."}]))

    run_one(session, library, answer)
    [attempt] = attempts(session)
    assert attempt.validation_outcome == "unsupported_content" and "schema_valid" in attempt.validation_outcomes
    assert {"schema_valid", "unsupported_content", "needs_review"} <= set(attempt.validation_outcomes)
    unit = session.scalars(select(KnowledgeUnit)).one()
    assert unit.validation_status == "uncertain"  # passing the schema did not make it valid


def test_an_answer_that_repeats_itself_is_marked_as_a_duplicate(session, library):
    run_one(session, library, lambda call: reply(item(call), item(call)))
    [attempt] = attempts(session)
    assert "duplicate" in attempt.validation_outcomes and attempt.validation_outcome == "duplicate"


def test_a_repeated_extraction_of_stored_knowledge_is_recorded_as_a_duplicate_on_its_attempt(session, library):
    service, _ = make_service(session, one_item)
    service.extract_section(library.nodes["ss1"].id, ["concept"])
    service.extract_section(library.nodes["ss1"].id, ["concept"], force=True)
    first, second = attempts(session)
    assert "duplicate" not in first.validation_outcomes
    assert second.validation_outcome == "duplicate" and "duplicate" in second.validation_outcomes


@pytest.mark.parametrize(
    "issues, kwargs, primary, everything",
    [
        ([], dict(answered=False), O.NOT_VALIDATED, []),
        ([], {}, O.SCHEMA_VALID, [O.SCHEMA_VALID]),
        ([issue("invalid_json", ValidationStage.STRUCTURAL, Severity.ERROR)], {}, O.SCHEMA_INVALID, [O.SCHEMA_INVALID]),
        ([issue("unknown_passage_id", ValidationStage.SOURCE, Severity.ERROR)], {}, O.SOURCE_INVALID, [O.SOURCE_INVALID, O.SCHEMA_VALID]),
        ([issue("quote_not_in_source", ValidationStage.SEMANTIC)], {}, O.UNSUPPORTED_CONTENT, [O.UNSUPPORTED_CONTENT, O.SCHEMA_VALID]),
        ([issue("expression_not_in_source", ValidationStage.SEMANTIC)], {}, O.UNSUPPORTED_CONTENT, [O.UNSUPPORTED_CONTENT, O.SCHEMA_VALID]),
        ([issue("target_passage_not_cited", ValidationStage.SOURCE)], {}, O.SEMANTIC_CONCERNS, [O.SEMANTIC_CONCERNS, O.SCHEMA_VALID]),
        ([issue("duplicate_item", ValidationStage.QUALITY)], {}, O.DUPLICATE, [O.DUPLICATE, O.SCHEMA_VALID]),
        ([], dict(needs_review=True), O.NEEDS_REVIEW, [O.NEEDS_REVIEW, O.SCHEMA_VALID]),
        ([], dict(duplicate=True, needs_review=True), O.DUPLICATE, [O.DUPLICATE, O.NEEDS_REVIEW, O.SCHEMA_VALID]),
        (
            [issue("quote_not_in_source", ValidationStage.SEMANTIC), issue("target_passage_not_cited", ValidationStage.SOURCE)],
            {}, O.UNSUPPORTED_CONTENT, [O.UNSUPPORTED_CONTENT, O.SEMANTIC_CONCERNS, O.SCHEMA_VALID],
        ),
    ],
)
def test_validation_outcomes_are_classified_separately_from_each_other(issues, kwargs, primary, everything):
    assert classify_outcomes(issues, **kwargs) == (primary, everything)


# ---------------------------------------------------------------- retry limits


def test_the_retry_limit_is_enforced_and_every_attempt_is_kept(session, library):
    _, providers, status = run_one(session, library, lambda call: "garbage", max_retries=2)
    rows = attempts(session)
    assert [(a.attempt_number, a.kind, a.validation_outcome) for a in rows] == [(1, "initial", "schema_invalid"), (2, "correction", "schema_invalid"), (3, "correction", "schema_invalid")]
    assert len(providers["local-test"].requests) == 3 and all(a.raw_response == "garbage" for a in rows)
    assert status.status == "failed" and status.run.retry_count == 2 and status.run.validation_failure_count == 3


def test_provider_errors_are_retried_within_the_same_limit_and_authentication_errors_are_not(session, library):
    _, providers, _ = run_one(session, library, lambda call: ProviderTimeoutError("slow"), max_retries=2)
    assert [a.kind for a in attempts(session)] == ["initial", "retry", "retry"] and {a.status for a in attempts(session)} == {"provider_error"}
    session.query(ExtractionExecution).delete()
    session.commit()
    service, providers = make_service(session, lambda call: ProviderAuthError("bad key"), max_retries=5)
    service.extract_section(library.nodes["ss1"].id, ["concept"], force=True)
    assert len(providers["local-test"].requests) == 1  # the section has one passage: one call, and no retry


def test_a_failed_extraction_never_leaves_a_unit_behind(session, library):
    run_one(session, library, lambda call: "garbage", max_retries=1)
    assert session.scalars(select(KnowledgeUnit)).all() == []
    [execution] = session.scalars(select(ExtractionExecution)).all()
    assert execution.status == "failed" and execution.needs_review and execution.failure_code == "validation_failed"


# ---------------------------------------------------------------- review flags on units


def unit_flags(session, library):
    unit = session.scalars(select(KnowledgeUnit)).one()
    return {f.code: f for f in flags(session, library, unit_id=unit.id)}


def test_a_clean_unit_raises_no_flags(session, library):
    run_one(session, library)
    assert flags(session, library) == []


def test_a_model_that_reports_uncertainty_is_flagged_with_its_reason(session, library):
    run_one(session, library, lambda call: reply(item(call, is_uncertain=True, uncertainty_notes="The definition is implied.", basis="inferred", confidence=0.3)))
    found = unit_flags(session, library)
    assert set(found) == {ReviewFlagCode.UNCERTAIN_EXTRACTION, ReviewFlagCode.INFERRED_CONTENT, ReviewFlagCode.LOW_CONFIDENCE}
    uncertain = found[ReviewFlagCode.UNCERTAIN_EXTRACTION]
    assert (uncertain.message, uncertain.raised_by, uncertain.severity, uncertain.status) == ("The definition is implied.", "model", FlagSeverity.WARNING, FlagStatus.OPEN)
    assert found[ReviewFlagCode.INFERRED_CONTENT].severity == FlagSeverity.INFO and "30%" in found[ReviewFlagCode.LOW_CONFIDENCE].message
    assert uncertain.extraction_run_id is not None and uncertain.target_kind == "unit"


def test_the_low_confidence_threshold_is_a_setting(session, library):
    run_one(session, library, lambda call: reply(item(call, confidence=0.6)), low_confidence_threshold=0.7)
    assert set(unit_flags(session, library)) == {ReviewFlagCode.LOW_CONFIDENCE}


def test_unsupported_content_is_flagged_as_an_unsupported_claim(session, library):
    run_one(session, library, lambda call: reply(item(call, sources=[{"passage_id": call.target_id, "role": "primary", "quote": "Words that are not in the passage."}])))
    found = unit_flags(session, library)
    assert ReviewFlagCode.UNSUPPORTED_CLAIM in found and "does not appear" in found[ReviewFlagCode.UNSUPPORTED_CLAIM].message
    assert ReviewFlagCode.SOURCE_QUOTE_MISMATCH in found  # the stored quote is checked against the passage as well


def test_a_unit_with_the_same_title_but_other_content_from_the_same_passage_is_flagged_not_merged(session, library):
    service, _ = make_service(session, one_item)
    service.extract_section(library.nodes["ss1"].id, ["concept"])
    other, _ = make_service(session, lambda call: reply(item(call, content={"definition": "In information theory, the average surprise of a message source."})))
    other.extract_section(library.nodes["ss1"].id, ["concept"], force=True)
    units = session.scalars(select(KnowledgeUnit).order_by(KnowledgeUnit.created_at)).all()
    assert len(units) == 2
    [flag] = flags(session, library, unit_id=units[1].id, code="potential_duplicate")
    assert flag.details["other_unit_id"] == str(units[0].id) and flags(session, library, unit_id=units[0].id) == []


def test_reduced_context_is_noted_on_the_units_it_produced(session):
    big = store_library(session, "Long", text_size=60)
    service, _ = make_service(session, one_item, configs=[make_config(context_window=3_000, max_output_tokens=500)], context_passages=3, context_safety_margin=0)
    service.extract_section(big.nodes["s1"].id, ["concept"])
    codes = {f.code for f in ReviewFlagService(session).list_flags(big.book.id, status=None)}
    assert ReviewFlagCode.CONTEXT_REDUCED in codes


# ---------------------------------------------------------------- review flags on attempts


def test_an_unresolved_failure_flags_the_last_attempt(session, library):
    _, _, status = run_one(session, library, lambda call: "garbage", max_retries=2)
    found = {f.code: f for f in flags(session, library)}
    flag = found[ReviewFlagCode.UNRESOLVED_EXTRACTION_ERROR]
    last = attempts(session)[-1]
    assert (flag.target_kind, flag.target_id, flag.severity) == ("attempt", last.id, FlagSeverity.ERROR)
    assert flag.details == {"code": "validation_failed", "attempts": 3} and "No valid response" in flag.message
    repeated = found[ReviewFlagCode.REPEATED_VALIDATION_FAILURE]
    assert repeated.target_id == last.id and repeated.details["validation_failures"] == 3 and status.status == "failed"


def test_a_single_corrected_failure_is_not_flagged(session, library):
    calls = {"n": 0}

    def answer(call):
        calls["n"] += 1
        return "garbage" if calls["n"] == 1 else one_item(call)

    run_one(session, library, answer)
    assert flags(session, library) == []


def test_two_validation_failures_before_success_are_flagged_as_repeated(session, library):
    calls = {"n": 0}

    def answer(call):
        calls["n"] += 1
        return "garbage" if calls["n"] <= 2 else one_item(call)

    run_one(session, library, answer, max_retries=3)
    [flag] = flags(session, library)
    assert flag.code == ReviewFlagCode.REPEATED_VALIDATION_FAILURE and flag.target_kind == "attempt"


def test_provider_failures_that_never_recover_are_flagged_as_unresolved(session, library):
    run_one(session, library, lambda call: ProviderTimeoutError("slow"), max_retries=1)
    [flag] = flags(session, library)
    assert flag.code == ReviewFlagCode.UNRESOLVED_EXTRACTION_ERROR and flag.details["code"] == "provider_timeout"


# ---------------------------------------------------------------- the flag service


def a_unit(session, library):
    run_one(session, library)
    return session.scalars(select(KnowledgeUnit)).one()


def test_flags_can_be_raised_listed_and_counted(session, library):
    unit = a_unit(session, library)
    service = ReviewFlagService(session)
    flag, created = service.flag_unit(library.book.id, unit.id, "low_confidence", "Looks doubtful.", details={"x": 1}, severity="info")
    assert created and flag.code == "low_confidence" and flag.status == "open" and flag.raised_by == "system"
    service.flag_unit(library.book.id, unit.id, ReviewFlagCode.NO_SOURCE, "No source.", severity=FlagSeverity.ERROR)
    assert [f.code for f in service.list_flags(library.book.id, unit_id=unit.id)] == [ReviewFlagCode.LOW_CONFIDENCE, ReviewFlagCode.NO_SOURCE]
    assert service.counts_by_code(library.book.id) == {"low_confidence": 1, "no_source": 1}
    assert [f.code for f in service.list_flags(library.book.id, code="no_source")] == [ReviewFlagCode.NO_SOURCE]
    assert service.flags_for_unit(library.book.id, unit.id)[0].details == {"x": 1}
    with pytest.raises(ValueError):
        service.flag_unit(library.book.id, unit.id, "made_up_reason", "x")


def test_the_same_open_flag_is_not_raised_twice_but_a_different_key_is(session, library):
    unit = a_unit(session, library)
    service = ReviewFlagService(session)
    first, created_first = service.flag_unit(library.book.id, unit.id, "invalid_source_reference", "A", dedupe_key="passage:1")
    again, created_again = service.flag_unit(library.book.id, unit.id, "invalid_source_reference", "A again", dedupe_key="passage:1")
    other, created_other = service.flag_unit(library.book.id, unit.id, "invalid_source_reference", "B", dedupe_key="passage:2")
    assert (created_first, created_again, created_other) == (True, False, True) and again.id == first.id and other.id != first.id


def test_a_flag_can_be_resolved_or_dismissed_with_a_note_and_raised_again_later(session, library):
    unit = a_unit(session, library)
    service = ReviewFlagService(session)
    flag, _ = service.flag_unit(library.book.id, unit.id, "low_confidence", "Doubtful.")
    resolved = service.resolve(flag.id, "  Checked against the page.  ")
    assert (resolved.status, resolved.resolution_note) == (FlagStatus.RESOLVED, "Checked against the page.") and resolved.resolved_at is not None
    assert service.list_flags(library.book.id) == [] and len(service.list_flags(library.book.id, status="resolved")) == 1
    second, created = service.flag_unit(library.book.id, unit.id, "low_confidence", "Doubtful again.")
    assert created and second.id != flag.id  # a resolved flag does not block a new one
    dismissed = service.resolve(second.id, "Not a problem.", dismiss=True)
    assert dismissed.status == FlagStatus.DISMISSED
    assert service.resolve(second.id, "twice").resolution_note == "Not a problem."  # closing again changes nothing
    with pytest.raises(LookupError):
        service.resolve(uuid.uuid4())


def test_a_flag_belongs_to_exactly_one_target(session, library):
    unit = a_unit(session, library)
    attempt = attempts(session)[0]

    def flag(**targets):
        return ReviewFlag(book_id=library.book.id, code="low_confidence", message="x", **targets)

    for bad in (flag(), flag(knowledge_unit_id=unit.id, extraction_attempt_id=attempt.id)):
        session.add(bad)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    session.add(flag(extraction_attempt_id=attempt.id))
    session.commit()


def test_the_details_screen_data_carries_flags_and_attempts(session, library):
    run_one(session, library, lambda call: reply(item(call, is_uncertain=True, uncertainty_notes="Implied.")))
    unit = session.scalars(select(KnowledgeUnit)).one()
    detail = KnowledgeService(session).detail(unit.id)
    assert [f.code for f in detail.flags] == [ReviewFlagCode.UNCERTAIN_EXTRACTION]
    [attempt] = detail.attempts
    assert (attempt.model_configuration_id, attempt.prompt, attempt.status, attempt.validation_outcome) == ("local-test", f"concept_extraction@{LATEST}", "completed", "needs_review")
    assert attempt.model_configuration["id"] == "local-test" and detail.source_status == "verified"
    resolved = KnowledgeService(session).resolve_flag(detail.flags[0].id, "Looked at it.")
    assert resolved.status == FlagStatus.RESOLVED and KnowledgeService(session).detail(unit.id).flags == []
