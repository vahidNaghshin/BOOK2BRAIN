"""The extraction pipeline end to end on the test database, with a mocked LLM."""

import json
from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.extraction.errors import ProviderAuthError, ProviderTimeoutError
from app.extraction.results import ExtractionStatus
from app.knowledge.types import KnowledgeType
from app.services import knowledge_extraction_service as pipeline
from app.services.errors import ExtractionRunError, ExtractionScopeError
from app.services.knowledge_extraction_service import NothingToExtractError
from database.models import (
    ExtractionAttempt,
    ExtractionExecution,
    ExtractionRun,
    KnowledgeUnit,
    KnowledgeUnitPassage,
    Passage,
)
from tests.extraction_db_helpers import Library, item, make_service, one_item, only_provider, reply, store_library
from tests.extraction_helpers import LATEST, hosted_config, make_config

ALL_PASSAGES = 8


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


def count(session: Session, model) -> int:
    return session.scalar(select(func.count()).select_from(model))


def units(session: Session) -> list[KnowledgeUnit]:
    return list(session.scalars(select(KnowledgeUnit).order_by(KnowledgeUnit.page_start, KnowledgeUnit.title)))


def executions(session: Session) -> list[ExtractionExecution]:
    return list(session.scalars(select(ExtractionExecution).order_by(ExtractionExecution.started_at)))


# ---------------------------------------------------------------- scopes: book, chapter, section


def test_a_book_run_extracts_every_passage_and_stores_everything_it_needs(session, library):
    service, providers = make_service(session)
    status = service.extract_book(library.book.id, ["concept"])
    run = status.run

    assert status.status == "completed" and run.scope_type == "book" and not status.is_active and status.progress == 1.0
    assert (run.total_passages, run.passages_processed, run.units_extracted, run.failure_count) == (ALL_PASSAGES, ALL_PASSAGES, ALL_PASSAGES, 0)
    assert (run.attempt_count, run.retry_count, run.validation_failure_count) == (ALL_PASSAGES, 0, 0)
    assert (run.input_tokens, run.output_tokens) == (100 * ALL_PASSAGES, 40 * ALL_PASSAGES)
    assert run.started_at and run.completed_at and status.duration_seconds is not None
    assert len(only_provider(providers).requests) == ALL_PASSAGES
    assert (count(session, ExtractionExecution), count(session, ExtractionAttempt), count(session, KnowledgeUnit)) == (8, 8, 8)


def test_a_chapter_run_covers_the_chapter_with_its_sections(session, library):
    service, providers = make_service(session)
    status = service.extract_chapter(library.nodes["ch1"].id, ["concept"])
    assert status.run.scope_type == "chapter" and status.run.structural_node_id == library.nodes["ch1"].id
    assert status.run.total_passages == 5
    stored = {u.title: u for u in units(session)}
    pages = sorted(u.page_start for u in units(session))
    assert pages == [2, 3, 4, 5, 6] and len(stored) == 1 or len(units(session)) == 5
    assert {u.chapter_node_id for u in units(session)} == {library.nodes["ch1"].id}


def test_a_section_run_covers_the_section_and_its_subsections_only(session, library):
    service, _ = make_service(session)
    status = service.extract_section(library.nodes["s1"].id, ["concept"])
    assert status.run.scope_type == "section" and status.run.total_passages == 3
    assert sorted(u.page_start for u in units(session)) == [4, 5, 6]
    sections = {u.section_node_id for u in units(session)}
    assert sections == {library.nodes["s1"].id, library.nodes["ss1"].id}  # each unit points at its own nearest section
    assert {u.chapter_node_id for u in units(session)} == {library.nodes["ch1"].id}


def test_the_wrong_kind_of_node_is_refused_before_anything_is_stored(session, library):
    service, providers = make_service(session)
    with pytest.raises(ExtractionScopeError, match="not a chapter"):
        service.extract_chapter(library.nodes["s1"].id, ["concept"])
    with pytest.raises(ExtractionScopeError, match="not a section"):
        service.extract_section(library.nodes["ch1"].id, ["concept"])
    assert count(session, ExtractionRun) == 0 and only_provider(providers).requests == []


def test_a_single_document_of_a_book_can_be_extracted(session, library):
    service, _ = make_service(session)
    scope = service.book_scope(library.book.id, library.document.id)
    status = service.extract_book(library.book.id, ["concept"], document_id=library.document.id, limit=2)
    assert scope.kind == "document" and status.run.scope_type == "document" and status.run.document_id == library.document.id


def test_unknown_scopes_and_empty_books_give_clear_errors(session, library):
    service, _ = make_service(session)
    with pytest.raises(ExtractionScopeError, match="book does not exist"):
        service.extract_book(__import__("uuid").uuid4(), ["concept"])
    empty = store_library(session, "Empty")
    session.query(Passage).filter(Passage.book_id == empty.book.id).delete()
    session.commit()
    with pytest.raises(ExtractionRunError, match="No passages"):
        service.extract_book(empty.book.id, ["concept"])
    with pytest.raises(ExtractionRunError, match="at least one"):
        service.extract_book(library.book.id, [])
    with pytest.raises(ExtractionRunError, match="Unknown knowledge category"):
        service.extract_book(library.book.id, ["poem"])


# ---------------------------------------------------------------- units, categories, sources


def test_one_passage_can_yield_several_units(session, library):
    def answer(call):
        names = [("Entropy", "A measure of disorder."), ("Temperature", "A measure of mean kinetic energy."), ("Heat", "Energy in transit.")]
        return reply(*[item(call, content={"name": n, "definition": d}) for n, d in names])

    service, _ = make_service(session, answer)
    status = service.extract_section(library.nodes["ss1"].id, ["concept"])
    assert status.run.units_extracted == 3 and status.run.passages_processed == 1
    assert sorted(u.title for u in units(session)) == ["Entropy", "Heat", "Temperature"]
    execution = executions(session)[0]
    assert execution.items_extracted == 3 and len({u.extraction_execution_id for u in units(session)}) == 1


def test_one_passage_can_yield_several_categories(session, library):
    service, _ = make_service(session)
    status = service.extract_section(library.nodes["ss1"].id, ["concept", "principle", "method", "framework", "evidence", "example", "equation"])
    assert status.run.units_extracted == 7 and status.run.attempt_count == 7
    assert {u.knowledge_type for u in units(session)} == {t.value for t in KnowledgeType}
    assert {e.prompt_name for e in executions(session)} == {f"{t.value}_extraction" for t in KnowledgeType}
    assert {e.prompt_version for e in executions(session)} == {LATEST}


def test_every_unit_keeps_its_source_passages_pages_and_place_in_the_book(session, library):
    target, neighbour = library.passages["s1a"], library.passages["c1b"]

    def answer(call):
        if call.target_id != str(target.id):
            return reply()
        sources = [{"passage_id": call.target_id, "role": "primary", "quote": "Entropy is a measure of the disorder of a system."}, {"passage_id": str(neighbour.id), "role": "supporting"}]
        return reply(item(call, sources=sources))

    service, _ = make_service(session, answer)
    service.extract_section(library.nodes["s1"].id, ["concept"])
    unit = units(session)[0]
    links = {(link.passage_id, link.role): link for link in session.scalars(select(KnowledgeUnitPassage).where(KnowledgeUnitPassage.knowledge_unit_id == unit.id))}
    assert set(links) == {(target.id, "primary"), (neighbour.id, "supporting")}
    assert links[(target.id, "primary")].quote.startswith("Entropy is a measure")
    assert (unit.page_start, unit.page_end, unit.page_status) == (3, 4, "exact")  # the cited neighbour is on page 3
    assert (unit.chapter_node_id, unit.section_node_id) == (library.nodes["ch1"].id, library.nodes["s1"].id)
    assert unit.extraction_run_id and unit.extraction_execution_id and unit.validation_status == "valid" and unit.confidence == 0.8
    assert unit.details["name"] == "Entropy" and unit.schema_version == 1 and unit.review_status == "unreviewed"


def test_passages_with_nothing_to_extract_are_processed_without_units(session, library):
    service, _ = make_service(session, lambda call: '{"items": [], "notes": "Nothing here."}')
    status = service.extract_book(library.book.id, ["equation"])
    assert status.status == "completed" and status.run.units_extracted == 0 and status.run.passages_processed == ALL_PASSAGES
    assert {e.status for e in executions(session)} == {"succeeded"} and executions(session)[0].notes == "Nothing here."


# ---------------------------------------------------------------- model selection and capabilities


def test_a_run_can_select_any_active_configured_model(session, library):
    local, hosted = make_config(), hosted_config()
    service, providers = make_service(session, configs=[local, hosted], default="local-test")
    first = service.extract_chapter(library.nodes["ch2"].id, ["concept"], model_id="hosted-test")
    assert (first.run.model_configuration_id, first.run.provider) == ("hosted-test", "openai")
    assert first.model_label == "hosted-test (gpt-test)" and first.run.model_snapshot["deployment"] == "hosted"
    second = service.extract_chapter(library.nodes["ch1"].id, ["concept"], model_id="local-test")
    assert (second.run.model_configuration_id, second.run.provider) == ("local-test", "ollama")
    assert len(providers["hosted-test"].requests) == 2 and len(providers["local-test"].requests) == 5
    assert {e.model_configuration_id for e in executions(session)} == {"hosted-test", "local-test"}


def test_the_application_default_is_used_when_no_model_is_selected(session, library):
    local, hosted = make_config(), hosted_config()
    service, providers = make_service(session, configs=[local, hosted], default="hosted-test")
    status = service.extract_section(library.nodes["ss1"].id, ["concept"])
    assert status.run.model_configuration_id == "hosted-test" and len(providers["hosted-test"].requests) == 1


def test_a_clear_error_when_no_model_is_selected_or_the_choice_is_invalid(session, library):
    service, _ = make_service(session, default=None)
    with pytest.raises(ExtractionRunError, match="EXTRACTION_MODEL_ID"):
        service.extract_book(library.book.id, ["concept"])

    service, _ = make_service(session, configs=[make_config(), make_config(id="off", active=False)])
    with pytest.raises(ExtractionRunError, match="inactive"):
        service.extract_book(library.book.id, ["concept"], model_id="off")
    with pytest.raises(ExtractionRunError, match="Unknown model configuration 'nope'"):
        service.extract_book(library.book.id, ["concept"], model_id="nope")

    service, providers = make_service(session, configs=[hosted_config()], default="hosted-test", environ={})
    with pytest.raises(ExtractionRunError, match="TEST_LLM_KEY"):
        service.extract_book(library.book.id, ["concept"])
    assert count(session, ExtractionRun) == 0 and only_provider(providers).requests == []


def test_available_models_say_which_can_be_used_and_why_not(session):
    configs = [make_config(), hosted_config(), make_config(id="off", active=False)]
    service, _ = make_service(session, configs=configs, environ={})
    options = {o.id: o for o in service.available_models()}
    assert options["local-test"].ready and options["local-test"].deployment == "local"
    assert not options["hosted-test"].ready and "TEST_LLM_KEY" in options["hosted-test"].reason
    assert not options["off"].ready and "inactive" in options["off"].reason
    assert options["hosted-test"].label == "hosted-test (gpt-test, hosted)"


def test_one_model_per_run_ignores_per_category_overrides_but_the_default_setup_keeps_them(session, library):
    local, hosted = make_config(), hosted_config()
    overrides = {"equation": "hosted-test"}
    service, _ = make_service(session, configs=[local, hosted], default="local-test", overrides=overrides)

    mixed = service.extract_section(library.nodes["ss1"].id, ["concept", "equation"])
    assert mixed.run.model_configuration_id is None and mixed.run.model_snapshot is None
    assert mixed.run.parameters["model_configurations"] == {"concept": "local-test", "equation": "hosted-test"}
    assert {(e.knowledge_type, e.model_configuration_id) for e in executions(session)} == {("concept", "local-test"), ("equation", "hosted-test")}

    single = service.extract_section(library.nodes["s1"].id, ["concept", "equation"], model_id="local-test")
    assert single.run.model_configuration_id == "local-test"
    assert {e.model_configuration_id for e in executions(session) if e.extraction_run_id == single.run.id} == {"local-test"}


def test_generation_parameters_and_unsupported_ones_are_respected(session, library):
    config = make_config(temperature=0.3, supports_temperature=True, extra_params={"seed": 7, "mirostat": 2}, unsupported_params=["mirostat"])
    service, _ = make_service(session, configs=[config])
    service.extract_section(library.nodes["ss1"].id, ["concept"])
    execution = executions(session)[0]
    assert execution.generation_parameters == {"max_output_tokens": 1024, "temperature": 0.3, "structured_output": "json_mode", "seed": 7}


def test_context_that_does_not_fit_the_model_is_reduced_and_recorded(session):
    library = store_library(session, "Long", text_size=60)
    service, providers = make_service(session, configs=[make_config(context_window=3_000, max_output_tokens=500)], context_passages=3, context_safety_margin=0)
    status = service.extract_section(library.nodes["s1"].id, ["concept"])
    assert status.status == "needs_review"
    execution = executions(session)[0]
    assert execution.context["reduced"] and execution.context["requested"] == 6 and execution.context["sent"] < 6
    assert execution.context_passages_dropped and any("context" in reason for reason in execution.review_reasons)
    assert execution.context["estimated_input_tokens"] <= execution.context["budget_tokens"] == 2_500
    sent = only_provider(providers).requests[0].messages[0].content
    assert str(library.passages["s1a"].id) in sent  # the primary passage is always there


def test_a_passage_too_large_for_the_model_fails_alone_and_is_never_truncated(session, library):
    library.passages["c1a"].text = "word " * 8_000
    session.commit()
    service, providers = make_service(session, configs=[make_config(context_window=4_000, max_output_tokens=500)])
    status = service.extract_chapter(library.nodes["ch1"].id, ["concept"])
    failed = [e for e in executions(session) if e.status == "failed"]
    assert [e.failure_code for e in failed] == ["context_window_exceeded"] and failed[0].passage_id == library.passages["c1a"].id
    assert "Nothing was truncated" in failed[0].failure_message and failed[0].attempt_count == 0
    assert status.status == "needs_review" and status.failed_passages == 1 and status.run.units_extracted == 4
    assert len(only_provider(providers).requests) == 4  # no call was made for the big passage


def test_the_book_the_chapter_and_the_section_reach_the_model(session, library):
    service, providers = make_service(session)
    service.extract_section(library.nodes["s1"].id, ["concept"], limit=1)
    content = only_provider(providers).requests[0].messages[0].content
    assert 'The passages come from the book "Thermodynamics" (A primer)' in content
    assert 'chapter="Entropy"' in content and 'section="The second law"' in content and 'pages="4"' in content


# ---------------------------------------------------------------- metadata


def test_every_execution_and_attempt_records_the_model_and_prompt_actually_used(session, library):
    config = make_config(model_version="2025-01", temperature=0.2, input_cost_per_million_tokens=2.0, output_cost_per_million_tokens=10.0)
    service, _ = make_service(session, configs=[config])
    status = service.extract_section(library.nodes["s1"].id, ["concept"])

    for execution in executions(session):
        assert (execution.model_configuration_id, execution.provider, execution.model_id) == ("local-test", "ollama", "test-model:1b")
        assert (execution.prompt_name, execution.prompt_version) == ("concept_extraction", LATEST)
        assert execution.model_configuration["model_version"] == "2025-01"
        assert execution.generation_parameters["temperature"] == 0.2
        assert execution.started_at <= execution.completed_at and execution.duration_ms >= 0
        assert (execution.input_tokens, execution.output_tokens, execution.attempt_count) == (100, 40, 1)
        assert execution.estimated_cost_usd == pytest.approx(100 * 2 / 1e6 + 40 * 10 / 1e6)
        attempt = execution.attempts[0]
        assert (attempt.prompt_version, attempt.model_id, attempt.kind, attempt.outcome) == (LATEST, "test-model:1b", "initial", "valid")
    assert status.run.estimated_cost_usd == pytest.approx(3 * (100 * 2 / 1e6 + 40 * 10 / 1e6))
    assert status.run.prompt_version == LATEST and status.run.model_name == "test-model:1b"


def test_history_keeps_the_configuration_that_was_used_when_the_configuration_changes(session, library):
    first, _ = make_service(session, configs=[make_config(temperature=0.1)])
    first.extract_chapter(library.nodes["ch2"].id, ["concept"])
    second, _ = make_service(session, configs=[make_config(model="other-model:7b", temperature=0.9)])
    second.extract_chapter(library.nodes["ch1"].id, ["concept"])
    rows = executions(session)
    assert [(r.model_id, r.generation_parameters["temperature"]) for r in rows][:2] == [("test-model:1b", 0.1)] * 2
    assert {(r.model_id, r.generation_parameters["temperature"]) for r in rows[2:]} == {("other-model:7b", 0.9)}
    runs = {r.model_snapshot["model"] for r in session.scalars(select(ExtractionRun))}
    assert runs == {"test-model:1b", "other-model:7b"}


def test_nothing_secret_is_stored(session, library):
    secret = "sai_do_not_store_this"
    service, _ = make_service(session, configs=[hosted_config()], default="hosted-test", environ={"TEST_LLM_KEY": secret})
    service.extract_section(library.nodes["ss1"].id, ["concept"])
    stored = json.dumps(
        [(e.model_configuration, e.generation_parameters, e.raw_response, e.failure_message, e.context) for e in executions(session)]
        + [(r.parameters, r.model_snapshot) for r in session.scalars(select(ExtractionRun))],
        default=str,
    )
    assert secret not in stored and "TEST_LLM_KEY" in stored


# ---------------------------------------------------------------- validation, retries and failures


def test_validation_failures_and_retries_are_counted_per_execution_and_per_run(session, library):
    def answer(call):
        return "not json" if call.number % 2 == 1 else one_item(call)

    service, _ = make_service(session, answer)
    status = service.extract_chapter(library.nodes["ch2"].id, ["concept"])
    assert (status.run.attempt_count, status.run.retry_count, status.run.validation_failure_count) == (4, 2, 2)
    execution = executions(session)[0]
    assert (execution.attempt_count, execution.retry_count, execution.validation_failures) == (2, 1, 1)
    first, second = execution.attempts
    assert (first.outcome, first.kind, first.raw_response) == ("invalid", "initial", "not json")
    assert first.issues[0]["code"] == "invalid_json" and (second.outcome, second.kind) == ("valid", "correction")
    assert status.status == "completed"


def test_invalid_data_that_cannot_be_corrected_fails_the_passage_with_its_evidence(session, library):
    def answer(call):
        return reply(item(call, content={"definition": None}))

    service, _ = make_service(session, answer, max_retries=1)
    status = service.extract_section(library.nodes["ss1"].id, ["concept"])
    assert status.status == "failed" and status.run.error_code == "all_extractions_failed" and status.failed_passages == 1
    execution = executions(session)[0]
    assert (execution.status, execution.needs_review, execution.attempt_count, execution.validation_failures) == ("failed", True, 2, 2)
    assert execution.failure_code == "validation_failed" and "definition" in execution.raw_response and execution.issues
    assert count(session, KnowledgeUnit) == 0


def test_a_failing_passage_does_not_stop_the_run(session, library):
    bad = str(library.passages["c1b"].id)

    def answer(call):
        return ProviderTimeoutError("slow") if call.target_id == bad else one_item(call)

    service, _ = make_service(session, answer, max_retries=0)
    status = service.extract_book(library.book.id, ["concept"])
    assert status.status == "needs_review" and status.failed_passages == 1
    assert (status.run.passages_processed, status.run.units_extracted, status.run.failure_count, status.run.needs_review_count) == (8, 7, 1, 1)
    failed = [e for e in executions(session) if e.status == "failed"]
    assert len(failed) == 1 and failed[0].failure_code == "provider_timeout" and str(failed[0].passage_id) == bad


def test_a_crash_inside_one_extraction_is_recorded_and_the_run_continues(session, library):
    bad = str(library.passages["c1a"].id)

    def answer(call):
        return RuntimeError("boom") if call.target_id == bad else one_item(call)

    service, _ = make_service(session, answer)
    status = service.extract_chapter(library.nodes["ch1"].id, ["concept"])
    failed = [e for e in executions(session) if e.status == "failed"]
    assert [e.failure_code for e in failed] == ["unexpected_error"] and "RuntimeError: boom" in failed[0].failure_message
    assert status.run.units_extracted == 4 and status.status == "needs_review"


def test_a_storage_failure_is_recorded_as_a_failed_execution_and_the_run_continues(session, library, monkeypatch):
    service, _ = make_service(session)
    original, calls = service.runs.store_execution, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return original(*args, **kwargs)

    monkeypatch.setattr(service.runs, "store_execution", flaky)
    status = service.extract_chapter(library.nodes["ch2"].id, ["concept"])
    rows = executions(session)
    assert [(r.status, r.failure_code) for r in rows] == [("failed", "storage_failed"), ("succeeded", None)]
    assert status.run.units_extracted == 1 and status.status == "needs_review"


def test_a_run_where_everything_fails_is_failed_and_says_why(session, library):
    service, _ = make_service(session, lambda call: "garbage", max_retries=0, max_consecutive_failures=0)
    status = service.extract_book(library.book.id, ["concept"], limit=3)
    assert status.status == "failed" and status.run.error_code == "all_extractions_failed" and "Every extraction failed" in status.error
    assert status.run.failure_count == 3 and status.failed_passages == 3 and status.run.completed_at


def test_a_run_stops_early_when_extractions_keep_failing_in_a_row(session, library):
    service, providers = make_service(session, lambda call: ProviderAuthError("bad key"), max_retries=0, max_consecutive_failures=2)
    status = service.extract_book(library.book.id, ["concept"])
    assert status.status == "failed" and status.run.error_code == "too_many_failures" and "2 consecutive failures" in status.error
    assert len(only_provider(providers).requests) == 2 and status.run.passages_processed == 2


def test_a_run_that_finds_things_to_review_ends_as_needs_review(session, library):
    service, _ = make_service(session, lambda call: reply(item(call, basis="inferred", is_uncertain=True, uncertainty_notes="Implied.")))
    status = service.extract_chapter(library.nodes["ch2"].id, ["concept"])
    assert status.status == "needs_review" and status.run.needs_review_count == 2 and status.run.failure_count == 0
    assert {u.validation_status for u in units(session)} == {"uncertain"} and all(u.is_uncertain for u in units(session))


def test_an_incomplete_equation_is_stored_for_review_never_as_valid(session, library):
    from tests.knowledge_helpers import equation_data

    def answer(call):
        content = equation_data(original_expression="dS = dQ /", latex=None, variables=[], parameters=[], assumptions=[], extraction_status="partial", reconstruction_status="not_attempted")
        return reply({"content": content, "sources": [{"passage_id": call.target_id}], "basis": "explicit"})

    service, _ = make_service(session, answer)
    status = service.extract_section(library.nodes["ss1"].id, ["equation"])
    unit = units(session)[0]
    assert status.status == "needs_review" and unit.knowledge_type == "equation" and unit.is_uncertain and unit.validation_status != "valid"


# ---------------------------------------------------------------- idempotency and duplicates


def test_repeating_a_run_does_not_repeat_finished_work(session, library):
    service, providers = make_service(session)
    service.extract_book(library.book.id, ["concept"], limit=3)
    calls = len(only_provider(providers).requests)
    with pytest.raises(NothingToExtractError, match="already been extracted"):
        service.extract_book(library.book.id, ["concept"], limit=3)
    assert len(only_provider(providers).requests) == calls and count(session, ExtractionRun) == 1

    more = service.extract_book(library.book.id, ["concept"], limit=5)
    assert more.run.skipped_count == 3 and more.run.units_extracted == 2 and more.run.passages_processed == 5
    assert len(only_provider(providers).requests) == calls + 2 and count(session, KnowledgeUnit) == 5

    other = service.extract_book(library.book.id, ["concept", "principle"], limit=2)  # a new category on finished passages
    assert other.run.units_extracted == 2 and other.run.skipped_count == 2


def test_forcing_a_rerun_does_not_create_duplicate_units(session, library):
    service, _ = make_service(session)
    service.extract_chapter(library.nodes["ch2"].id, ["concept"])
    forced = service.extract_chapter(library.nodes["ch2"].id, ["concept"], force=True)
    assert forced.run.units_extracted == 0 and forced.run.duplicate_count == 2 and forced.status == "completed"
    assert count(session, KnowledgeUnit) == 2 and count(session, ExtractionExecution) == 4
    second_run = [e for e in executions(session) if e.extraction_run_id == forced.run.id]
    assert all(e.items_extracted == 0 and "duplicate_skipped" in {i["code"] for i in e.issues} for e in second_run)


def test_the_same_knowledge_worded_slightly_differently_by_another_model_is_a_duplicate(session, library):
    first, _ = make_service(session)
    first.extract_section(library.nodes["ss1"].id, ["concept"])
    reworded = lambda call: reply(item(call, content={"definition": "A measure of the disorder of a physical system."}))  # noqa: E731
    second, _ = make_service(session, reworded, configs=[make_config(model="other-model")])
    status = second.extract_section(library.nodes["ss1"].id, ["concept"], force=True)
    assert status.run.duplicate_count == 1 and count(session, KnowledgeUnit) == 1


def test_different_knowledge_with_the_same_title_from_the_same_passage_is_kept(session, library):
    first, _ = make_service(session)
    first.extract_section(library.nodes["ss1"].id, ["concept"])
    other = lambda call: reply(item(call, content={"definition": "In information theory, the average surprise of a message source."}))  # noqa: E731
    second, _ = make_service(session, other)
    status = second.extract_section(library.nodes["ss1"].id, ["concept"], force=True)
    assert status.run.duplicate_count == 0 and count(session, KnowledgeUnit) == 2


def test_the_same_concept_in_different_passages_is_never_merged_but_is_noted(session, library):
    service, _ = make_service(session)
    service.extract_chapter(library.nodes["ch2"].id, ["concept"])
    first, second = units(session)
    assert first.title == second.title == "Entropy" and first.content_fingerprint != second.content_fingerprint
    assert "same_title_as" not in first.unit_metadata and second.unit_metadata["same_title_as"] == [str(first.id)]
    sources = {link.knowledge_unit_id: link.passage_id for link in session.scalars(select(KnowledgeUnitPassage))}
    assert sources[first.id] != sources[second.id]


def test_two_identical_items_in_one_response_are_stored_once(session, library):
    service, _ = make_service(session, lambda call: reply(item(call), item(call)))
    status = service.extract_section(library.nodes["ss1"].id, ["concept"])
    assert count(session, KnowledgeUnit) == 1 and status.run.duplicate_count == 1 and status.status == "needs_review"


def test_a_resumed_run_does_not_repeat_what_it_already_extracted(session, library):
    state = {"fail": True}

    def answer(call):
        return ProviderAuthError("bad key") if state["fail"] and call.number > 2 else one_item(call)

    service, providers = make_service(session, answer, max_retries=0, max_consecutive_failures=2)
    run_id = service.create_run(service.book_scope(library.book.id), ["concept"])
    stopped = service.execute_run(run_id)
    assert stopped.status == "failed" and stopped.run.units_extracted == 2 and stopped.run.error_code == "too_many_failures"
    calls_before = len(only_provider(providers).requests)

    state["fail"] = False
    finished = service.execute_run(run_id)
    assert finished.status == "completed"  # the status follows the passages now: the earlier failures were repaired
    assert finished.run.failure_count == 2  # the counters keep the history of every execution
    assert finished.run.units_extracted == ALL_PASSAGES and finished.failed_passages == 0
    assert len(only_provider(providers).requests) - calls_before == ALL_PASSAGES - 2
    assert count(session, KnowledgeUnit) == ALL_PASSAGES
    with pytest.raises(ExtractionRunError, match="already finished"):
        service.execute_run(run_id)


# ---------------------------------------------------------------- progress, status, cancellation


def test_progress_is_reported_after_every_extraction(session, library):
    events = []
    service, _ = make_service(session)
    run_id = service.create_run(service.chapter_scope(library.nodes["ch2"].id), ["concept", "principle"])
    service.execute_run(run_id, progress=events.append)
    assert [(e.passages_done, e.total_passages, e.category.value, e.status, e.items) for e in events] == [
        (1, 2, "concept", "succeeded", 1), (1, 2, "principle", "succeeded", 1),
        (2, 2, "concept", "succeeded", 1), (2, 2, "principle", "succeeded", 1),
    ]
    assert events[0].page_start == 7 and events[0].attempts == 1 and events[0].failure_code is None


def test_status_can_be_read_while_a_run_is_pending_and_after_it_finished(session, library):
    service, _ = make_service(session)
    run_id = service.create_run(service.book_scope(library.book.id), ["concept"])
    pending = service.get_run_status(run_id)
    assert pending.status == "pending" and pending.is_active and pending.progress == 0.0 and pending.run.total_passages == ALL_PASSAGES
    seen = []
    service.execute_run(run_id, progress=lambda e: seen.append(service.get_run_status(run_id).progress))
    assert seen == [i / ALL_PASSAGES for i in range(1, ALL_PASSAGES + 1)]
    assert not service.get_run_status(run_id).is_active
    with pytest.raises(ExtractionRunError, match="does not exist"):
        service.get_run_status(__import__("uuid").uuid4())


def test_a_run_can_be_cancelled_and_keeps_what_it_extracted(session, library):
    service, providers = make_service(session)
    run_id = service.create_run(service.book_scope(library.book.id), ["concept"])
    status = service.execute_run(run_id, progress=lambda event: service.cancel_run(run_id) if event.passages_done == 2 else None)
    assert status.status == "cancelled" and status.run.units_extracted == 2 and status.run.completed_at
    assert len(only_provider(providers).requests) == 2 and count(session, KnowledgeUnit) == 2
    assert service.cancel_run(run_id) is False  # already finished


def test_only_one_extraction_of_a_book_runs_at_a_time(session, library, monkeypatch):
    service, _ = make_service(session)
    first = service.create_run(service.book_scope(library.book.id), ["concept"], limit=2)
    with pytest.raises(ExtractionRunError, match="already running"):
        service.create_run(service.book_scope(library.book.id), ["concept"], limit=2, force=True)
    service.cancel_run(first)
    assert service.create_run(service.book_scope(library.book.id), ["concept"], limit=2, force=True)
    monkeypatch.setattr(pipeline, "ACTIVE_RUN_TIMEOUT", timedelta(seconds=-1))  # a run abandoned long ago does not block
    assert service.create_run(service.book_scope(library.book.id), ["principle"], limit=1)


def test_runs_and_knowledge_can_be_listed_for_a_book(session, library):
    service, _ = make_service(session)
    first = service.extract_chapter(library.nodes["ch2"].id, ["concept"])
    second = service.extract_chapter(library.nodes["ch1"].id, ["principle"])
    assert [s.run.id for s in service.list_runs(library.book.id)] == [second.run.id, first.run.id]
    assert service.list_runs(store_library(session, "Other").book.id) == []

    everything = service.get_book_knowledge(library.book.id)
    assert len(everything) == 7 and [u.page_start for u in everything] == sorted(u.page_start for u in everything)
    assert len(service.get_book_knowledge(library.book.id, knowledge_type="concept")) == 2
    assert len(service.get_book_knowledge(library.book.id, run_id=second.run.id)) == 5
    assert len(service.get_book_knowledge(library.book.id, review_status="approved")) == 0
    assert len(service.get_book_knowledge(library.book.id, validation_status="valid", limit=3, offset=1)) == 3
    unit = everything[0]
    assert unit.sources and unit.extraction_run_id in (first.run.id, second.run.id)


def test_a_dry_run_calls_the_model_and_stores_nothing(session, library):
    service, providers = make_service(session)
    results = service.dry_run(service.book_scope(library.book.id), ["concept"], limit=2)
    assert [r.status for _, r in results] == [ExtractionStatus.SUCCEEDED] * 2 and len(only_provider(providers).requests) == 2
    assert count(session, ExtractionRun) == count(session, ExtractionExecution) == count(session, KnowledgeUnit) == 0


def test_the_plan_says_what_a_run_would_do(session, library):
    service, _ = make_service(session)
    service.extract_book(library.book.id, ["concept"], limit=3)
    plan = service.plan(service.book_scope(library.book.id), ["concept", "principle"])
    assert (plan.total_passages, plan.model_calls, plan.skipped, plan.scope_kind) == (8, 13, 3, "book")
    assert plan.model_label == "local-test (test-model:1b)" and plan.model_configurations == {"concept": "local-test", "principle": "local-test"}
    assert service.plan(service.book_scope(library.book.id), ["concept"], force=True).model_calls == 8


def test_phase_1_data_is_not_modified(session, library):
    snapshot = lambda: [(p.id, p.text, p.page_start, p.document_sequence, p.structural_node_id) for p in session.scalars(select(Passage).order_by(Passage.document_sequence))]  # noqa: E731
    before = snapshot()
    service, _ = make_service(session, lambda call: reply(item(call), item(call, content={"name": "Heat", "definition": "Energy."})))
    service.extract_book(library.book.id, ["concept"])
    session.expire_all()
    assert snapshot() == before and count(session, Passage) == ALL_PASSAGES


def test_deleting_a_run_removes_its_history_but_keeps_the_knowledge(session, library):
    service, _ = make_service(session)
    status = service.extract_chapter(library.nodes["ch2"].id, ["concept"])
    session.delete(session.get(ExtractionRun, status.run.id))
    session.commit()
    session.expire_all()
    assert (count(session, ExtractionRun), count(session, ExtractionExecution), count(session, ExtractionAttempt), count(session, KnowledgeUnit)) == (0, 0, 0, 2)
    assert all(u.extraction_run_id is None and u.extraction_execution_id is None for u in units(session))


def test_regenerating_passages_keeps_the_executions(session, library):
    service, _ = make_service(session)
    service.extract_section(library.nodes["ss1"].id, ["concept"])
    for passage in session.scalars(select(Passage)):
        session.delete(passage)
    session.commit()
    session.expire_all()
    execution = executions(session)[0]
    assert execution.passage_id is None and execution.status == "succeeded" and count(session, KnowledgeUnit) == 1


def test_the_complete_workflow(session, library):
    """Select a model, extract a chapter with several categories, follow the progress, read the knowledge back."""
    local, hosted = make_config(), hosted_config()
    service, providers = make_service(session, configs=[local, hosted], default="local-test")

    options = [o for o in service.available_models() if o.ready]
    assert [o.id for o in options] == ["local-test", "hosted-test"]
    scope = service.chapter_scope(library.nodes["ch1"].id)
    plan = service.plan(scope, ["concept", "principle"], model_id="hosted-test")
    assert (plan.total_passages, plan.model_calls, plan.model_label) == (5, 10, "hosted-test (gpt-test)")

    run_id = service.create_run(scope, ["concept", "principle"], model_id="hosted-test")
    progress = []
    status = service.execute_run(run_id, progress=progress.append)

    assert status.status == "completed" and len(progress) == 10 and progress[-1].passages_done == 5
    assert (status.run.units_extracted, status.run.passages_processed, status.run.model_configuration_id) == (10, 5, "hosted-test")
    knowledge = service.get_book_knowledge(library.book.id, run_id=run_id)
    assert len(knowledge) == 10 and {k.knowledge_type.value for k in knowledge} == {"concept", "principle"}
    assert all(k.sources and k.page_start for k in knowledge)
    assert len(providers["hosted-test"].requests) == 10 and providers["local-test"].requests == []
