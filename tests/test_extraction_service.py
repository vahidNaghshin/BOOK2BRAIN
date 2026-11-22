"""Extraction service with scripted (mocked) providers: retries, limits, metadata, provider independence."""

import json
import uuid

import pytest

from app.extraction.errors import ProviderAuthError, ProviderRateLimitError, ProviderTimeoutError
from app.extraction.model_config import ModelRegistry
from app.extraction.providers import LlmResponse
from app.extraction.results import AttemptOutcome, ExtractionStatus, PassageInput
from app.extraction.service import ExtractionService
from app.extraction.settings import ExtractionSettings
from app.knowledge.schemas import ExtractionRunCreate, KnowledgeUnitCreate
from app.knowledge.types import KnowledgeType, ValidationStatus
from tests.extraction_helpers import (
    LATEST,
    ScriptedProvider,
    hosted_config,
    item,
    make_config,
    make_harness,
    make_request,
    response_json,
)
from tests.knowledge_helpers import BUILDERS, equation_data


def partial_equation_request():
    request = make_request("equation", text="Entropy change is dS = dQ / (the denominator is unreadable).")
    content = equation_data(
        original_expression="dS = dQ /", latex=None, variables=[], parameters=[], assumptions=[],
        extraction_status="partial", reconstruction_status="not_attempted",
    )
    return request, response_json(item(request, content=content))


def test_successful_extraction_returns_validated_knowledge():
    request = make_request()
    h = make_harness([response_json(item(request))])
    result = h.service.extract(request)
    assert result.status == ExtractionStatus.SUCCEEDED and result.is_valid and not result.needs_review
    assert result.knowledge_type == KnowledgeType.CONCEPT
    assert result.items[0].content.name == "Entropy"
    assert result.source_references == [request.passage.id]
    assert result.confidence == 0.9
    assert (result.items[0].page_start, result.items[0].page_end) == (12, 12)
    assert result.items[0].validation_status == ValidationStatus.VALID
    assert result.execution.attempt_count == 1 and result.execution.retry_count == 0
    assert h.sleeps == []


@pytest.mark.parametrize("knowledge_type", list(KnowledgeType))
def test_every_category_extracts_through_the_service(knowledge_type):
    request = make_request(knowledge_type)
    h = make_harness([response_json(item(request))])
    result = h.service.extract(request)
    assert result.status in (ExtractionStatus.SUCCEEDED, ExtractionStatus.NEEDS_REVIEW), result.issues
    assert result.items and result.execution.prompt_name.startswith(knowledge_type.value)
    assert h.provider.requests[0].json_schema is not None


def test_nothing_found_is_a_success_with_no_items():
    h = make_harness(['{"items": [], "notes": "Only a table of contents."}'])
    result = h.service.extract(make_request())
    assert result.status == ExtractionStatus.SUCCEEDED and result.items == [] and result.notes


def test_invalid_json_triggers_a_correction_attempt_with_errors_and_previous_response():
    request = make_request()
    h = make_harness(["this is { not json", response_json(item(request))])
    result = h.service.extract(request)
    assert result.status == ExtractionStatus.SUCCEEDED
    first, second = result.execution.attempts
    assert first.outcome == AttemptOutcome.INVALID and first.raw_response == "this is { not json"
    assert second.kind == "correction" and second.outcome == AttemptOutcome.VALID
    joined = "\n".join(m.content for m in h.provider.requests[1].messages)
    assert "not valid JSON" in joined and "this is { not json" in joined
    assert result.execution.validation_failures == 1 and result.execution.retry_count == 1


def test_missing_required_fields_are_retried_then_fixed():
    request = make_request()
    content = BUILDERS[KnowledgeType.CONCEPT]()
    del content["definition"]
    h = make_harness([response_json(item(request, content=content)), response_json(item(request))])
    assert h.service.extract(request).status == ExtractionStatus.SUCCEEDED
    assert "definition" in "\n".join(m.content for m in h.provider.requests[1].messages)


def test_invented_passage_id_is_corrected_not_accepted():
    request = make_request()
    bad = item(request, sources=[{"passage_id": str(uuid.uuid4()), "role": "primary"}])
    h = make_harness([response_json(bad), response_json(item(request))])
    result = h.service.extract(request)
    assert result.status == ExtractionStatus.SUCCEEDED
    assert result.execution.attempts[0].issues[0].code == "unknown_passage_id"


def test_maximum_retry_limit_is_enforced_and_failure_flagged_for_review():
    h = make_harness(["garbage"], max_retries=2)
    result = h.service.extract(make_request())
    assert result.status == ExtractionStatus.FAILED and result.needs_review and result.items == []
    assert len(h.provider.requests) == 3  # one attempt plus two retries, never more
    assert result.execution.attempt_count == 3 and result.execution.retry_count == 2
    assert result.execution.validation_failures == 3
    assert result.raw_response == "garbage" and result.issues and result.failure


def test_zero_retries_means_a_single_attempt():
    h = make_harness(["garbage"], max_retries=0)
    assert h.service.extract(make_request()).execution.attempt_count == 1
    assert len(h.provider.requests) == 1


def test_per_model_retry_limit_overrides_the_default():
    h = make_harness(["garbage"], config=make_config(max_retries=1), max_retries=5)
    assert h.service.extract(make_request()).execution.attempt_count == 2


def test_timeout_is_retried_with_backoff_then_succeeds():
    request = make_request()
    h = make_harness([ProviderTimeoutError("slow"), ProviderTimeoutError("slow"), response_json(item(request))], max_retries=3)
    result = h.service.extract(request)
    assert result.status == ExtractionStatus.SUCCEEDED
    assert [a.outcome for a in result.execution.attempts] == [
        AttemptOutcome.PROVIDER_ERROR, AttemptOutcome.PROVIDER_ERROR, AttemptOutcome.VALID,
    ]
    assert h.sleeps == [1.0, 2.0] and result.execution.attempts[0].provider_error_code
    assert result.execution.validation_failures == 0


def test_rate_limit_honours_retry_after():
    request = make_request()
    h = make_harness([ProviderRateLimitError("slow down", retry_after=7), response_json(item(request))])
    assert h.service.extract(request).status == ExtractionStatus.SUCCEEDED
    assert h.sleeps == [7.0]


def test_backoff_is_capped():
    h = make_harness([ProviderRateLimitError("slow down", retry_after=9999), "{}"], max_retries=1, max_backoff_seconds=5)
    h.service.extract(make_request())
    assert h.sleeps == [5.0]


def test_persistent_provider_failure_stops_at_the_limit():
    h = make_harness([ProviderTimeoutError("slow")], max_retries=2)
    result = h.service.extract(make_request())
    assert result.status == ExtractionStatus.FAILED and result.needs_review
    assert len(h.provider.requests) == 3 and result.failure.code
    assert result.raw_response is None


def test_authentication_errors_are_not_retried():
    h = make_harness([ProviderAuthError("bad key")], max_retries=5)
    result = h.service.extract(make_request())
    assert result.status == ExtractionStatus.FAILED and len(h.provider.requests) == 1


def test_truncated_output_is_recorded_and_retried():
    request = make_request()
    cut = LlmResponse(text='{"items": [{"content": ', model="m", finish_reason="length")
    h = make_harness([cut, response_json(item(request))])
    result = h.service.extract(request)
    assert result.status == ExtractionStatus.SUCCEEDED
    assert result.execution.attempts[0].finish_reason == "length"


def test_incomplete_equation_is_kept_and_flagged_for_review():
    request, payload = partial_equation_request()
    h = make_harness([payload])
    result = h.service.extract(request)
    assert result.status == ExtractionStatus.NEEDS_REVIEW and result.needs_review
    assert len(result.items) == 1 and result.items[0].uncertainty.is_uncertain
    assert result.items[0].validation_status != ValidationStatus.VALID
    assert len(h.provider.requests) == 1  # not a structural problem, so no retry


def test_ambiguous_result_needs_review_without_retry():
    request = make_request()
    h = make_harness([response_json(item(request, is_uncertain=True, uncertainty_notes="Implied.", basis="inferred"))])
    result = h.service.extract(request)
    assert result.status == ExtractionStatus.NEEDS_REVIEW and len(h.provider.requests) == 1
    assert result.review_reasons


def test_valid_schema_with_unsupported_quote_still_needs_review():
    request = make_request()
    entry = item(request, sources=[{"passage_id": str(request.passage.id), "quote": "Words that are not in the passage."}])
    result = make_harness([response_json(entry)]).service.extract(request)
    assert result.status == ExtractionStatus.NEEDS_REVIEW and result.quality.unsupported_content
    assert len(result.items) == 1


def test_duplicates_are_reported_in_quality():
    request = make_request()
    result = make_harness([response_json(item(request), item(request))]).service.extract(request)
    assert result.quality.duplicate_items == [(0, 1)]


def test_invalid_model_selection_fails_cleanly_without_calling_a_provider():
    service = ExtractionService(ModelRegistry([make_config()]), ExtractionSettings())
    result = service.extract(make_request())
    assert result.status == ExtractionStatus.FAILED and result.failure.code == "model_configuration_invalid"
    assert result.needs_review and result.execution.attempt_count == 0


def test_missing_api_key_fails_before_any_request():
    config = hosted_config()
    service = ExtractionService(ModelRegistry([config], default_id=config.id), environ={})
    result = service.extract(make_request())
    assert result.failure.code == "model_configuration_invalid" and "TEST_LLM_KEY" in result.failure.message
    with pytest.raises(Exception, match="TEST_LLM_KEY"):
        service.check_ready("concept")


def test_inactive_model_cannot_be_used():
    h = make_harness(["{}"], config=make_config(active=False))
    assert h.service.extract(make_request()).failure.code == "model_configuration_invalid"
    assert h.provider.requests == []


def test_context_is_trimmed_to_fit_a_small_context_window():
    context = [PassageInput(id=uuid.uuid4(), text="Filler sentence. " * 60) for _ in range(6)]
    request = make_request(context_before=context[:3], context_after=context[3:])
    config = make_config(context_window=2400, max_output_tokens=800)
    h = make_harness([response_json(item(request))], config=config)
    result = h.service.extract(request)
    assert result.status in (ExtractionStatus.SUCCEEDED, ExtractionStatus.NEEDS_REVIEW)
    dropped = result.execution.context_passages_dropped
    assert dropped and set(dropped) <= {p.id for p in context}
    assert str(request.passage.id) in h.provider.requests[0].messages[0].content
    assert result.review_reasons


def test_large_context_window_keeps_all_context():
    context = [PassageInput(id=uuid.uuid4(), text="Filler sentence. " * 60) for _ in range(4)]
    request = make_request(context_before=context[:2], context_after=context[2:])
    h = make_harness([response_json(item(request))], config=make_config(context_window=128_000))
    assert h.service.extract(request).execution.context_passages_dropped == []


def test_passage_that_cannot_fit_fails_without_calling_the_model():
    request = make_request(text="word " * 5000)
    h = make_harness(["{}"], config=make_config(context_window=2000, max_output_tokens=500))
    result = h.service.extract(request)
    assert result.failure.code == "context_window_exceeded" and h.provider.requests == []
    assert result.needs_review


def test_unknown_context_window_sends_everything():
    request = make_request(context_before=[PassageInput(id=uuid.uuid4(), text="x " * 1000)])
    h = make_harness([response_json(item(request))], config=make_config(context_window=None))
    assert h.service.extract(request).execution.context_passages_dropped == []


def test_pinned_prompt_version_missing_gives_a_clear_failure():
    h = make_harness(["{}"], prompt_versions={"concept": "42.0.0"})
    assert h.service.extract(make_request()).failure.code == "prompt_not_found"


def test_execution_metadata_records_model_prompt_parameters_and_usage():
    request = make_request()
    config = make_config(
        model_version="2025-01", temperature=0.3, top_p=0.8,
        input_cost_per_million_tokens=2.0, output_cost_per_million_tokens=10.0,
    )
    execution = make_harness([response_json(item(request))], config=config).service.extract(request).execution
    assert execution.model_configuration_id == "local-test" and execution.provider == "ollama"
    assert execution.model_id == "test-model:1b"
    assert execution.prompt_name == "concept_extraction" and execution.prompt_version == LATEST
    assert execution.prompt_identifier == "concept_extraction@" + LATEST + ""
    assert execution.generation_parameters["temperature"] == 0.3
    assert execution.generation_parameters["max_output_tokens"] == 1024
    assert execution.started_at <= execution.completed_at and execution.duration_ms >= 0
    assert (execution.input_tokens, execution.output_tokens) == (100, 50)
    assert execution.estimated_cost_usd == pytest.approx(100 * 2 / 1e6 + 50 * 10 / 1e6)
    attempt = execution.attempts[0]
    assert (attempt.prompt_version, attempt.model_id, attempt.generation_parameters) == (
        LATEST, "test-model:1b", execution.generation_parameters,
    )


def test_cost_is_unknown_without_prices():
    request = make_request()
    execution = make_harness([response_json(item(request))]).service.extract(request).execution
    assert execution.estimated_cost_usd is None


def test_tokens_and_cost_are_summed_across_attempts():
    request = make_request()
    config = make_config(input_cost_per_million_tokens=1.0, output_cost_per_million_tokens=1.0)
    execution = make_harness(["bad", response_json(item(request))], config=config).service.extract(request).execution
    assert execution.input_tokens == 200 and execution.output_tokens == 100
    assert execution.estimated_cost_usd == pytest.approx(300 / 1e6)
    assert sum(a.estimated_cost_usd for a in execution.attempts) == pytest.approx(execution.estimated_cost_usd)


def test_reported_model_identifier_wins_over_the_configured_alias():
    request = make_request()
    reply = LlmResponse(text=response_json(item(request)), model="test-model-2025-03-01", model_version="rev-9")
    execution = make_harness([reply]).service.extract(request).execution
    assert execution.model_id == "test-model-2025-03-01" and execution.model_version == "rev-9"
    assert execution.model_configuration["model"] == "test-model:1b"


def test_completed_records_keep_their_original_metadata_after_a_config_change():
    request = make_request()
    config = make_config(temperature=0.1)
    before = make_harness([response_json(item(request))], config=config).service.extract(request).execution
    snapshot = before.model_copy(deep=True)
    changed = config.model_copy(update={"temperature": 0.9, "model": "other-model"})
    other = ExtractionService(
        ModelRegistry([changed], default_id=changed.id),
        provider_factory=lambda c: ScriptedProvider(c, [response_json(item(request))]),
    )
    after = other.extract(request).execution
    assert before == snapshot
    assert before.generation_parameters["temperature"] == 0.1 and before.model_configuration["model"] == "test-model:1b"
    assert after.generation_parameters["temperature"] == 0.9 and after.model_id == "other-model"


def test_metadata_names_the_key_variable_but_never_a_value():
    request = make_request()
    secret = "super-secret-value"
    config = hosted_config()
    provider = ScriptedProvider(config, [response_json(item(request))])
    service = ExtractionService(
        ModelRegistry([config], default_id=config.id), provider_factory=lambda c: provider, environ={"TEST_LLM_KEY": secret}
    )
    result = service.extract(request)
    assert secret not in json.dumps(result.model_dump(mode="json"))
    assert result.execution.model_configuration["api_key_env"] == "TEST_LLM_KEY"


def test_run_fields_map_onto_the_day1_extraction_run_schema():
    request = make_request()
    execution = make_harness([response_json(item(request))]).service.extract(request).execution
    fields = execution.run_fields()
    assert fields["prompt_version"] == "concept_extraction@" + LATEST + ""
    ExtractionRunCreate(book_id=uuid.uuid4(), **fields)


def test_result_converts_to_a_day1_knowledge_unit():
    request = make_request()
    result = make_harness([response_json(item(request))]).service.extract(request)
    unit = result.items[0].to_unit_create(uuid.uuid4(), uuid.uuid4())
    assert isinstance(unit, KnowledgeUnitCreate) and unit.sources[0].passage_id == request.passage.id


def test_same_extraction_logic_works_with_any_provider_implementation():
    request = make_request()
    payload = response_json(item(request))
    configs = (make_config(), hosted_config(), make_config(id="compat", provider="openai_compatible", base_url="http://x/v1"))
    results = [make_harness([payload], config=c).service.extract(request) for c in configs]
    assert {r.status for r in results} == {ExtractionStatus.SUCCEEDED}
    assert {r.items[0].content.name for r in results} == {"Entropy"}
    assert [r.execution.provider for r in results] == ["ollama", "openai", "openai_compatible"]


def test_switching_the_selected_model_changes_the_recorded_execution():
    request = make_request()
    local, hosted = make_config(), hosted_config()
    providers = {c.id: ScriptedProvider(c, [response_json(item(request))]) for c in (local, hosted)}
    for default in ("local-test", "hosted-test"):
        service = ExtractionService(
            ModelRegistry([local, hosted], default_id=default),
            provider_factory=lambda c: providers[c.id], environ={"TEST_LLM_KEY": "k"},
        )
        assert service.extract(request).execution.model_configuration_id == default


def test_different_models_per_category():
    request_c, request_e = make_request("concept"), make_request("equation")
    local, hosted = make_config(), hosted_config()
    providers = {
        "local-test": ScriptedProvider(local, [response_json(item(request_c))]),
        "hosted-test": ScriptedProvider(hosted, [response_json(item(request_e))]),
    }
    service = ExtractionService(
        ModelRegistry([local, hosted], default_id="local-test", overrides={"equation": "hosted-test"}),
        provider_factory=lambda c: providers[c.id], environ={"TEST_LLM_KEY": "k"},
    )
    assert service.extract(request_c).execution.model_configuration_id == "local-test"
    assert service.extract(request_e).execution.model_configuration_id == "hosted-test"


@pytest.mark.parametrize("mode", ["json_schema", "json_mode", "none"])
def test_parsing_does_not_depend_on_the_structured_output_mode(mode):
    request = make_request()
    raw = "```json\n" + response_json(item(request)) + "\n```"
    h = make_harness([raw], config=make_config(structured_output=mode))
    assert h.service.extract(request).status == ExtractionStatus.SUCCEEDED
