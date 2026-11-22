"""Prompt templates, structural parsing, source and semantic validation. Pure functions; no LLM, no database."""

import json

import pytest

from app.extraction.parsing import parse_response
from app.extraction.prompts import PROMPTS, get_prompt, render_correction
from app.extraction.results import PassageInput, Severity, ValidationStage
from app.extraction.validation import (
    SourceTextSupportValidator,
    assess_quality,
    find_duplicates,
    run_semantic_validation,
    settle_status,
    validate_sources,
)
from app.knowledge.content import CONTENT_MODELS
from app.knowledge.types import KnowledgeType, ValidationStatus
from tests.extraction_helpers import LATEST, item, make_request, response_json
from tests.knowledge_helpers import BUILDERS, equation_data

ALL = list(KnowledgeType)


def codes(issues):
    return {i.code for i in issues}


# ---------- prompts ----------

def test_all_seven_categories_have_a_versioned_prompt():
    assert len(ALL) == 7
    for knowledge_type in ALL:
        prompt = get_prompt(knowledge_type)
        assert prompt.knowledge_type == knowledge_type
        assert prompt.version and prompt.identifier == f"{prompt.name}@{prompt.version}"
        assert knowledge_type.value in prompt.name


@pytest.mark.parametrize("knowledge_type", ALL)
def test_prompt_states_required_instructions(knowledge_type):
    text = get_prompt(knowledge_type).render_system().lower()
    for phrase in ["only", "invent", "explicit", "infer", "terminology", "passage_id", "page", "json", "uncertain"]:
        assert phrase in text, phrase
    assert "missing information" in text


@pytest.mark.parametrize("knowledge_type", ALL)
def test_prompt_skeleton_matches_the_day1_schema(knowledge_type):
    prompt = get_prompt(knowledge_type)
    fields = set(CONTENT_MODELS[knowledge_type].model_fields)
    assert set(prompt.skeleton) <= fields
    required = {n for n, f in CONTENT_MODELS[knowledge_type].model_fields.items() if f.is_required()}
    assert required <= set(prompt.skeleton) | {"knowledge_type"}
    schema = prompt.output_schema()
    assert schema["properties"]["items"]["items"]["properties"]["content"]


def test_equation_prompt_forbids_reconstruction_and_keeps_incomplete_equations():
    text = get_prompt("equation").render_system().lower()
    assert "reconstruct" in text and "not_attempted" in text and "partial" in text


def test_user_prompt_marks_target_and_context_with_real_ids():
    request = make_request(context_before=[PassageInput(id=make_request().passage.id, text="Earlier text.")])
    rendered = get_prompt("concept").render_user(request)
    assert str(request.passage.id) in rendered and 'role="target"' in rendered and 'role="context"' in rendered
    assert rendered.index("Earlier text.") < rendered.index(request.passage.text)


def test_prompt_versions_can_be_pinned_and_unknown_versions_fail():
    version = get_prompt("concept").version
    assert get_prompt("concept", version).version == version
    with pytest.raises(Exception, match="version"):
        get_prompt("concept", "99.0.0")
    assert all(PROMPTS[k] for k in ALL)


def test_correction_prompt_lists_the_problems():
    request = make_request()
    parsed = parse_response("{}", KnowledgeType.CONCEPT)
    assert "items" in render_correction(parsed.issues)


# ---------- structural parsing ----------

@pytest.mark.parametrize("knowledge_type", ALL)
def test_valid_response_parses_for_every_category(knowledge_type):
    request = make_request(knowledge_type)
    parsed = parse_response(response_json(item(request)), knowledge_type)
    assert not parsed.issues and len(parsed.items) == 1
    assert parsed.items[0].knowledge_type == knowledge_type


def test_markdown_fences_and_chatter_are_tolerated():
    request = make_request()
    raw = "Sure, here it is:\n```json\n" + response_json(item(request)) + "\n```\nHope that helps."
    assert len(parse_response(raw, KnowledgeType.CONCEPT).items) == 1


@pytest.mark.parametrize(
    "raw, code",
    [("", "empty_response"), ("not json at all", "invalid_json"), ('{"items": [', "invalid_json"),
     ("[1, 2]", "not_an_object"), ("{}", "missing_required_field")],
)
def test_malformed_responses_give_clear_errors(raw, code):
    parsed = parse_response(raw, KnowledgeType.CONCEPT)
    assert code in codes(parsed.issues) and parsed.items == []
    assert all(i.stage == ValidationStage.STRUCTURAL for i in parsed.issues)


def test_missing_required_field_is_reported_with_its_path():
    request = make_request()
    content = BUILDERS[KnowledgeType.CONCEPT]()
    del content["definition"]
    parsed = parse_response(response_json(item(request, content=content)), KnowledgeType.CONCEPT)
    issue = next(i for i in parsed.issues if i.code == "missing_required_field")
    assert "definition" in (issue.path or "") and parsed.items == []


def test_invalid_pydantic_data_and_unknown_fields_are_not_discarded_silently():
    request = make_request()
    content = BUILDERS[KnowledgeType.CONCEPT]() | {"definition": "   ", "invented": 1}
    parsed = parse_response(response_json(item(request, content=content)), KnowledgeType.CONCEPT)
    assert parsed.items == [] and {"invalid_value", "unexpected_field"} <= codes(parsed.issues)


def test_one_invalid_item_rejects_the_whole_response():
    request = make_request()
    bad = item(request, content={"name": "x"})
    parsed = parse_response(response_json(item(request), bad), KnowledgeType.CONCEPT)
    assert parsed.has_errors and parsed.items == []


def test_empty_item_list_means_nothing_found():
    parsed = parse_response('{"items": [], "notes": "No concept here."}', KnowledgeType.CONCEPT)
    assert parsed.items == [] and not parsed.issues and parsed.notes == "No concept here."


def test_wrong_knowledge_type_in_content_is_rejected():
    request = make_request("concept")
    content = BUILDERS[KnowledgeType.PRINCIPLE]()
    parsed = parse_response(response_json(item(request, content=content)), KnowledgeType.CONCEPT)
    assert parsed.items == [] and parsed.issues


def test_incomplete_equation_is_retained_and_marked_uncertain():
    request = make_request("equation", text="Entropy change is dS = dQ / ... (the rest of the line is cut off).")
    content = equation_data(
        original_expression="dS = dQ /", latex=None, variables=[], parameters=[], assumptions=[],
        extraction_status="partial", reconstruction_status="not_attempted",
    )
    parsed = parse_response(response_json(item(request, content=content)), KnowledgeType.EQUATION)
    assert not parsed.has_errors and len(parsed.items) == 1
    assert parsed.items[0].uncertainty.is_uncertain
    assert settle_status(parsed.items[0]).validation_status != ValidationStatus.VALID


def test_ambiguous_result_keeps_the_models_uncertainty():
    request = make_request()
    entry = item(request, is_uncertain=True, uncertainty_notes="The definition is implied.", basis="inferred", confidence=0.4)
    parsed = parse_response(response_json(entry), KnowledgeType.CONCEPT)
    unit = settle_status(parsed.items[0])
    assert unit.uncertainty.is_uncertain and unit.uncertainty.notes == "The definition is implied."
    assert unit.validation_status != ValidationStatus.VALID and unit.basis == "inferred"


def test_uncertainty_notes_without_flag_are_normalised():
    request = make_request()
    parsed = parse_response(response_json(item(request, uncertainty_notes="hmm")), KnowledgeType.CONCEPT)
    assert not parsed.has_errors


# ---------- source validation ----------

def validated(request, **fields):
    parsed = parse_response(response_json(item(request, **fields)), KnowledgeType.CONCEPT)
    assert parsed.items, parsed.issues
    return validate_sources(parsed.items, request)


def test_invented_passage_id_is_a_source_error():
    request = make_request()
    entry_sources = [{"passage_id": "6f0b6a0e-1c1c-4f6e-9d0e-000000000000", "role": "primary"}]
    _, issues = validated(request, sources=entry_sources)
    assert "unknown_passage_id" in codes(issues)
    assert all(i.stage == ValidationStage.SOURCE for i in issues)


def test_pages_come_from_the_cited_passage_when_the_model_gives_none():
    request = make_request(pages=(12, 13))
    items, issues = validated(request)
    assert (items[0].page_start, items[0].page_end) == (12, 13) and not items[0].pages_from_model
    assert not [i for i in issues if i.severity == Severity.ERROR]


def test_page_outside_the_cited_passages_is_rejected_and_never_invented():
    request = make_request(pages=(12, 12))
    _, issues = validated(request, page_start=99, page_end=99)
    assert "page_outside_passages" in codes(issues)
    items, issues = validated(make_request(pages=None))
    assert items[0].page_start is None
    _, issues = validated(make_request(pages=None), page_start=3)
    assert "page_not_available" in codes(issues)


def test_inverted_page_range_is_rejected():
    _, issues = validated(make_request(pages=(1, 20)), page_start=9, page_end=4)
    assert "invalid_page_range" in codes(issues)


# ---------- semantic and quality ----------

def test_unsupported_quote_is_flagged_not_rejected():
    request = make_request()
    entry = item(request, sources=[{"passage_id": str(request.passage.id), "quote": "This sentence is not in the text."}])
    parsed = parse_response(response_json(entry), KnowledgeType.CONCEPT)
    _, issues = run_semantic_validation(parsed.items, request, [SourceTextSupportValidator()])
    warning = next(i for i in issues if i.code == "quote_not_in_source")
    assert warning.severity == Severity.WARNING and warning.stage == ValidationStage.SEMANTIC


def test_supported_quote_passes_even_with_different_whitespace():
    request = make_request()
    quote = "Entropy is a measure  of the\ndisorder of a system."
    entry = item(request, sources=[{"passage_id": str(request.passage.id), "quote": quote}])
    parsed = parse_response(response_json(entry), KnowledgeType.CONCEPT)
    _, issues = run_semantic_validation(parsed.items, request, [SourceTextSupportValidator()])
    assert not [i for i in issues if i.severity != Severity.INFO]


def test_quality_reports_indicators_without_a_reliability_score():
    request = make_request()
    parsed = parse_response(response_json(item(request), item(request)), KnowledgeType.CONCEPT)
    assert find_duplicates(parsed.items) == [(0, 1)]
    quality = assess_quality(parsed.items, [])
    assert quality.schema_valid and quality.duplicate_items == [(0, 1)] and quality.semantic_accuracy is None
    assert 0 <= quality.completeness <= 1
    assert not any("reliab" in name for name in type(quality).model_fields)


def test_latest_prompt_is_target_only_and_the_released_version_is_unchanged():
    latest, first = get_prompt("concept"), get_prompt("concept", "1.0.0")
    assert latest.version == LATEST and first.version == "1.0.0"
    assert "ONLY from the passage with role=\"target\"" in latest.render_system()
    assert "13." not in first.render_system()  # the released text is not edited
    request = make_request()
    assert "only" in latest.render_user(request) and "only if the" in first.render_user(request)


def test_a_category_can_be_pinned_to_the_first_version_through_settings():
    from tests.extraction_helpers import make_harness, make_request, response_json

    request = make_request()
    h = make_harness([response_json(item(request))], prompt_versions={"concept": "1.0.0"})
    assert h.service.extract(request).execution.prompt_version == "1.0.0"


def test_source_header_is_only_in_the_newest_version_and_never_in_released_ones():
    from app.extraction.results import SourceInfo

    request = make_request(source=SourceInfo(title="Thermo", author="A. Author", publisher="Press", year=2001))
    newest = get_prompt("concept").render_user(request)
    assert 'The passages come from the book "Thermo" by A. Author [Press, 2001]' in newest
    for old in ("1.0.0", "1.1.0"):
        assert "come from the book" not in get_prompt("concept", old).render_user(request)
    assert "come from the book" not in get_prompt("concept").render_user(make_request())  # no source given
