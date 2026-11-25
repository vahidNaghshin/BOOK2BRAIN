"""The evaluation metrics: pure functions over a reference dataset and an observed extraction (no database, no model)."""

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import EvaluationDataset, available_datasets, load_dataset
from app.evaluation.metrics import evaluate, math_form, name_score, normalize, token_coverage
from app.evaluation.observation import (
    Observation,
    ObservedAttempt,
    ObservedEquation,
    ObservedExecution,
    ObservedRelationship,
    ObservedRun,
    ObservedUnit,
)

# -- a small reference ---------------------------------------------------------------------------


def reference(**changes) -> dict:
    data = {
        "name": "tiny", "version": "1", "description": "A tiny reference.",
        "source": {"kind": "inline", "book_title": "Tiny"},
        "review": {"prepared_by": "test", "prepared_on": "2026-01-01", "status": "draft"},
        "categories": ["concept", "principle", "equation"],
        "passages": [
            {
                "key": "A", "pages": [1, 2], "text": "The spacing effect means spaced study is remembered longer. R = e^(-t/S) describes forgetting.",
                "units": [
                    {"key": "spacing", "category": "concept", "names": ["Spacing effect"],
                     "facts": [["spaced", "distributed"], ["remembered longer", "retention"]], "forbidden": ["cramming works best"]},
                    {"key": "testing", "category": "principle", "accept_categories": ["concept"], "names": ["Testing effect"], "required": False,
                     "facts": [["test"], ["retention"]]},
                    {"key": "eq", "category": "equation", "names": ["Forgetting"],
                     "equation": {"tokens": ["r=e", "t/s"], "capture_status": ["complete"], "has_image": True,
                                  "phase1": {"latex": "R = e^{-t/S}", "plain_text": "R = e -t/S"}}},
                ],
                "relationships": [{"source": "eq", "target": "spacing", "types": ["describes", "related_to"]}],
            },
            {"key": "B", "pages": [3, 3], "text": "Chapter roadmap only.", "expect_nothing": True, "units": []},
            {
                "key": "C", "pages": [4, 4], "text": "Encoding turns experience into a stored form.", "exhaustive": True,
                "units": [{"key": "encoding", "category": "concept", "names": ["Encoding"], "facts": [["stored"], ["experience"]]}],
            },
        ],
    }
    return data | changes


def make_dataset(**changes) -> EvaluationDataset:
    return EvaluationDataset.model_validate(reference(**changes))


_ids = iter(range(10_000))
PAGE = {"A": 1, "B": 3, "C": 4}  # a page inside each passage of the reference


def unit(category, title, passage, text="", **fields) -> ObservedUnit:
    return ObservedUnit(
        id=f"u{next(_ids)}", category=category, title=title, text=f"{title}\n{text}", primary_passage=passage, source_passages=[passage],
        page_start=fields.pop("page_start", PAGE[passage]), page_end=fields.pop("page_end", PAGE[passage]), source_status=fields.pop("source_status", "verified"),
        validation_status=fields.pop("validation_status", "valid"), extracted_from=fields.pop("extracted_from", passage), **fields,
    )


def equation(passage="A", status="complete", flags=(), **fields) -> ObservedUnit:
    eq = ObservedEquation(
        capture_status=status, original_expression=fields.pop("original", "R = e^(-t/S)"), latex=fields.pop("latex", r"R = e^{-t/S}"),
        source_text=fields.pop("source_text", "R = e -t/S"), source_latex=r"R = e^{-t/S}", image_reference=fields.pop("image", "equations/x.png"),
        page_start=fields.pop("page", 1),
    )
    return unit("equation", "Forgetting", passage, flags=list(flags), equation=eq, **fields)


def good_units() -> list[ObservedUnit]:
    return [
        unit("concept", "Spacing effect", "A", "Spaced study gives better retention, remembered longer."),
        unit("principle", "Testing effect", "A", "A test improves retention."),
        equation(),
        unit("concept", "Encoding", "C", "Experience is turned into a stored form."),
    ]


def run_info(**fields) -> ObservedRun:
    values = dict(run_id="r1", status="completed", model_configuration_id="m1", provider="p", model="mm", model_snapshot=None, generation={"temperature": 0.0},
                  parameters={}, prompt_versions=["concept_extraction@1.2.0"], started_at=None, completed_at=None, duration_seconds=10.0,
                  input_tokens=100, output_tokens=50, cost_usd=None, passages=3, attempts=4, retries=1, validation_failures=0, duplicates=0,
                  failed=0, needs_review=0)
    return ObservedRun(**(values | fields))


def execution(passage, category="concept", status="succeeded", attempts=1, retries=0, ms=1000) -> ObservedExecution:
    return ObservedExecution(passage=passage, category=category, status=status, attempts=attempts, retries=retries, validation_failures=0,
                             duration_ms=ms, input_tokens=None, output_tokens=None, cost_usd=None, prompt=None)


def observation(units=None, **fields) -> Observation:
    return Observation(units=good_units() if units is None else units, **fields)


def value(evaluation, key):
    return evaluation.value(key)


# -- text helpers ----------------------------------------------------------------------------------


def test_math_forms_ignore_layout_and_spell_out_greek_letters():
    assert math_form("F_{w}(x, y)") == math_form("Fw(x,y)") == "fw(x,y)"
    assert math_form("exp[-βF(x, y)]") == "exp[-betaf(x,y)]"
    assert token_coverage(["r=e", "t/s"], "R = e^(-t/S)") == 1.0
    assert token_coverage(["r=e", "q"], "R = e^(-t/S)") == 0.5


def test_names_match_by_words_not_by_substring_accidents():
    assert name_score("Energy-based models (EBMs)", ["Energy-based model", "EBM"]) >= 0.5
    assert name_score("The Spacing Effect", ["Spacing effect"]) == 1.0
    assert name_score("Gibbs distribution", ["Spacing effect"]) == 0.0
    assert name_score("Energy function used only for inference", ["Inference"]) < 0.5, "a short name inside a long title is not a match"
    assert name_score("Self-supervised learning (SSL)", ["Self-supervised learning", "SSL"]) == 1.0
    assert normalize("Ségrégation — Test!") == "segregation test"


# -- recall, precision and matching -----------------------------------------------------------------


def test_a_complete_and_correct_extraction_scores_perfectly():
    result = evaluate(make_dataset(), observation())

    assert value(result, "unit_recall") == 1.0 and result.metric("unit_recall").numerator == 2
    assert value(result, "unit_precision") == 1.0
    assert value(result, "primary_source_accuracy") == 1.0 and value(result, "page_accuracy") == 1.0
    assert value(result, "duplicate_rate") == 0.0
    assert value(result, "equation_handled") == 1.0
    assert not result.missed or all(not m.required for m in result.missed)


def test_a_missing_required_unit_lowers_recall_and_is_listed():
    result = evaluate(make_dataset(), observation([u for u in good_units() if u.title != "Encoding"]))

    assert value(result, "unit_recall") == 0.5
    [missed] = [m for m in result.missed if m.required]
    assert (missed.passage, missed.reference, missed.reason) == ("C", "encoding", "not_extracted")
    assert value(result, "unit_recall.concept") == 0.5


def test_a_category_that_was_never_extracted_is_not_held_against_the_model():
    only_concepts = Observation(units=[u for u in good_units() if u.category == "concept"],
                                executions=[execution("A", "concept"), execution("C", "concept")])

    result = evaluate(make_dataset(), only_concepts)

    missed = {(m.reference, m.reason) for m in result.missed}
    assert ("testing", "not_attempted") in missed
    assert result.equations[0].covered is False and result.equations[0].found is False
    assert value(result, "unit_recall") == 1.0
    assert value(result, "equation_found") is None, "nothing to measure is n/a, never zero"


def test_a_name_alone_does_not_recognise_a_unit_that_lacks_the_facts():
    wrong = [unit("concept", "Encoding", "C", "Nothing relevant here.")]

    result = evaluate(make_dataset(), Observation(units=wrong))

    assert value(result, "unit_recall.concept") == 0.0
    assert result.extras[0].kind == "unmatched"


def test_extras_count_against_precision_only_where_the_reference_is_exhaustive():
    extra_in_open = unit("concept", "Something else entirely", "A", "unrelated words")
    extra_in_exhaustive = unit("concept", "Another invention", "C", "unrelated words")

    result = evaluate(make_dataset(), observation([*good_units(), extra_in_open, extra_in_exhaustive]))

    counted = {e.observed_id: e.counted for e in result.extras}
    assert counted == {extra_in_open.id: False, extra_in_exhaustive.id: True}
    precision = result.metric("unit_precision")
    assert (precision.numerator, precision.denominator) == (3, 4)
    assert "not counted either way" in (precision.note or "")


def test_the_same_knowledge_twice_is_a_duplicate_not_two_correct_units():
    twin = unit("concept", "The spacing effect", "A", "Spaced study, better retention, remembered longer.")

    result = evaluate(make_dataset(), observation([*good_units(), twin]))

    [extra] = [e for e in result.extras if e.kind == "redundant"]
    assert extra.of_reference == "A/spacing" and extra.counted
    assert value(result, "duplicate_rate") == pytest.approx(1 / 5)
    precision = result.metric("unit_precision")
    assert (precision.numerator, precision.denominator) == (3, 4), "the duplicate is not a correct extraction"


def test_a_forbidden_phrase_is_reported_on_the_matched_unit():
    bad = unit("concept", "Spacing effect", "A", "Spaced study, remembered longer, but cramming works best.")

    result = evaluate(make_dataset(), observation([bad, good_units()[3]]))

    assert value(result, "forbidden_phrase_units") == 1
    assert result.matches[0].forbidden_hits == ["cramming works best"]


def test_an_empty_passage_with_an_invented_unit_is_not_left_clean():
    invented = unit("concept", "A made-up concept", "B", "words")

    clean = evaluate(make_dataset(), observation())
    dirty = evaluate(make_dataset(), observation([*good_units(), invented]))

    assert value(clean, "empty_passage_accuracy") == 1.0 and value(dirty, "empty_passage_accuracy") == 0.0
    assert [e.counted for e in dirty.extras] == [True], "a passage that expects nothing is judged exhaustively"


# -- sources -------------------------------------------------------------------------------------


def test_a_unit_citing_the_wrong_passage_or_pages_is_counted_as_a_source_error():
    wrong_passage = unit("concept", "Encoding", "A", "stored form of experience", page_start=4, page_end=4)
    wrong_pages = unit("concept", "Spacing effect", "A", "spaced, remembered longer", page_start=9, page_end=9)

    result = evaluate(make_dataset(), observation([wrong_passage, wrong_pages, equation()]))

    assert value(result, "primary_source_accuracy") == 0.5
    assert value(result, "page_accuracy") == 0.5
    mismatch = next(m for m in result.matches if m.reference == "encoding")
    assert mismatch.passage == "C" and not mismatch.source_correct


def test_provenance_quotes_and_target_source_are_measured_from_the_units():
    verified = unit("concept", "Spacing effect", "A", "spaced remembered longer", quotes=2, quotes_verified=2)
    shaky = unit("concept", "Encoding", "C", "stored experience", source_status="invalid", quotes=2, quotes_verified=1, extracted_from="A")

    result = evaluate(make_dataset(), observation([verified, shaky]))

    assert value(result, "provenance_verified") == 0.5
    assert value(result, "quote_verification") == 0.75
    assert value(result, "target_source_rate") == 0.5


# -- equations -----------------------------------------------------------------------------------


def test_a_missing_equation_is_not_handled():
    result = evaluate(make_dataset(), observation([u for u in good_units() if u.category != "equation"]))

    assert value(result, "equation_found") == 0.0 and value(result, "equation_handled") == 0.0
    assert result.equations[0].found is False


def test_the_original_source_image_and_page_of_an_equation_are_checked():
    misread = equation(original="x = y", latex="x = y")  # the unit is tied to the right source, but the model read it wrongly
    partly = equation(image=None, page=9)

    outcome = evaluate(make_dataset(), observation([misread])).equations[0]
    assert (outcome.found, outcome.original_preserved, outcome.source_preserved) == (True, False, True)

    outcome = evaluate(make_dataset(), observation([partly])).equations[0]
    assert (outcome.found, outcome.original_preserved, outcome.image_retained, outcome.page_retained) == (True, True, False, False)
    assert outcome.source_preserved is True

    unrelated = equation(original="x = y", latex="x = y", source_text="E = mc2")
    unrelated.equation.source_latex = "E = mc^2"
    assert evaluate(make_dataset(), observation([unrelated])).equations[0].found is False


def test_incomplete_equations_must_be_flagged_when_the_reference_says_so():
    data = reference()
    data["passages"][0]["units"][2]["equation"] |= {"capture_status": ["incomplete", "needs_review"], "must_be_flagged": True}
    dataset = EvaluationDataset.model_validate(data)

    flagged = evaluate(dataset, observation([equation(status="incomplete", flags=["incomplete_equation"])])).equations[0]
    silent = evaluate(dataset, observation([equation(status="incomplete")])).equations[0]  # status says incomplete but nothing is flagged
    complete_and_clean = evaluate(dataset, observation([equation(status="complete")])).equations[0]

    assert flagged.handled_correctly is True
    assert silent.flagged is True, "an incomplete capture status is itself a flag"
    assert complete_and_clean.status_ok is False and complete_and_clean.handled_correctly is False


def test_image_only_equations_match_the_source_derived_unit():
    data = reference()
    data["passages"][0]["units"][2]["equation"] = {
        "tokens": ["(4)"], "capture_status": ["image_only"], "must_be_flagged": True, "has_image": True,
        "phase1": {"latex": "", "plain_text": None, "validation_status": "invalid"},
    }
    image_only = unit("equation", "Equation 4", "A", "", origin="source_derived", flags=["incomplete_equation"],
                      equation=ObservedEquation(capture_status="image_only", image_reference="x.png", page_start=1))

    [outcome] = evaluate(EvaluationDataset.model_validate(data), Observation(units=[image_only])).equations

    assert outcome.found and outcome.status_ok and outcome.flagged and outcome.image_retained and outcome.handled_correctly


# -- relationships -------------------------------------------------------------------------------


def test_relationships_are_matched_between_units_including_equations():
    units = good_units()
    equation_unit, concept = units[2], units[0]

    right = evaluate(make_dataset(), observation(units, relationships=[ObservedRelationship(equation_unit.id, concept.id, "describes", "extracted", "unreviewed", True)]))
    wrong_type = evaluate(make_dataset(), observation(units, relationships=[ObservedRelationship(equation_unit.id, concept.id, "contradicts", "extracted", "unreviewed", True)]))
    none = evaluate(make_dataset(), observation(units))

    assert value(right, "relationship_recall") == 1.0 and value(right, "relationship_type_accuracy") == 1.0
    assert value(wrong_type, "relationship_recall") == 1.0 and value(wrong_type, "relationship_type_accuracy") == 0.0
    assert value(none, "relationship_recall") == 0.0 and none.relationships[0].found is False


def test_relationships_the_reference_does_not_list_are_reported_not_scored_against_unrecognised_units():
    units = good_units()
    stray = unit("concept", "Unrelated thing", "A", "words")
    relations = [
        ObservedRelationship(units[0].id, units[1].id, "related_to", "extracted", "unreviewed", True),  # both recognised, not in the reference
        ObservedRelationship(units[0].id, stray.id, "related_to", "ai_suggested", "unreviewed", False),  # touches an unrecognised unit
    ]

    result = evaluate(make_dataset(), observation([*units, stray], relationships=relations))

    assert value(result, "relationship_precision") == 0.0 and result.metric("relationship_precision").denominator == 1
    assert {u.kind for u in result.unexpected_relationships} == {"not_in_reference", "unjudged"}


# -- reliability ---------------------------------------------------------------------------------


def test_reliability_metrics_come_from_executions_and_attempts():
    executions = [
        execution("A", "concept", attempts=1), execution("A", "principle", attempts=3, retries=2, status="needs_review", ms=4000),
        execution("C", "concept", attempts=2, retries=1, status="failed"),
    ]
    attempts = [
        ObservedAttempt("completed", "schema_valid", "concept"), ObservedAttempt("completed", "schema_invalid", "principle"),
        ObservedAttempt("completed", "schema_valid", "principle"), ObservedAttempt("provider_error", "not_validated", "concept"),
        ObservedAttempt("completed", "schema_valid", "relationship"),
    ]

    result = evaluate(make_dataset(), observation(executions=executions, attempts=attempts, runs=[run_info(input_tokens=None)]))

    assert value(result, "schema_valid_calls") == pytest.approx(2 / 3), "provider errors and relationship calls are not answers to a schema"
    assert value(result, "first_attempt_success") == pytest.approx(1 / 3)
    assert value(result, "retry_rate") == 1.0 and value(result, "failure_rate") == pytest.approx(1 / 3)
    assert value(result, "needs_review_rate") == pytest.approx(1 / 3)
    assert value(result, "model_calls") == 6 and value(result, "processing_seconds") == 6.0
    assert value(result, "input_tokens") is None, "tokens the provider did not report are unavailable, not zero"
    assert value(result, "estimated_cost") is None


def test_every_metric_states_how_it_is_counted():
    result = evaluate(make_dataset(), observation())

    assert all(m.definition for m in result.metrics)
    assert len({m.key for m in result.metrics}) == len(result.metrics)
    assert result.dataset_digest == make_dataset().digest != make_dataset(version="2").digest


# -- the dataset files ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", available_datasets())
def test_the_shipped_datasets_are_valid_and_say_who_reviewed_them(name):
    dataset = load_dataset(name)

    assert dataset.reference_units and dataset.review.prepared_by and dataset.review.status in ("draft", "reviewed", "approved")
    assert len({u.key for p in dataset.passages for u in p.units}) == len([u for p in dataset.passages for u in p.units]), "unit keys are unique"


def test_the_inline_dataset_covers_every_knowledge_category_and_the_equation_cases():
    dataset = load_dataset("study_guide_v1")

    assert set(dataset.counts()) == {"concept", "principle", "method", "framework", "evidence", "example", "equation"}
    statuses = {s.value for _, u in dataset.reference_units if u.equation for s in u.equation.capture_status}
    assert {"complete", "incomplete", "image_only"} <= statuses
    assert any(p.expect_nothing for p in dataset.passages) and any(p.relationships for p in dataset.passages)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["passages"].append(dict(d["passages"][0])), "Duplicate passage keys"),
        (lambda d: d["passages"][0]["units"][0].update(quote="words that are not in the passage"), "quote is not in the passage"),
        (lambda d: d["passages"][0]["units"][2].pop("equation"), "equation block is required"),
        (lambda d: d["passages"][0]["units"][0].update(equation={"tokens": ["x"], "capture_status": ["complete"]}), "only for them"),
        (lambda d: d["passages"][0]["relationships"].append({"source": "eq", "target": "nope", "types": ["related_to"]}), "not in the passage"),
        (lambda d: d["passages"][1]["units"].append({"key": "u", "category": "concept", "names": ["x"]}), "expects nothing"),
        (lambda d: d["passages"][0].update(pages=[5, 2]), "pages are reversed"),
        (lambda d: d["passages"][0].pop("text"), "needs the passage text"),
        (lambda d: d["passages"][0]["units"][2]["equation"].pop("phase1"), "needs its Phase 1 data"),
    ],
)
def test_a_malformed_dataset_is_refused_with_a_clear_message(change, message):
    data = reference()
    change(data)

    with pytest.raises(ValidationError, match=message):
        EvaluationDataset.model_validate(data)
