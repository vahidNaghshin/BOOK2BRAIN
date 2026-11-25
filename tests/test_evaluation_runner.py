"""Running an evaluation through the real extraction pipeline with scripted (mocked) models.

The scripted models answer from the reference itself or from a degraded copy of it. That proves the harness
measures what it should and is repeatable; it says nothing about the quality of any real model.
"""

import json
import re
import uuid
from argparse import Namespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.evaluation.dataset import EvaluationDataset, load_dataset
from app.evaluation.metrics import Evaluation
from app.evaluation.report import render_comparison, render_report, to_json
from app.evaluation.runner import (
    EVALUATION_PREFIX,
    EvaluationError,
    evaluate_stored,
    locate,
    materialise,
    remove_book,
    run_evaluation,
)
from app.extraction.errors import ProviderTimeoutError
from database.models import Book, Equation, ExtractionRun, KnowledgeUnit, Passage
from tests.extraction_db_helpers import Call, item, make_service, reply, relationship, relationships_reply
from tests.extraction_helpers import make_config
from tests.knowledge_helpers import BUILDERS
from tools import evaluate_extraction as cli

DATASET = load_dataset("study_guide_v1")
CHARACTERISTIC = {"concept": "definition", "principle": "statement", "method": "purpose", "framework": "purpose", "evidence": "description", "example": "scenario"}


def sentence(ref) -> str:
    """Text that carries every key fact of a reference unit (first alternative of each)."""
    return "; ".join(alternatives[0] for alternatives in ref.facts) or f"{ref.names[0]} as described in the passage"


def content_for(ref) -> dict:
    """A valid content of the reference unit's category that carries its key facts."""
    return BUILDERS[ref.category](**_overrides(ref))


def _overrides(ref) -> dict:
    if ref.equation is not None:
        data = ref.equation.phase1
        status = ref.equation.capture_status[0].value
        return dict(
            name=ref.names[0], original_expression=data.plain_text or data.latex, latex=data.latex or None,
            extraction_status={"complete": "complete", "incomplete": "partial"}.get(status, "ambiguous"),
            reconstruction_status="not_needed" if status == "complete" else "not_attempted",
        )
    field = CHARACTERISTIC[ref.category.value]
    base = {"name": ref.names[0]} if ref.category.value != "evidence" else {"claim": ref.names[0]}
    return base | {field: sentence(ref), "explanation": sentence(ref)} if ref.category.value in ("concept", "principle") else base | {field: sentence(ref)}


class Script:
    """A model that answers from the dataset's reference, optionally spoiling some answers."""

    def __init__(self, session: Session, dataset: EvaluationDataset = DATASET, *, skip=lambda ref: False, repeat=lambda ref: False, wrong_passage=lambda ref: False):
        self.session, self.dataset = session, dataset
        self.skip, self.repeat, self.wrong_passage = skip, repeat, wrong_passage
        self.calls: list[Call] = []

    def reference_passage(self, call: Call):
        text = self.session.get(Passage, uuid.UUID(call.target_id)).text
        return next(p for p in self.dataset.passages if p.text == text)

    def __call__(self, call: Call) -> str:
        self.calls.append(call)
        passage = self.reference_passage(call)
        if call.is_relationship:
            found = []
            titles = {u["title"] for u in call.units}
            for rel in passage.relationships:
                names = {u.key: u.names[0] for u in passage.units}
                if names[rel.source] in titles and names[rel.target] in titles:
                    kind = "related_to" if any(t.value == "related_to" for t in rel.types) else rel.types[0].value
                    found.append(relationship(call, names[rel.source], names[rel.target], kind))
            return relationships_reply(*found)
        items = []
        for ref in passage.units:
            if ref.category.value != call.category.value or self.skip(ref):
                continue
            if ref.equation is not None and not (ref.equation.phase1.plain_text or ref.equation.phase1.latex):
                continue  # an image-only equation: the model cannot read it
            source = {"passage_id": call.target_id, "role": "primary"} | ({"quote": ref.quote} if ref.quote else {})
            made = {"content": content_for(ref), "sources": [source], "basis": "explicit", "confidence": 0.8, "is_uncertain": False}
            items.append(made)
            if self.repeat(ref):
                items.append(made | {"content": content_for(ref) | {"name": ref.names[0].lower() + " (again)"}})
        return reply(*items)


def run(session, script, *, model="local-test", relationships=True, keep=False, **config) -> tuple:
    service, providers = make_service(session, script, configs=[make_config(id=model, **config)], default=model)
    outcome = run_evaluation(session, script.dataset, service, model_id=model, relationships=relationships, keep=keep, label=f"{model}-{uuid.uuid4().hex[:4]}")
    return outcome, providers[model]


# -- the dataset in the database -----------------------------------------------------------------


def test_an_inline_dataset_becomes_a_fresh_book_with_its_phase_1_equations(session):
    first = materialise(session, DATASET)
    second = materialise(session, DATASET)

    assert first.book_id != second.book_id, "every run gets its own book, so the pipeline's duplicate protection never hides earlier units"
    passages = list(session.scalars(select(Passage).where(Passage.book_id == first.book_id).order_by(Passage.document_sequence)))
    assert [p.text for p in passages] == [p.text for p in DATASET.passages]
    assert [(p.page_start, p.page_end) for p in passages] == [tuple(p.pages) for p in DATASET.passages]
    equations = list(session.scalars(select(Equation).where(Equation.document_id == first.document_id)))
    assert len(equations) == 3 and sum(not e.latex for e in equations) == 1, "the image-only equation has a crop but no text"
    assert all(e.crop_reference for e in equations)
    assert session.get(Book, first.book_id).title.startswith(EVALUATION_PREFIX)


def test_only_evaluation_books_can_be_removed(session):
    from tests.knowledge_helpers import store_book

    real = store_book(session)
    with pytest.raises(EvaluationError, match="Only evaluation books"):
        remove_book(session, real.book.id)

    loaded = materialise(session, DATASET)
    remove_book(session, loaded.book_id)
    assert session.get(Book, loaded.book_id) is None
    assert session.scalar(select(func.count()).select_from(Passage).where(Passage.book_id == loaded.book_id)) == 0


# -- a full run through the pipeline ---------------------------------------------------------------


def test_a_reference_following_model_is_measured_through_the_real_pipeline(session):
    script = Script(session)

    outcome, provider = run(session, script)
    evaluation = outcome.evaluation

    assert outcome.run_status.status in ("completed", "needs_review")
    assert evaluation.value("unit_recall") == 1.0, [m for m in evaluation.missed if m.required]
    assert evaluation.value("primary_source_accuracy") == 1.0 and evaluation.value("page_accuracy") == 1.0
    assert evaluation.value("provenance_verified") == 1.0 and evaluation.value("quote_verification") == 1.0
    assert evaluation.value("schema_valid_calls") == 1.0 and evaluation.value("failure_rate") == 0.0
    assert evaluation.value("target_source_rate") == 1.0
    assert evaluation.value("duplicate_rate") == 0.0
    assert evaluation.value("empty_passage_accuracy") == 1.0
    assert evaluation.value("forbidden_phrase_units") == 0
    assert evaluation.value("relationship_recall") == 1.0
    assert len(provider.requests) == outcome.run_status.run.attempt_count

    equations = {e.reference: e for e in evaluation.equations}
    assert equations["retention-equation"].handled_correctly and equations["retention-equation"].capture_status == "complete"
    assert equations["interval-equation"].handled_correctly and equations["interval-equation"].flagged, "a damaged equation is flagged"
    image = equations["image-equation"]
    assert image.found and image.capture_status == "image_only" and image.image_retained and image.handled_correctly, \
        "an equation the model could not read is kept from the source and marked image-only, not invented"
    assert evaluation.value("equation_handled") == 1.0 and evaluation.value("equation_source_preserved") == 1.0


def test_the_run_records_the_model_prompt_and_generation_settings_that_produced_the_numbers(session):
    outcome, _ = run(session, Script(session), temperature=0.2, max_output_tokens=900)

    [info] = outcome.evaluation.runs
    assert info.model_configuration_id == "local-test" and info.model == "test-model:1b" and info.provider == "ollama"
    assert info.generation["temperature"] == 0.2 and info.generation["max_output_tokens"] == 900
    assert info.prompt_versions and all(re.fullmatch(r"[a-z_]+@\d+\.\d+\.\d+", v) for v in info.prompt_versions)
    assert info.input_tokens and info.output_tokens and info.cost_usd is None, "tokens as reported; no price configured, so no invented cost"
    assert outcome.evaluation.dataset_digest == DATASET.digest


def test_the_evaluation_book_is_removed_unless_it_is_kept_for_inspection(session):
    gone, _ = run(session, Script(session))
    assert session.scalars(select(Book).where(Book.id == gone.materialised.book_id)).first() is None
    assert session.scalar(select(func.count()).select_from(KnowledgeUnit)) == 0

    kept, _ = run(session, Script(session), keep=True)
    assert kept.kept and session.get(Book, kept.materialised.book_id) is not None
    assert session.scalar(select(func.count()).select_from(KnowledgeUnit).where(KnowledgeUnit.book_id == kept.materialised.book_id)) > 0


def test_running_the_same_passages_twice_is_repeatable(session):
    first, _ = run(session, Script(session))
    second, _ = run(session, Script(session))

    assert first.run_status.run.id != second.run_status.run.id
    numbers = lambda e: {m.key: (m.numerator, m.denominator) for m in e.metrics if m.unit != "seconds" and not m.key.startswith("run_")}  # noqa: E731
    assert numbers(first.evaluation) == numbers(second.evaluation)
    assert {m.reference for m in first.evaluation.matches} == {m.reference for m in second.evaluation.matches}


# -- comparing models ----------------------------------------------------------------------------


def test_a_weaker_model_scores_lower_where_it_is_weaker_and_the_comparison_shows_both(session):
    strong, _ = run(session, Script(session), model="strong")
    weak_script = Script(
        session,
        skip=lambda ref: ref.category.value == "principle" or ref.key == "encoding",
        repeat=lambda ref: ref.key == "spacing",
    )
    weak, _ = run(session, weak_script, model="weak")

    assert strong.evaluation.value("unit_recall") > weak.evaluation.value("unit_recall")
    assert weak.evaluation.value("unit_recall.principle") == 0.0 and strong.evaluation.value("unit_recall.principle") == 1.0
    assert weak.evaluation.value("duplicates_suppressed") == 1 and strong.evaluation.value("duplicates_suppressed") == 0, \
        "the pipeline refuses to store a unit that repeats another one, and the evaluation counts what it refused"
    assert {e.reference for e in weak.evaluation.missed if e.required} >= {"encoding", "testing-effect"}

    table = render_comparison([strong.evaluation, weak.evaluation], ["strong", "weak"])
    assert "| Required units found (recall) |" in table and "strong | weak" in table
    assert "No column is ranked" in table


def test_a_model_that_cites_a_neighbouring_passage_is_caught_by_the_source_metrics(session):
    class Neighbour(Script):
        def __call__(self, call):
            answer = json.loads(super().__call__(call))
            if call.category and call.category.value == "concept":
                others = [p for p in self.session.scalars(select(Passage).where(Passage.id != uuid.UUID(call.target_id)).limit(1))]
                for found in answer.get("items", []):
                    found["sources"] = [{"passage_id": str(others[0].id), "role": "primary"}]
            return json.dumps(answer)

    # the neighbour is cited as the only source: the pipeline accepts it with a warning, the evaluation measures it
    script = Neighbour(session)
    outcome, _ = run(session, script, relationships=False)

    assert outcome.evaluation.value("primary_source_accuracy") < 1.0
    assert outcome.evaluation.value("target_source_rate") < 1.0
    assert any(not m.source_correct for m in outcome.evaluation.matches)


# -- failures ------------------------------------------------------------------------------------


def test_a_provider_that_is_down_is_reported_as_failures_not_as_missing_knowledge_silently(session):
    def down(call):
        raise ProviderTimeoutError("timed out")

    service, _ = make_service(session, down, max_retries=1, max_consecutive_failures=0)
    outcome = run_evaluation(session, DATASET, service, categories=["concept"], keep=False, label="down")

    evaluation = outcome.evaluation
    assert evaluation.value("failure_rate") == 1.0 and evaluation.value("unit_recall") == 0.0
    assert evaluation.value("retry_rate") == 1.0 and evaluation.value("first_attempt_success") == 0.0
    assert evaluation.value("schema_valid_calls") is None, "no answer was received, so there is nothing to check against a schema"
    assert outcome.run_status.status == "failed"


def test_an_invalid_model_choice_stops_before_anything_is_left_behind(session):
    service, _ = make_service(session, Script(session))

    with pytest.raises(Exception, match="nonexistent"):
        run_evaluation(session, DATASET, service, model_id="nonexistent")

    assert session.scalar(select(func.count()).select_from(Book)) == 0
    assert session.scalar(select(func.count()).select_from(ExtractionRun)) == 0


# -- evaluating extractions that already exist -----------------------------------------------------


def stored_dataset(prefix: str = "Passage c1a.", sequence: int | None = 1) -> EvaluationDataset:
    return EvaluationDataset.model_validate({
        "name": "stored-tiny", "version": "1", "description": "d",
        "source": {"kind": "stored_book", "book_title": "Thermodynamics"},
        "review": {"prepared_by": "t", "prepared_on": "2026-01-01", "status": "draft"},
        "categories": ["concept"],
        "passages": [{
            "key": "c1a", "pages": [2, 2], "anchor": prefix, "document_sequence": sequence,
            "units": [{"key": "entropy", "category": "concept", "names": ["Entropy"], "facts": [["disorder"], ["system"]]}],
        }],
    })


def test_stored_extractions_are_evaluated_without_changing_them(session):
    from tests.extraction_db_helpers import store_library

    library = store_library(session)
    service, _ = make_service(session, lambda call: reply(item(call)))
    service.extract_book(library.book.id, ["concept"])
    before = session.scalar(select(func.count()).select_from(KnowledgeUnit))

    evaluation, located = evaluate_stored(session, stored_dataset())

    assert located.book_id == library.book.id and located.passage_ids["c1a"] == library.passages["c1a"].id
    assert evaluation.value("unit_recall") == 1.0 and evaluation.value("primary_source_accuracy") == 1.0
    assert evaluation.passages_covered == 1
    assert session.scalar(select(func.count()).select_from(KnowledgeUnit)) == before


def test_a_stored_dataset_that_cannot_find_its_passages_says_so(session):
    from tests.extraction_db_helpers import store_library

    with pytest.raises(EvaluationError, match="needs one book titled"):
        locate(session, stored_dataset())

    store_library(session)
    with pytest.raises(EvaluationError, match="matches 0 passages"):
        locate(session, stored_dataset("Not the start of any passage"))
    with pytest.raises(EvaluationError, match="matches 8 passages"):
        locate(session, stored_dataset("Passage", None))
    with pytest.raises(EvaluationError, match="No extraction runs exist"):
        evaluate_stored(session, stored_dataset())


def test_an_inline_dataset_cannot_be_located_and_a_stored_one_cannot_be_loaded(session):
    with pytest.raises(EvaluationError, match="stored book"):
        materialise(session, stored_dataset())


# -- reports ---------------------------------------------------------------------------------------


def test_the_report_names_the_run_the_limits_and_every_metric_definition(session):
    outcome, _ = run(session, Script(session))

    text = render_report([outcome.evaluation], title="Test report")

    for expected in ("# Test report", "local-test", "Reference status: **draft**", "## How to read this report", "## Metric definitions",
                     "Required units not found", "Matched pairs for review", "Equations", "A value of `n/a` means"):
        assert expected in text
    assert all(m.label in text for m in outcome.evaluation.metrics if m.key.startswith("unit_"))
    payload = json.loads(to_json([outcome.evaluation], ["local-test"]))
    assert payload[0]["label"] == "local-test" and payload[0]["dataset_digest"] == DATASET.digest
    assert Evaluation.model_validate({k: v for k, v in payload[0].items() if k != "label"}).value("unit_recall") == 1.0


# -- the command line ----------------------------------------------------------------------------


def args(**values) -> Namespace:
    defaults = dict(dataset="study_guide_v1", validate=False, stored=False, run=None, model=["local-test"], types=None, relationships=False,
                    repeat=1, context=None, keep=False, out=None, yes=True)
    return Namespace(**{**defaults, **values})


def test_the_command_runs_models_repeatedly_and_writes_both_reports(session, tmp_path, capsys):
    service, _ = make_service(session, Script(session))
    out = tmp_path / "result"

    code = cli._run(args(types="concept,principle", repeat=2, out=str(out)), session, service)

    assert code == 0
    printed = capsys.readouterr().out
    assert "x 2 repeat(s)" in printed and "local-test #1" in printed and "local-test #2" in printed
    assert (tmp_path / "result.md").read_text().startswith("# Evaluation of study_guide_v1")
    assert len(json.loads((tmp_path / "result.json").read_text())) == 2


def test_the_command_checks_the_model_and_the_dataset_kind_before_running(session):
    service, _ = make_service(session, Script(session))

    with pytest.raises(EvaluationError, match="Unknown model"):
        cli._run(args(model=["nope"]), session, service)
    with pytest.raises(EvaluationError, match="stored book"):
        cli._run(args(dataset="latent_ebm_v1"), session, service)
    service2, _ = make_service(session, Script(session), environ={})
    assert session.scalar(select(func.count()).select_from(Book)) == 0


def test_listing_and_validating_run_nothing(capsys):
    assert cli.main(["--list"]) == 0
    listing = capsys.readouterr().out
    assert "study_guide_v1" in listing and "latent_ebm_v1" in listing

    assert cli.main(["--dataset", "study_guide_v1", "--validate"]) == 0
    assert "valid." in capsys.readouterr().out
    assert cli.main(["--dataset", "missing_dataset", "--validate"]) == 2
