"""Evaluate knowledge extraction against manually reviewed reference results.

    python -m tools.evaluate_extraction --list
    python -m tools.evaluate_extraction --dataset study_guide_v1 --validate         check the dataset file, run nothing
    python -m tools.evaluate_extraction --dataset study_guide_v1 --model stormy-claude-haiku-5.5 --types all --relationships
    python -m tools.evaluate_extraction --dataset study_guide_v1 --model MODEL_A --model MODEL_B   compare models
    python -m tools.evaluate_extraction --dataset study_guide_v1 --model MODEL_A --repeat 3        repeatability
    python -m tools.evaluate_extraction --dataset latent_ebm_v1 --stored            evaluate the extractions already stored
    python -m tools.evaluate_extraction --dataset latent_ebm_v1 --stored --run RUN_ID

An *inline* dataset carries its passages: each run loads them as a fresh evaluation book (titled
"[Evaluation] ...", removed afterwards unless --keep), runs the normal extraction pipeline with the model
and measures the result. A *stored-book* dataset only carries reference results for a book that is
already in the database, so it measures extractions that exist (--stored) and reads only.

Live runs call the selected model with the dataset's passages. A hosted model needs --yes. The reports
(Markdown and JSON) go to evaluation/results/ unless --out is given.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, get_settings
from app.evaluation.dataset import EvaluationDataset, available_datasets, load_dataset
from app.evaluation.metrics import Evaluation
from app.evaluation.report import format_value, render_report, to_json
from app.evaluation.runner import EvaluationError, evaluate_stored, run_evaluation
from app.extraction.errors import ExtractionError
from app.extraction.model_config import ModelRegistry
from app.extraction.settings import ExtractionSettings, load_environment
from app.services.errors import ExtractionRunError
from app.services.knowledge_extraction_service import KnowledgeExtractionService
from database.connection import session_scope
from tools.extract_knowledge import models_path, parse_types

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
HEADLINE = ["unit_recall", "unit_precision", "primary_source_accuracy", "schema_valid_calls", "duplicate_rate", "equation_handled", "retry_rate", "failure_rate"]


def list_datasets() -> None:
    print(f"{'DATASET':<20}{'KIND':<13}{'PASSAGES':>9}  {'REFERENCE UNITS':<45}  REVIEW")
    for name in available_datasets():
        dataset = load_dataset(name)
        counts = ", ".join(f"{k} {v}" for k, v in sorted(dataset.counts().items()))
        print(f"{name:<20}{dataset.source.kind:<13}{len(dataset.passages):>9}  {counts:<45}  {dataset.review.status}")


def validate(dataset: EvaluationDataset) -> None:
    required = sum(u.required for _, u in dataset.reference_units)
    print(f"{dataset.name} v{dataset.version} ({dataset.source.kind}), fingerprint {dataset.digest}: valid.")
    print(f"  {len(dataset.passages)} passages, {len(dataset.reference_units)} reference units ({required} required), review status: {dataset.review.status}")
    for kind, count in sorted(dataset.counts().items()):
        print(f"  {kind:<10}{count}")


def summarise(label: str, evaluation: Evaluation) -> None:
    print(f"\n{label}")
    for key in HEADLINE:
        metric = evaluation.metric(key)
        print(f"  {metric.label:<62}{format_value(metric)}")


def write_reports(evaluations: list[Evaluation], labels: list[str], title: str, out: str | None, dataset: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = Path(out) if out else RESULTS_DIR / f"{dataset}-{stamp}"
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".md").write_text(render_report(evaluations, title=title, labels=labels), encoding="utf-8")
    base.with_suffix(".json").write_text(to_json(evaluations, labels), encoding="utf-8")
    return base


def _stored(args, session: Session, dataset: EvaluationDataset) -> int:
    runs = [uuid.UUID(r) for r in args.run] if args.run else None
    evaluation, _ = evaluate_stored(session, dataset, runs)
    label = f"stored runs of {dataset.source.book_title}"
    summarise(label, evaluation)
    base = write_reports([evaluation], [label], f"Evaluation of stored extractions: {dataset.name}", args.out, dataset.name)
    print(f"\nReport: {base.with_suffix('.md')}\nData:   {base.with_suffix('.json')}")
    return 0


def _live(args, session: Session, service: KnowledgeExtractionService, dataset: EvaluationDataset) -> int:
    if dataset.source.kind != "inline":
        raise EvaluationError(
            f"Dataset '{dataset.name}' refers to a stored book and cannot be run again; use --stored to evaluate its existing extractions."
        )
    models = args.model or [service.registry.default_id]
    if not models or None in models:
        raise EvaluationError("Choose the model(s) with --model (see: python -m app.extraction.check).")
    options = {option.id: option for option in service.available_models()}
    for model_id in models:
        option = options.get(model_id)
        if option is None:
            raise EvaluationError(f"Unknown model '{model_id}'.")
        if not option.ready:
            raise EvaluationError(f"Model '{model_id}' cannot be used: {option.reason}")
    categories = parse_types(args.types) if args.types else list(dataset.categories)
    calls = len(dataset.passages) * len(categories) * args.repeat * len(models)
    print(f"Dataset {dataset.name} v{dataset.version}: {len(dataset.passages)} passages x {len(categories)} categories "
          f"x {args.repeat} repeat(s) x {len(models)} model(s) = {calls} extraction calls (before retries).")
    hosted = [m for m in models if options[m].deployment == "hosted"]
    if hosted and not args.yes:
        print(f"The dataset's passages will be sent to hosted model(s): {', '.join(hosted)}.")
        if input("Continue? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return 1

    evaluations: list[Evaluation] = []
    labels: list[str] = []
    for model_id in models:
        for repeat in range(1, args.repeat + 1):
            label = model_id if args.repeat == 1 else f"{model_id} #{repeat}"
            print(f"\nRunning {label} ...")
            outcome = run_evaluation(
                session, dataset, service, model_id=model_id, categories=categories, relationships=args.relationships,
                context_passages=args.context, label=f"{model_id}-{repeat}-{uuid.uuid4().hex[:4]}", keep=args.keep,
            )
            summarise(label, outcome.evaluation)
            if outcome.kept:
                print(f"  kept as book '{outcome.materialised.book_id}' for the Knowledge screen")
            evaluations.append(outcome.evaluation)
            labels.append(label)
    title = f"Evaluation of {dataset.name}: " + (" vs ".join(dict.fromkeys(models)) if len(models) > 1 else models[0])
    base = write_reports(evaluations, labels, title, args.out, dataset.name)
    print(f"\nReport: {base.with_suffix('.md')}\nData:   {base.with_suffix('.json')}")
    return 0


def _run(args, session: Session, service: KnowledgeExtractionService | None) -> int:
    dataset = load_dataset(args.dataset)
    if args.validate:
        validate(dataset)
        return 0
    if args.stored:
        return _stored(args, session, dataset)
    return _live(args, session, service, dataset)


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate knowledge extraction against reviewed reference results.")
    parser.add_argument("--list", action="store_true", help="list the evaluation datasets")
    parser.add_argument("--dataset", help="dataset name (evaluation/datasets/NAME.json) or file path")
    parser.add_argument("--validate", action="store_true", help="check the dataset file and stop")
    parser.add_argument("--stored", action="store_true", help="evaluate the extractions already stored for a stored-book dataset")
    parser.add_argument("--run", action="append", help="with --stored: only this extraction run (repeatable; default: every run of the book)")
    parser.add_argument("--model", action="append", help="model configuration id; repeat to compare models")
    parser.add_argument("--models-file", help="models file (default: EXTRACTION_MODELS_FILE)")
    parser.add_argument("--types", help="categories to extract (default: the dataset's), e.g. concept,equation or all")
    parser.add_argument("--relationships", action="store_true", help="also extract relationships")
    parser.add_argument("--repeat", type=int, default=1, help="run each model this many times (each on a fresh copy of the passages)")
    parser.add_argument("--context", type=int, help="neighbouring passages sent as context (default: setting)")
    parser.add_argument("--keep", action="store_true", help="keep the evaluation books so they can be inspected on the Knowledge screen")
    parser.add_argument("--out", help="report path without extension (default: evaluation/results/DATASET-TIMESTAMP)")
    parser.add_argument("--yes", action="store_true", help="do not ask before sending passages to a hosted model")
    args = parser.parse_args(argv)

    if args.list:
        list_datasets()
        return 0
    if not args.dataset:
        parser.error("choose --dataset (or --list)")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    try:
        if args.validate:
            return _run(args, None, None)  # type: ignore[arg-type]
        settings = get_settings()
        with session_scope() as session:
            if args.stored:
                return _run(args, session, None)
            registry = ModelRegistry.from_file(
                models_path(settings.extraction_models_file, args.models_file),
                default_id=settings.extraction_model_id,
                overrides=settings.extraction_model_overrides,
            )
            environ = load_environment() if environ is None else environ
            service = KnowledgeExtractionService(session, registry, ExtractionSettings.from_settings(settings), environ=environ)
            return _run(args, session, service)
    except (EvaluationError, ExtractionRunError, ExtractionError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
