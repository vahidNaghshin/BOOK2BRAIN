"""Extract knowledge from a stored book with the selected LLM and save it.

    python -m tools.extract_knowledge                              list books and models
    python -m tools.extract_knowledge --book Latent --outline      chapters and sections with ids and passage counts
    python -m tools.extract_knowledge --book Latent --model stormy-claude-haiku-5.5 --types concept --limit 3
    python -m tools.extract_knowledge --chapter NODE_ID --types all --yes
    python -m tools.extract_knowledge --section NODE_ID --types equation --dry-run   call the model, store nothing
    python -m tools.extract_knowledge --book Latent --runs         recent runs;  --status RUN_ID for one run

The API key is read from the environment or .env; it is never printed or stored. Passages that already
have a usable result for a category are skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, get_settings
from app.extraction.errors import ExtractionError
from app.extraction.model_config import ModelRegistry
from app.extraction.settings import ExtractionSettings, load_environment
from app.knowledge.types import KnowledgeType
from app.repositories.passage_repository import PassageRepository
from app.services.errors import ExtractionRunError
from app.services.provenance_service import ProvenanceService
from app.services.knowledge_extraction_service import (
    ExtractionProgress,
    KnowledgeExtractionService,
    NothingToExtractError,
    RunStatus,
)
from database.connection import session_scope
from database.models import Book, KnowledgeUnit, Passage


def list_books(session: Session) -> None:
    print(f"{'BOOK ID':<38}{'PASSAGES':>9}{'UNITS':>7}  TITLE")
    for book in session.scalars(select(Book).order_by(Book.title)):
        passages = session.scalar(select(func.count()).select_from(Passage).where(Passage.book_id == book.id))
        units = session.scalar(select(func.count()).select_from(KnowledgeUnit).where(KnowledgeUnit.book_id == book.id))
        print(f"{book.id!s:<38}{passages:>9}{units:>7}  {book.title}")


def find_book(session: Session, reference: str) -> Book:
    try:
        book = session.get(Book, uuid.UUID(reference))
        if book:
            return book
    except ValueError:
        pass
    matches = list(session.scalars(select(Book).where(Book.title.ilike(f"%{reference}%"))))
    if len(matches) != 1:
        raise ExtractionRunError(
            f"No book matches '{reference}'." if not matches else f"'{reference}' matches {len(matches)} books; use the id."
        )
    return matches[0]


def parse_types(value: str) -> list[KnowledgeType]:
    if value.strip().lower() == "all":
        return list(KnowledgeType)
    try:
        return [KnowledgeType(v.strip().lower()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        valid = ", ".join(t.value for t in KnowledgeType)
        raise ExtractionRunError(f"Unknown knowledge category ({exc}). Choose from: {valid}, or all.") from exc


def models_path(configured: str, override: str | None) -> Path:
    path = Path(override or configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def print_outline(session: Session, book: Book) -> None:
    repository = PassageRepository(session)
    for document in book.documents:
        print(f"Document {document.id}  {document.original_filename}")
        for node, passages in repository.outline(document.id):
            print(f"  {'  ' * node.level}{node.node_type:<10} {node.id}  {passages:>3} passages  {node.title}")


def _print_progress(event: ExtractionProgress) -> None:
    name = event.category.value if event.category else "relations"
    print(
        f"[{event.passages_done}/{event.total_passages}] {name:<10} p.{event.page_start or '?':<4} "
        f"{event.status:<12}{event.items} " + ("unit(s)" if event.category else "relationship(s)")
        + (f", {event.duplicates} duplicate(s)" if event.duplicates else "")
        + (f"  {event.duration_ms / 1000:.1f}s  attempts={event.attempts}" if event.category else "")
        + (f"  {event.failure_code}" if event.failure_code else "")
    )


def describe(status: RunStatus) -> str:
    run = status.run
    text = (
        f"Run {run.id}: {status.status}"
        f"  passages {run.passages_processed}/{run.total_passages}  units={run.units_extracted}"
        f"  needs_review={run.needs_review_count}  failed_passages={status.failed_passages}  duplicates={run.duplicate_count}"
        f"  relationships={run.relationships_extracted}  skipped={run.skipped_count}  attempts={run.attempt_count}  retries={run.retry_count}"
        f"  validation_failures={run.validation_failure_count}  tokens={run.input_tokens or 0}/{run.output_tokens or 0}"
    )
    if run.estimated_cost_usd is not None:
        text += f"  cost=${run.estimated_cost_usd:.4f}"
    text += f"\n  model: {status.model_label}  prompt: {run.prompt_version}"
    if status.duration_seconds is not None:
        text += f"  duration: {status.duration_seconds:.1f}s"
    if run.error_message:
        text += f"\n  error: {run.error_code}: {run.error_message}"
    return text


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract knowledge from a stored book.")
    parser.add_argument("--book", help="book id or part of its title")
    parser.add_argument("--document", type=uuid.UUID, help="only this document of the book")
    parser.add_argument("--chapter", type=uuid.UUID, help="a chapter id: extract this chapter and everything in it")
    parser.add_argument("--section", type=uuid.UUID, help="a section id: extract this section and everything in it")
    parser.add_argument("--outline", action="store_true", help="show the book's chapters and sections with their ids")
    parser.add_argument("--runs", action="store_true", help="list the recent extraction runs of the book")
    parser.add_argument("--status", type=uuid.UUID, help="show one run's status")
    parser.add_argument("--passages", type=uuid.UUID, metavar="RUN_ID", help="show the processing status of every passage of a run")
    parser.add_argument("--resume", type=uuid.UUID, metavar="RUN_ID", help="continue a failed, cancelled or interrupted run")
    parser.add_argument("--relationships", action="store_true", help="also extract the relationships between the units of each passage")
    parser.add_argument("--verify-sources", action="store_true", help="verify the source references of the book's knowledge units")
    parser.add_argument("--types", default="all", help="comma-separated categories, or all (default)")
    parser.add_argument("--model", help="model configuration id (default: EXTRACTION_MODEL_ID)")
    parser.add_argument("--models-file", help="models file (default: EXTRACTION_MODELS_FILE)")
    parser.add_argument("--limit", type=int, help="process at most this many passages")
    parser.add_argument("--offset", type=int, default=0, help="skip this many passages first")
    parser.add_argument("--context", type=int, help="neighbouring passages sent as context on each side (default: setting)")
    parser.add_argument("--force", action="store_true", help="extract again even where a result exists")
    parser.add_argument("--dry-run", action="store_true", help="call the model but store nothing")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args(argv)
    environ = load_environment() if environ is None else environ
    settings = get_settings()

    try:
        registry = ModelRegistry.from_file(
            models_path(settings.extraction_models_file, args.models_file),
            default_id=settings.extraction_model_id,
            overrides=settings.extraction_model_overrides,
        )
    except ExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    with session_scope() as session:
        service = KnowledgeExtractionService(
            session, registry, ExtractionSettings.from_settings(settings), environ=environ
        )
        try:
            return _run(args, session, service)
        except (ExtractionError, ExtractionRunError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2


def _run(args, session: Session, service: KnowledgeExtractionService) -> int:
    if args.status:
        print(describe(service.get_run_status(args.status)))
        info = service.get_resume_info(args.status)
        print(f"  passages: {info.completed} finished, {info.failed} failed, {info.pending} not reached, {info.processing} interrupted"
              f" - {'can resume' if info.can_resume else 'cannot resume'}: {info.reason}")
        return 0
    if args.passages:
        for view in service.get_passage_status(args.passages):
            line = f"{view.sequence:>4}  p.{view.page_start or '?':<4} {view.status:<13} attempts={view.attempts} units={view.units_extracted}"
            print(line + (f"  {view.error_code}: {view.error_message}" if view.error_code else ""))
        return 0
    if args.resume:
        status = service.resume_run(args.resume, progress=_print_progress)
        print("\n" + describe(status))
        return 0 if status.status in ("completed", "needs_review") else 1
    if not (args.book or args.chapter or args.section):
        list_books(session)
        print(f"\n{'MODEL ID':<34}STATUS")
        for option in service.available_models():
            print(f"{option.id:<34}{'ready' if option.ready else ('inactive' if not option.active else 'not ready: ' + str(option.reason))}")
        print("\nChoose --book (or --chapter/--section) and --model, for example: --types concept --limit 3")
        return 0

    if args.chapter:
        scope = service.chapter_scope(args.chapter)
    elif args.section:
        scope = service.section_scope(args.section)
    else:
        scope = service.book_scope(find_book(session, args.book).id, args.document)
    book = session.get(Book, scope.book_id)

    if args.outline:
        print_outline(session, book)
        return 0
    if args.verify_sources:
        counts = ProvenanceService(session).verify_book(book.id)
        print(f"Source verification: {counts['verified']} verified, {counts['invalid']} invalid (see the review flags).")
        return 0 if not counts["invalid"] else 1
    if args.runs:
        for status in service.list_runs(book.id):
            print(describe(status))
        return 0

    categories = parse_types(args.types)
    options = {"model_id": args.model, "limit": args.limit, "offset": args.offset, "context_passages": args.context}
    plan = service.plan(
        scope, categories, force=args.force, extract_relationships=args.relationships,
        **{k: v for k, v in options.items() if k != "context_passages"},
    )
    print(
        f"Book: {book.title}\nScope: {scope.kind}\nModel: {plan.model_label}\n"
        f"Plan: {plan.total_passages} passages x {len(categories)} categories; {plan.model_calls} model calls"
        f" ({plan.skipped} already done and skipped)"
        + (f", plus {plan.relationship_calls} relationship calls" if args.relationships else "")
        + (" - dry run, nothing is stored" if args.dry_run else "")
    )
    if plan.model_calls == 0 and plan.relationship_calls == 0 and not args.dry_run:
        print("Nothing to do. Use --force to extract again.")
        return 0
    if not args.yes and plan.model_calls > 10 and input("Continue? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1

    if args.dry_run:
        results = service.dry_run(scope, categories, **options, progress=lambda i, n, passage, result: print(
            f"[{i}/{n}] {result.knowledge_type.value:<10} p.{passage.page_start or '?':<4} {result.status.value:<12}"
            f"{len(result.items)} item(s)  {result.execution.duration_ms / 1000:.1f}s  attempts={result.execution.attempt_count}"
        ))
        print(f"\nDry run: {sum(len(r.items) for _, r in results)} items from {len(results)} extractions; nothing stored.")
        return 0

    try:
        run_id = service.create_run(scope, categories, force=args.force, extract_relationships=args.relationships, **options)
    except NothingToExtractError as exc:
        print(exc)
        return 0
    status = service.execute_run(run_id, progress=_print_progress)
    print("\n" + describe(status))
    return 0 if status.status in ("completed", "needs_review") else 1


if __name__ == "__main__":
    raise SystemExit(main())
