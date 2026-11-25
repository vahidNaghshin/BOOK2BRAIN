"""Running an evaluation: load the dataset, run the real pipeline, observe the result, measure it.

Two ways in:

* `run_evaluation` takes an *inline* dataset, loads it into the database as a fresh book, runs the
  normal `KnowledgeExtractionService` over it with the chosen model and measures the outcome. A
  fresh book each time keeps runs independent: the pipeline never stores a unit twice for the same
  passage, so repeating a run on one book would find the earlier units. The book is removed afterwards
  unless it is kept for inspection on the Knowledge screen.
* `evaluate_stored` measures extractions that already exist for a *stored-book* dataset. It reads and
  never writes.

Nothing is simulated here: the extraction goes through the same services, persistence and provenance
checks as any run. A test (or `--scripted`) can supply a fake provider through the service's
`provider_factory`, exactly as the other pipeline tests do.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.evaluation.dataset import EvaluationDataset, ReferencePassage
from app.evaluation.metrics import Evaluation, evaluate
from app.evaluation.observation import (
    Observation,
    ObservedAttempt,
    ObservedEquation,
    ObservedExecution,
    ObservedRelationship,
    ObservedRun,
    ObservedUnit,
)
from app.knowledge.types import KnowledgeType
from app.services.knowledge_extraction_service import (
    ExtractionProgress,
    KnowledgeExtractionService,
    RunStatus,
)
from database.models import (
    Book,
    Document,
    Equation,
    EquationSource,
    ExtractionAttempt,
    ExtractionExecution,
    ExtractionRun,
    KnowledgeRelationship,
    KnowledgeUnit,
    Passage,
    ReviewFlag,
    StructuralNode,
)

EVALUATION_PREFIX = "[Evaluation] "


class EvaluationError(Exception):
    """The dataset cannot be loaded or evaluated (a clear message for the person running it)."""


@dataclass
class Materialised:
    """The passages of a dataset as stored in the database."""

    book_id: uuid.UUID
    document_id: uuid.UUID
    passage_ids: dict[str, uuid.UUID]  # dataset passage key -> passage id

    @property
    def keys_by_id(self) -> dict[uuid.UUID, str]:
        return {pid: key for key, pid in self.passage_ids.items()}


@dataclass
class EvaluationOutcome:
    evaluation: Evaluation
    run_status: RunStatus
    materialised: Materialised
    kept: bool
    wall_seconds: float


# -- loading the dataset into the database -----------------------------------------------------------


def materialise(session: Session, dataset: EvaluationDataset, *, label: str | None = None) -> Materialised:
    """Store an inline dataset as a book of its own (with Phase 1 equations) and return where each passage went."""
    if dataset.source.kind != "inline":
        raise EvaluationError(f"Dataset '{dataset.name}' refers to a stored book; it cannot be loaded.")
    stamp = label or datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    book = Book(title=f"{EVALUATION_PREFIX}{dataset.source.book_title} ({stamp})", author=dataset.source.author or "Evaluation dataset")
    document = Document(
        book=book, original_filename=f"evaluation-{dataset.name}.pdf", stored_filename=f"{uuid.uuid4()}.pdf",
        file_path=f"evaluation/{uuid.uuid4()}/{dataset.name}.pdf", file_format="pdf", mime_type="application/pdf",
        file_size=1, sha256_hash=uuid.uuid4().hex + uuid.uuid4().hex, processing_status="completed",
    )
    session.add_all([book, document])
    session.flush()

    chapters: dict[str, StructuralNode] = {}
    sections: dict[tuple[str, str], StructuralNode] = {}
    sequence = 0

    def node(title: str, kind: str, parent: StructuralNode | None) -> StructuralNode:
        nonlocal sequence
        created = StructuralNode(
            book_id=book.id, document_id=document.id, parent_id=parent.id if parent else None, node_type=kind, title=title,
            level=0 if parent is None else parent.level + 1, sequence=sequence,
        )
        sequence += 1
        session.add(created)
        session.flush()
        return created

    passage_ids: dict[str, uuid.UUID] = {}
    in_node: dict[uuid.UUID, int] = {}
    equation_number = 0
    for index, ref in enumerate(dataset.passages):
        chapter_title = ref.chapter or "Evaluation passages"
        chapter = chapters.get(chapter_title) or chapters.setdefault(chapter_title, node(chapter_title, "chapter", None))
        owner = chapter
        if ref.section:
            owner = sections.get((chapter_title, ref.section)) or sections.setdefault(
                (chapter_title, ref.section), node(ref.section, "section", chapter)
            )
        passage = Passage(
            book_id=book.id, document_id=document.id, structural_node_id=owner.id, text=ref.text,
            sequence=in_node.get(owner.id, 0), document_sequence=index, page_start=ref.pages[0], page_end=ref.pages[1],
            page_status="exact", token_count=max(1, len(ref.text or "") // 4),
        )
        in_node[owner.id] = in_node.get(owner.id, 0) + 1
        session.add(passage)
        session.flush()
        passage_ids[ref.key] = passage.id
        equation_ids = []
        for unit in ref.units:
            if unit.equation is None or unit.equation.phase1 is None:
                continue
            data = unit.equation.phase1
            equation = Equation(
                document_id=document.id, sequence_number=equation_number, identity_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32],
                latex=data.latex or "", original_latex=data.original_latex, plain_text=data.plain_text, equation_type="display",
                extraction_method="evaluation_dataset", page_number=ref.pages[0], surrounding_text=data.surrounding_text,
                crop_reference=f"equations/{document.id}/{uuid.uuid4().hex[:8]}.png" if data.has_image else None,
                validation_status=data.validation_status,
            )
            equation_number += 1
            session.add(equation)
            session.flush()
            equation_ids.append(str(equation.id))
        if equation_ids:
            passage.passage_metadata = {"equation_ids": equation_ids, "has_equation": True}
    session.commit()
    return Materialised(book.id, document.id, passage_ids)


def locate(session: Session, dataset: EvaluationDataset) -> Materialised:
    """Find the passages of a stored-book dataset in the database by the start of their text."""
    books = list(session.scalars(select(Book).where(Book.title == dataset.source.book_title)))
    if len(books) != 1:
        raise EvaluationError(
            f"The dataset needs one book titled '{dataset.source.book_title}' in the database; found {len(books)}."
        )
    book = books[0]
    passages = list(session.scalars(select(Passage).where(Passage.book_id == book.id).order_by(Passage.document_sequence)))
    found: dict[str, uuid.UUID] = {}
    document_id: uuid.UUID | None = None
    for ref in dataset.passages:
        candidates = [p for p in passages if p.text.lstrip().startswith(ref.anchor.strip())]
        if ref.document_sequence is not None:
            candidates = [p for p in candidates if p.document_sequence == ref.document_sequence] or candidates
        if len(candidates) != 1:
            raise EvaluationError(
                f"Passage '{ref.key}': the anchor '{ref.anchor[:40]}' matches {len(candidates)} passages of '{book.title}'. "
                "The book may have been processed again; check the dataset."
            )
        found[ref.key] = candidates[0].id
        document_id = document_id or candidates[0].document_id
    return Materialised(book.id, document_id, found)


def remove_book(session: Session, book_id: uuid.UUID) -> None:
    """Delete an evaluation book and everything derived from it (the database cascades)."""
    book = session.get(Book, book_id)
    if book is None or not book.title.startswith(EVALUATION_PREFIX):
        raise EvaluationError("Only evaluation books can be removed by the evaluation tools.")
    session.execute(delete(Book).where(Book.id == book_id))
    session.commit()


# -- observing a run ---------------------------------------------------------------------------------


def _flatten(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten(item, out)
    elif isinstance(value, list):
        for item in value:
            _flatten(item, out)


def observe(session: Session, run_ids: Iterable[uuid.UUID], passage_keys: dict[uuid.UUID, str]) -> Observation:
    """Read what the runs stored (read-only) into the plain form the metrics use."""
    ids = list(run_ids)
    observation = Observation()
    runs = list(session.scalars(select(ExtractionRun).where(ExtractionRun.id.in_(ids)).order_by(ExtractionRun.created_at)))
    for run in runs:
        executions = list(session.scalars(select(ExtractionExecution).where(ExtractionExecution.extraction_run_id == run.id)))
        duration = (run.completed_at - run.started_at).total_seconds() if run.started_at and run.completed_at else None
        observation.runs.append(
            ObservedRun(
                run_id=str(run.id), status=run.status, model_configuration_id=run.model_configuration_id, provider=run.provider,
                model=run.model_name,
                model_snapshot=run.model_snapshot or next((e.model_configuration for e in executions if e.model_configuration), None),
                generation=next((e.generation_parameters for e in executions if e.generation_parameters), {}),
                parameters=run.parameters or {},
                prompt_versions=sorted({f"{e.prompt_name}@{e.prompt_version}" for e in executions if e.prompt_name and e.prompt_version}),
                started_at=run.started_at, completed_at=run.completed_at, duration_seconds=duration,
                input_tokens=run.input_tokens, output_tokens=run.output_tokens, cost_usd=run.estimated_cost_usd,
                passages=run.total_passages, attempts=run.attempt_count, retries=run.retry_count,
                validation_failures=run.validation_failure_count, duplicates=run.duplicate_count,
                failed=run.failure_count, needs_review=run.needs_review_count,
            )
        )
        for execution in executions:
            observation.executions.append(
                ObservedExecution(
                    passage=passage_keys.get(execution.passage_id), category=execution.knowledge_type, status=execution.status,
                    attempts=execution.attempt_count, retries=execution.retry_count, validation_failures=execution.validation_failures,
                    duration_ms=execution.duration_ms, input_tokens=execution.input_tokens, output_tokens=execution.output_tokens,
                    cost_usd=execution.estimated_cost_usd,
                    prompt=f"{execution.prompt_name}@{execution.prompt_version}" if execution.prompt_name else None,
                )
            )
            for attempt in session.scalars(select(ExtractionAttempt).where(ExtractionAttempt.execution_id == execution.id)):
                observation.attempts.append(
                    ObservedAttempt(status=attempt.status, validation_outcome=attempt.validation_outcome, category=execution.knowledge_type)
                )

    units = list(session.scalars(select(KnowledgeUnit).where(KnowledgeUnit.extraction_run_id.in_(ids))))
    execution_passage = {
        e.id: e.passage_id for e in session.scalars(select(ExtractionExecution).where(ExtractionExecution.extraction_run_id.in_(ids)))
    }
    unit_ids = [u.id for u in units]
    flags: dict[uuid.UUID, list[str]] = {}
    for flag in session.scalars(select(ReviewFlag).where(ReviewFlag.knowledge_unit_id.in_(unit_ids), ReviewFlag.status == "open")):
        flags.setdefault(flag.knowledge_unit_id, []).append(flag.code)
    sources = {s.knowledge_unit_id: s for s in session.scalars(select(EquationSource).where(EquationSource.knowledge_unit_id.in_(unit_ids)))}
    for unit in units:
        passages = sorted(unit.sources, key=lambda s: s.position)
        primary = next((s for s in passages if s.role == "primary"), passages[0] if passages else None)
        text: list[str] = [unit.title, unit.content]
        _flatten(unit.details, text)
        equation = None
        if unit.knowledge_type == "equation":
            source = sources.get(unit.id)
            equation = ObservedEquation(
                capture_status=source.capture_status if source else None,
                original_expression=str(unit.details.get("original_expression") or ""), latex=str(unit.details.get("latex") or ""),
                source_text=source.original_text if source else None, source_latex=source.source_latex if source else None,
                image_reference=source.image_reference if source else None, page_start=source.page_start if source else None,
                surrounding_text=source.surrounding_text if source else None,
            )
        observation.units.append(
            ObservedUnit(
                id=str(unit.id), category=unit.knowledge_type, title=unit.title, text="\n".join(text),
                primary_passage=passage_keys.get(primary.passage_id) if primary else None,
                source_passages=[passage_keys[s.passage_id] for s in passages if s.passage_id in passage_keys],
                page_start=unit.page_start, page_end=unit.page_end, source_status=unit.source_status,
                validation_status=unit.validation_status, origin=unit.origin, is_uncertain=unit.is_uncertain,
                confidence=unit.confidence, quotes=sum(bool(s.quote) for s in passages),
                quotes_verified=sum(bool(s.quote) and s.verification_status == "verified" for s in passages),
                flags=flags.get(unit.id, []), run_id=str(unit.extraction_run_id), equation=equation,
                extracted_from=passage_keys.get(execution_passage.get(unit.extraction_execution_id)) if unit.origin == "ai_extracted" else None,
            )
        )
    observation.units = [u for u in observation.units if u.primary_passage is not None]
    known = {u.id for u in observation.units}
    for rel in session.scalars(select(KnowledgeRelationship).where(KnowledgeRelationship.extraction_run_id.in_(ids))):
        observation.relationships.append(
            ObservedRelationship(
                source_id=str(rel.source_unit_id), target_id=str(rel.target_unit_id), type=rel.relationship_type, origin=rel.origin,
                review_status=rel.review_status, has_evidence=bool(rel.evidence_quote),
            )
        )
    observation.relationships = [r for r in observation.relationships if r.source_id in known or r.target_id in known]
    return observation


# -- the two entry points ----------------------------------------------------------------------------


def evaluate_stored(
    session: Session, dataset: EvaluationDataset, run_ids: Iterable[uuid.UUID] | None = None
) -> tuple[Evaluation, Materialised]:
    """Measure the extractions already stored for a stored-book dataset. Reads only."""
    located = locate(session, dataset)
    if run_ids is None:
        run_ids = list(session.scalars(select(ExtractionRun.id).where(ExtractionRun.book_id == located.book_id)))
    run_ids = list(run_ids)
    if not run_ids:
        raise EvaluationError(f"No extraction runs exist for '{dataset.source.book_title}'; run extraction first.")
    return evaluate(dataset, observe(session, run_ids, located.keys_by_id)), located


def run_evaluation(
    session: Session,
    dataset: EvaluationDataset,
    service: KnowledgeExtractionService,
    *,
    model_id: str | None = None,
    categories: Iterable[KnowledgeType | str] | None = None,
    relationships: bool = False,
    context_passages: int | None = None,
    label: str | None = None,
    keep: bool = False,
    progress: Callable[[ExtractionProgress], None] | None = None,
) -> EvaluationOutcome:
    """Load the dataset as a fresh book, run the pipeline with the model, measure, and (unless kept) remove the book."""
    kinds = list(categories) if categories is not None else list(dataset.categories)
    materialised = materialise(session, dataset, label=label)
    started = time.perf_counter()
    try:
        run_id = service.create_run(
            service.book_scope(materialised.book_id), kinds, model_id=model_id, force=True, context_passages=context_passages,
            extract_relationships=relationships,
        )
        status = service.execute_run(run_id, progress)
        wall = time.perf_counter() - started
        evaluation = evaluate(dataset, observe(session, [run_id], materialised.keys_by_id))
    except Exception:
        session.rollback()
        remove_book(session, materialised.book_id)
        raise
    if not keep:
        remove_book(session, materialised.book_id)
    return EvaluationOutcome(evaluation=evaluation, run_status=status, materialised=materialised, kept=keep, wall_seconds=wall)


def dumps(evaluations: list[Evaluation]) -> str:
    return json.dumps([e.model_dump(mode="json") for e in evaluations], indent=2, ensure_ascii=False)
