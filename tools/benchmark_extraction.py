"""Measure the extraction pipeline's own cost: time, database queries and memory per passage.

    python -m tools.benchmark_extraction                      # 60 passages, concept + equation + relationships
    python -m tools.benchmark_extraction --passages 200 --latency-ms 0
    python -m tools.benchmark_extraction --drop               # remove the benchmark database afterwards

No model is called. A scripted provider answers instantly (or after --latency-ms), so what is measured
is everything else: context preparation, validation, persistence, provenance, equation capture,
review flags and run bookkeeping. A real model's latency adds to it per call and is reported by the
runs themselves.

It uses its own database (the configured one with the suffix `_benchmark`, created and migrated on the
first run) and removes the book it creates, so your data is never touched.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
import tracemalloc
import uuid
from collections import Counter

from alembic import command
from alembic.config import Config
from sqlalchemy import event, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, get_settings
from app.extraction.model_config import LlmModelConfig, ModelRegistry
from app.extraction.providers import LlmResponse
from app.extraction.settings import ExtractionSettings
from app.knowledge.types import KnowledgeType
from app.repositories.knowledge_repository import KnowledgeFilter
from app.services.knowledge_extraction_service import KnowledgeExtractionService
from app.services.knowledge_service import KnowledgeService
from database.connection import build_engine
from database.models import Book, Document, Equation, Passage, StructuralNode

TEXT = (
    "Retention fades with time unless a memory is revisited, and the pace of fading depends on how well the material was learned. "
    "Spaced review strengthens a memory more than the same time spent re-reading it. "
)


def benchmark_url(base: str) -> str:
    url = make_url(base)
    return url.set(database=f"{url.database}_benchmark").render_as_string(hide_password=False)


def prepare_database(url: str) -> Engine:
    target = make_url(url)
    admin = build_engine(target.set(database="postgres").render_as_string(hide_password=False))
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        if not conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target.database}).scalar():
            conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    admin.dispose()
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["sqlalchemy_url"] = url
    command.upgrade(config, "head")
    return build_engine(url)


def build_book(session: Session, passages: int, chapters: int) -> Book:
    book = Book(title=f"[Evaluation] benchmark {uuid.uuid4().hex[:6]}", author="Benchmark")
    document = Document(
        book=book, original_filename="benchmark.pdf", stored_filename=f"{uuid.uuid4()}.pdf", file_path=f"benchmark/{uuid.uuid4()}.pdf",
        file_format="pdf", mime_type="application/pdf", file_size=1, sha256_hash=uuid.uuid4().hex + uuid.uuid4().hex, processing_status="completed",
    )
    session.add_all([book, document])
    session.flush()
    nodes = []
    for number in range(chapters):
        node = StructuralNode(book_id=book.id, document_id=document.id, node_type="chapter", title=f"Chapter {number + 1}", level=0, sequence=number)
        session.add(node)
        nodes.append(node)
    session.flush()
    for index in range(passages):
        node = nodes[index * chapters // passages]
        passage = Passage(
            book_id=book.id, document_id=document.id, structural_node_id=node.id, text=f"Passage {index}. " + TEXT * 4,
            sequence=index, document_sequence=index, page_start=index // 3 + 1, page_end=index // 3 + 1, page_status="exact", token_count=300,
        )
        session.add(passage)
        session.flush()
        if index % 5 == 0:  # one Phase 1 equation in every fifth passage
            equation = Equation(
                document_id=document.id, sequence_number=index, identity_hash=uuid.uuid4().hex + uuid.uuid4().hex[:32], latex="R = e^{-t/S}",
                plain_text="R = e -t/S", equation_type="display", extraction_method="benchmark", page_number=passage.page_start,
                validation_status="valid",
            )
            session.add(equation)
            session.flush()
            passage.passage_metadata = {"equation_ids": [str(equation.id)]}
    session.commit()
    return book


class Scripted:
    """An instant model: one concept per passage, the stored equation read back, one relationship."""

    def __init__(self, config: LlmModelConfig, latency: float) -> None:
        self.config, self.latency, self.calls = config, latency, 0

    def complete(self, request):
        import re

        self.calls += 1
        if self.latency:
            time.sleep(self.latency)
        content = request.messages[0].content
        target = re.search(r'<passage id="([0-9a-f-]+)" (?:pages="[^"]*" )?role="target"', content) or re.search(r'<passage id="([0-9a-f-]+)" role="target"', content)
        if "Knowledge units:" in content:
            return LlmResponse(text=json.dumps({"relationships": [], "notes": None}), model=self.config.model, input_tokens=400, output_tokens=20, finish_reason="stop")
        kind = re.search(r"Extract (\w+) knowledge", content).group(1)
        source = {"passage_id": target.group(1), "role": "primary"}
        if kind == "equation":
            if "e^{-t/S}" not in content and "R = e" not in content:
                items = []
            else:
                items = [{"content": {"knowledge_type": "equation", "name": "Retention", "original_expression": "R = e -t/S", "latex": "R = e^{-t/S}",
                                      "extraction_status": "complete", "reconstruction_status": "not_needed"}, "sources": [source], "basis": "explicit",
                          "confidence": 0.9, "is_uncertain": False}]
        else:
            items = [{"content": {"knowledge_type": "concept", "name": "Retention", "definition": "How well a memory lasts.", "explanation": "Memory fades with time."},
                      "sources": [source], "basis": "explicit", "confidence": 0.9, "is_uncertain": False}]
        return LlmResponse(text=json.dumps({"items": items}), model=self.config.model, input_tokens=900, output_tokens=120, finish_reason="stop")


class QueryCounter:
    def __init__(self, engine: Engine) -> None:
        self.engine, self.statements, self.seconds = engine, Counter(), Counter()
        self.total = 0
        self._started: list[float] = []
        event.listen(engine, "before_cursor_execute", self._before)
        event.listen(engine, "after_cursor_execute", self._after)

    def _before(self, conn, cursor, statement, parameters, context, executemany):
        self._started.append(time.perf_counter())

    def _after(self, conn, cursor, statement, parameters, context, executemany):
        key = " ".join(statement.split())[:90]
        self.total += 1
        self.statements[key] += 1
        self.seconds[key] += time.perf_counter() - self._started.pop()

    def reset(self) -> None:
        self.statements.clear(), self.seconds.clear()
        self.total = 0

    def close(self) -> None:
        event.remove(self.engine, "before_cursor_execute", self._before)
        event.remove(self.engine, "after_cursor_execute", self._after)


def timed(label: str, counter: QueryCounter, function):
    counter.reset()
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    print(f"  {label:<44}{elapsed * 1000:>9.1f} ms{counter.total:>8} queries")
    return result, elapsed, counter.total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the extraction pipeline's own cost, with no model.")
    parser.add_argument("--passages", type=int, default=60)
    parser.add_argument("--chapters", type=int, default=6)
    parser.add_argument("--latency-ms", type=float, default=0.0, help="pretend model latency per call")
    parser.add_argument("--no-relationships", action="store_true")
    parser.add_argument("--drop", action="store_true", help="drop the benchmark database afterwards")
    args = parser.parse_args(argv)

    base = get_settings().database_url.get_secret_value()
    url = benchmark_url(base)
    engine = prepare_database(url)
    counter = QueryCounter(engine)
    config = LlmModelConfig.model_validate({"id": "scripted", "provider": "ollama", "model": "scripted", "deployment": "local", "context_window": 32768, "max_output_tokens": 1024})
    provider = Scripted(config, args.latency_ms / 1000)
    categories = [KnowledgeType.CONCEPT, KnowledgeType.EQUATION]
    try:
        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            book = build_book(session, args.passages, args.chapters)
            service = KnowledgeExtractionService(
                session, ModelRegistry([config], default_id="scripted"), ExtractionSettings(retry_backoff_seconds=0), environ={},
                provider_factory=lambda _c: provider, sleep=lambda _s: None,
            )
            print(f"Benchmark database '{make_url(url).database}': {args.passages} passages, {args.chapters} chapters, categories {[c.value for c in categories]}"
                  f"{'' if args.no_relationships else ' + relationships'}, model latency {args.latency_ms:g} ms\n")
            tracemalloc.start()
            counter.reset()
            started = time.perf_counter()
            run_id = service.create_run(service.book_scope(book.id), categories, model_id="scripted", extract_relationships=not args.no_relationships)
            status = service.execute_run(run_id)
            total = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            run = status.run
            executions = run.passages_processed * len(categories) + (0 if args.no_relationships else run.passages_processed)
            print(f"Extraction run {run.status.value}: {run.units_extracted} units, {run.relationships_extracted} relationships, "
                  f"{run.attempt_count} model calls, {run.duplicate_count} duplicates, needs_review={run.needs_review_count}")
            print(f"  total {total:.2f} s   per passage {1000 * total / args.passages:.1f} ms   per model call {1000 * total / max(1, provider.calls):.1f} ms (model time excluded)")
            print(f"  queries {counter.total}   per passage {counter.total / args.passages:.1f}   per model call {counter.total / max(1, provider.calls):.1f}")
            print(f"  peak Python memory during the run {peak / 1e6:.1f} MB   process max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB (macOS bytes)")
            print("  most frequent statements:")
            for statement, count in counter.statements.most_common(6):
                print(f"    {count:>6} x {counter.seconds[statement] * 1000 / count:>6.2f} ms  {statement}")
            slowest = sorted(counter.statements, key=lambda s: -counter.seconds[s])[:3]
            print("  most database time:")
            for statement in slowest:
                print(f"    {counter.seconds[statement] * 1000:>8.1f} ms in {counter.statements[statement]:>5} x  {statement}")

            print("\nReading the result back (what the Knowledge screen does):")
            knowledge = KnowledgeService(session)
            f = KnowledgeFilter(book_id=book.id)
            _, _, _ = timed("summary (counts)", counter, lambda: knowledge.summary(f))
            page, _, _ = timed("one page of 15 units", counter, lambda: knowledge.list_units(f, page_size=15))
            timed("overview (categories, flags, failures)", counter, lambda: knowledge.overview(f))
            timed("search 'retention' (all units)", counter, lambda: knowledge.list_units(KnowledgeFilter(book_id=book.id, search="retention"), page_size=15))
            durations = []
            for item in page.items:
                _, elapsed, _ = timed(f"detail of one unit ({item.knowledge_type})", counter, lambda item=item: knowledge.detail(item.id))
                durations.append(elapsed)
            print(f"  median detail time {statistics.median(durations) * 1000:.1f} ms")
            timed("resume check on a finished run", counter, lambda: service.get_resume_info(run_id))
            timed("plan an extraction with nothing left", counter, lambda: service.plan(service.book_scope(book.id), categories, model_id="scripted"))
            session.execute(text("DELETE FROM books WHERE id = :id"), {"id": book.id})
            session.commit()
    finally:
        counter.close()
        engine.dispose()
        if args.drop:
            admin = build_engine(make_url(url).set(database="postgres").render_as_string(hide_password=False))
            with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text(f'DROP DATABASE IF EXISTS "{make_url(url).database}" WITH (FORCE)'))
            admin.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
