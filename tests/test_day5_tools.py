"""The Day 5 quality tools keep working: the benchmark's scripted model and the migration check's fingerprints."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.extraction.model_config import LlmModelConfig, ModelRegistry
from app.extraction.settings import ExtractionSettings
from app.knowledge.types import KnowledgeType
from app.services.knowledge_extraction_service import KnowledgeExtractionService
from database.models import Book, EquationSource, KnowledgeUnit
from tools import benchmark_extraction as bench
from tools import check_migrations_on_copy as migration_check
from tests.test_models import make_book


def test_the_benchmark_book_and_scripted_model_run_through_the_real_pipeline(session: Session):
    book = bench.build_book(session, passages=10, chapters=2)
    config = LlmModelConfig.model_validate({"id": "scripted", "provider": "ollama", "model": "scripted", "deployment": "local", "context_window": 32768, "max_output_tokens": 1024})
    provider = bench.Scripted(config, latency=0)
    service = KnowledgeExtractionService(
        session, ModelRegistry([config], default_id="scripted"), ExtractionSettings(retry_backoff_seconds=0), environ={},
        provider_factory=lambda _c: provider, sleep=lambda _s: None,
    )

    status = service.extract_book(book.id, [KnowledgeType.CONCEPT, KnowledgeType.EQUATION], model_id="scripted", extract_relationships=True)

    assert status.status == "completed" and status.run.attempt_count == provider.calls == 30
    assert session.scalar(select(func.count()).select_from(KnowledgeUnit).where(KnowledgeUnit.knowledge_type == "concept")) == 10
    assert session.scalar(select(func.count()).select_from(EquationSource)) == 2, "one stored equation in every fifth passage, each captured from its source"
    assert bench.benchmark_url("postgresql+psycopg://u:p@h/book2brain").endswith("/book2brain_benchmark")


def test_the_migration_check_fingerprints_change_when_data_changes(engine, session: Session):
    first = migration_check.fingerprint(engine)
    assert first == migration_check.fingerprint(engine) and "books" in first and "alembic_version" not in first

    session.add(make_book("Changed"))
    session.commit()
    second = migration_check.fingerprint(engine)

    assert second["books"][0] == first["books"][0] + 1 and second["books"][1:] != first["books"][1:]
    assert {t: v for t, v in second.items() if t != "books"} == {t: v for t, v in first.items() if t != "books"}
    assert session.scalar(select(func.count()).select_from(Book)) == 1
