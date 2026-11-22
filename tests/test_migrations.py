from collections.abc import Iterator

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from database.connection import build_engine
from app.knowledge.records import unit_to_record
from database.models import Base, Book, Document, KnowledgeUnit, Passage, StructuralNode
from tests.conftest import alembic_config, derive_test_url, ensure_database, migrate

EXPECTED_TABLES = {
    "books", "documents", "processing_runs", "processing_errors", "extracted_blocks", "equations",
    "structural_nodes", "paragraphs", "passages", "passage_paragraphs", "structure_analyses",
    "extraction_runs", "extraction_executions", "extraction_attempts", "knowledge_units", "knowledge_unit_passages", "knowledge_relationships",
    "knowledge_relationship_passages",
}
KNOWLEDGE_TABLES = {t for t in EXPECTED_TABLES if t.startswith(("knowledge_", "extraction_"))}
PHASE_1_HEAD = "e33c9630cffb"  # the last revision before the knowledge tables


@pytest.fixture
def migration_engine() -> Iterator[Engine]:
    """A throwaway database so downgrade tests never disturb the shared test database."""
    url = derive_test_url("_test_migrations")
    ensure_database(url)
    migration_engine = build_engine(url)
    command.downgrade(alembic_config(url), "base")
    yield migration_engine
    migration_engine.dispose()


def test_upgrade_creates_schema_and_vector_extension(migration_engine: Engine) -> None:
    migrate(str(migration_engine.url.render_as_string(hide_password=False)))

    assert EXPECTED_TABLES <= set(inspect(migration_engine).get_table_names())
    with migration_engine.connect() as conn:
        extension = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
    assert extension == 1


def test_migrations_match_models(migration_engine: Engine) -> None:
    migrate(migration_engine.url.render_as_string(hide_password=False))

    with migration_engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)

    assert diff == []


def test_downgrade_removes_tables_and_upgrade_is_repeatable(migration_engine: Engine) -> None:
    url = migration_engine.url.render_as_string(hide_password=False)
    migrate(url)
    command.downgrade(alembic_config(url), "base")
    assert not (EXPECTED_TABLES & set(inspect(migration_engine).get_table_names()))

    migrate(url)
    assert EXPECTED_TABLES <= set(inspect(migration_engine).get_table_names())


def _phase_1_data(engine: Engine) -> dict[str, int]:
    """Store a book with one document, one section and one passage; return the row counts."""
    from tests.test_models import make_book, make_document

    with Session(engine) as session:
        book = make_book("Migrated Book")
        document = make_document(book)
        session.add_all([book, document])
        session.flush()
        node = StructuralNode(
            book_id=book.id, document_id=document.id, node_type="chapter", title="One", level=0, sequence=0
        )
        session.add(node)
        session.flush()
        session.add(
            Passage(
                book_id=book.id, document_id=document.id, structural_node_id=node.id, text="Kept text.", sequence=0,
                document_sequence=0, token_count=2,
            )
        )
        session.commit()
    return {"books": 1, "documents": 1, "structural_nodes": 1, "passages": 1}


def _counts(engine: Engine, tables: dict[str, int]) -> dict[str, int]:
    with engine.connect() as conn:
        return {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in tables}


def test_knowledge_migration_preserves_existing_records(migration_engine: Engine) -> None:
    url = migration_engine.url.render_as_string(hide_password=False)
    migrate(url, PHASE_1_HEAD)
    assert not (KNOWLEDGE_TABLES & set(inspect(migration_engine).get_table_names()))
    expected = _phase_1_data(migration_engine)

    migrate(url)

    assert KNOWLEDGE_TABLES <= set(inspect(migration_engine).get_table_names())
    assert _counts(migration_engine, expected) == expected
    with Session(migration_engine) as session:
        assert session.scalar(select(Book.title)) == "Migrated Book"
        assert session.scalar(select(Passage.text)) == "Kept text."
        assert session.scalars(select(KnowledgeUnit)).all() == []


def test_knowledge_migration_rolls_back_without_touching_phase_1_data(migration_engine: Engine) -> None:
    from tests.knowledge_helpers import concept_data
    from app.knowledge.schemas import KnowledgeUnitCreate

    url = migration_engine.url.render_as_string(hide_password=False)
    migrate(url)
    expected = _phase_1_data(migration_engine)
    with Session(migration_engine) as session:
        book, document, passage = session.scalar(select(Book)), session.scalar(select(Document)), session.scalar(select(Passage))
        session.add(
            unit_to_record(
                KnowledgeUnitCreate(
                    book_id=book.id, document_id=document.id, content=concept_data(),
                    sources=[{"passage_id": passage.id}],
                )
            )
        )
        session.commit()

    command.downgrade(alembic_config(url), PHASE_1_HEAD)

    assert not (KNOWLEDGE_TABLES & set(inspect(migration_engine).get_table_names()))
    assert _counts(migration_engine, expected) == expected

    migrate(url)  # and forward again
    assert KNOWLEDGE_TABLES <= set(inspect(migration_engine).get_table_names())
    assert _counts(migration_engine, expected) == expected


def test_knowledge_foreign_keys_and_indexes(migration_engine: Engine) -> None:
    migrate(migration_engine.url.render_as_string(hide_password=False))
    inspector = inspect(migration_engine)

    def rules(table: str) -> dict[tuple[tuple[str, ...], str], str | None]:
        return {
            (tuple(fk["constrained_columns"]), fk["referred_table"]): fk["options"].get("ondelete")
            for fk in inspector.get_foreign_keys(table)
        }

    units = rules("knowledge_units")
    assert units[(("book_id",), "books")] == "CASCADE"
    assert units[(("document_id",), "documents")] == "CASCADE"
    assert units[(("extraction_run_id",), "extraction_runs")] == "SET NULL"
    assert units[(("chapter_node_id",), "structural_nodes")] == "SET NULL"
    assert units[(("section_node_id",), "structural_nodes")] == "SET NULL"
    assert units[(("equation_id",), "equations")] == "SET NULL"
    links = rules("knowledge_unit_passages")
    assert links[(("knowledge_unit_id",), "knowledge_units")] == "CASCADE"
    assert links[(("passage_id",), "passages")] == "CASCADE"
    edges = rules("knowledge_relationships")
    assert edges[(("source_unit_id", "book_id"), "knowledge_units")] == "CASCADE"
    assert edges[(("target_unit_id", "book_id"), "knowledge_units")] == "CASCADE"
    runs = rules("extraction_runs")
    assert runs[(("book_id",), "books")] == "CASCADE"
    assert runs[(("document_id",), "documents")] == "CASCADE"
    assert runs[(("structural_node_id",), "structural_nodes")] == "SET NULL"

    indexed = {tuple(i["column_names"]) for i in inspector.get_indexes("knowledge_units")}
    assert {("book_id", "knowledge_type"), ("document_id",), ("extraction_run_id",)} <= indexed
    assert ("passage_id",) in {tuple(i["column_names"]) for i in inspector.get_indexes("knowledge_unit_passages")}
    assert ("target_unit_id",) in {tuple(i["column_names"]) for i in inspector.get_indexes("knowledge_relationships")}
