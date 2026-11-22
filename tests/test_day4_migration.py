"""The Day 4 migration keeps existing data valid and fills in what can be derived for it."""

import json
import uuid
from collections.abc import Iterator

import pytest
from alembic import command
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from database.connection import build_engine
from database.models import Passage, StructuralNode
from tests.conftest import alembic_config, derive_test_url, ensure_database, migrate
from tests.test_models import make_book, make_document

BEFORE = "b9221d0781db"  # the last revision before Day 4


@pytest.fixture
def old_database() -> Iterator[Engine]:
    url = derive_test_url("_test_migrations_day4")
    ensure_database(url)
    engine = build_engine(url)
    command.downgrade(alembic_config(url), "base")
    migrate(url, BEFORE)
    yield engine
    engine.dispose()


def seed(engine: Engine) -> dict[str, uuid.UUID]:
    """Phase 1 data through the ORM (unchanged tables), extraction data through SQL (changed tables)."""
    ids = {k: uuid.uuid4() for k in ("run", "ex_ok", "ex_bad", "unit", "a_valid", "a_invalid", "a_error", "a_bad")}
    with Session(engine) as session:
        book = make_book("Old Book")
        document = make_document(book)
        session.add_all([book, document])
        session.flush()
        node = StructuralNode(book_id=book.id, document_id=document.id, node_type="chapter", title="One", level=0, sequence=0)
        session.add(node)
        session.flush()
        passages = [
            Passage(
                book_id=book.id, document_id=document.id, structural_node_id=node.id, text=f"Entropy is disorder number {i}.", sequence=i,
                document_sequence=i, page_start=i + 1, page_end=i + 1, page_status="exact", token_count=5,
            )
            for i in range(2)
        ]
        session.add_all(passages)
        session.commit()
        ids.update(book=book.id, document=document.id, node=node.id, p1=passages[0].id, p2=passages[1].id)

        snapshot = json.dumps({"id": "m1", "model": "x"})
        session.execute(text("INSERT INTO extraction_runs (id, book_id, status, passages_processed, units_extracted, failure_count) VALUES (:r, :b, 'completed', 2, 1, 1)"), {"r": ids["run"], "b": book.id})
        execution = (
            "INSERT INTO extraction_executions (id, extraction_run_id, passage_id, knowledge_type, status, needs_review, model_configuration_id, provider, model_id, "
            "model_configuration, prompt_name, prompt_version, started_at, completed_at, attempt_count) "
            "VALUES (:id, :r, :p, 'concept', :status, :nr, 'm1', 'ollama', 'x', CAST(:snap AS jsonb), 'concept_extraction', '1.0.0', now(), now(), :n)"
        )
        session.execute(text(execution), {"id": ids["ex_ok"], "r": ids["run"], "p": passages[0].id, "status": "needs_review", "nr": True, "snap": snapshot, "n": 2})
        session.execute(text(execution), {"id": ids["ex_bad"], "r": ids["run"], "p": passages[1].id, "status": "failed", "nr": True, "snap": snapshot, "n": 1})
        attempt = (
            "INSERT INTO extraction_attempts (id, execution_id, attempt_number, kind, outcome, model_configuration_id, provider, model_id, prompt_name, prompt_version, started_at, completed_at) "
            "VALUES (:id, :e, :n, :kind, :outcome, 'm1', 'ollama', 'x', 'concept_extraction', '1.0.0', now(), now())"
        )
        for key, execution_id, n, kind, outcome in (("a_invalid", "ex_ok", 1, "initial", "invalid"), ("a_valid", "ex_ok", 2, "correction", "valid"), ("a_error", "ex_bad", 1, "initial", "provider_error")):
            session.execute(text(attempt), {"id": ids[key], "e": ids[execution_id], "n": n, "kind": kind, "outcome": outcome})
        session.execute(
            text("INSERT INTO knowledge_units (id, book_id, document_id, extraction_run_id, knowledge_type, title, content, details, page_start, page_end) VALUES (:id, :b, :d, :r, 'concept', 'Entropy', 'Disorder.', CAST(:details AS jsonb), 1, 1)"),
            {"id": ids["unit"], "b": book.id, "d": document.id, "r": ids["run"], "details": json.dumps({"name": "Entropy", "definition": "Disorder."})},
        )
        session.execute(
            text("INSERT INTO knowledge_unit_passages (knowledge_unit_id, passage_id, position, role, quote) VALUES (:u, :p, 0, 'primary', 'Entropy is disorder')"),
            {"u": ids["unit"], "p": passages[0].id},
        )
        session.commit()
    return ids


def test_existing_data_survives_and_is_backfilled(old_database: Engine):
    ids = seed(old_database)
    url = old_database.url.render_as_string(hide_password=False)
    migrate(url)

    with old_database.connect() as conn:
        def one(sql, **params):
            return conn.execute(text(sql), params).mappings().one()

        # attempts gain their run, passage, status, validation outcome and the configuration snapshot
        valid, invalid, error = (one("SELECT * FROM extraction_attempts WHERE id = :i", i=ids[k]) for k in ("a_valid", "a_invalid", "a_error"))
        assert (valid["extraction_run_id"], valid["passage_id"]) == (ids["run"], ids["p1"])
        assert (valid["status"], valid["validation_outcome"], valid["validation_outcomes"]) == ("completed", "schema_valid", ["schema_valid"])
        assert (invalid["status"], invalid["validation_outcome"]) == ("completed", "schema_invalid")
        assert (error["status"], error["validation_outcome"], error["validation_outcomes"], error["passage_id"]) == ("provider_error", "not_validated", [], ids["p2"])
        assert valid["model_configuration"] == {"id": "m1", "model": "x"}

        # the unit keeps its content and gains origin, source status, link details and a snapshot of its source
        unit = one("SELECT * FROM knowledge_units WHERE id = :i", i=ids["unit"])
        assert (unit["title"], unit["origin"], unit["source_status"], unit["details"]["name"]) == ("Entropy", "ai_extracted", "unverified", "Entropy")
        [snap] = unit["source_snapshot"]
        assert snap["passage_id"] == str(ids["p1"]) and snap["excerpt"] == "Entropy is disorder" and snap["excerpt_is_quote"] is True and snap["pages"] == [1, 1]
        link = one("SELECT * FROM knowledge_unit_passages WHERE knowledge_unit_id = :i", i=ids["unit"])
        assert (link["page_start"], link["structural_node_id"], link["reference_type"], link["verification_status"]) == (1, ids["node"], "text", "unverified")

        # passages that were already processed get their records
        rows = conn.execute(text("SELECT passage_id, status, execution_count, attempt_count FROM extraction_passage_records WHERE extraction_run_id = :r"), {"r": ids["run"]}).all()
        assert {r[0]: (r[1], r[2], r[3]) for r in rows} == {ids["p1"]: ("needs_review", 1, 2), ids["p2"]: ("failed", 1, 1)}
        assert conn.execute(text("SELECT total_passages, relationships_extracted FROM extraction_runs")).one() == (0, 0)


def test_the_new_tables_and_columns_exist_and_the_old_data_can_be_read_by_the_models(old_database: Engine):
    ids = seed(old_database)
    migrate(old_database.url.render_as_string(hide_password=False))
    tables = set(inspect(old_database).get_table_names())
    assert {"review_flags", "equation_sources", "equation_reconstructions", "extraction_passage_records"} <= tables
    from database.models import KnowledgeUnit

    with Session(old_database) as session:
        unit = session.get(KnowledgeUnit, ids["unit"])
        assert unit.knowledge_type == "concept" and unit.sources[0].quote == "Entropy is disorder"


def test_downgrading_removes_only_what_the_old_schema_cannot_hold(old_database: Engine):
    ids = seed(old_database)
    url = old_database.url.render_as_string(hide_password=False)
    migrate(url)
    with Session(old_database) as session:
        session.execute(text("INSERT INTO knowledge_units (id, book_id, document_id, knowledge_type, title, content, details) VALUES (:id, :b, :d, 'principle', 'P', 'p', '{\"statement\": \"p\"}')"), {"id": ids["a_bad"], "b": ids["book"], "d": ids["document"]})
        session.execute(text("INSERT INTO knowledge_relationships (id, book_id, source_unit_id, target_unit_id, relationship_type) VALUES (gen_random_uuid(), :b, :s, :t, 'derived_from')"), {"b": ids["book"], "s": ids["a_bad"], "t": ids["unit"]})
        session.execute(text("INSERT INTO knowledge_relationships (id, book_id, source_unit_id, target_unit_id, relationship_type) VALUES (gen_random_uuid(), :b, :s, :t, 'depends_on')"), {"b": ids["book"], "s": ids["a_bad"], "t": ids["unit"]})
        session.commit()
    command.downgrade(alembic_config(url), BEFORE)
    with old_database.connect() as conn:
        assert conn.execute(text("SELECT relationship_type FROM knowledge_relationships")).scalars().all() == ["depends_on"]
        assert conn.execute(text("SELECT count(*) FROM knowledge_units")).scalar() == 2  # no knowledge was lost
        assert "review_flags" not in inspect(old_database).get_table_names()
    migrate(url)  # and it can go forward again
