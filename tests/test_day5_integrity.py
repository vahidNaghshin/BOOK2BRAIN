"""Day 5: the integrity contract of the whole Phase 2 schema, checked against the real migrated database.

Individual constraints are tested next to the models (test_knowledge_models, test_day4_models). This file
pins the contract that spans all of them: what happens to every dependent row when something is deleted.
"""

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.engine import Engine

from tests.test_day5_integration import book, book_items, knowledge_answer, passages_of  # noqa: F401  (the processed-book fixture)
from tests.extraction_db_helpers import make_service

PHASE_2_TABLES = (
    "knowledge_units", "knowledge_unit_passages", "knowledge_relationships", "knowledge_relationship_passages", "extraction_runs",
    "extraction_executions", "extraction_attempts", "extraction_passage_records", "review_flags", "equation_sources", "equation_reconstructions",
)

# (table, constrained columns) -> ON DELETE rule of the parent. Rows that are only meaningful with their parent
# cascade; references to things that can be regenerated (passages, structure, runs) are cleared, never the knowledge.
FOREIGN_KEYS = {
    ("equation_reconstructions", ("knowledge_unit_id",)): "CASCADE",
    ("equation_sources", ("equation_id",)): "SET NULL",
    ("equation_sources", ("knowledge_unit_id",)): "CASCADE",
    ("equation_sources", ("passage_id",)): "SET NULL",
    ("equation_sources", ("structural_node_id",)): "SET NULL",
    ("extraction_attempts", ("execution_id",)): "CASCADE",
    ("extraction_attempts", ("extraction_run_id",)): "CASCADE",
    ("extraction_attempts", ("passage_id",)): "SET NULL",
    ("extraction_executions", ("extraction_run_id",)): "CASCADE",
    ("extraction_executions", ("passage_id",)): "SET NULL",
    ("extraction_passage_records", ("extraction_run_id",)): "CASCADE",
    ("extraction_passage_records", ("passage_id",)): "SET NULL",
    ("extraction_runs", ("book_id",)): "CASCADE",
    ("extraction_runs", ("document_id",)): "CASCADE",
    ("extraction_runs", ("structural_node_id",)): "SET NULL",
    ("knowledge_relationship_passages", ("passage_id",)): "CASCADE",
    ("knowledge_relationship_passages", ("relationship_id",)): "CASCADE",
    ("knowledge_relationships", ("book_id", "source_unit_id")): "CASCADE",
    ("knowledge_relationships", ("book_id", "target_unit_id")): "CASCADE",
    ("knowledge_relationships", ("extraction_execution_id",)): "SET NULL",
    ("knowledge_relationships", ("extraction_run_id",)): "SET NULL",
    ("knowledge_unit_passages", ("knowledge_unit_id",)): "CASCADE",
    ("knowledge_unit_passages", ("passage_id",)): "CASCADE",
    ("knowledge_unit_passages", ("structural_node_id",)): "SET NULL",
    ("knowledge_units", ("book_id",)): "CASCADE",
    ("knowledge_units", ("chapter_node_id",)): "SET NULL",
    ("knowledge_units", ("document_id",)): "CASCADE",
    ("knowledge_units", ("equation_id",)): "SET NULL",
    ("knowledge_units", ("extraction_execution_id",)): "SET NULL",
    ("knowledge_units", ("extraction_run_id",)): "SET NULL",
    ("knowledge_units", ("section_node_id",)): "SET NULL",
    ("review_flags", ("book_id",)): "CASCADE",
    ("review_flags", ("extraction_attempt_id",)): "CASCADE",
    ("review_flags", ("extraction_run_id",)): "SET NULL",
    ("review_flags", ("knowledge_unit_id",)): "CASCADE",
    ("review_flags", ("relationship_id",)): "CASCADE",
}


def actual_foreign_keys(engine: Engine) -> dict:
    inspector = inspect(engine)
    found = {}
    for table in PHASE_2_TABLES:
        for fk in inspector.get_foreign_keys(table):
            found[(table, tuple(sorted(fk["constrained_columns"])))] = (fk.get("options") or {}).get("ondelete", "NO ACTION").upper()
    return found


def test_every_foreign_key_of_phase_2_has_the_documented_delete_rule(engine):
    actual = actual_foreign_keys(engine)
    expected = {(t, tuple(sorted(c))): rule for (t, c), rule in FOREIGN_KEYS.items()}

    assert actual == expected, "a foreign key was added, removed or changed: decide its delete rule and record it here"


def test_no_foreign_key_of_phase_2_can_block_a_delete(engine):
    assert not [key for key, rule in actual_foreign_keys(engine).items() if rule in ("NO ACTION", "RESTRICT")]


def counts(session) -> dict[str, int]:
    session.rollback()
    return {table: session.scalar(text(f'SELECT count(*) FROM "{table}"')) for table in PHASE_2_TABLES}


def test_deleting_a_document_deletes_its_knowledge_and_deleting_the_book_deletes_the_run_history(session, book):  # noqa: F811
    service, _ = make_service(session, knowledge_answer(session))
    service.extract_book(book.book_id, ["concept", "equation"], extract_relationships=True)
    populated = counts(session)
    assert all(populated[t] > 0 for t in ("knowledge_units", "knowledge_unit_passages", "extraction_runs", "extraction_executions",
                                          "extraction_attempts", "extraction_passage_records", "equation_sources", "review_flags")), populated

    session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": book.id})
    session.commit()

    knowledge = ("knowledge_units", "knowledge_unit_passages", "knowledge_relationships", "knowledge_relationship_passages", "review_flags",
                 "equation_sources", "equation_reconstructions")
    left = counts(session)
    assert {t: left[t] for t in knowledge} == dict.fromkeys(knowledge, 0), "everything extracted from the document is gone"
    assert left["extraction_runs"] == 1 and left["extraction_executions"] == populated["extraction_executions"], \
        "a book-wide run is the book's history: it stays, with its passage references cleared, until the book is deleted"
    assert session.scalar(text("SELECT count(*) FROM extraction_executions WHERE passage_id IS NOT NULL")) == 0

    session.execute(text("DELETE FROM books WHERE id = :id"), {"id": book.book_id})
    session.commit()
    assert {t: n for t, n in counts(session).items() if n} == {}, "deleting the book leaves no Phase 2 row behind"


def test_deleting_a_run_keeps_the_knowledge_and_regenerating_passages_keeps_the_runs_history(session, book):  # noqa: F811
    service, _ = make_service(session, knowledge_answer(session))
    run = service.extract_book(book.book_id, ["concept"])
    units = counts(session)["knowledge_units"]

    session.execute(text("DELETE FROM extraction_runs WHERE id = :id"), {"id": run.run.id})
    session.commit()
    after = counts(session)

    assert after["knowledge_units"] == units and after["extraction_executions"] == after["extraction_attempts"] == 0
    assert session.scalar(text("SELECT count(*) FROM knowledge_units WHERE extraction_run_id IS NOT NULL")) == 0, "the units no longer point at a run that is gone"
    assert after["knowledge_unit_passages"] == units, "and each unit keeps its source"
