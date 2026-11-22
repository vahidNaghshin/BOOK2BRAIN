"""Database rules for extraction runs, executions and attempts."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import ExtractionAttempt, ExtractionExecution, ExtractionRun
from tests.extraction_db_helpers import store_library

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def run_for(library, **fields) -> ExtractionRun:
    return ExtractionRun(book_id=library.book.id, **fields)


def execution_for(run, **fields) -> ExtractionExecution:
    values = {"extraction_run_id": run.id, "knowledge_type": "concept", "status": "succeeded", "started_at": NOW, "completed_at": NOW}
    return ExtractionExecution(**(values | fields))


def test_runs_can_end_as_needs_review_and_carry_progress_and_model_data(session: Session):
    library = store_library(session)
    run = run_for(
        library, status="needs_review", scope_type="chapter", model_configuration_id="m", provider="ollama",
        model_snapshot={"id": "m"}, total_passages=5, needs_review_count=2, estimated_cost_usd=0.5, input_tokens=10,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    assert (run.status, run.attempt_count, run.duplicate_count, run.skipped_count, run.output_tokens) == ("needs_review", 0, 0, 0, None)


@pytest.mark.parametrize("fields", [{"status": "paused"}, {"scope_type": "paragraph"}, {"total_passages": -1}, {"retry_count": -1}, {"duplicate_count": -2}])
def test_invalid_run_values_are_rejected(session: Session, fields):
    library = store_library(session)
    session.add(run_for(library, **fields))
    with pytest.raises(IntegrityError):
        session.commit()


def test_executions_must_be_consistent(session: Session):
    library = store_library(session)
    run = run_for(library)
    session.add(run)
    session.commit()
    for bad in (
        {"status": "unknown"}, {"knowledge_type": "poem"}, {"attempt_count": -1},
        {"completed_at": NOW - timedelta(seconds=1)}, {"status": "failed", "needs_review": False},
    ):
        session.add(execution_for(run, **bad))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    session.add(execution_for(run, status="failed", needs_review=True))
    session.commit()


def test_attempts_are_numbered_once_per_execution(session: Session):
    library = store_library(session)
    run = run_for(library)
    session.add(run)
    session.flush()
    execution = execution_for(run)
    session.add(execution)
    session.flush()

    def attempt(number, **fields):
        values = dict(
            execution_id=execution.id, attempt_number=number, kind="initial", outcome="valid", model_configuration_id="m", provider="p",
            model_id="x", prompt_name="n", prompt_version="1", started_at=NOW, completed_at=NOW,
        )
        return ExtractionAttempt(**(values | fields))

    session.add_all([attempt(1), attempt(2, kind="correction")])
    session.commit()
    for bad in (attempt(1), attempt(3, outcome="maybe"), attempt(4, kind="again"), attempt(0)):
        session.add(bad)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    assert uuid.UUID(str(execution.id))
