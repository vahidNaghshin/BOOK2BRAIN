"""The background job runner: validation happens before the worker starts, the work on the worker."""

from contextlib import contextmanager

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.services.errors import ExtractionRunError
from app.services.extraction_job_runner import ExtractionJobRunner
from app.services.knowledge_extraction_service import NothingToExtractError
from database.models import ExtractionRun, KnowledgeUnit
from tests.extraction_db_helpers import make_service, one_item, store_library
from tests.extraction_helpers import make_config


@pytest.fixture
def runner_for(engine: Engine):
    runners = []

    def build(answer=one_item, **options):
        @contextmanager
        def scope():
            local = Session(bind=engine, autoflush=False, expire_on_commit=False)
            try:
                yield local
            finally:
                local.close()

        runner = ExtractionJobRunner(session_factory=scope, service_factory=lambda s: make_service(s, answer, **options)[0])
        runners.append(runner)
        return runner

    yield build
    for runner in runners:
        runner.shutdown()


def test_a_run_is_recorded_at_once_and_executed_on_the_worker(session, runner_for):
    library = store_library(session)
    runner = runner_for()
    scope = make_service(session)[0].chapter_scope(library.nodes["ch1"].id)
    run_id = runner.start(scope, ["concept"], model_id="local-test")
    runner.shutdown(wait=True)

    session.expire_all()
    run = session.get(ExtractionRun, run_id)
    assert (run.status, run.units_extracted, run.passages_processed, run.model_configuration_id) == ("completed", 5, 5, "local-test")
    assert session.scalars(select(KnowledgeUnit)).all().__len__() == 5


def test_problems_are_reported_to_the_caller_before_anything_is_queued(session, runner_for):
    library = store_library(session)
    runner = runner_for(configs=[make_config(), make_config(id="off", active=False)])
    scope = make_service(session)[0].book_scope(library.book.id)
    with pytest.raises(ExtractionRunError, match="inactive"):
        runner.start(scope, ["concept"], model_id="off")
    runner.start(scope, ["concept"], model_id="local-test", limit=2)
    runner.shutdown(wait=True)
    with pytest.raises(NothingToExtractError):
        runner_for(configs=[make_config(), make_config(id="off", active=False)]).start(scope, ["concept"], limit=2)


def test_a_crash_on_the_worker_is_logged_and_does_not_break_the_runner(session, runner_for, caplog):
    library = store_library(session)

    def exploding(call):
        raise SystemError("worker bug")

    runner = runner_for(exploding)
    scope = make_service(session)[0].chapter_scope(library.nodes["ch2"].id)
    run_id = runner.start(scope, ["concept"])
    runner.shutdown(wait=True)
    session.expire_all()
    run = session.get(ExtractionRun, run_id)
    assert run.status in ("failed", "needs_review") and run.failure_count == 2  # recorded per passage; the worker survived
