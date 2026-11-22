"""Resumable processing: per-passage status, recovering failed and interrupted runs, no repeated work, no duplicates."""

from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.extraction.errors import ProviderTimeoutError
from app.services import knowledge_extraction_service as pipeline
from app.services.errors import ExtractionRunError
from database.models import (
    ExtractionAttempt,
    ExtractionExecution,
    ExtractionPassageRecord,
    ExtractionRun,
    KnowledgeRelationship,
    KnowledgeUnit,
)
from tests.extraction_db_helpers import (
    Library,
    item,
    make_service,
    one_item,
    only_provider,
    relationship,
    relationships_reply,
    reply,
    store_library,
    with_relationships,
)

QUOTE = "Entropy is a measure of the disorder of a system."
CHAPTER_PASSAGES = 5  # chapter ch1: c1a, c1b, s1a, s1b, ss1a


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


def count(session, model) -> int:
    return session.scalar(select(func.count()).select_from(model))


def records(session, run_id):
    return list(session.scalars(select(ExtractionPassageRecord).where(ExtractionPassageRecord.extraction_run_id == run_id).order_by(ExtractionPassageRecord.document_sequence)))


def statuses(session, run_id):
    return [r.status for r in records(session, run_id)]


def new_run(service, library, categories=("concept",), **options):
    return service.create_run(service.chapter_scope(library.nodes["ch1"].id), list(categories), **options)


class Switch:
    """A provider answer that fails for the passages in `failing` while `on` is true."""

    def __init__(self, library, *names, on=True, error=None):
        self.ids = {str(library.passages[n].id) for n in names}
        self.on, self.error = on, error or ProviderTimeoutError("slow")

    def __call__(self, call):
        if self.on and call.target_id in self.ids:
            return self.error
        return one_item(call)


# ---------------------------------------------------------------- passage records and their statuses


def test_every_selected_passage_is_recorded_as_pending_when_the_run_is_created(session, library):
    service, _ = make_service(session)
    run_id = new_run(service, library)
    rows = records(session, run_id)
    assert [r.passage_id for r in rows] == [p.id for p in library.ordered if p.document_sequence in (1, 2, 3, 4, 5)]
    assert {r.status for r in rows} == {"pending"} and [r.document_sequence for r in rows] == [1, 2, 3, 4, 5]
    assert all(r.execution_count == r.attempt_count == r.units_extracted == 0 and r.started_at is None for r in rows)
    status = service.get_run_status(run_id)
    assert status.status == "pending" and status.passage_counts == {"pending": CHAPTER_PASSAGES}
    info = service.get_resume_info(run_id)
    assert (info.pending, info.completed, info.failed, info.can_resume, info.total) == (CHAPTER_PASSAGES, 0, 0, False, CHAPTER_PASSAGES)
    assert info.reason == "The run has not started yet."


def test_a_limited_run_records_only_the_passages_it_will_process(session, library):
    service, _ = make_service(session)
    run_id = service.create_run(service.book_scope(library.book.id), ["concept"], limit=3, offset=2)
    assert [r.document_sequence for r in records(session, run_id)] == [2, 3, 4]


def test_a_passage_is_processing_while_it_is_worked_on_and_completed_after(session, library):
    service, _ = make_service(session)
    run_id = new_run(service, library)
    seen = []
    service.execute_run(run_id, progress=lambda event: seen.append(statuses(session, run_id)))
    session.expire_all()
    assert seen[0] == ["processing", "pending", "pending", "pending", "pending"]
    assert seen[2] == ["completed", "completed", "processing", "pending", "pending"]
    assert statuses(session, run_id) == ["completed"] * CHAPTER_PASSAGES


def test_a_finished_passage_records_its_timestamps_counts_and_units(session, library):
    service, _ = make_service(session)
    run_id = new_run(service, library, ("concept", "principle"))
    service.execute_run(run_id)
    session.expire_all()
    for record in records(session, run_id):
        assert record.started_at <= record.completed_at and record.error_code is None
        assert (record.execution_count, record.attempt_count, record.retry_count, record.validation_failures, record.units_extracted) == (2, 2, 0, 0, 2)
    view = service.get_passage_status(run_id)
    assert [v.sequence for v in view] == [1, 2, 3, 4, 5] and [v.page_start for v in view] == [2, 3, 4, 5, 6]
    assert view[0].categories == {"concept": "succeeded", "principle": "succeeded"} and view[0].units_extracted == 2


def test_a_passage_with_findings_needs_review_and_one_that_failed_says_why(session, library):
    failing = str(library.passages["c1b"].id)

    def answer(call):
        if call.target_id == failing:
            return ProviderTimeoutError("slow")
        if call.target_id == str(library.passages["s1a"].id):
            return reply(item(call, is_uncertain=True, uncertainty_notes="Implied."))
        return reply(item(call))

    service, _ = make_service(session, answer, max_retries=0)
    run_id = new_run(service, library)
    status = service.execute_run(run_id)
    by_sequence = {r.document_sequence: r for r in records(session, run_id)}
    assert [by_sequence[n].status for n in (1, 2, 3, 4, 5)] == ["completed", "failed", "needs_review", "completed", "completed"]
    failed = by_sequence[2]
    assert (failed.error_code, failed.error_message) == ("provider_timeout", "slow") and failed.attempt_count == 1 and failed.units_extracted == 0
    assert status.status == "needs_review" and status.failed_passages == 1 and status.passage_counts == {"completed": 3, "failed": 1, "needs_review": 1}
    assert {v.status for v in service.get_passage_status(run_id)} == {"completed", "failed", "needs_review"}
    assert service.get_passage_status(run_id)[1].categories == {"concept": "failed"}


# ---------------------------------------------------------------- when a run can be resumed


def test_what_can_and_cannot_be_resumed(session, library, monkeypatch):
    service, _ = make_service(session)
    run_id = new_run(service, library)
    assert not service.get_resume_info(run_id).can_resume  # not started

    run = session.get(ExtractionRun, run_id)
    run.status = "processing"
    session.commit()
    info = service.get_resume_info(run_id)
    assert not info.can_resume and info.reason == "The run is still being processed."
    with pytest.raises(ExtractionRunError, match="still being processed"):
        service.resume_run(run_id)
    with pytest.raises(ExtractionRunError, match="still being processed"):
        service.execute_run(run_id)

    monkeypatch.setattr(pipeline, "ACTIVE_RUN_TIMEOUT", timedelta(seconds=-1))  # it has been silent for too long
    assert service.get_resume_info(run_id).can_resume
    monkeypatch.undo()

    service.execute_run(run_id, takeover=True)  # or the user says it is dead
    assert service.get_run_status(run_id).status == "completed"
    info = service.get_resume_info(run_id)
    assert not info.can_resume and info.reason == "Every passage has finished; there is nothing to resume."
    with pytest.raises(ExtractionRunError, match="nothing to resume"):
        service.resume_run(run_id)
    with pytest.raises(ExtractionRunError, match="already finished"):
        service.execute_run(run_id)


# ---------------------------------------------------------------- resuming


def test_resuming_retries_only_what_failed_and_keeps_every_earlier_attempt(session, library):
    switch = Switch(library, "c1b", "s1b")
    service, providers = make_service(session, switch, max_retries=0)
    run_id = new_run(service, library)
    first = service.execute_run(run_id)
    assert first.status == "needs_review" and first.failed_passages == 2 and first.can_resume
    done_before = {u.id for u in session.scalars(select(KnowledgeUnit))}
    calls_before = len(only_provider(providers).requests)
    info = service.get_resume_info(run_id)
    assert (info.completed, info.failed, info.pending, info.can_resume) == (3, 2, 0, True)

    switch.on = False
    finished = service.resume_run(run_id)
    assert len(only_provider(providers).requests) - calls_before == 2  # the two failed passages, nothing else
    assert finished.status == "completed" and finished.failed_passages == 0 and not finished.can_resume
    assert statuses(session, run_id) == ["completed"] * CHAPTER_PASSAGES and all(r.error_code is None for r in records(session, run_id))
    assert done_before <= {u.id for u in session.scalars(select(KnowledgeUnit))} and count(session, KnowledgeUnit) == CHAPTER_PASSAGES

    # the failed attempts are history, not overwritten
    rows = session.scalars(select(ExtractionAttempt).where(ExtractionAttempt.extraction_run_id == run_id)).all()
    assert len(rows) == CHAPTER_PASSAGES + 2 and sum(a.status == "provider_error" for a in rows) == 2
    assert count(session, ExtractionExecution) == CHAPTER_PASSAGES + 2
    run = finished.run
    assert (run.passages_processed, run.total_passages, run.units_extracted, run.failure_count) == (CHAPTER_PASSAGES, CHAPTER_PASSAGES, CHAPTER_PASSAGES, 2)


def test_resuming_continues_with_passages_that_were_never_reached(session, library):
    switch = Switch(library, "c1a", "c1b", error=ProviderTimeoutError("down"))
    service, providers = make_service(session, switch, max_retries=0, max_consecutive_failures=2)
    run_id = new_run(service, library)
    stopped = service.execute_run(run_id)
    assert stopped.status == "failed" and stopped.run.error_code == "too_many_failures"
    assert statuses(session, run_id) == ["failed", "failed", "pending", "pending", "pending"]
    info = service.get_resume_info(run_id)
    assert (info.completed, info.failed, info.pending, info.can_resume) == (0, 2, 3, True)
    calls_before = len(only_provider(providers).requests)

    switch.on = False
    finished = service.resume_run(run_id)
    assert len(only_provider(providers).requests) - calls_before == CHAPTER_PASSAGES  # two retries and three new passages
    assert finished.status == "completed" and finished.run.passages_processed == CHAPTER_PASSAGES and finished.run.error_code is None
    assert count(session, KnowledgeUnit) == CHAPTER_PASSAGES


def test_an_interrupted_run_keeps_its_results_and_resumes_without_repeating_them(session, library):
    service, providers = make_service(session)
    run_id = new_run(service, library)

    def interrupt(event):
        if event.passages_done == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        service.execute_run(run_id, progress=interrupt)
    session.expire_all()
    run = session.get(ExtractionRun, run_id)
    assert (run.status, run.error_code, run.units_extracted, run.passages_processed) == ("cancelled", "interrupted", 2, 2)
    assert count(session, KnowledgeUnit) == 2  # nothing completed was lost
    assert statuses(session, run_id) == ["completed", "processing", "pending", "pending", "pending"]
    info = service.get_resume_info(run_id)
    assert (info.completed, info.processing, info.pending, info.can_resume) == (1, 1, 3, True)

    calls_before = len(only_provider(providers).requests)
    finished = service.resume_run(run_id)
    assert len(only_provider(providers).requests) - calls_before == 3  # the interrupted passage had its result already
    assert finished.status == "completed" and statuses(session, run_id) == ["completed"] * CHAPTER_PASSAGES
    assert count(session, KnowledgeUnit) == CHAPTER_PASSAGES and finished.run.started_at is not None and finished.run.completed_at is not None


def test_a_cancelled_run_can_be_resumed(session, library):
    service, providers = make_service(session)
    run_id = new_run(service, library)
    cancelled = service.execute_run(run_id, progress=lambda e: service.cancel_run(run_id) if e.passages_done == 2 else None)
    assert cancelled.status == "cancelled" and cancelled.can_resume
    finished = service.resume_run(run_id)
    assert finished.status == "completed" and count(session, KnowledgeUnit) == CHAPTER_PASSAGES and len(only_provider(providers).requests) == CHAPTER_PASSAGES


def test_resuming_twice_in_a_row_does_nothing_the_second_time(session, library):
    switch = Switch(library, "c1b")
    service, providers = make_service(session, switch, max_retries=0)
    run_id = new_run(service, library)
    service.execute_run(run_id)
    switch.on = False
    service.resume_run(run_id)
    calls = len(only_provider(providers).requests)
    with pytest.raises(ExtractionRunError, match="nothing to resume"):
        service.resume_run(run_id)
    assert len(only_provider(providers).requests) == calls


def test_a_failure_while_resuming_is_recorded_and_the_run_can_be_resumed_again(session, library):
    switch = Switch(library, "c1b")
    service, _ = make_service(session, switch, max_retries=0)
    run_id = new_run(service, library)
    service.execute_run(run_id)
    again = service.resume_run(run_id)  # still failing
    assert again.status == "needs_review" and again.failed_passages == 1 and again.can_resume
    switch.on = False
    assert service.resume_run(run_id).status == "completed"
    assert count(session, ExtractionAttempt) == CHAPTER_PASSAGES + 2  # both failures kept, then the success


# ---------------------------------------------------------------- redoing completed work, explicitly


def test_completed_passages_are_only_processed_again_when_that_is_asked_for(session, library):
    service, providers = make_service(session)
    run_id = new_run(service, library)
    service.execute_run(run_id)
    units_before = {u.id for u in session.scalars(select(KnowledgeUnit))}
    calls = len(only_provider(providers).requests)

    redone = service.resume_run(run_id, redo_completed=True)
    assert len(only_provider(providers).requests) - calls == CHAPTER_PASSAGES  # asked again, on request
    assert {u.id for u in session.scalars(select(KnowledgeUnit))} == units_before  # and nothing was stored twice
    assert redone.status == "completed" and redone.run.duplicate_count == CHAPTER_PASSAGES and redone.run.units_extracted == CHAPTER_PASSAGES


# ---------------------------------------------------------------- relationships on resume


def test_resuming_does_not_create_relationships_twice(session, library):
    def relations(call):
        return relationships_reply(relationship(call, "Second law", "Entropy", "depends_on", quote=QUOTE))

    switch = Switch(library, "s1a")

    def knowledge(call):
        return switch(call)

    service, providers = make_service(session, with_relationships(knowledge, relations), max_retries=0)
    run_id = new_run(service, library, ("concept", "principle"), extract_relationships=True)
    first = service.execute_run(run_id)
    created = count(session, KnowledgeRelationship)
    assert first.failed_passages == 1 and created == CHAPTER_PASSAGES - 1  # the failed passage has no units to relate
    switch.on = False
    finished = service.resume_run(run_id)
    assert count(session, KnowledgeRelationship) == CHAPTER_PASSAGES and finished.status == "completed"
    again = service.resume_run(run_id, redo_completed=True)
    assert count(session, KnowledgeRelationship) == CHAPTER_PASSAGES and again.run.relationships_extracted == CHAPTER_PASSAGES


# ---------------------------------------------------------------- the job runner


def test_the_background_runner_reports_what_cannot_be_resumed_at_once(session, library, engine):
    from contextlib import contextmanager

    from app.services.extraction_job_runner import ExtractionJobRunner

    @contextmanager
    def scope():
        local = Session(bind=engine, autoflush=False, expire_on_commit=False)
        try:
            yield local
        finally:
            local.close()

    switch = Switch(library, "c1b")
    runner = ExtractionJobRunner(session_factory=scope, service_factory=lambda s: make_service(s, switch, max_retries=0)[0])
    try:
        service, _ = make_service(session, switch, max_retries=0)
        run_id = new_run(service, library)
        with pytest.raises(ExtractionRunError, match="not started"):
            runner.resume(run_id)
        service.execute_run(run_id)
        switch.on = False
        runner.resume(run_id)  # validated here, executed on the worker
        runner.shutdown(wait=True)
        session.expire_all()
        assert session.get(ExtractionRun, run_id).status == "completed" and count(session, KnowledgeUnit) == CHAPTER_PASSAGES
    finally:
        runner.shutdown()
