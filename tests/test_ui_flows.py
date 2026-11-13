"""The screens working together: library status and actions, editions, explorer and viewer."""

import io
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from app.config import Settings
from app.services.document_service import DocumentService
from app.services.pipeline_service import ProcessingPipeline
from database.models import Book, Document, Passage, StructuralNode
from tests.fakes import StubParser, StubParserFactory
from tests.structure_helpers import content_blocks, numbered_textbook
from tests.test_pipeline import break_structure, upload
from tests.ui_helpers import button, has_button, patch_ui, run_app, screen


@pytest.fixture
def settings(test_database_url: str, tmp_path: Path) -> Settings:
    return Settings(
        database_url=test_database_url, storage_root=tmp_path / "storage", max_upload_size_mb=5,
        equation_extraction_enabled=False, _env_file=None,
    )


class FakeRunner:
    def __init__(self) -> None:
        self.submitted: list[uuid.UUID] = []

    def submit(self, document_id: uuid.UUID) -> uuid.UUID:
        self.submitted.append(document_id)
        return uuid.uuid4()


def process(session, storage, settings, document, *, offset: int = 0):
    builder = numbered_textbook()
    for block in builder.blocks:
        block.page = (block.page or 0) + offset
    built = builder.build(page_count=9 + offset)
    factory = StubParserFactory(StubParser(content_blocks(built), page_count=built.page_count))
    return ProcessingPipeline(session, storage, settings, factory).process_document(document.id)


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


def texts(at: AppTest) -> str:
    return " ".join([*(m.value for m in at.markdown), *(c.value for c in at.caption), *(t.value for t in at.text)])


# -- library -------------------------------------------------------------------------------------


def test_empty_library_and_empty_screens_say_so(engine, settings, storage, monkeypatch, runner) -> None:
    patch_ui(monkeypatch, engine, settings, storage, runner)
    at = run_app()
    assert any("library is empty" in i.value for i in at.info)

    for name in ("Structure", "Passages"):
        at.radio(key="screen").set_value(name).run()
        assert any("No " in i.value and "yet" in i.value for i in at.info), name
        assert has_button(at, f"{name.lower()}_go_library")
    button(at, "passages_go_library").click().run()
    assert screen(at) == "Library"


def test_library_shows_uploaded_failed_and_completed_documents(
    session, document_service, storage, settings, engine, monkeypatch, runner
) -> None:
    uploaded = upload(document_service, "u1", title="Fresh Book", author=None)
    failed = upload(document_service, "u2", title="Broken Book", author=None)
    done = upload(document_service, "u3", title="Done Book", author=None)
    process(session, storage, settings, done)
    with monkeypatch.context() as m:
        break_structure(m)
        factory = StubParserFactory(StubParser(content_blocks(numbered_textbook().build()), page_count=9))
        ProcessingPipeline(session, storage, settings, factory).process_document(failed.id)
    patch_ui(monkeypatch, engine, settings, storage, runner)

    at = run_app()

    rows = {r["Title"]: r for table in at.dataframe for r in table.value.to_dict("records")}
    assert rows["Fresh Book"]["Status"] == "Uploaded, not processed"
    assert rows["Broken Book"]["Status"] == "Failed during detecting structure"
    assert rows["Done Book"]["Status"] == "Completed" and rows["Done Book"]["Passages"] > 0
    assert {r["Format"] for r in rows.values()} == {"PDF"}
    assert uploaded.id  # registered


def test_a_failed_document_shows_its_stage_and_can_be_retried(
    session, document_service, storage, settings, engine, monkeypatch, runner
) -> None:
    failed = upload(document_service, "u4", title="Broken Book", author=None)
    with monkeypatch.context() as m:
        break_structure(m)
        factory = StubParserFactory(StubParser(content_blocks(numbered_textbook().build()), page_count=9))
        ProcessingPipeline(session, storage, settings, factory).process_document(failed.id)
    patch_ui(monkeypatch, engine, settings, storage, runner)
    at = run_app()

    assert any("Structural detection failed." in e.value and "Failed during: Detecting structure" in e.value for e in at.error)
    assert not has_button(at, f"library_open_{failed.id}")  # nothing to open yet
    button(at, f"library_process_{failed.id}").click().run()
    assert runner.submitted == [failed.id]


def test_uploaded_documents_can_be_started_from_the_library(
    document_service, storage, settings, engine, monkeypatch, runner
) -> None:
    fresh = upload(document_service, "u5", title="Fresh Book", author=None)
    patch_ui(monkeypatch, engine, settings, storage, runner)
    at = run_app()

    button(at, f"library_process_{fresh.id}").click().run()

    assert runner.submitted == [fresh.id]


def test_processing_documents_show_their_stage_and_cannot_be_deleted(
    session, document_service, storage, settings, engine, monkeypatch, runner
) -> None:
    from database.models import ProcessingRun
    from database.models.base import utcnow

    busy = upload(document_service, "u6", title="Busy Book", author=None)
    session.get(Document, busy.id).processing_status = "processing"
    session.add(ProcessingRun(document_id=busy.id, status="processing", stage="passage_segmentation", started_at=utcnow()))
    session.commit()
    patch_ui(monkeypatch, engine, settings, storage, runner)

    at = run_app()

    assert any("Generating passages" in p.proto.text for p in at.get("progress"))
    assert next(b for b in at.button if b.key == f"library_delete_{busy.id}").disabled
    rows = [r for t in at.dataframe for r in t.value.to_dict("records")]
    assert rows[0]["Status"] == "Processing: generating passages"


# -- editions --------------------------------------------------------------------------------------


@pytest.fixture
def editions(session, document_service, storage, settings, engine, monkeypatch, runner):
    first = upload(document_service, "ed1", title="Dune", author="Frank Herbert")
    second = upload(document_service, "ed2", title="Dune", author="Frank Herbert")
    process(session, storage, settings, first, offset=0)
    process(session, storage, settings, second, offset=10)  # a different edition paginates differently
    patch_ui(monkeypatch, engine, settings, storage, runner)
    return first, second


def pages_of(session: Session, document) -> list[int]:
    return [p for p in session.scalars(select(Passage.page_start).where(Passage.document_id == document.id)) if p]


def test_the_library_lists_both_editions_of_one_book(editions, session) -> None:
    first, second = editions
    at = run_app()

    rows = [r for t in at.dataframe for r in t.value.to_dict("records")]
    assert [r["Title"] for r in rows] == ["Dune", "Dune"] and {r["Documents"] for r in rows} == {2}
    assert has_button(at, f"library_open_{first.id}") or has_button(at, f"library_open_{second.id}")


def test_the_explorer_and_viewer_follow_the_selected_edition(editions, session) -> None:
    first, second = editions
    at = run_app()
    at.radio(key="screen").set_value("Structure").run()

    at.selectbox(key="selector_documentstructure").select(str(first.id)).run()
    first_nodes = {n.id for n in session.scalars(select(StructuralNode).where(StructuralNode.document_id == first.id))}
    button(at, f"expand-all-{first.id}").click().run()
    assert all(b.key.removeprefix("select-") in {str(i) for i in first_nodes} for b in at.button if b.key and b.key.startswith("select-"))

    at.selectbox(key="selector_documentstructure").select(str(second.id)).run()
    second_nodes = {str(n.id) for n in session.scalars(select(StructuralNode).where(StructuralNode.document_id == second.id))}
    button(at, f"expand-all-{second.id}").click().run()
    shown = {b.key.removeprefix("select-") for b in at.button if b.key and b.key.startswith("select-")}
    assert shown and shown <= second_nodes  # nothing from the other edition leaks in

    at.radio(key="screen").set_value("Passages").run()  # the selection carries over to the viewer
    at.selectbox(key="selector_documentpassages").select(str(second.id)).run()
    opened = [b.key.removeprefix("open-passage-") for b in at.button if b.key and b.key.startswith("open-passage-")]
    ids = {str(p) for p in session.scalars(select(Passage.id).where(Passage.document_id == second.id))}
    assert opened and set(opened) <= ids
    button(at, f"open-passage-{opened[0]}").click().run()
    pages = " ".join(m.value for m in at.markdown)
    assert any(f"p. {p}" in pages or f"pp. {p}" in pages for p in pages_of(session, second))
    assert not any(f"p. {p}\n" in pages for p in range(1, 10) if p in pages_of(session, first) and p not in pages_of(session, second))


def test_a_failure_in_one_edition_does_not_hide_the_other(
    session, document_service, storage, settings, engine, monkeypatch, runner
) -> None:
    good = upload(document_service, "ed3", title="Dune", author="Frank Herbert")
    bad = upload(document_service, "ed4", title="Dune", author="Frank Herbert")
    process(session, storage, settings, good)
    with monkeypatch.context() as m:
        break_structure(m)
        factory = StubParserFactory(StubParser(content_blocks(numbered_textbook().build()), page_count=9))
        ProcessingPipeline(session, storage, settings, factory).process_document(bad.id)
    patch_ui(monkeypatch, engine, settings, storage, runner)

    at = run_app()
    at.radio(key="screen").set_value("Passages").run()

    options = at.selectbox(key="selector_documentpassages").options
    assert len(options) == 1 and "ed3" in options[0]  # only the processed edition is offered


# -- navigation ------------------------------------------------------------------------------------


def test_open_passages_from_the_library_selects_that_document(editions) -> None:
    first, second = editions
    at = run_app()
    at.selectbox(key="library_book").select(at.selectbox(key="library_book").value).run()
    at.radio(key=f"library_document_{next(b.value for b in at.selectbox if b.key == 'library_book')}").set_value(str(second.id)).run()

    button(at, f"library_passages_{second.id}").click().run()

    assert screen(at) == "Passages"
    assert at.selectbox(key="selector_documentpassages").value == str(second.id)


# -- upload -----------------------------------------------------------------------------------------


def upload_app() -> None:
    import io

    import streamlit as st

    from app.ui.upload import _handle_submit
    from tests.factories import make_pdf

    class FakeUpload(io.BytesIO):
        name = "dune.pdf"
        type = "application/pdf"

    data = FakeUpload(make_pdf("ui-up"))
    fields = {"title": "Dune", "author": "Frank Herbert", "publisher": "Chilton", "publication_year": "1965"}
    _handle_submit(data, fields, st.session_state.get("start", True))


def test_upload_registers_the_book_and_starts_processing(session, engine, settings, storage, monkeypatch, runner) -> None:
    patch_ui(monkeypatch, engine, settings, storage, runner)
    monkeypatch.setattr("app.ui.upload.check_connection", lambda: (True, "ok"))

    at = AppTest.from_function(upload_app, default_timeout=30)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    assert any("Uploaded dune.pdf" in s.value for s in at.success)
    document = session.scalar(select(Document))
    book = session.get(Book, document.book_id)
    assert (book.title, book.publisher, book.publication_year) == ("Dune", "Chilton", 1965)
    assert runner.submitted == [document.id]
    assert any("Processing started" in i.value for i in at.info)


def test_upload_does_not_start_processing_when_not_asked(session, engine, settings, storage, monkeypatch, runner) -> None:
    patch_ui(monkeypatch, engine, settings, storage, runner)
    monkeypatch.setattr("app.ui.upload.check_connection", lambda: (True, "ok"))

    at = AppTest.from_function(upload_app, default_timeout=30)
    at.session_state["start"] = False
    at.run()

    assert not at.exception and runner.submitted == []
    assert session.scalar(select(Document)).processing_status == "uploaded"


def test_an_upload_that_succeeds_stays_successful_if_processing_cannot_start(
    session, engine, settings, storage, monkeypatch
) -> None:
    from app.services.errors import ProcessingServiceError

    class Broken:
        def submit(self, document_id):
            raise ProcessingServiceError("The queue is full.")

    patch_ui(monkeypatch, engine, settings, storage, Broken())
    monkeypatch.setattr("app.ui.upload.check_connection", lambda: (True, "ok"))

    at = AppTest.from_function(upload_app, default_timeout=30)
    at.run()

    assert any("Uploaded dune.pdf" in s.value for s in at.success)
    assert any("processing could not start: The queue is full." in w.value for w in at.warning)
    assert session.scalar(select(Document)) is not None
