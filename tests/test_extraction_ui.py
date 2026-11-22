"""The Extraction screen, driven with AppTest against the test database and a mocked LLM."""

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from tests.extraction_db_helpers import item, make_service, one_item, reply, store_library
from tests.extraction_helpers import hosted_config, make_config
from tests.ui_helpers import patch_ui


def extraction_app() -> None:
    from app.ui.extraction import render_extraction_page

    render_extraction_page()


class SyncRunner:
    """Stands in for the background job runner: runs the whole extraction before returning."""

    def __init__(self, engine: Engine, **service_options) -> None:
        self.engine, self.options, self.started = engine, service_options, []

    def start(self, scope, categories, **kwargs):
        with Session(bind=self.engine, autoflush=False, expire_on_commit=False) as session:
            service, _ = make_service(session, **self.options)
            run_id = service.create_run(scope, categories, **kwargs)
            self.started.append((scope.kind, kwargs))
            service.execute_run(run_id)
        return run_id

    def resume(self, run_id):
        with Session(bind=self.engine, autoflush=False, expire_on_commit=False) as session:
            make_service(session, **self.options)[0].execute_run(run_id)


@pytest.fixture
def ui(session: Session, settings, engine: Engine, storage, monkeypatch):
    library = store_library(session)
    library.document.processing_status = "completed"
    session.commit()
    patch_ui(monkeypatch, engine, settings, storage)

    def setup(answer=one_item, **options):
        options.setdefault("configs", [make_config(), hosted_config()])
        runner = SyncRunner(engine, answer=answer, **options)
        monkeypatch.setattr("app.ui.extraction.get_job_runner", lambda: runner)
        monkeypatch.setattr("app.ui.extraction._service", lambda s: make_service(s, answer=answer, **options)[0])
        at = AppTest.from_function(extraction_app, default_timeout=30)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at, runner

    setup.library = library
    return setup


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_the_screen_offers_models_scope_and_categories_and_shows_the_plan(ui):
    at, _ = ui(environ={})  # no key for the hosted model
    assert [o.split(" ")[0] for o in at.selectbox(key="extraction_model").options] == ["local-test"]
    assert at.selectbox(key="extraction_model").value == "local-test"
    assert any(e.label == "Unavailable models (1)" for e in at.expander)
    assert any("hosted-test" in m.value and "TEST_LLM_KEY" in m.value for m in at.markdown)
    assert at.radio(key="extraction_scope").options == ["Whole document", "A chapter", "A section"]
    assert len(at.multiselect(key="extraction_categories").value) == 7
    assert metrics(at) == {"Passages": "8", "Model calls": "56", "Already extracted": "0"}
    assert any("local-test (test-model:1b)" in c.value for c in at.caption)
    assert any("56 model calls" in w.value for w in at.warning)
    assert not at.button(key="extraction_start").disabled


def test_the_model_can_be_switched(ui):
    at, _ = ui(environ={"TEST_LLM_KEY": "k"})
    assert [o.split(" ")[0] for o in at.selectbox(key="extraction_model").options] == ["local-test", "hosted-test"]
    at.selectbox(key="extraction_model").set_value("hosted-test").run()
    assert any("hosted-test (gpt-test)" in c.value for c in at.caption)


def test_scope_can_be_a_chapter_or_a_section(ui):
    at, _ = ui()
    at.radio(key="extraction_scope").set_value("A chapter").run()
    chapter = at.selectbox(key=f"extraction_node_{ui.library.document.id}_A chapter")
    assert len(chapter.options) == 2
    chapter.set_value(str(ui.library.nodes["ch1"].id)).run()
    assert metrics(at)["Passages"] == "5"
    at.radio(key="extraction_scope").set_value("A section").run()
    section = at.selectbox(key=f"extraction_node_{ui.library.document.id}_A section")
    section.set_value(str(ui.library.nodes["s1"].id)).run()
    assert metrics(at)["Passages"] == "3"


def test_starting_an_extraction_shows_the_model_progress_and_results(ui):
    at, runner = ui()
    at.multiselect(key="extraction_categories").set_value(["concept", "principle"]).run()
    at.number_input(key="extraction_limit").set_value(2).run()
    assert metrics(at)["Model calls"] == "4"
    at.button(key="extraction_start").click().run()
    assert not at.exception

    assert runner.started[0][0] == "document" and runner.started[0][1]["limit"] == 2 and runner.started[0][1]["model_id"] == "local-test"
    assert any("Completed" in s.value for s in at.success)
    assert any("local-test (test-model:1b)" in m.value for m in at.markdown)
    assert metrics(at)["Knowledge units"] == "4" and metrics(at)["Failed passages"] == "0"
    assert any("Extracted knowledge (4 units)" in e.label for e in at.expander)
    # a second identical run has nothing left to do, so it cannot be started
    assert metrics(at)["Model calls"] == "0" and at.button(key="extraction_start").disabled


def test_extracted_again_only_when_asked(ui):
    at, runner = ui()
    at.multiselect(key="extraction_categories").set_value(["concept"]).run()
    at.number_input(key="extraction_limit").set_value(2).run()
    at.button(key="extraction_start").click().run()
    at.run()
    assert metrics(at)["Model calls"] == "0" and metrics(at)["Already extracted"] == "2"
    assert any("already been extracted" in i.value for i in at.info) and at.button(key="extraction_start").disabled
    at.checkbox(key="extraction_force").check().run()
    assert metrics(at)["Model calls"] == "2" and not at.button(key="extraction_start").disabled


def test_a_run_that_needs_review_and_a_failed_run_are_explained(ui):
    at, _ = ui(lambda call: reply(item(call, basis="inferred", is_uncertain=True, uncertainty_notes="Implied.")))
    at.multiselect(key="extraction_categories").set_value(["concept"]).run()
    at.number_input(key="extraction_limit").set_value(2).run()
    at.button(key="extraction_start").click().run()
    assert any("needs review" in w.value for w in at.warning) and metrics(at)["Need review"] == "2"

    failing, _ = ui(lambda call: "garbage", max_retries=0, max_consecutive_failures=0)
    failing.multiselect(key="extraction_categories").set_value(["equation"]).run()
    failing.number_input(key="extraction_limit").set_value(1).run()
    failing.button(key="extraction_start").click().run()
    assert any("Failed" in e.value for e in failing.error) and any("all_extractions_failed" in e.value for e in failing.error)
    assert any(b.key and b.key.startswith("extraction_resume_") for b in failing.button)


def test_without_any_ready_model_the_screen_says_so_and_cannot_start(ui):
    at, _ = ui(configs=[hosted_config()], default="hosted-test", environ={})
    assert any("No model is ready" in e.value for e in at.error)
    assert at.button(key="extraction_start").disabled


def test_a_book_without_processed_documents_gets_a_hint(session, settings, engine, storage, monkeypatch):
    patch_ui(monkeypatch, engine, settings, storage)
    at = AppTest.from_function(extraction_app, default_timeout=30)
    at.run()
    assert not at.exception and any("No processed documents" in i.value for i in at.info)
