"""Helpers for driving the Streamlit screens with `AppTest` against the test database."""

from contextlib import contextmanager

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

UI_MODULES = (
    "app.ui.library",
    "app.ui.selection",
    "app.ui.structure",
    "app.ui.passages",
    "app.ui.processing",
    "app.ui.upload",
    "app.ui.equations",
    "app.ui.extraction",
    "app.ui.knowledge",
)


def patch_ui(monkeypatch, engine: Engine, settings, storage, runner=None) -> None:
    """Point every screen at the test database, settings and storage."""

    @contextmanager
    def scope():
        local = Session(bind=engine, autoflush=False, expire_on_commit=False)
        try:
            yield local
        finally:
            local.close()

    import importlib

    for name in UI_MODULES:
        module = importlib.import_module(name)
        for attribute, value in (("session_scope", scope), ("get_settings", lambda: settings), ("get_storage", lambda *a, **k: storage)):
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, value)
    if runner is not None:
        monkeypatch.setattr("app.ui.processing.get_job_runner", lambda: runner)
        monkeypatch.setattr("app.ui.library.get_job_runner", lambda: runner)
        monkeypatch.setattr("app.ui.upload.get_job_runner", lambda: runner)


def full_app() -> None:
    """The application as main.py wires it: a screen selector and the chosen screen."""
    from app.ui.equations import render_equations_page
    from app.ui.extraction import render_extraction_page
    from app.ui.knowledge import render_knowledge_page
    from app.ui.library import render_library_page
    from app.ui.navigation import render_navigation
    from app.ui.passages import render_passages_page
    from app.ui.processing import render_processing_page
    from app.ui.structure import render_structure_page
    from app.ui.upload import render_upload_page

    screens = {
        "Library": render_library_page,
        "Upload": render_upload_page,
        "Structure": render_structure_page,
        "Passages": render_passages_page,
        "Processing": render_processing_page,
        "Equations": render_equations_page,
        "Extraction": render_extraction_page,
        "Knowledge": render_knowledge_page,
    }
    screens[render_navigation()]()


def run_app(timeout: int = 60) -> AppTest:
    at = AppTest.from_function(full_app, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def button(at: AppTest, key: str):
    return next(b for b in at.button if b.key == key)


def has_button(at: AppTest, key: str) -> bool:
    return any(b.key == key for b in at.button)


def screen(at: AppTest) -> str:
    return at.radio(key="screen").value
