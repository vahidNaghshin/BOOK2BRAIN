"""The extraction command line: argument helpers (no model calls, no production database)."""

import uuid
from argparse import Namespace

import pytest
from sqlalchemy import select

from app.knowledge.types import KnowledgeType
from app.services.knowledge_extraction_service import ExtractionRunError
from database.models import KnowledgeUnit
from tests.extraction_db_helpers import Library, item, make_service, one_item, reply, store_library
from tests.extraction_helpers import make_config
from tests.knowledge_helpers import store_book
from tools import extract_knowledge as cli


def test_parse_types():
    assert cli.parse_types("all") == list(KnowledgeType)
    assert cli.parse_types("Concept, equation") == [KnowledgeType.CONCEPT, KnowledgeType.EQUATION]
    with pytest.raises(ExtractionRunError, match="Choose from"):
        cli.parse_types("concept,poem")


def test_find_book_by_id_or_title(session):
    stored = store_book(session, "Thermodynamics of Entropy")
    assert cli.find_book(session, str(stored.book.id)).id == stored.book.id
    assert cli.find_book(session, "thermodynamics").id == stored.book.id
    with pytest.raises(ExtractionRunError, match="No book matches"):
        cli.find_book(session, "nothing like this")
    store_book(session, "Thermodynamics Again")
    with pytest.raises(ExtractionRunError, match="matches 2 books"):
        cli.find_book(session, "thermodynamics")


def test_models_path_is_relative_to_the_project(tmp_path):
    assert cli.models_path("config/llm_models.json", None).name == "llm_models.json"
    assert cli.models_path("config/x.json", str(tmp_path / "m.json")) == tmp_path / "m.json"


# The Day 4 options run against the test database with a mocked model.


def args(**values) -> Namespace:
    defaults = dict(
        book=None, document=None, chapter=None, section=None, outline=False, runs=False, status=None, passages=None, resume=None,
        relationships=False, verify_sources=False, types="concept", model=None, limit=None, offset=0, context=None, force=False,
        dry_run=False, yes=True,
    )
    return Namespace(**{**defaults, **values})


@pytest.fixture
def library(session) -> Library:
    return store_library(session)


def test_relationships_are_planned_and_extracted(session, library, capsys):
    service, _ = make_service(session, lambda call: reply() if call.is_relationship else one_item(call))
    code = cli._run(args(section=library.nodes["ss1"].id, relationships=True, limit=1), session, service)
    out = capsys.readouterr().out
    assert code == 0 and "plus 1 relationship calls" in out and "relations" in out and "relationships=0" in out
    assert len(session.scalars(select(KnowledgeUnit)).all()) == 1


def test_status_passages_and_resume_report_on_a_run(session, library, capsys):
    from app.extraction.errors import ProviderTimeoutError

    failing = str(library.passages["c1b"].id)
    service, _ = make_service(session, lambda call: ProviderTimeoutError("slow") if call.target_id == failing else one_item(call), max_retries=0)
    scope = service.book_scope(library.book.id)
    run_id = service.create_run(scope, ["concept"], limit=3)
    service.execute_run(run_id)

    assert cli._run(args(status=run_id), session, service) == 0
    out = capsys.readouterr().out
    assert "1 failed" in out and "can resume" in out

    assert cli._run(args(passages=run_id), session, service) == 0
    out = capsys.readouterr().out
    assert "failed" in out and "completed" in out and "provider_timeout: slow" in out

    service, _ = make_service(session, one_item)
    assert cli._run(args(resume=run_id), session, service) == 0
    out = capsys.readouterr().out
    assert "completed" in out and "failed_passages=0" in out
    assert cli._run(args(status=run_id), session, service) == 0
    assert "cannot resume" in capsys.readouterr().out


def test_resuming_a_run_that_cannot_be_resumed_is_an_error(session, library):
    service, _ = make_service(session, one_item)
    run_id = service.create_run(service.book_scope(library.book.id), ["concept"], limit=1)
    service.execute_run(run_id)
    with pytest.raises(ExtractionRunError):
        cli._run(args(resume=run_id), session, service)
    with pytest.raises(ExtractionRunError):
        cli._run(args(passages=uuid.uuid4()), session, service)


def test_verify_sources_reports_the_counts_and_fails_on_problems(session, library, capsys):
    service, _ = make_service(session, lambda call: reply(item(call, sources=[{"passage_id": call.target_id, "role": "primary", "quote": "Entropy is a measure of the disorder of a system."}])))
    service.extract_section(library.nodes["ss1"].id, ["concept"])
    assert cli._run(args(section=library.nodes["ss1"].id, verify_sources=True), session, service) == 0
    assert "Source verification: 1 verified, 0 invalid" in capsys.readouterr().out

    unit = session.scalars(select(KnowledgeUnit)).one()
    unit.page_start = unit.page_end = 40
    session.commit()
    assert cli._run(args(section=library.nodes["ss1"].id, verify_sources=True), session, service) == 1
    assert "1 invalid" in capsys.readouterr().out
