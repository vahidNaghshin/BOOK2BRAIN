"""Day 5: regressions found while evaluating real extractions against reviewed references.

Both defects attached the wrong Phase 1 equation to a unit as its source:

* a reading that cited a neighbouring passage was matched against the equations of the passage being
  processed instead of the passage it cites;
* the only stored equation of a passage was bound to any reading, however unrelated.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services.equation_service import related_expressions
from database.models import EquationSource, KnowledgeUnit
from tests.extraction_db_helpers import Library, add_equation, item, make_service, reply, store_library
from tests.knowledge_helpers import equation_data

TEXT = "Before the equation. dS = dQ / T. After the equation."


@pytest.fixture
def library(session: Session) -> Library:
    lib = store_library(session)
    lib.passages["ss1a"].text = TEXT
    session.commit()
    return lib


def equation_units(session):
    return list(session.scalars(select(KnowledgeUnit).where(KnowledgeUnit.knowledge_type == "equation")))


def test_an_unrelated_reading_is_not_bound_to_the_only_stored_equation(session, library):
    stored = add_equation(session, library, "ss1a")
    unrelated = equation_data(name="Gibbs", original_expression="P(y|x) = exp(-bF(x,y)) / Z", latex=r"P(y|x)=\frac{e^{-\beta F}}{Z}")
    service, _ = make_service(session, lambda call: reply(item(call, content=unrelated)))

    service.extract_section(library.nodes["ss1"].id, ["equation"])

    model_unit = next(u for u in equation_units(session) if u.origin == "ai_extracted")
    capture = session.get(EquationSource, model_unit.id)
    assert capture.equation_id is None and capture.original_text is None
    assert capture.capture_status == "needs_review"
    assert any("could not be matched" in note for note in capture.capture_notes)
    # the stored equation is not lost: it becomes a unit of its own, from the source
    source_unit = next(u for u in equation_units(session) if u.origin == "source_derived")
    assert source_unit.equation_id == stored.id


def test_a_related_reading_still_finds_the_only_stored_equation(session, library):
    stored = add_equation(session, library, "ss1a")
    service, _ = make_service(session, lambda call: reply(item(call, content=equation_data(original_expression="dS=dQ/T", latex=r"dS = \frac{dQ}{T}"))))

    service.extract_section(library.nodes["ss1"].id, ["equation"])

    [unit] = equation_units(session)
    assert unit.equation_id == stored.id and session.get(EquationSource, unit.id).equation_id == stored.id


def test_an_equation_cited_from_a_neighbouring_passage_takes_its_source_from_that_passage(session, library):
    neighbour = library.passages["ss1a"]
    stored = add_equation(session, library, "ss1a")
    target = library.passages["s1b"]

    def answer(call):
        assert call.target_id == str(target.id)
        cited = {"content": equation_data(), "sources": [{"passage_id": str(neighbour.id), "role": "primary"}], "basis": "explicit", "confidence": 0.8, "is_uncertain": False}
        return reply(cited)

    service, _ = make_service(session, answer)
    service.extract_section(library.nodes["s1"].id, ["equation"], offset=1, limit=1)

    [unit] = [u for u in equation_units(session) if u.origin == "ai_extracted"]
    capture = session.get(EquationSource, unit.id)
    assert capture.passage_id == neighbour.id, "the capture belongs to the passage the unit cites, not the one being processed"
    assert capture.equation_id == stored.id and capture.page_start == neighbour.page_start


@pytest.mark.parametrize(
    "first, second, related",
    [
        (["dS = dQ / T"], [r"\frac{dQ}{T}", "dS = dQ / T"], True),
        ([r"F(x, y) = argmin_z E_w(x, y, z)"], ["F_w(x,y) = argminz Ew(x, y, z)"], True),
        (["P(y | x) = exp[-βF(x, y)]"], ["-log Pw(y1, ..., yn | x1, ..., xn)"], False),
        ([None], ["x"], False),
    ],
)
def test_related_expressions_share_symbols(first, second, related):
    assert related_expressions(first, second) is related


# -- a unit read from a neighbouring passage is visible, not silent ------------------------------------


def flags_of(session, library):
    from app.services.review_flag_service import ReviewFlagService

    return {(f.code, f.message) for f in ReviewFlagService(session).list_flags(library.book.id)}


def test_a_unit_that_cites_a_neighbouring_passage_as_its_primary_source_is_flagged(session, library):
    target, neighbour = library.passages["s1b"], library.passages["ss1a"]

    def answer(call):
        cited = {"passage_id": str(neighbour.id), "role": "primary"}
        return reply(item(call, sources=[cited]))

    service, _ = make_service(session, answer)
    service.extract_section(library.nodes["s1"].id, ["concept"], offset=1, limit=1)

    codes = {code for code, _ in flags_of(session, library)}
    assert "source_outside_target" in codes
    [unit] = session.scalars(select(KnowledgeUnit)).all()
    assert unit.sources[0].passage_id == neighbour.id and unit.source_status == "verified", "the citation is true; it is just not the passage being processed"
    assert target.id != neighbour.id


def test_a_unit_that_cites_the_passage_it_was_extracted_from_is_not_flagged(session, library):
    service, _ = make_service(session, lambda call: reply(item(call)))
    service.extract_section(library.nodes["s1"].id, ["concept"], offset=1, limit=1)

    assert "source_outside_target" not in {code for code, _ in flags_of(session, library)}
