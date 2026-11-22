"""Context preparation: what the model sees for one passage, and how it is reduced to fit."""

import pytest
from sqlalchemy.orm import Session

from app.extraction.prompts import get_prompt
from app.extraction.settings import ExtractionSettings
from app.knowledge.types import KnowledgeType
from app.repositories.passage_repository import PassageRepository
from app.services.context_preparation import ContextPreparer, source_info
from tests.extraction_db_helpers import Library, store_library
from tests.extraction_helpers import make_config

CONCEPT = KnowledgeType.CONCEPT


def preparer(session, **settings) -> ContextPreparer:
    return ContextPreparer(PassageRepository(session), ExtractionSettings(**settings))


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


def test_a_prepared_request_carries_the_passage_its_neighbours_and_where_they_come_from(session, library):
    source = source_info(library.book)
    prepared = preparer(session).prepare(library.passages["s1a"], CONCEPT, make_config(), source)
    request = prepared.request

    assert request.passage.id == library.passages["s1a"].id and request.passage.text == library.passages["s1a"].text
    assert (request.passage.page_start, request.passage.page_end) == (4, 4)
    assert (request.passage.chapter_title, request.passage.section_title) == ("Entropy", "The second law")
    assert [p.id for p in request.context_before] == [library.passages["c1b"].id]
    assert [p.id for p in request.context_after] == [library.passages["s1b"].id]
    assert request.source.title == "Thermodynamics" and request.source.publisher == "Test Press" and request.source.year == 1999
    assert prepared.fits and not prepared.info.reduced and (prepared.info.requested, prepared.info.sent) == (2, 2)

    text = get_prompt(CONCEPT).render_user(request)
    assert 'come from the book "Thermodynamics" (A primer) by Frank Herbert [Test Press, 1999]' in text
    assert f'id="{library.passages["s1a"].id}" role="target"' in text


def test_neighbours_can_be_set_per_preparer_and_by_setting(session, library):
    passage = library.passages["c1b"]
    assert len(preparer(session, context_passages=2).prepare(passage, CONCEPT, make_config()).request.context_before) == 2
    assert preparer(session, context_passages=0).prepare(passage, CONCEPT, make_config()).request.context_before == []
    override = ContextPreparer(PassageRepository(session), ExtractionSettings(context_passages=3), neighbours=1)
    prepared = override.prepare(passage, CONCEPT, make_config())
    assert len(prepared.request.context_before) == 1 and len(prepared.request.context_after) == 1


def test_the_first_passage_has_no_context_before_it(session, library):
    prepared = preparer(session).prepare(library.passages["f1"], CONCEPT, make_config())
    assert prepared.request.context_before == [] and len(prepared.request.context_after) == 1
    assert prepared.request.passage.chapter_title is None


def test_context_is_reduced_before_the_passage_and_the_reduction_is_recorded(session):
    library = store_library(session, "Long", text_size=60)  # about 1 000 characters per passage
    target = library.passages["s1a"]
    wide = preparer(session, context_passages=3, chars_per_token=4.0, context_safety_margin=0)
    full = wide.prepare(target, CONCEPT, make_config(context_window=128_000)).request
    assert len(full.context_before) + len(full.context_after) == 6

    small = make_config(context_window=3_000, max_output_tokens=500)  # about 1 500 tokens for a prompt
    prepared = wide.prepare(target, CONCEPT, small)
    info = prepared.info
    assert prepared.fits and info.reduced and info.requested == 6 and info.sent < 6
    assert len(info.dropped) == 6 - info.sent and info.reason
    assert prepared.request.passage.text == target.text  # the primary passage is whole
    assert info.estimated_input_tokens <= info.budget_tokens == 2_500
    kept = {p.id for p in [*prepared.request.context_before, *prepared.request.context_after]}
    assert kept.isdisjoint(info.dropped)
    # the furthest passages go first: the nearest neighbours are the last to be dropped
    assert library.passages["c1b"].id in kept or library.passages["s1b"].id in kept


def test_furthest_context_goes_first_alternating_around_the_passage(session):
    library = store_library(session, "Long", text_size=60)
    target = library.passages["s1b"]
    seen = []
    for window in (6_000, 4_000, 2_600, 2_000):
        prepared = preparer(session, context_passages=2, context_safety_margin=0).prepare(
            target, CONCEPT, make_config(context_window=window, max_output_tokens=500)
        )
        seen.append((len(prepared.request.context_before), len(prepared.request.context_after)))
    assert seen[0] == (2, 2)
    assert all(a[0] >= b[0] and a[1] >= b[1] for a, b in zip(seen, seen[1:]))  # only ever fewer
    assert seen[-1][0] + seen[-1][1] < 4


def test_a_passage_that_cannot_fit_is_reported_and_never_truncated(session):
    library = store_library(session, "Huge", text_size=2_000)
    target = library.passages["c1a"]
    prepared = preparer(session, context_safety_margin=0).prepare(target, CONCEPT, make_config(context_window=1_500, max_output_tokens=500))
    assert not prepared.fits and prepared.info.sent == 0 and prepared.info.reduced
    assert prepared.request.passage.text == target.text  # nothing silently cut
    assert prepared.info.estimated_input_tokens > prepared.info.budget_tokens


def test_the_safety_margin_keeps_room_free(session):
    library = store_library(session, "Long", text_size=60)
    config = make_config(context_window=3_000, max_output_tokens=500)
    strict = preparer(session, context_passages=3, context_safety_margin=0.4).prepare(library.passages["s1a"], CONCEPT, config)
    loose = preparer(session, context_passages=3, context_safety_margin=0.0).prepare(library.passages["s1a"], CONCEPT, config)
    assert strict.info.budget_tokens < loose.info.budget_tokens and strict.info.sent <= loose.info.sent


def test_an_unknown_context_window_sends_everything(session, library):
    prepared = preparer(session, context_passages=3).prepare(library.passages["s1a"], CONCEPT, make_config(context_window=None))
    assert prepared.fits and not prepared.info.reduced and prepared.info.budget_tokens is None


def test_an_unknown_prompt_version_is_left_for_the_extraction_service_to_report(session, library):
    prepared = preparer(session, prompt_versions={"concept": "99.0.0"}).prepare(library.passages["c1a"], CONCEPT, make_config())
    assert prepared.fits and prepared.info.requested == prepared.info.sent
