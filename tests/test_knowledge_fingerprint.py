"""Identity of extracted knowledge: what counts as the same extraction."""

import uuid

import pytest

from app.knowledge.fingerprint import content_fingerprint, normalise, primary_passage_id, similarity
from app.knowledge.schemas import KnowledgeUnitCreate
from tests.knowledge_helpers import concept_data, method_data, unit_data


def unit(content=None, passage=None, **overrides) -> KnowledgeUnitCreate:
    data = unit_data(content or concept_data(), **overrides)
    if passage is not None:
        data["sources"] = [{"passage_id": passage}]
    return KnowledgeUnitCreate(**data)


def test_case_spacing_and_edge_punctuation_do_not_make_different_knowledge():
    passage = uuid.uuid4()
    plain = unit(passage=passage)
    changed = unit(concept_data(name="  ENTROPY ", definition="A measure   of the disorder of a system"), passage=passage)
    assert content_fingerprint(plain) == content_fingerprint(changed)
    assert normalise("  Hello,   World! ") == "hello, world"


def test_other_content_passage_or_category_gives_another_fingerprint():
    passage = uuid.uuid4()
    base = content_fingerprint(unit(passage=passage))
    assert content_fingerprint(unit(concept_data(definition="Something else entirely."), passage=passage)) != base
    assert content_fingerprint(unit(passage=uuid.uuid4())) != base
    assert content_fingerprint(unit(method_data(), passage=passage)) != base


def test_the_primary_source_is_the_one_that_identifies_the_occurrence():
    primary, supporting = uuid.uuid4(), uuid.uuid4()
    first = unit(passage=None, sources=[{"passage_id": supporting, "role": "supporting"}, {"passage_id": primary, "role": "primary"}])
    assert primary_passage_id(first) == primary
    only = unit(sources=[{"passage_id": supporting}])
    assert primary_passage_id(only) == supporting


@pytest.mark.parametrize(
    "first, second, expected",
    [("a b c", "a b c", 1.0), ("a b c d", "a b c e", 0.6), ("a b", "c d", 0.0), ("", "", 1.0), ("Hello World", "hello world", 1.0)],
)
def test_similarity_is_the_share_of_words_in_common(first, second, expected):
    assert similarity(first, second) == pytest.approx(expected)
