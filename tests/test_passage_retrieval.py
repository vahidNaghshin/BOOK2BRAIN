"""Retrieving passages for a book, chapter or section, in reading order, without touching Phase 1 data."""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.repositories.passage_repository import PassageRepository
from app.services.errors import ExtractionScopeError
from database.models import Passage, StructuralNode
from tests.extraction_db_helpers import Library, add_second_document, store_library


@pytest.fixture
def library(session: Session) -> Library:
    return store_library(session)


def names(library: Library, passages) -> list[str]:
    by_id = {p.id: name for name, p in library.passages.items()}
    return [by_id.get(p.id, "other") for p in passages]


def test_a_book_returns_all_its_passages_in_reading_order(session, library):
    repository = PassageRepository(session)
    scope = repository.book_scope(library.book.id)
    assert scope.kind == "book" and repository.count(scope) == 8
    assert names(library, repository.select(scope)) == ["f1", "c1a", "c1b", "s1a", "s1b", "ss1a", "c2a", "c2b"]


def test_a_chapter_includes_the_sections_and_subsections_inside_it(session, library):
    repository = PassageRepository(session)
    scope = repository.chapter_scope(library.nodes["ch1"].id)
    assert (scope.kind, scope.book_id, scope.document_id, scope.node_id) == ("chapter", library.book.id, library.document.id, library.nodes["ch1"].id)
    assert names(library, repository.select(scope)) == ["c1a", "c1b", "s1a", "s1b", "ss1a"]
    assert names(library, repository.select(repository.chapter_scope(library.nodes["ch2"].id))) == ["c2a", "c2b"]


def test_a_section_includes_its_subsections_but_not_its_chapter(session, library):
    repository = PassageRepository(session)
    assert names(library, repository.select(repository.section_scope(library.nodes["s1"].id))) == ["s1a", "s1b", "ss1a"]
    assert names(library, repository.select(repository.section_scope(library.nodes["ss1"].id))) == ["ss1a"]


def test_scopes_are_checked(session, library):
    repository = PassageRepository(session)
    with pytest.raises(ExtractionScopeError, match="not a chapter"):
        repository.chapter_scope(library.nodes["s1"].id)
    with pytest.raises(ExtractionScopeError, match="not a section"):
        repository.section_scope(library.nodes["ch1"].id)
    with pytest.raises(ExtractionScopeError, match="not a section"):
        repository.section_scope(library.nodes["front"].id)
    with pytest.raises(ExtractionScopeError, match="chapter does not exist"):
        repository.chapter_scope(uuid.uuid4())
    with pytest.raises(ExtractionScopeError, match="book does not exist"):
        repository.book_scope(uuid.uuid4())
    with pytest.raises(ExtractionScopeError, match="does not belong"):
        repository.book_scope(library.book.id, uuid.uuid4())


def test_limit_and_offset_page_through_the_scope(session, library):
    repository = PassageRepository(session)
    scope = repository.book_scope(library.book.id)
    assert names(library, repository.select(scope, offset=2, limit=3)) == ["c1b", "s1a", "s1b"]
    assert names(library, repository.select(scope, offset=7, limit=5)) == ["c2b"]


def test_documents_are_read_oldest_first_and_can_be_selected_alone(session, library):
    older = add_second_document(session, library, older=True)
    repository = PassageRepository(session)
    scope = repository.book_scope(library.book.id)
    assert [p.id for p in repository.select(scope)[:2]] == [p.id for p in older]
    assert repository.count(scope) == 10
    own = repository.book_scope(library.book.id, library.document.id)
    assert own.kind == "document" and repository.count(own) == 8


def test_neighbours_come_from_the_same_document_in_order(session, library):
    repository = PassageRepository(session)
    add_second_document(session, library, older=False)
    before, after = repository.neighbours(library.passages["c1b"], 2)
    assert names(library, before) == ["f1", "c1a"] and names(library, after) == ["s1a", "s1b"]
    before, after = repository.neighbours(library.passages["f1"], 1)
    assert before == [] and names(library, after) == ["c1a"]
    assert repository.neighbours(library.passages["c1b"], 0) == ([], [])


def test_the_chapter_and_section_of_a_passage_are_found_through_the_hierarchy(session, library):
    repository = PassageRepository(session)
    located = repository.source_of(library.passages["ss1a"])
    assert located.chapter.title == "Entropy" and located.section.title == "Statistical view"  # nearest section, even a subsection
    located = repository.source_of(library.passages["c1a"])
    assert located.chapter.title == "Entropy" and located.section is None
    located = repository.source_of(library.passages["f1"])
    assert located.chapter is None and located.section is None


def test_the_outline_lists_chapters_and_sections_with_passage_counts(session, library):
    outline = {node.title: count for node, count in PassageRepository(session).outline(library.document.id)}
    assert outline == {"Entropy": 5, "The second law": 3, "Statistical view": 1, "Engines": 2}


def test_retrieval_does_not_change_phase_1_data(session, library):
    before = [(p.id, p.text, p.sequence, p.document_sequence, p.page_start, p.structural_node_id) for p in library.ordered]
    repository = PassageRepository(session)
    for scope in (repository.book_scope(library.book.id), repository.chapter_scope(library.nodes["ch1"].id)):
        repository.select(scope)
        repository.neighbours(library.passages["c1a"], 2)
    session.rollback()
    after = [(p.id, p.text, p.sequence, p.document_sequence, p.page_start, p.structural_node_id) for p in session.scalars(select(Passage).order_by(Passage.document_sequence))]
    assert after == before
    assert session.scalar(select(StructuralNode.id).limit(1)) is not None
