import uuid

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import (
    Paragraph,
    Passage,
    PassageParagraph,
    StructuralNode,
    StructureAnalysis,
)
from tests.test_models import make_book, make_document


def make_node(document, **overrides) -> StructuralNode:
    values = {
        "book_id": document.book_id,
        "document_id": document.id,
        "node_type": "chapter",
        "title": "Introduction",
        "level": 0,
        "sequence": 0,
    }
    values.update(overrides)
    return StructuralNode(**values)


def make_paragraph(document, node, **overrides) -> Paragraph:
    values = {
        "book_id": document.book_id,
        "document_id": document.id,
        "structural_node_id": node.id,
        "sequence": 0,
        "document_sequence": 0,
        "kind": "paragraph",
        "text": "Some text.",
        "page_status": "exact",
        "page_start": 1,
        "page_end": 1,
    }
    values.update(overrides)
    return Paragraph(**values)


def make_passage(document, node, **overrides) -> Passage:
    values = {
        "book_id": document.book_id,
        "document_id": document.id,
        "structural_node_id": node.id,
        "text": "Some text.",
        "sequence": 0,
        "document_sequence": 0,
        "token_count": 3,
        "page_status": "exact",
        "page_start": 1,
        "page_end": 1,
    }
    values.update(overrides)
    return Passage(**values)


@pytest.fixture
def document(session: Session):
    doc = make_document(make_book())
    session.add(doc)
    session.commit()
    return doc


@pytest.fixture
def node(session: Session, document):
    value = make_node(document)
    session.add(value)
    session.commit()
    return value


def test_nodes_nest_to_any_depth_and_know_their_children(session: Session, document) -> None:
    parent = make_node(document, node_type="part", title="Part I", sequence=0)
    session.add(parent)
    session.flush()
    chain = [parent]
    for depth in range(1, 8):
        child = make_node(document, parent_id=chain[-1].id, level=depth, sequence=depth, node_type="section", title=f"S{depth}")
        session.add(child)
        session.flush()
        chain.append(child)
    session.commit()

    session.expire_all()
    loaded = session.get(StructuralNode, parent.id)
    assert [c.title for c in loaded.children] == ["S1"]
    assert session.get(StructuralNode, chain[-1].id).parent.title == "S6"
    assert session.get(StructuralNode, chain[-1].id).level == 7


def test_node_defaults_and_optional_fields(session: Session, node) -> None:
    assert isinstance(node.id, uuid.UUID)
    assert node.created_at.tzinfo is not None and node.updated_at.tzinfo is not None
    assert node.parent_id is None and node.number is None and node.original_title is None
    assert node.page_start is None and node.start_offset is None and node.confidence is None
    assert node.node_metadata is None


def test_node_metadata_is_stored_as_json_in_a_column_named_metadata(session: Session, document) -> None:
    value = make_node(document, node_metadata={"signals": ["label_pattern"], "style": "L1|S0"})
    session.add(value)
    session.commit()

    session.expire_all()
    assert session.get(StructuralNode, value.id).node_metadata == {"signals": ["label_pattern"], "style": "L1|S0"}
    assert "metadata" in {c["name"] for c in inspect(session.get_bind()).get_columns("structural_nodes")}


def test_updated_at_changes_on_update(session: Session, node) -> None:
    created = node.updated_at
    node.title = "Changed"
    session.commit()

    assert node.updated_at > created


@pytest.mark.parametrize(
    "overrides",
    [
        {"level": -1},
        {"sequence": -1},
        {"page_start": 0},
        {"page_start": 5, "page_end": 4},
        {"confidence": 1.5},
        {"confidence": -0.1},
    ],
)
def test_node_constraints(session: Session, document, overrides) -> None:
    session.add(make_node(document, **overrides))
    with pytest.raises(IntegrityError):
        session.commit()


def test_node_sequence_is_unique_within_a_document(session: Session, document, node) -> None:
    session.add(make_node(document, title="Other", sequence=0))

    with pytest.raises(IntegrityError):
        session.commit()


def test_different_documents_of_a_book_have_separate_structures(session: Session, document) -> None:
    second = make_document(document.book)
    session.add(second)
    session.flush()
    session.add_all([make_node(document), make_node(second)])
    session.commit()  # the same sequence in two editions is fine

    assert len(session.scalars(select(StructuralNode)).all()) == 2


def test_deleting_a_node_removes_its_subtree_paragraphs_and_passages(session: Session, document, node) -> None:
    child = make_node(document, parent_id=node.id, level=1, sequence=1, node_type="section", title="Child")
    session.add(child)
    session.flush()
    paragraph = make_paragraph(document, child)
    passage = make_passage(document, child)
    session.add_all([paragraph, passage])
    session.flush()
    session.add(PassageParagraph(passage_id=passage.id, paragraph_id=paragraph.id, position=0))
    session.commit()

    session.execute(StructuralNode.__table__.delete().where(StructuralNode.id == node.id))
    session.commit()

    for model in (StructuralNode, Paragraph, Passage, PassageParagraph):
        assert session.scalars(select(model)).all() == []


def test_paragraph_constraints_and_ordering(session: Session, document, node) -> None:
    session.add_all([
        make_paragraph(document, node, sequence=0, document_sequence=0),
        make_paragraph(document, node, sequence=1, document_sequence=1, page_start=None, page_end=None, page_status="unknown"),
    ])
    session.commit()

    session.add(make_paragraph(document, node, sequence=1, document_sequence=2))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(make_paragraph(document, node, sequence=5, document_sequence=1))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(make_paragraph(document, node, sequence=6, document_sequence=6, page_status="guessed"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_passage_requires_valid_pages_and_unique_sequence(session: Session, document, node) -> None:
    session.add(make_passage(document, node))
    session.commit()

    for overrides in (
        {"sequence": 0, "document_sequence": 1},  # duplicate sequence within the node
        {"sequence": 1, "document_sequence": 0},  # duplicate sequence within the document
        {"sequence": 2, "document_sequence": 2, "page_start": 3, "page_end": 2},
        {"sequence": 3, "document_sequence": 3, "token_count": -1},
        {"sequence": 4, "document_sequence": 4, "page_status": "approx"},
    ):
        session.add(make_passage(document, node, **overrides))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_passages_and_paragraphs_are_many_to_many_and_ordered(session: Session, document, node) -> None:
    paragraphs = [make_paragraph(document, node, sequence=i, document_sequence=i) for i in range(3)]
    passages = [make_passage(document, node, sequence=i, document_sequence=i) for i in range(2)]
    session.add_all(paragraphs + passages)
    session.flush()
    session.add_all([
        PassageParagraph(passage_id=passages[0].id, paragraph_id=paragraphs[0].id, position=0),
        PassageParagraph(passage_id=passages[0].id, paragraph_id=paragraphs[1].id, position=1),
        # The end of paragraph 1 is repeated at the start of the second passage, which also continues it.
        PassageParagraph(passage_id=passages[1].id, paragraph_id=paragraphs[1].id, position=0, is_overlap=True, text_start=10, text_end=40),
        PassageParagraph(passage_id=passages[1].id, paragraph_id=paragraphs[1].id, position=1, text_start=40, text_end=90),
        PassageParagraph(passage_id=passages[1].id, paragraph_id=paragraphs[2].id, position=2),
    ])
    session.commit()

    session.expire_all()
    first, second = (session.get(Passage, p.id) for p in passages)
    assert [link.paragraph_id for link in first.paragraph_links] == [paragraphs[0].id, paragraphs[1].id]
    assert [(link.is_overlap, link.text_start) for link in second.paragraph_links] == [(True, 10), (False, 40), (False, None)]
    uses_of_paragraph_1 = session.scalars(select(PassageParagraph).where(PassageParagraph.paragraph_id == paragraphs[1].id)).all()
    assert len(uses_of_paragraph_1) == 3


def test_passage_paragraph_text_range_must_be_valid(session: Session, document, node) -> None:
    paragraph = make_paragraph(document, node)
    passage = make_passage(document, node)
    session.add_all([paragraph, passage])
    session.flush()
    session.add(PassageParagraph(passage_id=passage.id, paragraph_id=paragraph.id, position=0, text_start=9, text_end=3))

    with pytest.raises(IntegrityError):
        session.commit()


def test_one_analysis_per_document(session: Session, document) -> None:
    session.add(StructureAnalysis(document_id=document.id, structure_status="completed"))
    session.commit()

    session.add(StructureAnalysis(document_id=document.id))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(StructureAnalysis(document_id=uuid.uuid4()))
    with pytest.raises(IntegrityError):
        session.commit()


def test_analysis_status_values_are_constrained(session: Session, document) -> None:
    session.add(StructureAnalysis(document_id=document.id, passage_status="nonsense"))

    with pytest.raises(IntegrityError):
        session.commit()


def test_foreign_keys_and_indexes_exist(session: Session) -> None:
    inspector = inspect(session.get_bind())

    node_indexes = {tuple(i["column_names"]) for i in inspector.get_indexes("structural_nodes")}
    assert {("parent_id",), ("document_id", "node_type"), ("document_id", "parent_id"), ("book_id",)} <= node_indexes
    fks = {tuple(fk["constrained_columns"]): fk["referred_table"] for fk in inspector.get_foreign_keys("structural_nodes")}
    assert fks == {("book_id",): "books", ("document_id",): "documents", ("parent_id",): "structural_nodes"}
    passage_fks = {fk["referred_table"] for fk in inspector.get_foreign_keys("passages")}
    assert passage_fks == {"books", "documents", "structural_nodes"}
