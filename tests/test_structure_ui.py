import uuid
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from streamlit.testing.v1 import AppTest

from app.services.structure_service import StructureService
from app.structure.tree import ancestor_ids, default_expanded, descendant_ids, visible_rows
from app.ui.formatting import format_pages, node_type_label
from database.models import StructuralNode
from tests.structure_helpers import numbered_textbook, persist_document
from tests.ui_helpers import patch_ui


@dataclass
class N:
    id: uuid.UUID
    parent_id: uuid.UUID | None
    sequence: int
    level: int = 0
    name: str = ""
    node_type: str = "chapter"


def tree() -> list[N]:
    ids = {name: uuid.uuid4() for name in "abcdefg"}
    return [
        N(ids["a"], None, 0, name="front", node_type="front_matter"),
        N(ids["b"], None, 1, name="part", node_type="part"),
        N(ids["c"], ids["b"], 2, 1, "chapter 1"),
        N(ids["d"], ids["c"], 3, 2, "section 1.1"),
        N(ids["e"], ids["c"], 4, 2, "section 1.2"),
        N(ids["f"], ids["b"], 5, 1, "chapter 2"),
        N(ids["g"], None, 6, name="back", node_type="back_matter"),
    ]


def names(rows) -> list[str]:
    return [r.node.name for r in rows]


def test_collapsed_tree_shows_only_top_level_nodes() -> None:
    assert names(visible_rows(tree(), set())) == ["front", "part", "back"]


def test_expanding_a_node_reveals_its_children_in_document_order() -> None:
    nodes = tree()
    part = next(n for n in nodes if n.name == "part")
    chapter = next(n for n in nodes if n.name == "chapter 1")

    rows = visible_rows(nodes, {part.id})
    assert names(rows) == ["front", "part", "chapter 1", "chapter 2", "back"]
    assert [r.depth for r in rows] == [0, 0, 1, 1, 0]
    assert [r.has_children for r in rows] == [False, True, True, False, False]

    deeper = visible_rows(nodes, {part.id, chapter.id})
    assert names(deeper) == ["front", "part", "chapter 1", "section 1.1", "section 1.2", "chapter 2", "back"]
    assert next(r for r in deeper if r.node.name == "chapter 1").expanded is True


def test_children_of_a_collapsed_node_stay_hidden_even_if_expanded_themselves() -> None:
    nodes = tree()
    chapter = next(n for n in nodes if n.name == "chapter 1")

    assert "section 1.1" not in names(visible_rows(nodes, {chapter.id}))


def test_descendants_and_ancestors() -> None:
    nodes = tree()
    by_name = {n.name: n for n in nodes}
    expected = [by_name[n].id for n in ("part", "chapter 1", "section 1.1", "section 1.2", "chapter 2")]

    assert descendant_ids(nodes, by_name["part"].id) == expected
    assert descendant_ids(nodes, by_name["section 1.1"].id) == [by_name["section 1.1"].id]
    assert ancestor_ids(nodes, by_name["section 1.2"].id) == [by_name["chapter 1"].id, by_name["part"].id]
    assert ancestor_ids(nodes, by_name["front"].id) == []


def test_containers_start_expanded_and_chapters_collapsed() -> None:
    nodes = tree()
    by_name = {n.name: n for n in nodes}

    assert default_expanded(nodes) == {by_name[n].id for n in ("front", "part", "back")}
    assert names(visible_rows(nodes, default_expanded(nodes))) == ["front", "part", "chapter 1", "chapter 2", "back"]


def test_page_references_never_invent_pages() -> None:
    assert format_pages(None, None) == "page unknown"
    assert format_pages(4, 4) == "p. 4"
    assert format_pages(4, None) == "p. 4"
    assert format_pages(4, 6) == "pp. 4–6"
    assert format_pages(4, 6, "partial") == "pp. 4–6 (partly known)"


def test_node_type_labels() -> None:
    assert node_type_label("chapter") == "Chapter"
    assert node_type_label("other_heading") == "Other heading"
    assert node_type_label("table_of_contents") == "Table of contents"
    assert node_type_label("something_new") == "Something new"


# -- the Streamlit page ------------------------------------------------------------------------


def structure_app() -> None:
    from app.ui.structure import render_structure_page

    render_structure_page()


def wire_database(engine: Engine, settings, monkeypatch) -> None:
    patch_ui(monkeypatch, engine, settings, None)


@pytest.fixture
def ui(session: Session, settings, engine: Engine, monkeypatch):
    """The structure page wired to the test database, with a processed and analysed book."""
    document = persist_document(session, numbered_textbook().build())
    StructureService(session, settings).detect_structure(document.id)
    wire_database(engine, settings, monkeypatch)
    return document


def run_page() -> AppTest:
    at = AppTest.from_function(structure_app, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def button(at: AppTest, key: str):
    return next(b for b in at.button if b.key == key)


def toc_labels(at: AppTest) -> list[str]:
    return [b.label for b in at.button if b.key and b.key.startswith("select-")]


def test_page_without_documents_says_so(session: Session, settings, engine: Engine, monkeypatch) -> None:
    wire_database(engine, settings, monkeypatch)

    at = run_page()

    assert any("No processed documents" in i.value for i in at.info)


def test_unanalysed_book_offers_to_detect_its_structure(session: Session, settings, engine: Engine, monkeypatch) -> None:
    document = persist_document(session, numbered_textbook().build())
    wire_database(engine, settings, monkeypatch)

    at = run_page()

    assert any("not been detected" in i.value for i in at.info)
    button(at, f"structure-detect-{document.id}").click().run()
    assert not at.exception
    assert "Introduction" in " ".join(toc_labels(at))


def test_table_of_contents_is_shown_collapsed_below_the_top_level(ui) -> None:
    at = run_page()

    labels = toc_labels(at)
    assert "Front Matter" in labels and "1  Introduction" in labels and "Back Matter" in labels
    assert "1.1  Background" not in labels  # sections stay hidden until the chapter is expanded


def test_expanding_and_collapsing_a_chapter(ui, session: Session) -> None:
    at = run_page()
    chapter = session.scalar(select(StructuralNode).where(StructuralNode.title == "Introduction"))

    button(at, f"toggle-{chapter.id}").click().run()
    assert "1.1  Background" in toc_labels(at) and "1.2  Motivation" in toc_labels(at)

    button(at, f"toggle-{chapter.id}").click().run()
    assert "1.1  Background" not in toc_labels(at)


def test_expand_all_and_collapse_all(ui) -> None:
    at = run_page()

    button(at, f"expand-all-{ui.id}").click().run()
    assert "2.1.1  Least Squares" in toc_labels(at)

    button(at, f"collapse-all-{ui.id}").click().run()
    assert "2.1.1  Least Squares" not in toc_labels(at) and "Front Matter" in toc_labels(at)


def test_selecting_a_section_shows_its_passages_with_pages(ui, session: Session) -> None:
    at = run_page()
    button(at, f"expand-all-{ui.id}").click().run()
    section = session.scalar(select(StructuralNode).where(StructuralNode.title == "Background"))

    button(at, f"select-{section.id}").click().run()

    assert not at.exception
    markdown = [m.value for m in at.markdown]
    assert any(m.startswith("### 1.1  Background") for m in markdown)
    assert any(m.startswith("**Passage 1**") and "p. 5" in m for m in markdown)
    assert any("paragraph(s) and" in c.value for c in at.caption)


def test_summary_metrics_separate_structure_paragraphs_and_passages(ui) -> None:
    at = run_page()

    labels = {m.label for m in at.metric}
    assert {"Structural nodes", "Paragraphs", "Passages", "Warnings", "Errors"} <= labels
    assert next(m for m in at.metric if m.label == "Errors").value == "0"
