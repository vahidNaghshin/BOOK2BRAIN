"""Passage viewer: read the passages of a chapter or section and trace them to the source document.

Passages are the processing units that later steps will work on. This screen shows them exactly as
stored (never shortened or rewritten) together with where they came from: the document, the
chapter and section, the page range and the paragraphs they were built from.
"""

import logging
import math
import uuid

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.schemas import DocumentRead, ParagraphRead, PassageRead, StructuralNodeRead
from app.services.errors import ProcessingServiceError
from app.services.structure_service import StructureService
from app.structure.tree import ancestor_ids, descendant_ids
from app.ui.formatting import format_pages, node_type_label
from app.ui.navigation import NODE_KEY, PASSAGE_KEY, goto
from app.ui.selection import render_document_selector, safe_load_library
from database.connection import session_scope

logger = logging.getLogger(__name__)

PAGE_SIZE = 10
_WHOLE = "__whole__"
_CONTAINERS = ("front_matter", "back_matter", "part")


def _is_chapter_level(node: StructuralNodeRead, by_id: dict[uuid.UUID, StructuralNodeRead]) -> bool:
    """Chapters are the top-level nodes, or the children of front/back matter and parts."""
    if node.node_type in _CONTAINERS:
        return False
    parent = by_id.get(node.parent_id) if node.parent_id else None
    return parent is None or parent.node_type in _CONTAINERS


def _chapters(nodes: list[StructuralNodeRead], counts: dict[uuid.UUID, int]) -> list[StructuralNodeRead]:
    by_id = {n.id: n for n in nodes}
    return [
        n
        for n in nodes
        if _is_chapter_level(n, by_id) or (n.node_type in _CONTAINERS and counts.get(n.id, 0) > 0)
    ]


def _label(node: StructuralNodeRead, *, indent: int = 0) -> str:
    number = f"{node.number}  " if node.number and node.node_type not in ("part", "appendix") else ""
    if node.node_type == "appendix" and node.number:
        number = f"Appendix {node.number}: "
    if node.node_type == "part" and node.number:
        number = f"Part {node.number}: "
    return "\u00a0\u00a0\u00a0\u00a0" * indent + number + node.title


def _load_overview(document_id: uuid.UUID):
    with session_scope() as session:
        return StructureService(session, get_settings()).get_overview(document_id)


def render_passages_page() -> None:
    st.header("Passages")
    books = safe_load_library()
    if books is None:
        return
    selection = render_document_selector(books, predicate=lambda d: d.is_processed, key="passages")
    if selection is None:
        st.info("No document has passages yet. Process a book first; its status is shown in the Library.")
        if st.button("Go to Library", key="passages_go_library"):
            goto("Library")
        return
    _, entry = selection
    document = entry.document

    try:
        overview = _load_overview(document.id)
    except (SQLAlchemyError, ProcessingServiceError):
        logger.exception("Could not load the structure of document %s", document.id)
        st.error("Could not load the passages. Check the database connection and the logs.")
        return
    if not overview.nodes:
        st.info("This document has no structure yet.")
        return

    scope = _select_scope(document, overview)
    if scope is None:
        return
    node_ids, scope_node = scope

    selected_passage = st.session_state.get(PASSAGE_KEY)
    if selected_passage is not None:
        _render_passage(document, overview, node_ids, selected_passage)
    else:
        _render_list(document, overview, node_ids, scope_node)


# -- choosing chapter and section ----------------------------------------------------------------


def _select_scope(document: DocumentRead, overview) -> tuple[list[uuid.UUID], StructuralNodeRead] | None:
    nodes: list[StructuralNodeRead] = overview.nodes
    by_id = {n.id: n for n in nodes}
    chapters = _chapters(nodes, overview.passage_counts)
    if not chapters:
        st.info("This document has no chapters or sections with passages.")
        return None
    key = str(document.id)
    chapter_widget, section_widget = f"passages_chapter_{key}", f"passages_section_{key}"

    requested = st.session_state.pop(NODE_KEY, None)
    if requested in by_id:
        # Arrived from the structure explorer: start at that chapter/section.
        chain = [requested, *ancestor_ids(nodes, requested)]
        chapter = next((i for i in chain if by_id[i] in chapters), None)
        if chapter is not None:
            st.session_state[chapter_widget] = chapter
            st.session_state[section_widget] = requested if requested != chapter else _WHOLE

    chapter_ids = [c.id for c in chapters]
    if st.session_state.get(chapter_widget) not in chapter_ids:
        with_passages = [
            c.id for c in chapters if any(overview.passage_counts.get(i, 0) for i in descendant_ids(nodes, c.id))
        ]
        st.session_state[chapter_widget] = (with_passages or chapter_ids)[0]

    def passages_in(node_id: uuid.UUID) -> int:
        return sum(overview.passage_counts.get(n, 0) for n in descendant_ids(nodes, node_id))

    left, right = st.columns(2)
    chapter_id = left.selectbox(
        "Chapter",
        options=chapter_ids,
        format_func=lambda i: f"{_label(by_id[i])} ({passages_in(i)})",
        key=chapter_widget,
    )
    chapter_tree = descendant_ids(nodes, chapter_id)
    sections = [_WHOLE, *[i for i in chapter_tree if i != chapter_id]]
    if st.session_state.get(section_widget) not in sections:
        st.session_state[section_widget] = _WHOLE

    def section_label(i) -> str:
        if i == _WHOLE:
            return "Whole chapter"
        node = by_id[i]
        return f"{_label(node, indent=max(0, node.level - by_id[chapter_id].level - 1))} ({passages_in(i)})"

    section_id = right.selectbox("Section", options=sections, format_func=section_label, key=section_widget)
    scope_node = by_id[chapter_id if section_id == _WHOLE else section_id]

    include = True
    if len(descendant_ids(nodes, scope_node.id)) > 1:
        include = st.checkbox(
            "Include subsections",
            value=True,
            key=f"passages_include_{scope_node.id}",
            help="Also show the passages of the sections inside the selected one.",
        )
    node_ids = descendant_ids(nodes, scope_node.id) if include else [scope_node.id]

    # Changing the chapter or section leaves any opened passage and returns to the list.
    scope_state = (str(scope_node.id), include)
    if st.session_state.get("_passages_scope") not in (None, scope_state):
        st.session_state[PASSAGE_KEY] = None
    st.session_state["_passages_scope"] = scope_state
    return node_ids, scope_node


# -- the list --------------------------------------------------------------------------------------


def _breadcrumb(overview, node_id: uuid.UUID) -> str:
    by_id = {n.id: n for n in overview.nodes}
    chain = [*reversed(ancestor_ids(overview.nodes, node_id)), node_id]
    return " › ".join(by_id[i].title for i in chain if i in by_id)


def _render_list(
    document: DocumentRead, overview, node_ids: list[uuid.UUID], scope_node: StructuralNodeRead
) -> None:
    st.caption(f"{_breadcrumb(overview, scope_node.id)} · {format_pages(scope_node.page_start, scope_node.page_end)}")
    with session_scope() as session:
        service = StructureService(session, get_settings())
        total = service.count_passages(document.id, node_ids)
        if total == 0:
            st.info("This chapter or section has no passages of its own. Try including its subsections.")
            return
        pages = max(1, math.ceil(total / PAGE_SIZE))
        page = int(st.number_input("Page", 1, pages, 1, key=f"passages_page_{scope_node.id}")) if pages > 1 else 1
        passages = service.list_passages(document.id, node_ids, offset=(page - 1) * PAGE_SIZE, limit=PAGE_SIZE)
    st.caption(f"{total} passage(s)" + (f" · page {page} of {pages}" if pages > 1 else ""))

    by_id = {n.id: n for n in overview.nodes}
    for passage in passages:
        node = by_id.get(passage.structural_node_id)
        with st.container(border=True):
            cols = st.columns([5, 1])
            cols[0].markdown(
                f"**{node.title if node else 'Section'}** · passage {passage.sequence + 1} · "
                f"{passage.token_count} tokens · "
                f"{format_pages(passage.page_start, passage.page_end, passage.page_status)}"
            )
            preview = " ".join(passage.text.split())
            cols[0].caption(preview[:220] + ("…" if len(preview) > 220 else ""))
            if cols[1].button("Open", key=f"open-passage-{passage.id}"):
                st.session_state[PASSAGE_KEY] = passage.id
                st.rerun()


# -- one passage -----------------------------------------------------------------------------------


def _render_passage(document: DocumentRead, overview, node_ids: list[uuid.UUID], passage_id: uuid.UUID) -> None:
    with session_scope() as session:
        service = StructureService(session, get_settings())
        passage = service.get_passage(document.id, passage_id)
        if passage is None:
            st.session_state[PASSAGE_KEY] = None
            st.warning("That passage does not exist in this document any more.")
            return
        in_scope = passage.structural_node_id in node_ids
        previous_id, next_id, position, total = service.passage_context(
            document.id, passage, node_ids if in_scope else None
        )
        paragraphs = service.get_paragraphs(document.id, passage.paragraph_ids)

    nav = st.columns([1, 1, 2, 2])
    if nav[0].button("◀ Previous", key="passage-prev", disabled=previous_id is None):
        st.session_state[PASSAGE_KEY] = previous_id
        st.rerun()
    if nav[1].button("Next ▶", key="passage-next", disabled=next_id is None):
        st.session_state[PASSAGE_KEY] = next_id
        st.rerun()
    if nav[2].button("Back to the list", key="passage-back"):
        st.session_state[PASSAGE_KEY] = None
        st.rerun()
    if nav[3].button("Show in structure", key="passage-structure"):
        goto("Structure", document_id=document.id, node_id=passage.structural_node_id)
    st.caption(f"Passage {position} of {total} " + ("in this selection" if in_scope else "in the document"))

    _render_context(document, overview, passage)
    st.markdown("#### Text")
    overlap = int((passage.metadata or {}).get("overlap_chars") or 0)
    if overlap:
        st.caption(f"The first {overlap} characters repeat the end of the previous passage (overlap).")
    st.text(passage.text)  # the stored text in full, exactly as saved
    _render_paragraphs(paragraphs, passage)


def _render_context(document: DocumentRead, overview, passage: PassageRead) -> None:
    by_id = {n.id: n for n in overview.nodes}
    node = by_id.get(passage.structural_node_id)
    chain = [
        by_id[i]
        for i in [*reversed(ancestor_ids(overview.nodes, passage.structural_node_id)), passage.structural_node_id]
        if i in by_id
    ]
    chapter = next((n for n in chain if _is_chapter_level(n, by_id)), None)
    section = node if node is not None and chapter is not None and node.id != chapter.id else None

    st.markdown("#### Where this comes from")
    rows = {
        "Document": f"{document.original_filename} ({document.file_format.upper()})",
        "Chapter": _label(chapter) if chapter else "—",
        "Section": _label(section) if section else "—",
        "Section type": node_type_label(node.node_type) if node else "—",
        "Pages": format_pages(passage.page_start, passage.page_end, passage.page_status),
        "Passage": f"#{passage.sequence + 1} in its section · {passage.token_count} tokens",
        "Passage ID": str(passage.id),
    }
    for label, value in rows.items():
        st.markdown(f"**{label}:** {value}")
    if passage.page_status == "unknown":
        st.caption("This format has no fixed pages, so no page reference is available. None has been guessed.")
    elif passage.page_status == "partial":
        st.caption("Only some of the source text had a page number, so the page range is partly known.")
    else:
        st.caption("Page numbers are the PDF's own page numbers (the first page is 1), copied from the extracted text.")


def _render_paragraphs(paragraphs: list[ParagraphRead], passage: PassageRead) -> None:
    st.markdown("#### Paragraphs in this passage")
    if not paragraphs:
        st.caption("No paragraph records are linked to this passage.")
        return
    st.dataframe(
        [
            {
                "Sequence": p.sequence + 1,
                "Kind": p.kind,
                "Pages": format_pages(p.page_start, p.page_end, p.page_status),
                "Text": " ".join(p.text.split())[:140],
            }
            for p in paragraphs
        ],
        hide_index=True,
        width="stretch",
    )
    metadata = passage.metadata or {}
    with st.expander("Source details (for tracing and later extraction)"):
        st.json(
            {
                "passage_id": str(passage.id),
                "book_id": str(passage.book_id),
                "document_id": str(passage.document_id),
                "structural_node_id": str(passage.structural_node_id),
                "paragraph_ids": [str(i) for i in passage.paragraph_ids],
                "page_start": passage.page_start,
                "page_end": passage.page_end,
                "page_status": passage.page_status,
                "start_offset": passage.start_offset,
                "end_offset": passage.end_offset,
                "equation_ids": metadata.get("equation_ids", []),
                "figure_refs": metadata.get("figure_refs", []),
                "table_refs": metadata.get("table_refs", []),
                "bounding_boxes": metadata.get("bboxes", []),
            },
            expanded=False,
        )
