"""Book structure view: table of contents, paragraphs and passages of a processed book.

The three layers are kept visibly apart: the *structure* (what the book itself is organised
into), its *paragraphs* (source text) and the *passages* (derived units for AI processing).
"""

import logging
import math
import uuid

import streamlit as st
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

from app.config import get_settings
from app.schemas import (
    BookRead,
    DocumentRead,
    StructuralNodeRead,
    StructureIssueRead,
    StructureOverview,
)
from app.services.book_service import BookService
from app.services.errors import ProcessingServiceError
from app.services.structure_service import StructureService
from app.structure.segmentation import SegmentationConfig
from app.structure.tree import ancestor_ids, default_expanded, descendant_ids, visible_rows
from app.ui.formatting import format_pages, format_timestamp, node_type_label
from database.connection import session_scope

logger = logging.getLogger(__name__)

PARAGRAPH_PAGE_SIZE = 10
PASSAGE_PAGE_SIZE = 5
LOW_CONFIDENCE = 0.6
_SEVERITY_ICONS = {"error": "🔴", "warning": "🟠", "info": "🔵"}
_TOKENIZERS = ["approx", "words", "whitespace", "tiktoken"]
# The outline is a column of buttons; this left-aligns their labels and tightens the rows.
_TOC_STYLE = (
    "<style>"
    ".st-key-structure_toc button,.st-key-structure_toc button div,.st-key-structure_toc button span"
    "{justify-content:flex-start !important;text-align:left !important;}"
    ".st-key-structure_toc button span{width:100%;}"
    ".st-key-structure_toc button p{text-align:left !important;}"
    ".st-key-structure_toc [data-testid='stVerticalBlock']{gap:0.15rem;}"
    ".st-key-structure_toc [data-testid='stHorizontalBlock']{gap:0.25rem;}"
    "</style>"
)



def _load_documents() -> list[tuple[BookRead, DocumentRead]]:
    with session_scope() as session:
        books = BookService(session).list_books()
    return [
        (book, document)
        for book in books
        for document in book.documents
        if document.processing_status == "completed"
    ]


def _load_overview(document_id: uuid.UUID) -> StructureOverview:
    with session_scope() as session:
        return StructureService(session, get_settings()).get_overview(document_id)


def render_structure_page() -> None:
    st.header("Book structure")
    try:
        entries = _load_documents()
    except SQLAlchemyError:
        logger.exception("Could not load documents for the structure view")
        st.error("Could not load documents. Check the database connection and migrations.")
        return
    if not entries:
        st.info("No processed documents yet. Upload a book and process it first (Processing tab).")
        return

    labels = {
        str(document.id): f"{book.title} — {document.original_filename} ({document.file_format.upper()})"
        for book, document in entries
    }
    selected = st.selectbox(
        "Select a book", options=list(labels), format_func=lambda doc_id: labels[doc_id], key="structure_document"
    )
    document = next(doc for _, doc in entries if str(doc.id) == selected)

    flash = st.session_state.pop("structure_flash", None)
    if flash:
        st.success(flash)

    try:
        overview = _load_overview(document.id)
    except ProcessingServiceError as exc:
        st.error(str(exc))
        return
    except ProgrammingError:
        logger.exception("The structure tables are missing (document %s)", document.id)
        st.error("The structure tables do not exist yet. Run `alembic upgrade head`, then reload this page.")
        return
    except SQLAlchemyError:
        logger.exception("Could not load the structure of document %s", document.id)
        st.error("Could not load the book structure. Check the database connection and the logs.")
        return

    analysis = overview.analysis
    _render_detect_controls(document, overview)
    if analysis is None or analysis.structure_status != "completed":
        if analysis and analysis.structure_error:
            st.error(f"The last structure detection failed: {analysis.structure_error}")
        else:
            st.info("The structure of this book has not been detected yet.")
        return

    if overview.is_outdated:
        st.warning(
            "This document was processed again after its structure was detected. "
            "Detect the structure again to bring it up to date."
        )
    _render_summary(overview)
    _render_passage_controls(document, overview)

    toc_tab, details_tab, warnings_tab = st.tabs(["Table of contents", "Book details", "Warnings"])
    with toc_tab:
        _render_toc_and_node(document, overview)
    with details_tab:
        _render_book_details(overview)
    with warnings_tab:
        _render_warnings(overview)


# -- actions ------------------------------------------------------------------------------


def _run(action, success: str) -> None:
    try:
        with st.spinner("Working…"):
            with session_scope() as session:
                action(StructureService(session, get_settings()))
    except ProcessingServiceError as exc:
        st.error(str(exc))
        return
    except SQLAlchemyError:
        logger.exception("Database error during a structure action")
        st.error("A database error occurred. Check the logs for details.")
        return
    st.session_state["structure_flash"] = success
    st.rerun()


def _render_detect_controls(document: DocumentRead, overview: StructureOverview) -> None:
    detected = overview.analysis is not None and overview.analysis.structure_status == "completed"
    label = "Detect structure again" if detected else "Detect structure"
    if st.button(label, key=f"structure-detect-{document.id}", type="secondary" if detected else "primary"):
        _run(lambda service: service.detect_structure(document.id), "Structure detected.")
    if detected:
        st.caption(
            "Re-detecting replaces the structure, paragraphs and passages of this document. "
            "It reads the already extracted text; the file is not parsed again."
        )


def _render_passage_controls(document: DocumentRead, overview: StructureOverview) -> None:
    analysis = overview.analysis
    assert analysis is not None
    stored = analysis.passage_config or get_settings_defaults()
    with st.expander("Passage settings"):
        st.caption(
            "Passages are derived from the paragraphs. Changing these settings only regenerates "
            "passages: the document is not parsed again and the structure is not detected again."
        )
        key = str(document.id)
        cols = st.columns(4)
        target = cols[0].number_input("Target tokens", 1, 20000, int(stored["target_tokens"]), key=f"ps-target-{key}")
        minimum = cols[1].number_input("Minimum tokens", 1, 20000, int(stored["min_tokens"]), key=f"ps-min-{key}")
        maximum = cols[2].number_input("Maximum tokens", 1, 20000, int(stored["max_tokens"]), key=f"ps-max-{key}")
        overlap = cols[3].number_input("Overlap %", 0, 40, round(float(stored["overlap_ratio"]) * 100), key=f"ps-overlap-{key}")
        tokenizer = st.selectbox(
            "Tokenizer",
            _TOKENIZERS,
            index=_TOKENIZERS.index(stored.get("tokenizer", "approx")) if stored.get("tokenizer", "approx") in _TOKENIZERS else 0,
            key=f"ps-tokenizer-{key}",
        )
        if st.button("Regenerate passages", key=f"ps-run-{key}"):
            try:
                config = SegmentationConfig(
                    target_tokens=int(target),
                    min_tokens=int(minimum),
                    max_tokens=int(maximum),
                    overlap_ratio=overlap / 100,
                    tokenizer=tokenizer,
                    chars_per_token=float(stored.get("chars_per_token", 4.0)),
                    tiktoken_encoding=str(stored.get("tiktoken_encoding", "cl100k_base")),
                    table_split_factor=float(stored.get("table_split_factor", 2.0)),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                _run(lambda service: service.regenerate_passages(document.id, config), "Passages regenerated.")


def get_settings_defaults() -> dict:
    return SegmentationConfig.from_settings(get_settings()).to_dict()


# -- summary ------------------------------------------------------------------------------


def _all_issues(overview: StructureOverview) -> list[tuple[str, StructureIssueRead]]:
    analysis = overview.analysis
    if analysis is None:
        return []
    return [("Structure", i) for i in analysis.structure_issues] + [("Passages", i) for i in analysis.passage_issues]


def _render_summary(overview: StructureOverview) -> None:
    analysis = overview.analysis
    assert analysis is not None
    issues = _all_issues(overview)
    errors = sum(1 for _, i in issues if i.severity == "error")
    warnings = sum(1 for _, i in issues if i.severity == "warning")
    passage_summary = analysis.passage_summary or {}
    cols = st.columns(5)
    cols[0].metric("Structural nodes", len(overview.nodes))
    cols[1].metric("Paragraphs", sum(overview.paragraph_counts.values()))
    cols[2].metric("Passages", sum(overview.passage_counts.values()))
    cols[3].metric("Warnings", warnings)
    cols[4].metric("Errors", errors)
    if analysis.structure_detected_at:
        parts = [f"Structure detected {format_timestamp(analysis.structure_detected_at)}"]
        if analysis.passage_generated_at:
            parts.append(f"passages generated {format_timestamp(analysis.passage_generated_at)}")
        if passage_summary.get("mean_tokens"):
            parts.append(
                f"passage size {passage_summary['min_tokens']}–{passage_summary['max_tokens']} tokens "
                f"(mean {passage_summary['mean_tokens']})"
            )
        st.caption(" · ".join(parts))
    if analysis.passage_status == "failed" and analysis.passage_error:
        st.error(f"The last passage generation failed: {analysis.passage_error}")


# -- table of contents and node details ---------------------------------------------------


def _flagged_nodes(overview: StructureOverview) -> dict[uuid.UUID, list[StructureIssueRead]]:
    flagged: dict[uuid.UUID, list[StructureIssueRead]] = {}
    for _, issue in _all_issues(overview):
        if issue.entity_type == "node" and issue.entity_id and issue.severity in ("error", "warning"):
            flagged.setdefault(issue.entity_id, []).append(issue)
    return flagged


def _node_label(node: StructuralNodeRead, flagged: bool) -> str:
    text = f"{node.number}  {node.title}" if node.number and node.node_type not in ("part", "appendix") else node.title
    if node.node_type == "part" and node.number:
        text = f"Part {node.number}: {node.title}"
    if node.node_type == "appendix" and node.number:
        text = f"Appendix {node.number}: {node.title}"
    return f"{'⚠️ ' if flagged else ''}{text}"


def _render_toc_and_node(document: DocumentRead, overview: StructureOverview) -> None:
    nodes = overview.nodes
    if not nodes:
        st.info("No structure was detected.")
        return
    doc_key = str(document.id)
    expanded_key, selected_key = f"structure_expanded_{doc_key}", f"structure_selected_{doc_key}"
    if expanded_key not in st.session_state:
        st.session_state[expanded_key] = default_expanded(nodes)
    by_id = {n.id: n for n in nodes}
    if st.session_state.get(selected_key) not in by_id:
        st.session_state[selected_key] = nodes[0].id
    expanded: set[uuid.UUID] = st.session_state[expanded_key]
    flagged = _flagged_nodes(overview)

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("#### Structure")
        buttons = st.columns(2)
        if buttons[0].button("Expand all", key=f"expand-all-{doc_key}"):
            st.session_state[expanded_key] = {n.id for n in nodes}
            st.rerun()
        if buttons[1].button("Collapse all", key=f"collapse-all-{doc_key}"):
            st.session_state[expanded_key] = set()
            st.rerun()
        st.markdown(_TOC_STYLE, unsafe_allow_html=True)  # a fixed stylesheet, no user content
        with st.container(key="structure_toc"):
            for row in visible_rows(nodes, expanded):
                node = by_id[row.node.id]
                indent, toggle, label = st.columns([0.35 * row.depth + 0.01, 0.6, 9], vertical_alignment="center")
                if row.has_children:
                    if toggle.button("▾" if row.expanded else "▸", key=f"toggle-{node.id}", type="tertiary"):
                        expanded.symmetric_difference_update({node.id})
                        st.session_state[expanded_key] = expanded
                        st.rerun()
                is_selected = st.session_state[selected_key] == node.id
                if label.button(
                    _node_label(node, node.id in flagged),
                    key=f"select-{node.id}",
                    type="primary" if is_selected else "tertiary",
                    width="stretch",
                ):
                    st.session_state[selected_key] = node.id
                    st.rerun()
    with right:
        _render_node(document, overview, by_id[st.session_state[selected_key]], flagged, selected_key)


def _render_node(
    document: DocumentRead,
    overview: StructureOverview,
    node: StructuralNodeRead,
    flagged: dict[uuid.UUID, list[StructureIssueRead]],
    selected_key: str,
) -> None:
    nodes = overview.nodes
    by_id = {n.id: n for n in nodes}
    trail = [by_id[i].title for i in reversed(ancestor_ids(nodes, node.id)) if i in by_id]
    if trail:
        st.caption(" › ".join(trail))
    st.markdown(f"### {_node_label(node, False)}")
    if node.original_title and node.original_title != node.title:
        st.caption(f"Heading as written: {node.original_title}")

    subtree = descendant_ids(nodes, node.id)
    details = [
        f"**Type:** {node_type_label(node.node_type)}",
        f"**Level:** {node.level}",
        f"**Pages:** {format_pages(node.page_start, node.page_end, 'exact' if node.page_start else 'unknown')}",
    ]
    if node.confidence is not None:
        details.append(f"**Confidence:** {node.confidence:.2f}")
    st.markdown(" · ".join(details))
    own_paragraphs = overview.paragraph_counts.get(node.id, 0)
    own_passages = overview.passage_counts.get(node.id, 0)
    st.caption(
        f"{own_paragraphs} paragraph(s) and {own_passages} passage(s) directly in this "
        f"{node_type_label(node.node_type).lower()}; "
        f"{sum(overview.paragraph_counts.get(i, 0) for i in subtree)} paragraph(s) and "
        f"{sum(overview.passage_counts.get(i, 0) for i in subtree)} passage(s) including everything below."
    )
    for issue in flagged.get(node.id, []):
        st.warning(f"{issue.code}: {issue.message}")
    metadata = node.metadata or {}
    if metadata.get("llm"):
        st.caption(f"LLM classification: {metadata['llm'].get('reasoning') or 'no reasoning given'}")

    children = [n for n in nodes if n.parent_id == node.id]
    contents_tab, passages_tab = st.tabs(["Paragraphs and contents", "Passages"])
    with contents_tab:
        if children:
            st.markdown("**Contains**")
            for child in children:
                if st.button(
                    f"{_node_label(child, child.id in flagged)} · {format_pages(child.page_start, child.page_end)}",
                    key=f"child-{node.id}-{child.id}",
                    type="tertiary",
                ):
                    st.session_state[selected_key] = child.id
                    st.rerun()
        _render_paragraphs(document, node, subtree)
    with passages_tab:
        _render_passages(document, node, subtree)


def _pager(key: str, total: int, size: int) -> int:
    pages = max(1, math.ceil(total / size))
    page = 1
    if pages > 1:
        page = int(st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=key))
    st.caption(f"{total} item(s)" + (f" · page {page} of {pages}" if pages > 1 else ""))
    return page


def _render_paragraphs(document: DocumentRead, node: StructuralNodeRead, subtree: list[uuid.UUID]) -> None:
    st.markdown("**Paragraphs** (source text)")
    include = st.checkbox("Include subsections", value=False, key=f"par-sub-{node.id}") if len(subtree) > 1 else False
    node_ids = subtree if include else [node.id]
    with session_scope() as session:
        service = StructureService(session, get_settings())
        total = service.count_paragraphs(document.id, node_ids)
        if total == 0:
            st.caption("No paragraphs directly under this node." if not include else "No paragraphs.")
            return
        page = _pager(f"par-page-{node.id}-{include}", total, PARAGRAPH_PAGE_SIZE)
        paragraphs = service.list_paragraphs(
            document.id, node_ids, offset=(page - 1) * PARAGRAPH_PAGE_SIZE, limit=PARAGRAPH_PAGE_SIZE
        )
    for paragraph in paragraphs:
        with st.container(border=True):
            st.caption(
                f"¶ {paragraph.sequence + 1} · {paragraph.kind} · "
                f"{format_pages(paragraph.page_start, paragraph.page_end, paragraph.page_status)}"
            )
            st.text(paragraph.text)


def _render_passages(document: DocumentRead, node: StructuralNodeRead, subtree: list[uuid.UUID]) -> None:
    st.markdown("**Passages** (derived for AI processing)")
    include = st.checkbox("Include subsections", value=False, key=f"pas-sub-{node.id}") if len(subtree) > 1 else False
    node_ids = subtree if include else [node.id]
    with session_scope() as session:
        service = StructureService(session, get_settings())
        total = service.count_passages(document.id, node_ids)
        if total == 0:
            st.caption("No passages directly under this node." if not include else "No passages.")
            return
        page = _pager(f"pas-page-{node.id}-{include}", total, PASSAGE_PAGE_SIZE)
        passages = service.list_passages(
            document.id, node_ids, offset=(page - 1) * PASSAGE_PAGE_SIZE, limit=PASSAGE_PAGE_SIZE
        )
    for passage in passages:
        metadata = passage.metadata or {}
        with st.container(border=True):
            st.markdown(
                f"**Passage {passage.sequence + 1}** · {passage.token_count} tokens · "
                f"{format_pages(passage.page_start, passage.page_end, passage.page_status)}"
            )
            overlap = int(metadata.get("overlap_chars") or 0)
            if overlap:
                st.caption(f"The first {overlap} characters repeat the end of the previous passage.")
            if metadata.get("oversized"):
                st.caption(f"Larger than the maximum on purpose, to keep a {metadata['oversized']} in one piece.")
            st.text(passage.text)
            with st.expander("Source details"):
                st.json(
                    {
                        "passage_id": str(passage.id),
                        "book_id": str(passage.book_id),
                        "document_id": str(passage.document_id),
                        "structural_node_id": str(passage.structural_node_id),
                        "paragraph_ids": [str(p) for p in passage.paragraph_ids],
                        "page_start": passage.page_start,
                        "page_end": passage.page_end,
                        "page_status": passage.page_status,
                        "start_offset": passage.start_offset,
                        "end_offset": passage.end_offset,
                        "equation_ids": metadata.get("equation_ids", []),
                        "figure_refs": metadata.get("figure_refs", []),
                        "table_refs": metadata.get("table_refs", []),
                        "bboxes": metadata.get("bboxes", []),
                    },
                    expanded=False,
                )


# -- book details and warnings -------------------------------------------------------------


def _render_book_details(overview: StructureOverview) -> None:
    analysis = overview.analysis
    assert analysis is not None
    book = overview.book
    detected = (analysis.structure_summary or {}).get("detected_metadata") or {}
    st.markdown("#### Title and metadata")
    st.caption("Stored values were entered by you and are never overwritten by uncertain detections.")

    def confidence(key: str) -> str:
        value = detected.get(f"{key}_confidence")
        return "" if value is None else f"{value:.2f}"

    st.dataframe(
        [
            {"Field": "Title", "Stored": book.title, "Detected": detected.get("title") or "", "Confidence": confidence("title")},
            {"Field": "Subtitle", "Stored": book.subtitle or "", "Detected": detected.get("subtitle") or "", "Confidence": confidence("subtitle")},
            {"Field": "Author", "Stored": book.author or "", "Detected": detected.get("author") or "", "Confidence": confidence("author")},
        ],
        hide_index=True,
        width="stretch",
    )
    if detected.get("title_source"):
        st.caption(f"Title found from: {detected['title_source'].replace('_', ' ')}.")
    if detected.get("copyright_page"):
        st.caption(f"Copyright information found on page {detected['copyright_page']}.")
    applied = (analysis.structure_summary or {}).get("applied_book_metadata") or {}
    if applied:
        st.info("Filled in empty book fields: " + ", ".join(f"{k} = {v}" for k, v in applied.items()))

    st.markdown("#### What was detected")
    summary = analysis.structure_summary or {}
    types = summary.get("node_types") or {}
    st.dataframe(
        [{"Type": node_type_label(t), "Count": c} for t, c in sorted(types.items(), key=lambda kv: -kv[1])],
        hide_index=True,
        width="stretch",
    )
    extras = []
    if summary.get("toc_entries"):
        extras.append(f"{summary['toc_entries']} table-of-contents entries found")
    if summary.get("llm_classified"):
        extras.append(f"{summary['llm_classified']} heading(s) classified with the LLM")
    if summary.get("headings_rejected"):
        extras.append(f"{summary['headings_rejected']} heading-like block(s) kept as text")
    if extras:
        st.caption(" · ".join(extras))


def _render_warnings(overview: StructureOverview) -> None:
    issues = _all_issues(overview)
    by_id = {n.id: n for n in overview.nodes}
    if not issues:
        st.success("Validation found no problems.")
    else:
        chosen = st.multiselect("Show", ["error", "warning", "info"], default=["error", "warning"], key="structure_severity")
        shown = [(step, i) for step, i in issues if i.severity in chosen]
        st.caption(f"{len(shown)} of {len(issues)} finding(s)")
        st.dataframe(
            [
                {
                    "": _SEVERITY_ICONS.get(issue.severity, ""),
                    "Step": step,
                    "Code": issue.code,
                    "Where": by_id[issue.entity_id].title if issue.entity_id in by_id else (issue.entity_type or ""),
                    "Message": issue.message,
                }
                for step, issue in shown
            ],
            hide_index=True,
            width="stretch",
        )

    uncertain = [
        n
        for n in overview.nodes
        if (n.confidence is not None and n.confidence < LOW_CONFIDENCE)
        or (n.metadata or {}).get("status") == "ambiguous"
    ]
    st.markdown("#### Low-confidence structure")
    if not uncertain:
        st.caption("Every structural element was classified with confidence.")
        return
    st.dataframe(
        [
            {
                "Title": n.title,
                "Type": node_type_label(n.node_type),
                "Confidence": "" if n.confidence is None else f"{n.confidence:.2f}",
                "Pages": format_pages(n.page_start, n.page_end),
                "Signals": ", ".join((n.metadata or {}).get("signals", [])),
            }
            for n in uncertain
        ],
        hide_index=True,
        width="stretch",
    )
