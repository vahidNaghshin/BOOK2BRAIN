import logging
import math
import uuid

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.schemas import BookRead, DocumentRead, EquationRead, ProcessingInfo
from app.services.book_service import BookService
from app.services.document_processing_service import DocumentProcessingService
from app.services.errors import ProcessingServiceError
from app.services.storage_service import get_storage
from app.ui.formatting import format_timestamp
from app.ui.processing import get_job_runner
from database.connection import session_scope

logger = logging.getLogger(__name__)

PAGE_SIZE = 10
POLL_SECONDS = 2
_ALL = "All"
_TYPES = [_ALL, "inline", "display", "numbered", "unknown"]
_STATUSES = [_ALL, "valid", "uncertain", "invalid", "pending"]
_STATUS_BADGES = {"valid": ":green[valid]", "uncertain": ":orange[uncertain]", "invalid": ":red[invalid]", "pending": "pending"}
_METHOD_LABELS = {
    "docling_enrichment": "Docling formula model",
    "math_ocr": "Math OCR",
    "mathml": "MathML",
    "omml": "Word equation (OMML)",
    "latex_delimiters": "LaTeX in the text",
    "image_alt_text": "Image alt text (LaTeX)",
    "tex_annotation": "TeX annotation in MathML",
}
_STAGE_LABELS = {
    "starting": "Starting",
    "formula_enrichment": "Recognizing formulas (Docling)",
    "recognizing": "Recognizing equations",
    "math_ocr": "Recognizing equation images",
    "parsing": "Re-reading the document",
    "ocr": "Re-reading scanned pages",
    "normalizing": "Normalizing",
}


def _load_documents() -> list[tuple[BookRead, DocumentRead]]:
    with session_scope() as session:
        books = BookService(session).list_books()
    return [(book, document) for book in books for document in book.documents]


def _service(session) -> DocumentProcessingService:
    return DocumentProcessingService(session, get_storage(), get_settings())


def _load_info(document_id: uuid.UUID) -> ProcessingInfo:
    with session_scope() as session:
        return _service(session).get_info(document_id)


def render_equations_page() -> None:
    st.header("Equations")
    try:
        entries = _load_documents()
    except SQLAlchemyError:
        logger.exception("Could not load documents for equations")
        st.error("Could not load documents. Check the database connection and migrations.")
        return
    if not entries:
        st.info("No documents yet. Upload a book first.")
        return

    # Labels stay constant while work runs, or Streamlit resets the selection.
    labels = {
        str(document.id): f"{book.title} — {document.original_filename} ({document.file_format.upper()})"
        for book, document in entries
    }
    selected = st.selectbox(
        "Select a document",
        options=list(labels),
        format_func=lambda doc_id: labels[doc_id],
        key="equations_document",
    )
    document = next(doc for _, doc in entries if str(doc.id) == selected)

    info = _load_info(document.id)
    active = _extracting(info)
    st.fragment(run_every=POLL_SECONDS if active else None)(_render_document)(document, was_active=active)


def _extracting(info: ProcessingInfo) -> bool:
    run = info.latest_run
    return bool(run and run.equation_status == "processing" and not info.equation_is_stale)


def _render_document(document: DocumentRead, *, was_active: bool) -> None:
    info = _load_info(document.id)
    active = _extracting(info)
    if was_active and not active:
        st.rerun()  # extraction finished: refresh the whole page and stop polling

    if info.document_status != "completed" or info.latest_run is None:
        st.info("Process this document first (Processing tab). Equations are extracted once its text is saved.")
        return

    _render_status(document, info)
    if active:
        return
    _render_list(document)


# -- status and actions -------------------------------------------------------------------


def _start(document_id: uuid.UUID) -> None:
    try:
        get_job_runner().submit_equations(document_id)
    except ProcessingServiceError as exc:
        st.error(str(exc))
        return
    st.rerun()


def _render_status(document: DocumentRead, info: ProcessingInfo) -> None:
    run = info.latest_run
    status = run.equation_status
    summary = run.equation_summary or {}

    if status == "processing":
        if info.equation_is_stale:
            st.warning("A previous equation extraction looks interrupted. You can start it again.")
        else:
            progress = summary.get("progress") or {}
            label = _STAGE_LABELS.get(progress.get("stage", "starting"), progress.get("stage", "Working"))
            total = progress.get("total") or 0
            if total:
                st.progress(min((progress.get("current") or 0) / total, 1.0), text=f"{label} ({progress.get('current', 0)} of {total})")
            else:
                st.progress(0.0, text=f"{label}…")
            st.caption("Extracting equations… this page updates automatically. Your text is already saved.")
            return
    elif status == "not_started":
        st.info("Equations have not been extracted for this document yet.")
    elif status == "failed":
        st.error("Equation extraction failed. The document text is unaffected.")
    elif status == "unavailable":
        st.warning("Some equations could not be recognized because no recognition engine could run.")
    elif status == "completed_with_warnings":
        st.warning("Equations extracted with warnings. Check the entries marked uncertain or invalid.")
    elif status == "completed":
        st.success("Equations extracted.")

    if summary.get("total") is not None:
        cols = st.columns(4)
        cols[0].metric("Equations", summary.get("total", 0))
        cols[1].metric("Valid", summary.get("valid", 0))
        cols[2].metric("Uncertain", summary.get("uncertain", 0))
        cols[3].metric("Invalid", summary.get("invalid", 0))
        _render_methods(summary)
    _render_issues(summary)

    label = "Extract equations" if status in ("not_started",) else "Re-extract equations"
    if st.button(label, type="primary" if status == "not_started" else "secondary", key=f"equations-start-{document.id}"):
        _start(document.id)
    if status not in ("not_started", "processing"):
        st.caption("Re-extracting replaces the stored equations; it never changes the document text.")


def _render_methods(summary: dict) -> None:
    enrichment = summary.get("enrichment") or {}
    ocr = summary.get("math_ocr") or {}
    parts = []
    if enrichment.get("executed"):
        parts.append(
            f"Docling formula model on page(s) {', '.join(map(str, enrichment.get('pages', [])))} "
            f"({enrichment.get('formulas_enriched', 0)} of {enrichment.get('formulas_seen', 0)} recognized, "
            f"{enrichment.get('device')}/{enrichment.get('dtype')}, {enrichment.get('seconds')} s)"
        )
    if ocr.get("attempted"):
        parts.append(f"math OCR '{ocr.get('engine')}' on {ocr['attempted']} region(s) ({ocr.get('succeeded', 0)} succeeded)")
    if ocr.get("pictures_accepted") or ocr.get("pictures_rejected"):
        parts.append(
            f"{ocr.get('pictures_accepted', 0)} picture region(s) turned out to be equations, "
            f"{ocr.get('pictures_rejected', 0)} were figures"
        )
    if parts:
        st.caption("Method: " + "; ".join(parts) + f". Total {summary.get('seconds')} s.")


def _render_issues(summary: dict) -> None:
    issues = summary.get("issues") or []
    if issues:
        with st.expander(f"Warnings ({len(issues)})"):
            for issue in issues:
                prefix = f"Page {issue['page_number']}: " if issue.get("page_number") else ""
                st.markdown(f"- **{issue['code']}** — {prefix}{issue['message']}")


# -- the equation list ----------------------------------------------------------------------


def _render_list(document: DocumentRead) -> None:
    key = str(document.id)
    cols = st.columns(3)
    equation_type = cols[0].selectbox("Type", _TYPES, key=f"eq-type-{key}")
    status = cols[1].selectbox("Validation", _STATUSES, key=f"eq-status-{key}")
    page = cols[2].number_input("Page (0 = all)", min_value=0, value=0, step=1, key=f"eq-page-{key}")
    filters = {
        "equation_type": None if equation_type == _ALL else equation_type,
        "validation_status": None if status == _ALL else status,
        "page_number": int(page) or None,
    }

    with session_scope() as session:
        service = _service(session)
        total = service.count_equations(document.id, **filters)
        if total == 0:
            st.info("No equations match." if any(filters.values()) else "No equations were found in this document.")
            return
        pages = max(1, math.ceil(total / PAGE_SIZE))
        current = (
            st.number_input("Result page", min_value=1, max_value=pages, value=1, step=1, key=f"eq-pg-{key}") if pages > 1 else 1
        )
        equations = service.list_equations(document.id, offset=(int(current) - 1) * PAGE_SIZE, limit=PAGE_SIZE, **filters)
        crops = {e.id: service.read_crop(e.crop_reference) for e in equations if e.crop_reference}
    st.caption(f"{total} equation(s)" + (f" · page {current} of {pages}" if pages > 1 else ""))
    for equation in equations:
        _render_equation(equation, crops.get(equation.id))


def _render_equation(equation: EquationRead, crop: bytes | None) -> None:
    with st.container(border=True):
        header = [f"**#{equation.sequence_number + 1}**", equation.equation_type]
        if equation.equation_number:
            header.append(f"({equation.equation_number})")
        if equation.page_number:
            header.append(f"page {equation.page_number}")
        header.append(_STATUS_BADGES.get(equation.validation_status, equation.validation_status))
        st.markdown(" · ".join(header))

        if equation.latex and equation.validation_status != "invalid":
            st.latex(equation.latex)
        elif equation.latex:
            st.warning("This LaTeX failed validation and is not rendered. See the details below.")
        else:
            st.warning("No LaTeX could be recovered for this equation.")
        for warning in equation.validation_warnings or []:
            st.caption(f"⚠ {warning}")

        with st.expander("Details"):
            if equation.latex:
                st.caption("LaTeX (normalized)")
                st.code(equation.latex, language="latex")
            if equation.original_latex and equation.original_latex != equation.latex:
                st.caption("LaTeX as extracted")
                st.code(equation.original_latex, language="latex")
            if equation.plain_text:
                st.caption("Text found in the source")
                st.code(equation.plain_text, language="text")
            confidence = "not reported by the engine" if equation.confidence is None else f"{equation.confidence:.2f}"
            st.markdown(
                f"**Method:** {_METHOD_LABELS.get(equation.extraction_method, equation.extraction_method)} · "
                f"**Confidence:** {confidence} · **Extracted:** {format_timestamp(equation.created_at)}"
            )
            if equation.section_title:
                st.markdown(f"**Section:** {equation.section_title}")
            if equation.surrounding_text:
                st.caption("Surrounding text")
                st.write(equation.surrounding_text)
            if equation.source_location:
                st.caption("Source location")
                st.json(equation.source_location, expanded=False)
            if crop:
                st.caption("Region image")
                st.image(crop)
