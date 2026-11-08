import logging
import math
import uuid

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.schemas import BookRead, DocumentRead, ProcessingInfo
from app.services.book_service import BookService
from app.services.document_processing_service import DocumentProcessingService
from app.services.errors import ProcessingServiceError
from app.services.job_runner import ProcessingJobRunner
from app.services.storage_service import get_storage
from app.ui.formatting import format_size, format_timestamp
from database.connection import session_scope

logger = logging.getLogger(__name__)

PREVIEW_PAGE_SIZE = 20
POLL_SECONDS = 2
_STAGE_LABELS = {
    "parsing": "Parsing the document",
    "ocr": "Running OCR on scanned pages",
    "normalizing": "Normalizing text",
}


@st.cache_resource
def get_job_runner() -> ProcessingJobRunner:
    return ProcessingJobRunner(get_settings())


def _load_documents() -> list[tuple[BookRead, DocumentRead]]:
    with session_scope() as session:
        books = BookService(session).list_books()
    return [(book, document) for book in books for document in book.documents]


def _load_info(document_id: uuid.UUID) -> ProcessingInfo:
    with session_scope() as session:
        service = DocumentProcessingService(session, get_storage(), get_settings())
        return service.get_info(document_id)


def render_processing_page() -> None:
    st.header("Processing")
    try:
        entries = _load_documents()
    except SQLAlchemyError:
        logger.exception("Could not load documents for processing")
        st.error("Could not load documents. Check the database connection and migrations.")
        return
    if not entries:
        st.info("No documents yet. Upload a book first.")
        return

    # Labels must not contain anything that changes while processing runs (such as the
    # status), or Streamlit treats the selectbox as a new widget and resets the selection.
    labels = {
        str(document.id): (
            f"{book.title} — {document.original_filename} ({document.file_format.upper()})"
        )
        for book, document in entries
    }
    selected = st.selectbox(
        "Select a document",
        options=list(labels),
        format_func=lambda doc_id: labels[doc_id],
        key="processing_document",
    )
    document_id = uuid.UUID(selected)
    document = next(doc for _, doc in entries if doc.id == document_id)

    info = _load_info(document_id)
    active = info.document_status in ("pending", "processing") and not info.is_stale
    # Poll while a run is active; a full rerun (see below) turns polling off again.
    st.fragment(run_every=POLL_SECONDS if active else None)(_render_status)(document, was_active=active)


def _render_status(document: DocumentRead, *, was_active: bool) -> None:
    info = _load_info(document.id)
    status = info.document_status
    active = status in ("pending", "processing") and not info.is_stale

    if was_active and not active:
        st.rerun()  # the run finished: refresh the whole page and stop polling

    st.caption(
        f"{document.original_filename} · {document.file_format.upper()} · "
        f"{format_size(document.file_size)} · uploaded {format_timestamp(document.created_at)}"
    )

    if active:
        _render_running(info)
    elif status == "completed":
        _render_completed(document, info)
    elif status == "failed":
        _render_failed(document, info)
    else:
        _render_not_processed(document, info)


# -- states -----------------------------------------------------------------------


def _start(document_id: uuid.UUID) -> None:
    try:
        get_job_runner().submit(document_id)
    except ProcessingServiceError as exc:
        st.error(str(exc))
        return
    st.rerun()


def _render_not_processed(document: DocumentRead, info: ProcessingInfo) -> None:
    st.info("Not processed yet. Processing extracts clean text and keeps its source locations.")
    if info.is_stale:
        st.warning("A previous run looks interrupted. You can start processing again.")
    if st.button("Start processing", type="primary", key=f"start-{document.id}"):
        _start(document.id)


def _render_running(info: ProcessingInfo) -> None:
    run = info.latest_run
    stage = (run.stage if run else None) or "parsing"
    label = _STAGE_LABELS.get(stage, stage)
    total = run.progress_total if run and run.progress_total else 0
    current = run.progress_current if run and run.progress_current else 0
    if info.document_status == "pending":
        st.progress(0.0, text="Queued…")
    elif total:
        st.progress(min(current / total, 1.0), text=f"{label} ({current} of {total})")
    else:
        st.progress(0.0, text=f"{label}…")
    st.caption(f"Processing… this page updates automatically. Attempt {info.run_count}.")


def _render_failed(document: DocumentRead, info: ProcessingInfo) -> None:
    run = info.latest_run
    st.error(f"Processing failed. {run.error_message if run and run.error_message else ''}".strip())
    failed_stage = run.stage if run else None
    if failed_stage:
        st.caption(f"Failed during: {_STAGE_LABELS.get(failed_stage, failed_stage)}")

    run_errors = [e for e in info.errors if run and e.processing_run_id == run.id]
    for error in run_errors:
        with st.expander(f"{error.error_type}: {error.error_message[:120]}"):
            st.write(error.error_message)
            details = {k: v for k, v in (error.details or {}).items() if k != "traceback"}
            if details:
                st.json(details)
            if error.details and error.details.get("traceback"):
                st.code(error.details["traceback"], language="text")
    if info.run_count > 1:
        st.caption(f"{info.run_count} attempts so far.")
    if st.button("Retry processing", type="primary", key=f"retry-{document.id}"):
        _start(document.id)


def _render_completed(document: DocumentRead, info: ProcessingInfo) -> None:
    run = info.latest_run
    summary = run.summary if run else None
    if summary is None:
        st.warning("Processed, but no summary was stored.")
        return

    if summary.status.value == "partial":
        st.warning("Processed with some content missing. See the warnings below.")
    else:
        st.success("Processed successfully.")

    cols = st.columns(4)
    cols[0].metric("Characters", f"{summary.character_count:,}")
    cols[1].metric("Pages", summary.page_count if summary.page_count is not None else "n/a")
    cols[2].metric("Blocks", f"{summary.block_count:,}")
    cols[3].metric("OCR pages", len(summary.ocr.pages_processed) if summary.ocr else 0)
    st.caption(
        f"Extracted with {summary.extraction_method} on {format_timestamp(summary.extracted_at)}."
    )
    if summary.page_count is None:
        st.caption("This format has no fixed pages, so blocks are located by chapter or position instead.")

    _render_ocr(document, summary)
    if summary.warnings:
        with st.expander(f"Warnings ({len(summary.warnings)})"):
            for warning in summary.warnings:
                prefix = f"Page {warning.page_number}: " if warning.page_number else ""
                st.markdown(f"- **{warning.code}** — {prefix}{warning.message}")

    _render_preview(document)


def _render_ocr(document: DocumentRead, summary) -> None:
    if document.file_format != "pdf":
        return
    ocr = summary.ocr
    if ocr is None:
        st.info("No OCR was needed: every page had extractable text.")
        return
    used = [p for p in ocr.pages_processed if p not in ocr.pages_kept_digital]
    if used:
        confidence = f", mean confidence {ocr.mean_confidence}%" if ocr.mean_confidence is not None else ""
        st.info(
            f"OCR ({ocr.engine}, language '{ocr.language}', {ocr.dpi} DPI) was used on "
            f"{len(used)} page(s): {', '.join(map(str, used))}{confidence}."
        )
    if ocr.pages_kept_digital:
        st.caption(
            "OCR ran but the extracted text was kept on page(s): "
            + ", ".join(map(str, ocr.pages_kept_digital))
        )
    if ocr.pages_failed:
        st.error("OCR failed on page(s): " + ", ".join(map(str, ocr.pages_failed)))


def _render_preview(document: DocumentRead) -> None:
    st.subheader("Extracted text preview")
    settings = get_settings()
    with session_scope() as session:
        service = DocumentProcessingService(session, get_storage(settings), settings)

        controls = st.columns([2, 2, 3])
        page_filter = None
        if document.file_format == "pdf":
            value = controls[1].number_input(
                "PDF page (0 = all)", min_value=0, value=0, step=1, key=f"pdfpage-{document.id}"
            )
            page_filter = int(value) or None
        show_original = controls[2].checkbox(
            "Show original text (before normalization)", key=f"raw-{document.id}"
        )

        total = service.count_blocks(document.id, page_filter)
        pages = max(1, math.ceil(total / PREVIEW_PAGE_SIZE))
        preview_page = controls[0].number_input(
            f"Preview page (1–{pages})",
            min_value=1,
            max_value=pages,
            value=1,
            step=1,
            key=f"preview-{document.id}-{page_filter}",
        )
        blocks = service.list_blocks(
            document.id,
            offset=(int(preview_page) - 1) * PREVIEW_PAGE_SIZE,
            limit=PREVIEW_PAGE_SIZE,
            page_number=page_filter,
        )

    if not blocks:
        st.caption("No blocks to show.")
        return
    st.caption(f"Showing {len(blocks)} of {total:,} blocks.")
    for block in blocks:
        parts = [f"#{block.sequence_number}", block.block_type]
        if block.page_number is not None:
            parts.append(f"page {block.page_number}")
        location = block.source_location or {}
        if "item_id" in location:
            parts.append(f"chapter item {location['item_id']}")
        if block.metadata and block.metadata.get("ocr"):
            parts.append(f"OCR {block.metadata.get('ocr_confidence')}%")
        if block.section_title:
            parts.append(f"section: {block.section_title[:60]}")
        st.caption(" · ".join(parts))
        st.text(block.raw_text if show_original and block.raw_text is not None else block.text)
