"""Library screen: every book and document with its processing state, and what you can do next."""

import logging

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.schemas import BookRead
from app.services.document_service import DocumentService
from app.services.errors import ProcessingServiceError
from app.services.library_service import LibraryBook, LibraryDocument
from app.services.storage_service import get_storage
from app.ui.formatting import STAGE_LABELS, format_size, format_timestamp, stage_label, status_label
from app.ui.navigation import DOCUMENT_KEY, goto
from app.ui.processing import get_job_runner
from app.ui.selection import safe_load_library
from database.connection import session_scope

logger = logging.getLogger(__name__)

POLL_SECONDS = 3
_STAGE_ORDER = list(STAGE_LABELS)


def _book_label(book: BookRead) -> str:
    return f"{book.title} — {book.author}" if book.author else book.title


def render_library_page() -> None:
    st.header("Library")
    books = safe_load_library()
    if books is None:
        return
    active = any(book.has_active for book in books)
    # Poll while anything is processing, so the status is never stale; the fragment
    # reruns the whole page once the last run finishes, which turns polling off.
    st.fragment(run_every=POLL_SECONDS if active else None)(_render_library)(was_active=active)


def _render_library(*, was_active: bool) -> None:
    books = safe_load_library()
    if books is None:
        return
    if was_active and not any(book.has_active for book in books):
        st.rerun()

    if not books:
        st.info("The library is empty. Upload a book to get started.")
        if st.button("Go to Upload", key="library_go_upload"):
            goto("Upload")
        return

    st.dataframe(
        [
            {
                "Title": item.book.title,
                "Author": item.book.author or "",
                "Documents": len(item.documents),
                "Format": document.document.file_format.upper(),
                "Size": format_size(document.document.file_size),
                "Uploaded": format_timestamp(document.document.created_at),
                "Status": status_label(
                    document.document.processing_status, document.stage, incomplete=document.extraction_incomplete
                ),
                "Passages": document.passage_count,
            }
            for item in books
            for document in item.documents
        ],
        hide_index=True,
        width="stretch",
        column_config={"Title": st.column_config.TextColumn(width="large")},
    )

    st.subheader("Open a book")
    by_id = {str(item.book.id): item for item in books}
    wanted = st.session_state.get(DOCUMENT_KEY)
    options = list(by_id)
    default = next(
        (book_id for book_id, item in by_id.items() if any(str(d.document.id) == wanted for d in item.documents)),
        None,
    )
    selected_id = st.selectbox(
        "Select a book",
        options=options,
        index=options.index(default) if default in options else 0,
        format_func=lambda book_id: _book_label(by_id[book_id].book),
        key="library_book",
    )
    _render_book(by_id[selected_id])


def _render_book(item: LibraryBook) -> None:
    book = item.book
    st.markdown(f"### {book.title}")
    if book.subtitle:
        st.markdown(f"*{book.subtitle}*")
    details = {
        "Author": book.author,
        "Publisher": book.publisher,
        "Year": book.publication_year,
        "ISBN": book.isbn,
        "Language": book.language,
        "Added": format_timestamp(book.created_at),
    }
    st.markdown("  ·  ".join(f"**{label}:** {value}" for label, value in details.items() if value))
    if book.description:
        st.markdown(book.description)

    st.markdown(f"**Documents and editions ({len(item.documents)})**")
    if not item.documents:
        st.caption("This book has no documents.")
    else:
        labels = {
            str(d.document.id): f"{d.document.original_filename} · {d.document.file_format.upper()} · "
            f"uploaded {format_timestamp(d.document.created_at)}"
            for d in item.documents
        }
        ids = list(labels)
        wanted = st.session_state.get(DOCUMENT_KEY)
        chosen = st.radio(
            "Select a document or edition",
            options=ids,
            index=ids.index(wanted) if wanted in ids else 0,
            format_func=lambda doc_id: labels[doc_id],
            key=f"library_document_{book.id}",
        )
        st.session_state[DOCUMENT_KEY] = chosen
        selected = next(d for d in item.documents if str(d.document.id) == chosen)
        _render_document(item, selected)

    _render_delete_book(item)


def _progress_fraction(stage: str | None) -> float:
    if stage in _STAGE_ORDER:
        return _STAGE_ORDER.index(stage) / (len(_STAGE_ORDER) - 1)
    return 0.0


def _render_document(item: LibraryBook, entry: LibraryDocument) -> None:
    document = entry.document
    status = document.processing_status
    with st.container(border=True):
        st.markdown(
            f"**{document.original_filename}** · {document.file_format.upper()} · {format_size(document.file_size)}"
        )
        st.markdown(f"**Status:** {status_label(status, entry.stage, incomplete=entry.extraction_incomplete)}")

        if entry.is_active:
            text = "Queued…" if status == "pending" else f"{stage_label(entry.stage)}…"
            if entry.progress_total:
                text += f" ({entry.progress_current or 0} of {entry.progress_total})"
            st.progress(_progress_fraction(entry.stage), text=text)

        times = []
        if entry.started_at:
            times.append(f"started {format_timestamp(entry.started_at)}")
        if entry.completed_at:
            times.append(f"finished {format_timestamp(entry.completed_at)}")
        if entry.updated_at and entry.is_active:
            times.append(f"updated {format_timestamp(entry.updated_at)}")
        if times:
            st.caption(" · ".join(times))

        if status == "failed":
            st.error(f"{entry.error_message or 'Processing failed.'}  \nFailed during: {stage_label(entry.stage)}.")
            if entry.retry_count:
                st.caption(f"{entry.retry_count} retry attempt(s) so far.")
        elif status == "completed":
            if entry.extraction_incomplete:
                st.warning("Some pages could not be read, so the text of this document is incomplete.")
            st.caption(f"{entry.node_count} structural nodes and {entry.passage_count} passages.")

        _render_actions(item, entry)


def _render_actions(item: LibraryBook, entry: LibraryDocument) -> None:
    document = entry.document
    status = document.processing_status
    key = str(document.id)
    columns = st.columns(4)

    if entry.is_processed:
        if columns[0].button("Open structure", key=f"library_open_{key}", type="primary"):
            goto("Structure", document_id=document.id)
        if columns[1].button("Open passages", key=f"library_passages_{key}"):
            goto("Passages", document_id=document.id)
    elif status == "completed":
        # Text was extracted before structure detection existed: it can be detected from the stored text.
        if columns[0].button(
            "Detect structure", key=f"library_open_{key}", type="primary",
            help="This document has extracted text but no structure yet. Detecting it does not parse the file again.",
        ):
            goto("Structure", document_id=document.id)
    elif status in ("uploaded", "failed"):
        label = "Retry processing" if status == "failed" else "Start processing"
        if columns[0].button(label, key=f"library_process_{key}", type="primary"):
            try:
                get_job_runner().submit(document.id)
            except ProcessingServiceError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    if columns[2].button("Processing details", key=f"library_details_{key}"):
        goto("Processing", document_id=document.id)
    _render_delete_document(item, entry, columns[3])


def _render_delete_document(item: LibraryBook, entry: LibraryDocument, column) -> None:
    document = entry.document
    confirm_key = f"library_confirm_delete_{document.id}"
    label = f"{document.original_filename} ({document.file_format.upper()})"
    if not st.session_state.get(confirm_key):
        if column.button(
            "Delete", key=f"library_delete_{document.id}", help=f"Delete {label}", disabled=entry.is_active
        ):
            st.session_state[confirm_key] = True
            st.rerun()
        return

    extra = " It is the only document of this book, so the book is removed too." if len(item.documents) == 1 else ""
    st.warning(f"Delete **{label}** with its text, structure and passages? This cannot be undone.{extra}")
    yes, no = st.columns(2)
    if yes.button("Yes, delete", key=f"library_delete_yes_{document.id}", type="primary"):
        st.session_state.pop(confirm_key, None)
        _delete(lambda service: service.delete_document(document.id), document.id)
    if no.button("Cancel", key=f"library_delete_no_{document.id}"):
        st.session_state.pop(confirm_key, None)
        st.rerun()


def _render_delete_book(item: LibraryBook) -> None:
    book = item.book
    confirm_key = f"library_confirm_delete_book_{book.id}"
    if not st.session_state.get(confirm_key):
        if st.button("Delete this book", key=f"library_delete_book_{book.id}", disabled=item.has_active):
            st.session_state[confirm_key] = True
            st.rerun()
        return
    st.warning(
        f"Delete **{book.title}** and all {len(item.documents)} of its document(s), with their text, "
        "structure and passages? This cannot be undone."
    )
    yes, no = st.columns(2)
    if yes.button("Yes, delete the book", key=f"library_delete_book_yes_{book.id}", type="primary"):
        st.session_state.pop(confirm_key, None)
        _delete(lambda service: service.delete_book(book.id), None)
    if no.button("Cancel", key=f"library_delete_book_no_{book.id}"):
        st.session_state.pop(confirm_key, None)
        st.rerun()


def _delete(action, document_id) -> None:
    try:
        with session_scope() as session:
            action(DocumentService(session, get_storage(), get_settings()))
    except ProcessingServiceError as exc:
        st.error(str(exc))
        return
    except SQLAlchemyError:
        logger.exception("Could not delete")
        st.error("The deletion failed. Check the database connection and the logs.")
        return
    if document_id is not None and st.session_state.get(DOCUMENT_KEY) == str(document_id):
        st.session_state.pop(DOCUMENT_KEY, None)
    st.rerun()
