"""Book and document (edition) selectors shared by the screens that work on one document."""

import logging
import uuid
from collections.abc import Callable

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.services.library_service import LibraryBook, LibraryDocument, LibraryService
from app.ui.formatting import format_timestamp
from app.ui.navigation import DOCUMENT_KEY, SYNC_KEY
from database.connection import session_scope

logger = logging.getLogger(__name__)

_BOOK_WIDGET = "selector_book"
_DOCUMENT_WIDGET = "selector_document"


def load_library() -> list[LibraryBook]:
    with session_scope() as session:
        return LibraryService(session).list_books()


def document_label(item: LibraryDocument) -> str:
    """Identifies an edition without anything that changes while it is processed (the status)."""
    document = item.document
    return f"{document.original_filename} · {document.file_format.upper()} · uploaded {format_timestamp(document.created_at)}"


def render_document_selector(
    books: list[LibraryBook], *, predicate: Callable[[LibraryDocument], bool], key: str = "doc"
) -> tuple[LibraryBook, LibraryDocument] | None:
    """Select a book, then one of its documents. Returns the pair, or None when nothing qualifies.

    The choice is remembered across screens. Only documents for which `predicate` is true are
    offered, and books with none are left out.
    """
    eligible: list[tuple[LibraryBook, list[LibraryDocument]]] = []
    for book in books:
        documents = [d for d in book.documents if predicate(d)]
        if documents:
            eligible.append((book, documents))
    if not eligible:
        return None

    by_book = {str(book.book.id): (book, documents) for book, documents in eligible}
    wanted = st.session_state.get(DOCUMENT_KEY)
    if st.session_state.pop(SYNC_KEY, False) or _BOOK_WIDGET + key not in st.session_state:
        for book_id, (_, documents) in by_book.items():
            if any(str(d.document.id) == wanted for d in documents):
                st.session_state[_BOOK_WIDGET + key] = book_id
                st.session_state[_DOCUMENT_WIDGET + key] = wanted

    cols = st.columns(2)
    book_id = cols[0].selectbox(
        "Book",
        options=list(by_book),
        format_func=lambda value: by_book[value][0].book.title
        + (f" — {by_book[value][0].book.author}" if by_book[value][0].book.author else ""),
        key=_BOOK_WIDGET + key,
    )
    book, documents = by_book[book_id]
    options = {str(d.document.id): d for d in documents}
    if st.session_state.get(_DOCUMENT_WIDGET + key) not in options:
        st.session_state.pop(_DOCUMENT_WIDGET + key, None)
    document_id = cols[1].selectbox(
        "Document / edition",
        options=list(options),
        format_func=lambda value: document_label(options[value]),
        key=_DOCUMENT_WIDGET + key,
        help="A book can have several documents (formats or editions); each has its own structure and page numbers.",
    )
    st.session_state[DOCUMENT_KEY] = document_id
    return book, options[document_id]


def safe_load_library() -> list[LibraryBook] | None:
    """The library, or None after showing an error (so a screen can simply stop)."""
    try:
        return load_library()
    except SQLAlchemyError:
        logger.exception("Could not load the library")
        st.error(
            "Could not load the library. Check the database connection and that migrations "
            "have been applied (`alembic upgrade head`)."
        )
        return None


def as_uuid(value: str) -> uuid.UUID:
    return uuid.UUID(value)
