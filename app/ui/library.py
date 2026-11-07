import logging

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from app.schemas import BookRead
from app.services.book_service import BookService
from app.ui.formatting import format_size, format_timestamp
from database.connection import session_scope

logger = logging.getLogger(__name__)


def _book_label(book: BookRead) -> str:
    return f"{book.title} — {book.author}" if book.author else book.title


def render_library_page() -> None:
    st.header("Library")
    try:
        with session_scope() as session:
            books = BookService(session).list_books()
    except SQLAlchemyError:
        logger.exception("Could not load the library")
        st.error(
            "Could not load the library. Check the database connection and that "
            "migrations have been applied (`alembic upgrade head`)."
        )
        return

    if not books:
        st.info("The library is empty. Upload a book to get started.")
        return

    rows = [
        {
            "Title": book.title,
            "Author": book.author or "",
            "Format": document.file_format.upper(),
            "Size": format_size(document.file_size),
            "Uploaded": format_timestamp(document.created_at),
            "Status": document.processing_status,
        }
        for book in books
        for document in book.documents
    ]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={"Title": st.column_config.TextColumn(width="large")},
    )

    st.subheader("Book details")
    by_id = {str(book.id): book for book in books}
    selected_id = st.selectbox(
        "Select a book",
        options=list(by_id),
        format_func=lambda book_id: _book_label(by_id[book_id]),
    )
    _render_book_details(by_id[selected_id])


def _render_book_details(book: BookRead) -> None:
    st.markdown(f"### {book.title}")
    if book.subtitle:
        st.markdown(f"*{book.subtitle}*")

    details = {
        "Author": book.author,
        "ISBN": book.isbn,
        "Language": book.language,
        "Added": format_timestamp(book.created_at),
    }
    for label, value in details.items():
        if value:
            st.markdown(f"**{label}:** {value}")
    if book.description:
        st.markdown(book.description)

    st.markdown("**Documents**")
    st.dataframe(
        [
            {
                "File": document.original_filename,
                "Format": document.file_format.upper(),
                "Size": format_size(document.file_size),
                "Status": document.processing_status,
                "Uploaded": format_timestamp(document.created_at),
                "SHA-256": document.sha256_hash,
            }
            for document in book.documents
        ],
        hide_index=True,
        width="stretch",
    )
