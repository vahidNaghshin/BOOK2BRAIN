import logging

import streamlit as st
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import BookMetadata
from app.services.document_service import DocumentService
from app.services.errors import DuplicateDocumentError, UploadError
from app.services.file_validation import SUPPORTED_EXTENSIONS
from app.services.storage_service import get_storage
from app.ui.formatting import format_size
from database.connection import check_connection, session_scope

logger = logging.getLogger(__name__)

_FIELD_LABELS = {
    "title": "Title",
    "author": "Author",
    "subtitle": "Subtitle",
    "isbn": "ISBN",
    "language": "Language",
    "description": "Description",
}


def _validation_messages(error: ValidationError) -> list[str]:
    messages = []
    for item in error.errors():
        field = str(item["loc"][0]) if item["loc"] else "metadata"
        label = _FIELD_LABELS.get(field, field)
        if field == "title" and item["type"] in ("missing", "string_too_short"):
            messages.append("Title is required.")
        else:
            messages.append(f"{label}: {str(item['msg']).removeprefix('Value error, ')}")
    return messages


def render_upload_page() -> None:
    settings = get_settings()
    st.header("Upload a book")

    with st.form("upload_form"):
        uploaded = st.file_uploader(
            "Book file *",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            help=f"PDF, EPUB or DOCX, up to {settings.max_upload_size_mb} MB.",
        )
        title = st.text_input("Title *")
        left, right = st.columns(2)
        author = left.text_input("Author")
        subtitle = right.text_input("Subtitle")
        isbn = left.text_input("ISBN")
        language = right.text_input("Language", placeholder="e.g. English or en")
        description = st.text_area("Description")
        submitted = st.form_submit_button("Upload", type="primary")

    if submitted:
        _handle_submit(uploaded, title, author, subtitle, isbn, language, description)


def _handle_submit(uploaded, title, author, subtitle, isbn, language, description) -> None:
    if uploaded is None:
        st.error("Select a book file to upload.")
        return

    try:
        metadata = BookMetadata(
            title=title,
            author=author,
            subtitle=subtitle,
            isbn=isbn,
            language=language,
            description=description,
        )
    except ValidationError as exc:
        for message in _validation_messages(exc):
            st.error(message)
        return

    connected, detail = check_connection()
    if not connected:
        st.error(f"Upload unavailable: database not connected ({detail}).")
        return

    progress = st.progress(0.0, text="Starting upload")

    def on_progress(stage: str, fraction: float) -> None:
        progress.progress(fraction, text=stage)

    try:
        with session_scope() as session:
            service = DocumentService(session, get_storage(), get_settings())
            result = service.upload(
                file=uploaded,
                filename=uploaded.name,
                declared_mime_type=uploaded.type,
                metadata=metadata,
                on_progress=on_progress,
            )
    except DuplicateDocumentError as exc:
        progress.empty()
        st.warning(f"Duplicate file: {exc}")
    except UploadError as exc:
        progress.empty()
        st.error(f"Upload failed: {exc}")
    except Exception:
        logger.exception("Unexpected error during upload")
        progress.empty()
        st.error("Upload failed because of an unexpected error. Nothing was saved. See the logs.")
    else:
        progress.empty()
        document = result.document
        kind = "as a new book" if result.book_created else "to the existing book"
        st.success(
            f"Uploaded {document.original_filename} ({document.file_format.upper()}, "
            f"{format_size(document.file_size)}) {kind} '{result.book.title}'."
        )
