import logging
import uuid

import streamlit as st
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import BookMetadata
from app.services.document_service import DocumentService
from app.services.errors import DuplicateDocumentError, ProcessingServiceError, UploadError
from app.services.file_validation import SUPPORTED_EXTENSIONS
from app.services.storage_service import get_storage
from app.ui.formatting import format_size
from app.ui.navigation import goto
from app.ui.processing import get_job_runner
from database.connection import check_connection, session_scope

logger = logging.getLogger(__name__)

_FIELD_LABELS = {
    "title": "Title",
    "author": "Author",
    "subtitle": "Subtitle",
    "isbn": "ISBN",
    "language": "Language",
    "description": "Description",
    "publisher": "Publisher",
    "publication_year": "Publication year",
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
        publisher = left.text_input("Publisher")
        year = right.text_input("Publication year", placeholder="e.g. 2016")
        description = st.text_area("Description")
        start_processing = st.checkbox(
            "Start processing after upload",
            value=True,
            help="Extracts the text, detects chapters and sections and generates passages.",
        )
        submitted = st.form_submit_button("Upload", type="primary")

    if submitted:
        _handle_submit(
            uploaded,
            {
                "title": title,
                "author": author,
                "subtitle": subtitle,
                "isbn": isbn,
                "language": language,
                "description": description,
                "publisher": publisher,
                "publication_year": year.strip() or None,
            },
            start_processing,
        )

    if st.session_state.get("upload_started_document"):
        if st.button("Follow progress in the Library", key="upload_go_library", type="primary"):
            goto("Library", document_id=uuid.UUID(st.session_state.pop("upload_started_document")))


def _handle_submit(uploaded, fields: dict, start_processing: bool) -> None:
    st.session_state.pop("upload_started_document", None)
    if uploaded is None:
        st.error("Select a book file to upload.")
        return

    try:
        metadata = BookMetadata(**fields)
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
        if start_processing:
            _start_processing(document.id)


def _start_processing(document_id) -> None:
    """Queue the pipeline. An upload that succeeded stays successful if processing cannot start."""
    try:
        get_job_runner().submit(document_id)
    except ProcessingServiceError as exc:
        st.warning(f"The file was uploaded, but processing could not start: {exc} Start it from the Library.")
    except Exception:
        logger.exception("Could not start processing after upload")
        st.warning("The file was uploaded, but processing could not start. Start it from the Library.")
    else:
        st.info("Processing started. It runs in the background; follow its stage in the Library.")
        st.session_state["upload_started_document"] = str(document_id)
