import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from pydantic import ValidationError

from app import __version__
from app.config import get_settings
from app.logging_config import configure_logging
from app.ui.library import render_library_page
from app.ui.upload import render_upload_page
from database.connection import check_connection

st.set_page_config(page_title="BOOK2BRAIN", page_icon="🧠")


def render_header() -> None:
    st.title("BOOK2BRAIN")
    st.caption(f"Version {__version__}")

    connected, detail = check_connection()
    if connected:
        st.success(f"Database connected: {detail}")
    else:
        st.error(f"Database not connected: {detail}")


def render_processing() -> None:
    st.header("Processing status")
    st.info("Processing status placeholder. Ingestion progress will be shown here.")


try:
    configure_logging(get_settings().log_level)
except ValidationError:
    st.title("BOOK2BRAIN")
    st.error(
        "Configuration error: DATABASE_URL is not set. "
        "Copy .env.example to .env and set your values, then restart the app."
    )
    st.stop()

render_header()
upload_tab, library_tab, processing_tab = st.tabs(["Upload", "Library", "Processing status"])
with upload_tab:
    render_upload_page()
with library_tab:
    render_library_page()
with processing_tab:
    render_processing()
