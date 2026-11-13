import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from pydantic import ValidationError

from app import __version__
from app.config import get_settings
from app.logging_config import configure_logging
from app.ui.equations import render_equations_page
from app.ui.library import render_library_page
from app.ui.navigation import render_navigation
from app.ui.passages import render_passages_page
from app.ui.processing import render_processing_page
from app.ui.structure import render_structure_page
from app.ui.upload import render_upload_page
from database.connection import check_connection

st.set_page_config(page_title="BOOK2BRAIN", page_icon="🧠")

SCREENS = {
    "Library": render_library_page,
    "Upload": render_upload_page,
    "Structure": render_structure_page,
    "Passages": render_passages_page,
    "Processing": render_processing_page,
    "Equations": render_equations_page,
}


def render_header() -> None:
    st.title("BOOK2BRAIN")
    st.caption(f"Version {__version__}")

    connected, detail = check_connection()
    if connected:
        st.success(f"Database connected: {detail}")
    else:
        st.error(f"Database not connected: {detail}")


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
# Only the chosen screen runs, so a screen never works in the background while hidden.
SCREENS[render_navigation()]()
