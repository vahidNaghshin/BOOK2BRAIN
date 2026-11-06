import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app import __version__
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


def render_upload() -> None:
    st.header("Upload")
    st.info("Upload page placeholder. Document upload will be added here.")


def render_processing() -> None:
    st.header("Processing status")
    st.info("Processing status placeholder. Ingestion progress will be shown here.")


render_header()
upload_tab, processing_tab = st.tabs(["Upload", "Processing status"])
with upload_tab:
    render_upload()
with processing_tab:
    render_processing()
