"""Screen navigation and the selection shared between screens (book, document, node, passage).

Streamlit tabs cannot be switched from code, so the screens are chosen with a radio and only the
chosen screen runs. Any screen can send the user to another one with `goto`, carrying the
document / section / passage they were looking at.
"""

import uuid

import streamlit as st

SCREENS = ["Library", "Upload", "Structure", "Passages", "Processing", "Equations"]
SCREEN_KEY = "screen"
DOCUMENT_KEY = "selected_document_id"
NODE_KEY = "selected_node_id"
PASSAGE_KEY = "selected_passage_id"
_GOTO_KEY = "_goto_screen"
# Set when another screen chose the document/section, so selectors adopt it once.
SYNC_KEY = "_selection_changed"


def goto(
    screen: str,
    *,
    document_id: uuid.UUID | None = None,
    node_id: uuid.UUID | None = None,
    passage_id: uuid.UUID | None = None,
) -> None:
    """Switch to `screen` on the next run, optionally selecting a document, section or passage."""
    st.session_state[_GOTO_KEY] = screen
    if document_id is not None:
        st.session_state[DOCUMENT_KEY] = str(document_id)
    st.session_state[NODE_KEY] = node_id
    st.session_state[PASSAGE_KEY] = passage_id
    st.session_state[SYNC_KEY] = True
    st.rerun()


def render_navigation() -> str:
    """Draw the screen selector and return the chosen screen."""
    if _GOTO_KEY in st.session_state:
        st.session_state[SCREEN_KEY] = st.session_state.pop(_GOTO_KEY)
    st.session_state.setdefault(SCREEN_KEY, SCREENS[0])
    return st.radio("Screen", SCREENS, key=SCREEN_KEY, horizontal=True, label_visibility="collapsed")
