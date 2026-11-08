"""Cached canvas component shared by manual annotation and prompt workspaces."""

import hashlib
from typing import Any

import streamlit as st


@st.cache_resource
def eye_canvas_component(javascript: str) -> Any:
    return st.components.v2.component(
        "animal_eye_polygon_editor_" + hashlib.sha256(javascript.encode()).hexdigest()[:12],
        js=javascript,
    )
