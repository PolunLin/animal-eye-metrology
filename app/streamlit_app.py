"""Single-image segmentation and measurement demo."""

import logging

import streamlit as st

from animal_eye_metrology.paths import PROJECT_ROOT
from animal_eye_metrology.ui.inference_view import inference_workspace

logging.basicConfig(level=logging.INFO, format="%(message)s")
st.set_page_config(page_title="Animal Eye Metrology", layout="wide")
st.title("Animal Eye Metrology")
inference_workspace(PROJECT_ROOT)
