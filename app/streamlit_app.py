"""Single-image segmentation and measurement demo."""

import logging

import streamlit as st

from animal_eye_metrology.paths import PROJECT_ROOT
from animal_eye_metrology.ui.inference_view import inference_workspace
from animal_eye_metrology.ui.training_view import training_workspace

logging.basicConfig(level=logging.INFO, format="%(message)s")
st.set_page_config(page_title="Animal Eye Metrology", layout="wide")
st.title("Animal Eye Metrology")
tabs = st.tabs(
    ["影像推論", "訓練與實驗", "眼睛標註與審核"], key="main_workspace", on_change="rerun"
)
if tabs[0].open:
    with tabs[0]:
        inference_workspace(PROJECT_ROOT)
if tabs[1].open:
    with tabs[1]:
        training_workspace(PROJECT_ROOT)
if tabs[2].open:
    with tabs[2]:
        from animal_eye_metrology.ui.annotation_workspace import annotation_workspace

        annotation_workspace(PROJECT_ROOT)
