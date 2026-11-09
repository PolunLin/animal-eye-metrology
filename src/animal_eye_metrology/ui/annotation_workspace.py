"""Load only an explicitly prepared, versioned v2 annotation session."""

import json
import os
from dataclasses import replace
from pathlib import Path

import streamlit as st

from animal_eye_metrology.annotation.annotation_session import read_session, verify_session
from animal_eye_metrology.core.config import load_config
from animal_eye_metrology.data.coco import load_coco
from animal_eye_metrology.ui.annotation_view import eye_mask_workspace


def annotation_workspace(root: Path) -> None:
    st.header("眼睛標註與審核")
    st.caption("僅提供 v2 眼部遮罩；每次儲存建立新版本，人工審核後才有資格作為參考。")
    source = os.environ.get(
        "AEM_ANNOTATION_SESSION", str(root / "configs/annotation_workspace.toml")
    )
    if "AEM_ANNOTATION_SESSION" not in os.environ:
        with st.expander("新增標註批次：選擇 train／val", expanded=True):
            st.caption(
                "來源不等於訓練用途。至少兩個非 crowd 動物的圖片才列入；眼睛是否可標仍待人工審核。"
            )
            with st.form("new_annotation_batch"):
                split = st.radio("COCO 圖片來源", ["val2017", "train2017"], horizontal=True)
                count = st.number_input("本批照片數", min_value=1, max_value=100, value=20, step=1)
                seed = st.number_input(
                    "抽取順序 Seed", min_value=0, max_value=2**32 - 1, value=42, step=1
                )
                create = st.form_submit_button("建立待標註批次", type="primary")
            st.caption(
                "會核對原圖、排除已選圖片與完全相同檔案；不保證眼睛清楚或場景獨立。沒有自動訓練／驗收切分。"
            )
            if create:
                from animal_eye_metrology.annotation.batches import create_batch

                try:
                    with st.spinner("核對本機來源並建立獨立標註批次…"):
                        selected = create_batch(root, split, int(count), int(seed))
                    st.session_state["active_annotation_batch"] = str(selected)
                    st.rerun()
                except (OSError, ValueError, KeyError) as error:
                    st.error(f"無法建立批次：{error}")
        options = {source: "既有示範 · 3 張照片"}
        for path in sorted(
            (root / "outputs/phase3/eye_annotation_batches").glob("*/batch_manifest.json")
        ):
            record = json.loads(path.read_text())
            options[str(path.parent / "session.toml")] = (
                f"{record['source_split']} · {record['requested_images']} 張 · {path.parent.name}"
            )
        if st.session_state.get("active_annotation_batch") not in options:
            st.session_state["active_annotation_batch"] = source
        source = st.selectbox(
            "目前標註批次", list(options), format_func=options.get, key="active_annotation_batch"
        )
    try:
        session = read_session(Path(source))
        manifest = verify_session(session)
        config = load_config(session.phase1_config)
        dataset = load_coco(
            config.annotations,
            expected_sha256=manifest["dataset_annotations_sha256"],
            expected_animal_categories=config.expected_animal_categories,
        )
        config = replace(config, images=session.images)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"無法開啟獨立重標工作區：{error}")
        st.info("請先依眼睛標註說明準備工作區。缺少資料時不會下載、選取新資料或產生標註。")
        st.code(
            "python -m animal_eye_metrology.annotation.annotation_session "
            f"--config {source} --prepare",
            language="sh",
        )
        return
    st.info(
        "此頁只建立與審核標註，不會啟動訓練。train／val 是原始來源名稱；"
        "批次尚未指定訓練或驗收用途。"
    )
    eye_mask_workspace(dataset, config, session, assets_directory=root / "app/assets")
