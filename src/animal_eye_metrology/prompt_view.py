"""Manual point-and-box prompts workspace, independent of model predictions."""

import hashlib
import os
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from animal_eye_metrology.canvas_view import eye_canvas_component
from animal_eye_metrology.coco import CocoDataset
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.visualization import render_instance_overlay


def eye_prompt_workspace(dataset: CocoDataset, config: Any, *, assets_directory: Path) -> None:
    """Independent manual prompts; this workspace never loads an eye Mask or centroid."""
    import base64
    import copy
    import io

    from animal_eye_metrology.eye_annotations import empty_geometry, load_targets
    from animal_eye_metrology.eye_poc import settings
    from animal_eye_metrology.eye_prompts import latest_prompt, load_prompts, save_prompt
    from animal_eye_metrology.verification import sha256_file

    cfg = settings(
        Path(
            os.environ.get(
                "AEM_EYE_POC_CONFIG", assets_directory.parent / "configs/eye_segmentation_poc.toml"
            )
        )
    )
    directory = cfg["prompts_directory"]
    targets = load_targets(cfg["proposal_csv"], dataset)
    rows = load_prompts(directory, targets)
    st.subheader("Eye Segmentation · 固定人工提示")
    if message := st.session_state.pop("eye_prompt_saved_message", ""):
        st.success(message)
    st.caption("只看原圖提供一框＋一正點；不載入人工 Mask、舊輪廓、重心或模型結果。")
    st.info("相容性試跑先保存 Image 125405 / Animal 13692 / left 的提示即可。")
    st.caption(f"工作目錄：{directory}")
    identities = {(int(r["image_id"]), int(r["annotation_id"]), r["eye_side"]) for r in rows}
    st.metric("已保存提示的不同眼睛", f"{len(identities)} / {2 * len(targets)}")
    nav = st.columns(3)
    image_id = nav[0].selectbox(
        "提示 Image ID",
        list(dict.fromkeys(int(t["image_id"]) for t in targets)),
        key="prompt_image",
    )
    annotation_ids = [int(t["annotation_id"]) for t in targets if int(t["image_id"]) == image_id]
    if st.session_state.get("prompt_annotation") not in annotation_ids:
        st.session_state["prompt_annotation"] = annotation_ids[0]
    annotation_id = nav[1].selectbox(
        "提示 Animal Instance", annotation_ids, key="prompt_annotation"
    )
    side = nav[2].selectbox("提示解剖眼側", ("left", "right"), key="prompt_side")
    key = image_id, annotation_id, side
    target = next(t for t in targets if (int(t["image_id"]), int(t["annotation_id"])) == key[:2])
    source = config.images / target["file_name"]
    if sha256_file(source) != target["image_sha256"]:
        raise Phase1Error("Prompt original-image digest differs.")
    with Image.open(source) as opened:
        original = opened.convert("RGB")
    if original.size != (int(target["image_width"]), int(target["image_height"])):
        raise Phase1Error("Prompt original-image dimensions differ.")
    row = latest_prompt(rows, key)
    prefix = "eye_prompt_" + hashlib.sha256(f"{directory}:{key}".encode()).hexdigest()[:16]
    initial_view = empty_geometry(*original.size, dataset.annotations[annotation_id].bbox)["view"]
    drafts = st.session_state.setdefault("eye_prompt_drafts", {})
    if prefix not in drafts:
        drafts[prefix] = {
            "geometry": {
                "view": initial_view,
                "box": [float(row[k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1")]
                if row
                else None,
                "point": [float(row[k]) for k in ("point_x", "point_y")] if row else None,
            },
            "version": int(row["prompt_version"]) if row else 0,
        }
    draft = drafts[prefix]
    if row and int(row["prompt_version"]) != draft["version"]:
        st.warning("另一視窗已保存較新提示；請重新整理後再保存。")
    st.caption(
        f"Image {image_id} · Animal {annotation_id} · anatomical {side} · "
        f"本輪提示版本 {draft['version']}"
    )
    with st.expander("完整原圖與動物身份"):
        st.image(
            render_instance_overlay(
                original,
                dataset.images[image_id],
                [dataset.annotations[annotation_id]],
                dataset,
                show_masks=False,
                selected_id=annotation_id,
            )
        )
    left, right = st.columns([2.2, 1])
    with left:
        canvas_key = prefix + "_canvas"
        geometry = st.session_state.get(canvas_key, {}).get("geometry")
        if geometry is not None:
            draft["geometry"] = copy.deepcopy(geometry)
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        value = eye_canvas_component((assets_directory / "eye_prompt_canvas.js").read_text())(
            data={
                "image_url": "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode(),
                "width": original.width,
                "height": original.height,
                "geometry": draft["geometry"],
                "initial_view": initial_view,
            },
            default={"geometry": draft["geometry"]},
            on_geometry_change=lambda: None,
            key=canvas_key,
        )
        if value.geometry is not None:
            draft["geometry"] = copy.deepcopy(value.geometry)
        st.caption("每眼固定一個粗框與一個框內正點；重新整理前請保存。左右依動物自身定義。")
    with right:
        with st.form(prefix + "_save"):
            reviewer = st.text_input("提示 Reviewer", value=row["reviewer"] if row else "")
            confirmed = st.checkbox("我只依原圖設定提示，沒有參考 GT Mask 或模型輸出")
            submitted = st.form_submit_button("保存固定提示", type="primary", width="stretch")
        if submitted:
            if not confirmed:
                st.error("請先確認提示來源。")
            else:
                try:
                    saved = save_prompt(
                        directory, targets, key, draft["geometry"], reviewer, draft["version"]
                    )
                    draft["version"] = int(saved["prompt_version"])
                    st.session_state["eye_prompt_saved_message"] = (
                        f"已保存固定提示 v{draft['version']}；可關閉頁面。"
                    )
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(str(error))
        if (directory / "prompts.csv").exists():
            st.download_button(
                "下載提示 CSV", (directory / "prompts.csv").read_bytes(), file_name="prompts.csv"
            )
