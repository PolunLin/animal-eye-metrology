"""Manual eye-mask annotation workspace, including independent v2 relabeling."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.data.coco import CocoDataset
from animal_eye_metrology.ui.canvas_view import eye_canvas_component
from animal_eye_metrology.ui.visualization import render_instance_overlay

LOGGER = logging.getLogger(__name__)


def eye_mask_workspace(
    dataset: CocoDataset, config: Any, session: Any = None, *, assets_directory: Path
) -> None:
    """Manual annotation only; the existing Phase 1 CSVs are never initialized here."""
    import base64
    import copy
    import io

    from animal_eye_metrology.annotation.eye_annotations import (
        PROTOCOL_VERSION,
        empty_geometry,
        initialize_store,
        latest_annotation,
        load_annotations,
        load_geometry,
        load_targets,
        save_annotation,
    )
    from animal_eye_metrology.core.verification import sha256_file

    protocol = session.protocol_version if session else PROTOCOL_VERSION
    st.subheader(
        "Eye Mask Annotation · v2 獨立重標" if session else "Eye Mask Annotation · Phase 3B"
    )
    st.caption(
        "人工描繪可見眼部區域；不會自動辨識眼睛。左右依動物自身定義，Mask 重心不等於瞳孔中心。"
    )
    phase3 = config.output.parent / "phase3"
    proposal = (
        session.proposal
        if session
        else Path(os.environ.get("AEM_EYE_PROPOSAL", phase3 / "phase3a/proposed_instances.csv"))
    )
    output = (
        session.ground_truth
        if session
        else Path(os.environ.get("AEM_EYE_OUTPUT", phase3 / "ground_truth")).resolve()
    )
    if session:
        st.info(
            "v2：眼裂＋緊鄰且可辨識的眼瞼邊緣；排除眼周毛髮、睫毛、淚痕與色塊。"
            "本頁不載入 v1 或模型輪廓。請暫時關閉舊成果視窗，從原圖獨立描繪。"
        )
        with st.expander("v2 標註規範與操作流程"):
            st.markdown((session.directory / "protocol.md").read_text(encoding="utf-8"))
    try:
        if not output.is_relative_to(phase3.resolve()) or output.is_relative_to(
            proposal.parent.resolve()
        ):
            raise Phase1Error("Eye GT 須使用獨立的 Phase 3 輸出目錄，不能覆寫提案或 Phase 1／2。")
        targets = load_targets(proposal, dataset)
        if session:
            from animal_eye_metrology.annotation.annotation_session import verify_session

            manifest = verify_session(session)
            if manifest["dataset_annotations_sha256"] != dataset.source_sha256:
                raise Phase1Error("COCO 身份來源與 v2 準備紀錄不同。")
        initialize_store(output, targets, dataset.source_sha256, protocol_version=protocol)
        rows = load_annotations(output, targets, protocol_version=protocol)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"無法載入眼睛標註：{error}")
        return
    identities = [
        (int(t["image_id"]), int(t["annotation_id"]), side)
        for t in targets
        for side in ("left", "right")
    ]
    latest = [latest_annotation(rows, key) for key in identities]
    cols = st.columns(4)
    for col, label, count in zip(
        cols,
        ("試標圖片", "眼睛槽位", "已保存", "Mask 已接受"),
        (
            len({k[0] for k in identities}),
            len(identities),
            sum(r is not None for r in latest),
            sum(r is not None and r["mask_review_status"] == "Accepted" for r in latest),
        ),
        strict=True,
    ):
        col.metric(label, count)
    if session:
        eligible = sum(
            bool(
                r
                and r["visibility"] == "Visible"
                and r["side_confirmed"] == "Yes"
                and r["mask_review_status"] == "Accepted"
                and r["measurement_quality"] == "Pass"
                and r["mask_path"]
            )
            for r in latest
        )
        st.caption(
            f"完整比較的凍結資格：{eligible}/{len(identities)} 眼。"
            "12 槽位皆審閱不等於 12 眼可測；外界不清請保留 Uncertain 與原因。"
        )
        with st.expander("12 眼重標進度"):
            st.dataframe(
                [
                    {
                        "圖片": key[0],
                        "動物": key[1],
                        "眼側": key[2],
                        "審核": row["mask_review_status"] if row else "尚未標註",
                        "量測品質": row["measurement_quality"] if row else "未定",
                        "原因": row["exclusion_reason"] if row else "",
                    }
                    for key, row in zip(identities, latest, strict=True)
                ],
                hide_index=True,
                width="stretch",
            )
    with st.sidebar:
        st.subheader("Eye GT 工作檔")
        st.caption(str(output))
        st.download_button(
            "下載 Eye Mask CSV",
            (output / "eye_mask_annotations.csv").read_bytes(),
            file_name="eye_mask_annotations.csv",
            mime="text/csv",
        )
        st.caption(f"規範：{protocol}；每次保存建立新修訂，歷史保留。")
        if session:
            st.caption("獨立重標，不代表完全盲測；你先前已看過模型輸出。")
    nav = st.columns([1.1, 2, 1.3])
    image_ids = list(dict.fromkeys(int(t["image_id"]) for t in targets))
    image_id = nav[0].selectbox("試標 COCO Image ID", image_ids, key="eye_gt_image")
    image_targets = [t for t in targets if int(t["image_id"]) == image_id]
    annotation_ids = [int(t["annotation_id"]) for t in image_targets]
    annotation_key = f"eye_gt_annotation_{image_id}"
    if st.session_state.get(annotation_key) not in annotation_ids:
        st.session_state[annotation_key] = annotation_ids[0]
    annotation_id = nav[1].selectbox(
        "試標 Animal Instance",
        annotation_ids,
        key=annotation_key,
        format_func=lambda value: (
            f"{dataset.categories[dataset.annotations[value].category_id].name} · #{value}"
        ),
    )
    side = nav[2].selectbox(
        "動物自身的眼側",
        ("left", "right"),
        key="eye_gt_side",
        format_func=lambda value: "Left · 解剖左眼" if value == "left" else "Right · 解剖右眼",
    )
    target = next(t for t in image_targets if int(t["annotation_id"]) == annotation_id)
    image_record = dataset.images[image_id]
    source_path = (config.images / image_record.file_name).resolve()
    try:
        if not source_path.is_relative_to(config.images.resolve()):
            raise Phase1Error("原圖路徑超出資料目錄。")
        if sha256_file(source_path) != target["image_sha256"]:
            raise Phase1Error("原圖 digest 與試標提案不同，停止標註。")
        with Image.open(source_path) as im:
            original = im.convert("RGB")
        if original.size != (image_record.width, image_record.height):
            raise Phase1Error("原圖尺寸不符。")
    except (OSError, ValueError) as error:
        st.error(str(error))
        return
    key = image_id, annotation_id, side
    prefix = "eye_gt_" + hashlib.sha256(f"{output}:{key}".encode()).hexdigest()[:18]
    row = latest_annotation(rows, key)
    state_map = st.session_state.setdefault("eye_gt_drafts", {})
    blank = empty_geometry(
        image_record.width, image_record.height, dataset.annotations[annotation_id].bbox
    )
    if prefix not in state_map:
        state_map[prefix] = {
            "geometry": load_geometry(output, row) if row else blank,
            "version": int(row["annotation_version"]) if row else 0,
            "record": dict(row) if row else {},
            "generation": 0,
        }
    draft = state_map[prefix]
    canvas_key = f"{prefix}_canvas_{draft['generation']}"
    if st.button("重新載入此眼睛最新版本", key=prefix + "_reload"):
        draft.update(
            geometry=load_geometry(output, row) if row else blank,
            version=int(row["annotation_version"]) if row else 0,
            record=dict(row) if row else {},
            generation=draft["generation"] + 1,
        )
        for field in (
            "visibility",
            "side_confirmed",
            "measurement_quality",
            "reviewer",
            "exclusion_reason",
            "mask_review_status",
            "confirm",
        ):
            st.session_state.pop(prefix + "_" + field, None)
        st.rerun()
    if row and int(row["annotation_version"]) != draft["version"]:
        st.warning("另一個視窗已更新此眼睛。你的草稿仍保留，儲存前請重新載入並人工整合。")
    if message := st.session_state.pop("eye_gt_saved_message", ""):
        st.success(message)
    st.caption(
        f"Image {image_id} · Annotation {annotation_id} · {side} · "
        f"已載入版本 {draft['version']} · {protocol}"
    )
    with st.expander("查看完整原圖與動物身份"):
        st.image(
            render_instance_overlay(
                original,
                image_record,
                [dataset.annotations[annotation_id]],
                dataset,
                show_masks=False,
                selected_id=annotation_id,
            ),
            caption="原圖加 COCO 動物框；不以動物 Mask 抹除眼睛像素。",
        )
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        # Read component state before mounting, as required by the controlled v2 pattern.
        value = st.session_state.get(canvas_key, {}).get("geometry")
        if value is not None:
            draft["geometry"] = value
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        payload = {
            "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
            "width": original.width,
            "height": original.height,
            "geometry": draft["geometry"],
            "initial_view": blank["view"],
            "color": "#00b9c9" if side == "left" else "#ffbd59",
            "fill": "rgba(0,185,201,.25)" if side == "left" else "rgba(255,189,89,.25)",
        }
        value = eye_canvas_component(
            (assets_directory / "eye_canvas.js").read_text(encoding="utf-8")
        )(
            data=payload,
            default={"geometry": draft["geometry"]},
            on_geometry_change=lambda: None,
            key=canvas_key,
        )
        if value.geometry is not None:
            draft["geometry"] = copy.deepcopy(value.geometry)
        st.caption(
            "輪廓以原圖像素中心轉為 Mask。切換眼側會保留本次連線草稿；"
            "重新整理或關閉前請按「儲存草稿」。"
        )
        if row and row["mask_path"]:
            with st.expander("已保存的原圖尺寸 Mask"):
                path = output / row["mask_path"]
                st.image(str(path), caption=f"版本 {row['annotation_version']} · 黑底白色眼部")
                st.download_button(
                    "下載此版本 Mask",
                    path.read_bytes(),
                    file_name=path.name,
                    mime="image/png",
                    key=prefix + "_download",
                )
    with right:
        st.markdown("**人工審核**")
        st.caption(
            "標眼裂與可辨識眼瞼邊緣，排除毛髮／睫毛／淚痕／色塊；外緣不清請選 Uncertain。"
            if session
            else "標可見眼裂內區域，排除眼瞼與毛髮；反光若邊界不明請選 Uncertain。"
        )
        st.caption(
            "Accepted 表示輪廓審核通過；Pass 另要求完整可見且左右已確認。"
            "Partial 可保留 Mask，量測請選 Fail 或 Unknown。"
        )
        saved = draft["record"]

        def choose(label: str, field: str, options: list[str], default: str | None = None) -> Any:
            previous = saved.get(field, default)
            return st.selectbox(
                label,
                options,
                index=options.index(previous) if previous in options else None,
                key=prefix + "_" + field,
                placeholder="尚未審核",
            )

        with st.form(prefix + "_review", enter_to_submit=False):
            updates = {
                "visibility": choose(
                    "眼部可見性", "visibility", ["Visible", "Partial", "NotVisible", "Uncertain"]
                ),
                "side_confirmed": choose(
                    "已確認動物的解剖左右", "side_confirmed", ["Unknown", "Yes", "No"], "Unknown"
                ),
                "measurement_quality": choose(
                    "量測品質", "measurement_quality", ["Unknown", "Pass", "Fail"], "Unknown"
                ),
                "mask_review_status": choose(
                    "Mask 審核結果", "mask_review_status", ["Accepted", "Rejected", "Uncertain"]
                ),
                "reviewer": st.text_input(
                    "Reviewer", value=saved.get("reviewer", ""), key=prefix + "_reviewer"
                ),
                "exclusion_reason": st.text_area(
                    "原因／備註",
                    value=saved.get("exclusion_reason", ""),
                    key=prefix + "_exclusion_reason",
                ),
            }
            confirmed = st.checkbox("我已確認眼側及輪廓符合規範", key=prefix + "_confirm")
            save_draft = st.form_submit_button("儲存草稿", width="stretch")
            save_reviewed = st.form_submit_button("儲存審核結果", type="primary", width="stretch")
        if save_draft or save_reviewed:
            if save_reviewed and (
                not updates["mask_review_status"]
                or (updates["mask_review_status"] == "Accepted" and not confirmed)
            ):
                st.error("請選擇審核結果；接受 Mask 前需勾選人工確認。")
            else:
                if save_draft:
                    updates["mask_review_status"] = "Pending"
                    updates["measurement_quality"] = "Unknown"
                updates = {k: v if v is not None else "" for k, v in updates.items()}
                try:
                    saved_row = save_annotation(
                        output,
                        targets,
                        key,
                        draft["geometry"],
                        updates,
                        expected_version=draft["version"],
                        protocol_version=protocol,
                    )
                    draft.update(version=int(saved_row["annotation_version"]), record=saved_row)
                    LOGGER.info(
                        json.dumps(
                            {
                                "event": "eye_annotation_saved",
                                "image_id": image_id,
                                "annotation_id": annotation_id,
                                "eye_side": side,
                                "annotation_version": draft["version"],
                                "output": str(output),
                            }
                        )
                    )
                    st.session_state["eye_gt_saved_message"] = (
                        f"已保存 {side} · Annotation {annotation_id} · "
                        f"v{draft['version']}，舊版本保留。"
                    )
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(str(error))
    with st.expander("保存版本紀錄"):
        st.dataframe(
            [
                r
                for r in rows
                if (int(r["image_id"]), int(r["annotation_id"]), r["eye_side"]) == key
            ],
            hide_index=True,
        )
