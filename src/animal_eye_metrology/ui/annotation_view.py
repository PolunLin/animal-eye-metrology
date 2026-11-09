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
from animal_eye_metrology.ui.annotation_overlay import render_instance_overlay
from animal_eye_metrology.ui.canvas_view import eye_canvas_component

LOGGER = logging.getLogger(__name__)
SPECIES_LABELS = {
    "bird": "鳥",
    "cat": "貓",
    "dog": "狗",
    "horse": "馬",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑馬",
    "giraffe": "長頸鹿",
}
REVIEW_LABELS = {
    "visibility": {
        "Visible": "完整可見",
        "Partial": "部分可見",
        "NotVisible": "看不到",
        "Uncertain": "無法判斷",
    },
    "side_confirmed": {"Unknown": "尚未確認", "Yes": "已確認", "No": "判斷不符"},
    "measurement_quality": {"Unknown": "待判定", "Pass": "可量測", "Fail": "不可量測"},
    "mask_review_status": {"Accepted": "通過", "Rejected": "不通過", "Uncertain": "待確認"},
}


REASON_CHOICES = [
    "眼睛被遮擋",
    "眼睛不在可見側",
    "影像模糊",
    "眼部像素不足",
    "眼部邊界不清楚",
    "無法確認左右眼",
    "輪廓需要修正",
]
BLUR_REVIEW_PRESET = {
    "visibility": "Uncertain",
    "side_confirmed": "Unknown",
    "measurement_quality": "Fail",
    "mask_review_status": "Rejected",
    "exclusion_reason": "影像模糊",
}


def review_dropdown(label: str, field: str, saved: dict, options: list[str], prefix: str) -> str:
    """Keep stored free text intact while offering common values and a custom option."""
    previous = saved.get(field, "")
    choice_key = prefix + "_" + field + "_choice"
    text_key = prefix + "_" + field
    if field == "reviewer":
        remembered = st.session_state.get("eye_gt_last_reviewer", "")
        previous = previous or remembered
        options = [*options, *st.session_state.get("eye_gt_reviewer_names", [])]

    def remember_reviewer() -> None:
        if field != "reviewer":
            return
        value = st.session_state.get(choice_key, "")
        if value == "__custom__":
            value = st.session_state.get(text_key, "").strip()
            if not value:
                return
        st.session_state["eye_gt_last_reviewer"] = value
        names = st.session_state.setdefault("eye_gt_reviewer_names", [])
        if value and value not in names:
            names.append(value)

    values = ["", *dict.fromkeys(value for value in options if value), "__custom__"]
    selected = st.selectbox(
        label,
        values,
        index=values.index(previous) if previous in values else len(values) - 1,
        format_func=lambda value: "其他" if value == "__custom__" else value or "未指定",
        key=choice_key,
        on_change=remember_reviewer,
    )
    if selected == "__custom__":
        return st.text_input(
            "其他審核者姓名" if field == "reviewer" else "其他原因／備註",
            value=previous if previous not in options else "",
            key=text_key,
            on_change=remember_reviewer,
        )
    return selected


def animal_choices(
    dataset: CocoDataset, identifiers: list[int]
) -> tuple[list[int], dict[int, str], dict[int, str]]:
    """Match human-readable labels to stable COCO IDs; image order never assigns eye side."""
    ordered = sorted(identifiers, key=lambda key: (dataset.annotations[key].bbox[0], key))
    markers = {
        key: chr(65 + index) if index < 26 else str(index + 1) for index, key in enumerate(ordered)
    }
    labels = {}
    for key in ordered:
        species = dataset.categories[dataset.annotations[key].category_id].name
        labels[key] = f"動物 {markers[key]} · {SPECIES_LABELS.get(species, species)}"
    return ordered, markers, labels


def photo_progress_labels(
    identities: list[tuple[int, int, str]], rows: list[dict[str, str]]
) -> dict[int, str]:
    """Summarize saved work per photo without treating drafts as accepted references."""
    latest: dict[tuple[int, int, str], dict[str, str]] = {}
    for row in rows:
        key = int(row["image_id"]), int(row["annotation_id"]), row["eye_side"]
        if key not in latest or int(row["annotation_version"]) > int(
            latest[key]["annotation_version"]
        ):
            latest[key] = row
    photos: dict[int, list[dict[str, str] | None]] = {}
    for key in identities:
        photos.setdefault(key[0], []).append(latest.get(key))
    labels = {}
    for image_id, slots in photos.items():
        marked = [row for row in slots if row and (row["mask_path"] or row["visibility"])]
        count, total = len(marked), len(slots)
        state = "未標註" if not count else "已標註" if count == total else "部分已標註"
        parts = [f"{state} {count}/{total}"]
        pending = sum(row["mask_review_status"] in {"Pending", "Uncertain"} for row in marked)
        rejected = [row for row in marked if row["mask_review_status"] == "Rejected"]
        excluded = sum(row["visibility"] == "NotVisible" for row in rejected)
        needs_revision = len(rejected) - excluded
        if count == total and all(row["mask_review_status"] == "Accepted" for row in marked):
            parts.append("審核通過")
        elif count == total and not pending and not needs_revision:
            parts.append("已審閱")
        if pending:
            parts.append(f"待審核 {pending}")
        if needs_revision:
            parts.append(f"需修正 {needs_revision}")
        if excluded:
            parts.append(f"已排除 {excluded}")
        labels[image_id] = " · ".join(parts)
    return labels


def eye_review_label(row: dict[str, str] | None) -> str:
    """Describe the latest saved eye, independently of measurement eligibility."""
    if not row or not (row["mask_path"] or row["visibility"]):
        return "未標註"
    status = row["mask_review_status"]
    if status == "Accepted":
        return "審核通過"
    if status == "Rejected":
        return "已排除" if row["visibility"] == "NotVisible" else "需修正"
    return "待審核"


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
        rasterize,
        save_annotation,
    )
    from animal_eye_metrology.core.verification import sha256_file

    protocol = session.protocol_version if session else PROTOCOL_VERSION
    st.subheader("眼睛標註與審核 · v2" if session else "Eye Mask Annotation · Phase 3B")
    st.caption("在原圖上描繪或修正可見眼部區域。左右依動物自身定義，Mask 重心不等於瞳孔中心。")
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
            "本頁僅提供 v2；每份草稿都需對照原圖確認輪廓、可見性與解剖左右，再儲存審核結果。"
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
        ("照片", "眼睛標註位置（左右各一）", "已保存", "Mask 已接受"),
        (
            len({k[0] for k in identities}),
            len(identities),
            sum(r is not None for r in latest),
            sum(r is not None and r["mask_review_status"] == "Accepted" for r in latest),
        ),
        strict=True,
    ):
        col.metric(label, count)
    st.caption(
        f"{len({k[0] for k in identities})} 張照片 · "
        f"{len(targets)} 隻動物 · {len(identities)} 個眼睛標註位置；不是照片數量。"
    )
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
            "所有槽位皆審閱不等於所有眼睛可測；外界不清請保留 Uncertain 與原因。"
        )
        with st.expander("標註進度"):
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
    with st.expander("工作檔與下載"):
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
    context = st.container()
    left, right = st.columns([2.2, 1], gap="large")
    with right:
        st.markdown("**標註對象**")
        nav = [st.container(), st.container(), st.container()]
    image_ids = list(dict.fromkeys(int(t["image_id"]) for t in targets))
    progress_labels = photo_progress_labels(identities, rows)
    photo_labels = {}
    for index, value in enumerate(image_ids, start=1):
        parts = progress_labels[value].split(" · ")
        summary = parts[0]
        if len(parts) > 1:
            summary += f"；{parts[1].split()[0]}"
        photo_labels[value] = f"第 {index}／{len(image_ids)} 張 · #{value}（{summary}）"
    # Selectbox browser values are formatted labels. Progress can change while an
    # older label is still in flight from an animal/eye/canvas interaction.
    # Keep navigation by image ID, scoped to this annotation workspace, and
    # translate any previously rendered label before Streamlit validates it.
    workspace_key = str(output)
    navigation = st.session_state.setdefault("eye_gt_photo_navigation", {}).setdefault(
        workspace_key, {"image_id": image_ids[0], "label_ids": {}}
    )
    previous_workspace = st.session_state.get("eye_gt_photo_workspace")
    selected = (
        st.session_state.get("eye_gt_image", navigation["image_id"])
        if previous_workspace in (None, workspace_key)
        else navigation["image_id"]
    )
    if isinstance(selected, str):
        selected = navigation["label_ids"].get(selected)
    if selected not in image_ids:
        selected = navigation["image_id"]
    if selected not in image_ids:
        selected = image_ids[0]
    navigation["label_ids"].update({label: value for value, label in photo_labels.items()})
    st.session_state["eye_gt_photo_workspace"] = workspace_key
    # Explicitly synchronize the new display label, including on canvas reruns.
    st.session_state["eye_gt_image"] = selected
    image_id = nav[0].selectbox(
        "選擇照片",
        image_ids,
        key="eye_gt_image",
        format_func=photo_labels.get,
    )
    navigation["image_id"] = image_id
    nav[0].caption(f"此照片：{progress_labels[image_id]}。")
    nav[0].caption(
        "進度依已儲存的最新版本更新；有輪廓或已填可見性才計入。"
        "已標註包含待審核草稿；審核通過也不代表每眼均可量測。"
    )
    image_targets = [t for t in targets if int(t["image_id"]) == image_id]
    annotation_ids, markers, animal_labels = animal_choices(
        dataset, [int(t["annotation_id"]) for t in image_targets]
    )
    annotation_key = f"eye_gt_annotation_{image_id}"
    if st.session_state.get(annotation_key) not in annotation_ids:
        st.session_state[annotation_key] = annotation_ids[0]
    annotation_id = nav[1].radio(
        "要標哪隻動物？",
        annotation_ids,
        key=annotation_key,
        format_func=animal_labels.get,
        horizontal=True,
    )
    side = nav[2].radio(
        "要標牠的哪一眼？",
        ("left", "right"),
        key="eye_gt_side",
        format_func=lambda value: "牠的左眼" if value == "left" else "牠的右眼",
        horizontal=True,
    )
    with nav[2]:
        st.markdown("**這張照片：逐眼狀態**")
        st.table(
            [
                {
                    "動物": animal_labels[value],
                    "牠的左眼": eye_review_label(
                        latest_annotation(rows, (image_id, value, "left"))
                    ),
                    "牠的右眼": eye_review_label(
                        latest_annotation(rows, (image_id, value, "right"))
                    ),
                }
                for value in annotation_ids
            ]
        )
        st.caption("依已儲存的最新版本顯示；尚未儲存的操作不會改變狀態。")
        with st.expander("狀態怎麼看？"):
            st.markdown(
                "- **待審核**：草稿或尚未確認，請對照原圖檢查。\n"
                "- **需修正**：已判為不通過，且未標為看不到；請查看原因再處理。\n"
                "- **已排除**：已確認看不到，並儲存為不通過。\n"
                "- **審核通過**：輪廓已接受；能否量測仍以量測品質為準。\n"
                "- **未標註**：尚未保存輪廓或可見性判讀。"
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

    def stage_blurry_review(keys: list[tuple[int, int, str]]) -> None:
        """Fill review fields in session drafts; never modify masks or saved revisions."""
        for eye_key in keys:
            eye_prefix = "eye_gt_" + hashlib.sha256(f"{output}:{eye_key}".encode()).hexdigest()[:18]
            if eye_prefix not in state_map:
                stored = latest_annotation(rows, eye_key)
                state_map[eye_prefix] = {
                    "geometry": load_geometry(output, stored)
                    if stored
                    else empty_geometry(
                        image_record.width,
                        image_record.height,
                        dataset.annotations[eye_key[1]].bbox,
                    ),
                    "version": int(stored["annotation_version"]) if stored else 0,
                    "record": dict(stored) if stored else {},
                    "generation": 0,
                }
            eye_draft = state_map[eye_prefix]
            reviewer_choice = st.session_state.get(eye_prefix + "_reviewer_choice")
            if reviewer_choice is not None:
                eye_draft["record"]["reviewer"] = (
                    st.session_state.get(eye_prefix + "_reviewer", "")
                    if reviewer_choice == "__custom__"
                    else reviewer_choice
                )
            eye_draft["record"].update(BLUR_REVIEW_PRESET)
            eye_draft["quick_fill_pending"] = True
            for field, value in BLUR_REVIEW_PRESET.items():
                st.session_state[eye_prefix + "_" + field] = value
            st.session_state[eye_prefix + "_exclusion_reason_choice"] = "影像模糊"
            st.session_state[eye_prefix + "_confirm"] = False

    with nav[0].expander("整張照片快捷填寫"):
        st.caption(
            "套用到本照片全部眼睛（含已審核項目）：無法判斷、左右尚未確認、不可量測、"
            "不通過，原因為影像模糊。保留輪廓；仍需逐眼按「儲存審核結果」才會寫入紀錄。"
        )
        if st.button("整張照片太模糊：全部眼睛", key=f"{workspace_key}_{image_id}_blur_all"):
            stage_blurry_review([eye_key for eye_key in identities if eye_key[0] == image_id])
            st.rerun()
    pending_blur = sum(
        bool(
            state_map.get(
                "eye_gt_" + hashlib.sha256(f"{output}:{eye_key}".encode()).hexdigest()[:18], {}
            ).get("quick_fill_pending")
        )
        for eye_key in identities
        if eye_key[0] == image_id
    )
    if pending_blur:
        nav[0].warning(f"本照片有 {pending_blur} 隻眼睛已快捷填寫、尚未儲存。")
    with context:
        if st.button("重新載入此眼睛最新版本", key=prefix + "_reload"):
            draft.update(
                geometry=load_geometry(output, row) if row else blank,
                version=int(row["annotation_version"]) if row else 0,
                record=dict(row) if row else {},
                generation=draft["generation"] + 1,
                quick_fill_pending=False,
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
                st.session_state.pop(prefix + "_" + field + "_choice", None)
            st.rerun()
        if row and int(row["annotation_version"]) != draft["version"]:
            st.warning("另一個視窗已更新此眼睛。你的草稿仍保留，儲存前請重新載入並人工整合。")
        if message := st.session_state.pop("eye_gt_saved_message", ""):
            st.success(message)
        eye_label = "左眼" if side == "left" else "右眼"
        st.markdown(f"**目前標註：{animal_labels[annotation_id]} · 牠的{eye_label}**")
        saved_status = eye_review_label(row)
        message = f"已儲存狀態：{saved_status}"
        if row and row["exclusion_reason"]:
            message += f"。原因／備註：{row['exclusion_reason']}"
        if saved_status == "需修正":
            st.warning(message)
        elif saved_status == "審核通過":
            st.success(message)
        else:
            st.info(message)
        st.caption(f"已載入版本 {draft['version']} · v2 眼部遮罩")
        if st.button("此眼影像模糊：自動填寫", key=prefix + "_blur"):
            stage_blurry_review([key])
            st.rerun()
        if draft.get("quick_fill_pending"):
            st.info(
                "此眼已填入影像模糊的審核選項，尚未儲存。可調整後按「儲存審核結果」；"
                "按「重新載入此眼睛最新版本」可取消未儲存的修改。"
            )
        with st.expander("原圖動物對照（粗框為目前選取）", expanded=False):
            st.image(
                render_instance_overlay(
                    original,
                    image_record,
                    [dataset.annotations[value] for value in annotation_ids],
                    dataset,
                    show_masks=False,
                    selected_id=annotation_id,
                    instance_labels=markers,
                ),
                caption="A／B 只用來辨認動物，不代表眼睛側別。",
                width=640,
            )
            st.caption(f"原始資料 ID：照片 {image_id} · 動物 {annotation_id} · {side}")
    with left:
        # Read component state before mounting, as required by the controlled v2 pattern.
        value = st.session_state.get(canvas_key, {}).get("geometry")
        if value is not None:
            draft["geometry"] = value
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        import numpy as np

        from animal_eye_metrology.measurement.metrology import mask_centroid

        try:
            preview = rasterize(draft["geometry"], original.width, original.height)
            centroid = mask_centroid(np.asarray(preview) > 0)
        except ValueError as error:
            st.error(f"輪廓尚無法產生重心：{error}")
            centroid = None
        payload = {
            "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
            "centroid": list(centroid) if centroid is not None else None,
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
        if centroid is not None:
            st.caption(
                f"遮罩重心預覽：({centroid[0]:.2f}, {centroid[1]:.2f}) px；"
                "由已完成輪廓自動計算，十字標記不是人工眼點。"
            )
        else:
            st.caption("完成非空輪廓後顯示遮罩重心；空白草稿不產生中心點。")
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
        with st.expander("標註與審核說明"):
            st.caption("A／B 對照原圖標記；左右眼以動物自身為準，不是照片的左右。")
            st.caption(
                "標眼裂與可辨識眼瞼邊緣，排除毛髮／睫毛／淚痕／色塊；外緣不清請選「無法判斷」。"
                if session
                else "標可見眼裂內區域，排除眼瞼與毛髮；反光若邊界不明請選 Uncertain。"
            )
            st.caption(
                "「通過」表示輪廓符合規範；「可量測」還要求完整可見且左右已確認。"
                "部分可見仍可保存輪廓，量測品質請選「不可量測」或「待判定」。"
            )
        saved = draft["record"]

        def choose(label: str, field: str, options: list[str], default: str | None = None) -> Any:
            previous = saved.get(field, default)
            return st.radio(
                label,
                options,
                index=options.index(previous) if previous in options else None,
                key=prefix + "_" + field,
                format_func=REVIEW_LABELS[field].get,
                horizontal=True,
            )

        with st.container(border=True):
            updates = {
                "visibility": choose(
                    "眼部可見性", "visibility", ["Visible", "Partial", "NotVisible", "Uncertain"]
                ),
                "side_confirmed": choose(
                    "左右眼判斷", "side_confirmed", ["Unknown", "Yes", "No"], "Unknown"
                ),
                "measurement_quality": choose(
                    "量測品質", "measurement_quality", ["Unknown", "Pass", "Fail"], "Unknown"
                ),
                "mask_review_status": choose(
                    "輪廓審核結果", "mask_review_status", ["Accepted", "Rejected", "Uncertain"]
                ),
                "reviewer": review_dropdown(
                    "標註／審核者",
                    "reviewer",
                    saved,
                    sorted({r["reviewer"] for r in rows if r.get("reviewer")}),
                    prefix,
                ),
                "exclusion_reason": review_dropdown(
                    "原因／備註",
                    "exclusion_reason",
                    saved,
                    REASON_CHOICES,
                    prefix,
                ),
            }
            confirmed = st.checkbox("我已確認眼側及輪廓符合規範", key=prefix + "_confirm")
            save_draft = st.button("儲存草稿", key=prefix + "_save_draft", width="stretch")
            save_reviewed = st.button(
                "儲存審核結果", key=prefix + "_save_reviewed", type="primary", width="stretch"
            )
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
                    draft.update(
                        version=int(saved_row["annotation_version"]),
                        record=saved_row,
                        quick_fill_pending=False,
                    )
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
                        f"已保存 {animal_labels[annotation_id]} · 牠的{eye_label} · "
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
