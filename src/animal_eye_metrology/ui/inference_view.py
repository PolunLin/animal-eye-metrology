"""Streamlit entry point for new-image inference without reference annotations."""

import hashlib
import io
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

from animal_eye_metrology.inference import inference_artifacts as artifacts
from animal_eye_metrology.inference import inference_pipeline as pipeline
from animal_eye_metrology.inference.inference_capabilities import COCO_ANIMAL_CATEGORIES

LOGGER = logging.getLogger(__name__)
BASELINE = "A_r100"
STAGES = {
    "detection": "動物偵測／實例分割",
    "localization": "HRNet 眼點定位",
    "segmentation": "SAM 眼部分割",
    "measurement": "重心與候選距離",
}
STATUS_LABELS = {
    "completed": "執行完成",
    "completed_with_issues": "完成，含未完成項目",
    "passed": "輸出檢查通過",
    "ready": "可執行",
    "missing": "缺少必要資源",
    "failed": "執行失敗",
    "skipped": "未執行",
    "pending": "等待執行",
    "not_run": "尚未執行",
    "running": "執行中",
    "empty_mask": "空遮罩",
    "no_animals": "沒有達門檻的動物",
    "unsupported_category": "本次未啟用眼部流程",
    "below_threshold": "未達偵測門檻",
    "candidate": "候選量測",
    "unavailable": "無法計算",
    "unverified": "尚未驗證",
}
CATEGORIES = {
    "bird": "鳥",
    "cat": "貓",
    "dog": "狗",
    "horse": "馬",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "象",
    "bear": "熊",
    "zebra": "斑馬",
    "giraffe": "長頸鹿",
}
SIDES = {"left": "左眼", "right": "右眼"}
POSE_DOMAINS = {
    "ap10k_mammal_mapping": "AP-10K 哺乳類對應（未驗證）",
    "out_of_training_domain": "AP-10K 訓練範圍外",
    "unsupported": "模型流程未涵蓋",
}


def label(value: Any) -> str:
    return STATUS_LABELS.get(str(value), str(value or "—"))


def pose_domain_label(value: Any) -> str:
    return POSE_DOMAINS.get(value, str(value)) if value else "舊結果未記錄"


def experimental_label(value: Any) -> str:
    if value is None:
        return "舊結果未記錄"
    return "是（域外試驗）" if value else "否（仍未驗證）"


def request_signature(image_bytes: bytes, conditions: tuple[str, ...], config_path: Path) -> str:
    """Invalidate a displayed run when image, conditions, or configuration changes."""
    digest = hashlib.sha256(image_bytes)
    digest.update(json.dumps(conditions).encode())
    digest.update(str(config_path.absolute()).encode())
    digest.update(config_path.read_bytes())
    return digest.hexdigest()


def artifact_path(run_dir: Path, relative: str | None) -> Path | None:
    """Only read result assets inside their own run directory."""
    if not relative:
        return None
    root = run_dir.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path


def stage_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "階段": title,
            "執行狀態": label(result.get("stages", {}).get(stage, {}).get("status", "pending")),
            "秒數": result.get("stages", {}).get(stage, {}).get("seconds"),
            "原因": result.get("stages", {}).get(stage, {}).get("reason", ""),
        }
        for stage, title in STAGES.items()
    ]


def animal_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "動物 ID": row["prediction_id"],
            "類別": CATEGORIES.get(row.get("category"), row.get("category")),
            "偵測分數": row.get("score"),
            "達偵測門檻": row.get("accepted"),
            "眼部流程支援": row.get("supported"),
            "眼點模型範圍": pose_domain_label(row.get("pose_domain")),
            "域外試驗": experimental_label(row.get("experimental")),
            "偵測／實例狀態": label(row.get("status")),
            "動物遮罩狀態": label(row.get("mask_status")),
            "動物遮罩原因": row.get("mask_failure_reason", ""),
            "原因": row.get("failure_reason", ""),
        }
        for row in result.get("animals", [])
    ]


def eye_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "條件": row["condition_id"],
            "動物 ID": row["prediction_id"],
            "預測眼別": SIDES.get(row.get("eye_side"), row.get("eye_side")),
            "定位狀態": label(row.get("localization_status")),
            "分割狀態": label(row.get("segmentation_status")),
            "眼點模型範圍": pose_domain_label(row.get("pose_domain")),
            "域外試驗": experimental_label(row.get("experimental")),
            "眼點 x px": row.get("point_x_px"),
            "眼點 y px": row.get("point_y_px"),
            "熱圖分數": row.get("point_score"),
            "遮罩面積 px²": row.get("mask_area_px"),
            "重心 x px": row.get("centroid_x_px"),
            "重心 y px": row.get("centroid_y_px"),
            "可見性": label(row.get("visibility_status", "unverified")),
            "左右眼資格": label(row.get("side_assignment_status", "unverified")),
            "原因": row.get("failure_reason", ""),
        }
        for row in result.get("eyes", [])
    ]


def measurement_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "條件": row["condition_id"],
            "距離類型": "同隻雙眼"
            if row.get("measurement_type") == "binocular"
            else "不同動物右眼",
            "動物 A": row.get("prediction_id_a"),
            "動物 B": row.get("prediction_id_b"),
            "距離 px": row.get("distance_px"),
            "量測狀態": label(row.get("status")),
            "量測資格": label(row.get("qualification", "unverified")),
            "端點 A 模型範圍": pose_domain_label(row.get("pose_domain_a")),
            "端點 B 模型範圍": pose_domain_label(row.get("pose_domain_b")),
            "域外試驗": experimental_label(row.get("experimental")),
            "原因": row.get("reason", ""),
        }
        for row in result.get("measurements", [])
    ]


def _table(rows: list[dict[str, Any]], empty_message: str) -> None:
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info(empty_message)


def _download_assets(run_dir: Path, result: dict[str, Any]) -> None:
    artifacts = result.get("artifacts", {})
    st.markdown("**下載本次結果**")
    st.caption("CSV／JSON 保留原始狀態與失敗原因；空值表示無法計算，不以 0 代替。")
    items = [
        ("animals_csv", "動物 CSV", "text/csv"),
        ("eyes_csv", "定位與分割 CSV", "text/csv"),
        ("measurements_csv", "候選距離 CSV", "text/csv"),
    ]
    for column, (key, title, mime) in zip(st.columns(3), items, strict=True):
        path = artifact_path(run_dir, artifacts.get(key))
        if path is not None:
            with column:
                st.download_button(
                    title,
                    path.read_bytes(),
                    f"{result['run_id']}_{path.name}",
                    mime,
                    key=f"inference_download_{key}",
                    on_click="ignore",
                )
    path = artifact_path(run_dir, "result.json")
    if path is not None:
        st.download_button(
            "完整結果 JSON",
            path.read_bytes(),
            f"{result['run_id']}_result.json",
            "application/json",
            key="inference_download_json",
            on_click="ignore",
        )


def _overlays(run_dir: Path, result: dict[str, Any]) -> None:
    conditions = result.get("conditions", [BASELINE])
    artifacts = result.get("artifacts", {})
    for column, condition in zip(st.columns(len(conditions)), conditions, strict=True):
        with column:
            st.markdown(f"**{condition}{' · 保留基線' if condition == BASELINE else ' · 對照'}**")
            relative = artifacts.get(f"{condition}_overlay")
            if condition == BASELINE:
                relative = relative or artifacts.get("overlay")
            path = artifact_path(run_dir, relative)
            if path is None:
                st.info("此條件未產生疊圖；請查看各階段原因。")
            else:
                st.image(str(path), width="stretch")
                st.download_button(
                    f"下載 {condition} 疊圖 PNG",
                    path.read_bytes(),
                    f"{result['run_id']}_{condition}.png",
                    "image/png",
                    key=f"inference_download_overlay_{condition}",
                    on_click="ignore",
                )


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _animal_segmentation(run_dir: Path, result: dict[str, Any]) -> None:
    """Show existing instance masks; display controls never rewrite inference artifacts."""
    st.subheader("動物輪廓（實例分割）")
    st.caption(
        "彩色區域與輪廓線來自 YOLO 的逐隻動物二值遮罩；矩形框只是偵測框，可另外開啟。"
        "這裡呈現模型預測的可見區域，不補畫遮擋部分，也不代表已通過準確度驗證。"
    )
    animals = [row for row in result.get("animals", []) if row.get("accepted")]
    if not animals:
        st.info("沒有達門檻的動物，因此沒有動物輪廓可顯示。")
        return
    raw_masks: dict[str, tuple[Path, Image.Image]] = {}
    display_animals = []
    for row in animals:
        display_row = dict(row)
        path = artifact_path(run_dir, row.get("mask_path"))
        reason = ""
        if row.get("mask_status") not in ("completed", "empty_mask"):
            reason = row.get("mask_failure_reason") or label(row.get("mask_status", "missing"))
        elif path is None:
            reason = "遮罩檔案缺失或路徑不屬於本次結果。"
        else:
            try:
                with Image.open(path) as source:
                    values = np.asarray(source)
                    expected = (result["input"]["width"], result["input"]["height"])
                    if (
                        source.size != expected
                        or values.ndim != 2
                        or not np.isin(values, [0, 1, 255]).all()
                    ):
                        raise ValueError("遮罩不是原圖尺寸的二值影像。")
                    empty = not np.any(values)
                    if empty and row.get("mask_status") == "completed":
                        reason = "保存狀態為完成，但遮罩沒有前景。"
                    elif not empty and row.get("mask_status") == "empty_mask":
                        reason = "保存狀態為空遮罩，但檔案含有前景；請核對輸出。"
                    else:
                        preview = Image.fromarray((values > 0).astype(np.uint8) * 255)
                        raw_masks[row["prediction_id"]] = (path, preview)
                        if empty:
                            reason = "空遮罩，沒有可繪製的動物輪廓。"
            except (OSError, KeyError, ValueError) as error:
                reason = str(error)
        if reason:
            display_row["mask_path"] = None
            st.warning(f"{row['prediction_id']}：{reason} 不會以偵測框代替輪廓。")
        display_animals.append(display_row)

    mode = st.radio(
        "動物分割顯示",
        ["輪廓疊圖", "實例遮罩"],
        horizontal=True,
        key=f"inference_animals_mode_{result['run_id']}",
    )
    show_boxes = st.checkbox(
        "另顯示矩形偵測框",
        key=f"inference_animals_boxes_{result['run_id']}",
    )
    try:
        views = artifacts.animal_segmentation_views(
            run_dir, result | {"animals": display_animals}, show_boxes=show_boxes
        )
        st.image(views["overlay" if mode == "輪廓疊圖" else "instances"], width="stretch")
        for column, (key, title) in zip(
            st.columns(2),
            (("overlay", "下載動物輪廓疊圖 PNG"), ("instances", "下載動物實例遮罩 PNG")),
            strict=True,
        ):
            with column:
                st.download_button(
                    title,
                    _png_bytes(views[key]),
                    f"{result['run_id']}_animal_{key}.png",
                    "image/png",
                    key=f"inference_animals_download_{key}",
                    on_click="ignore",
                )
    except (OSError, KeyError, ValueError) as error:
        LOGGER.exception("animal_segmentation_display_failed")
        st.warning(f"無法繪製動物輪廓：{error}。眼部結果仍可在下方檢視。")
    with st.expander("單隻動物的原始二值遮罩"):
        index = {row["prediction_id"]: row for row in animals}

        def animal_label(identity: str) -> str:
            category = index[identity].get("category")
            return f"{identity} · {CATEGORIES.get(category, category)}"

        selected = st.selectbox(
            "選擇動物遮罩",
            list(index),
            format_func=animal_label,
            key=f"inference_animals_selected_{result['run_id']}",
        )
        if selected in raw_masks:
            path, image = raw_masks[selected]
            st.caption("白色為該動物的預測前景，黑色為背景；下載保留原始二值 PNG 檔案。")
            st.image(image, width="stretch")
            st.download_button(
                "下載此動物原始二值遮罩 PNG",
                path.read_bytes(),
                f"{result['run_id']}_{selected}_mask.png",
                "image/png",
                key="inference_animals_download_raw",
                on_click="ignore",
            )
        else:
            st.info("此動物沒有可用的原始二值遮罩；請查看上方原因及動物狀態表。")


def _eye_zoom(run_dir: Path, result: dict[str, Any]) -> None:
    eyes = result.get("eyes", [])
    choices = sorted({(row["prediction_id"], row["eye_side"]) for row in eyes})
    if not choices:
        return
    st.markdown("**眼部局部檢視**")
    selected = st.selectbox(
        "選擇動物與預測眼別",
        choices,
        format_func=lambda pair: f"{pair[0]} · {SIDES.get(pair[1], pair[1])}",
        key=f"inference_zoom_{result['run_id']}",
    )
    rows = [row for row in eyes if (row["prediction_id"], row["eye_side"]) == tuple(selected)]
    rois = [row["roi_xyxy"] for row in rows if row.get("roi_xyxy")]
    if not rois:
        st.info("此眼沒有有效定位／ROI，因此沒有可顯示的局部範圍。")
        return
    width, height = result["input"]["width"], result["input"]["height"]
    box = (
        max(0, min(roi[0] for roi in rois) - 8),
        max(0, min(roi[1] for roi in rois) - 8),
        min(width, max(roi[2] for roi in rois) + 8),
        min(height, max(roi[3] for roi in rois) + 8),
    )
    st.caption("各條件使用相同顯示範圍；這只是疊圖放大，不會改變推論 ROI 或輸出。")
    for column, condition in zip(
        st.columns(len(result["conditions"])), result["conditions"], strict=True
    ):
        with column:
            st.markdown(f"**{condition}**")
            relative = result.get("artifacts", {}).get(f"{condition}_overlay")
            if condition == BASELINE:
                relative = relative or result.get("artifacts", {}).get("overlay")
            path = artifact_path(run_dir, relative)
            if path:
                with Image.open(path) as image:
                    st.image(image.crop(box), width="stretch")


def _eye_execution_summary(result: dict[str, Any]) -> None:
    """Surface saved eye outcomes before large overlays, without reinterpreting old runs."""
    animals = [row for row in result.get("animals", []) if row.get("accepted")]
    eyes = result.get("eyes", [])
    skipped_categories = sorted(
        {row["category"] for row in animals if row.get("supported") is False}
    )
    if skipped_categories:
        names = "、".join(CATEGORIES.get(category, category) for category in skipped_categories)
        old_scope = set(result.get("config", {}).get("supported_categories", [])) == {
            "dog",
            "bear",
            "giraffe",
        }
        if old_scope:
            st.warning(
                f"這是舊版三類範圍的保存結果：當次僅啟用狗、熊、長頸鹿，{names}的眼部流程被跳過。"
                "目前可對 COCO 十類重新執行；舊結果不會自動補跑或改寫。"
            )
        else:
            st.warning(
                f"保存結果顯示{names}當次未啟用眼部流程。此狀態保留原紀錄，"
                "若屬 COCO 十類，請重新執行以套用目前設定。"
            )
    if any(row.get("experimental") is True for row in [*animals, *eyes]):
        st.warning(
            "鳥類眼部輸出屬 AP-10K 訓練範圍外試驗。即使產生眼點、遮罩或距離，"
            "仍未經本專案驗證，不能視為可靠左右眼或合格量測。"
        )
    st.markdown("**眼部處理摘要**")
    if not eyes:
        st.info("本次沒有眼部結果紀錄；請查看上方偵測與定位階段的狀態。")
        return
    rows = []
    counts: Counter[str] = Counter()
    for condition in dict.fromkeys(row["condition_id"] for row in eyes):
        cases = [row for row in eyes if row["condition_id"] == condition]
        completed = sum(row.get("segmentation_status") == "completed" for row in cases)
        empty = sum(row.get("segmentation_status") == "empty_mask" for row in cases)
        rows.append(
            {
                "條件": condition,
                "預期逐眼結果": len(cases),
                "定位完成": sum(row.get("localization_status") == "completed" for row in cases),
                "非空眼部遮罩": completed,
                "空遮罩": empty,
                "失敗／未執行": len(cases) - completed - empty,
            }
        )
    _table(rows, "尚無逐眼處理摘要。")
    st.caption("以上按條件列出保存的執行結果；兩個 ROI 條件不代表增加獨立眼睛數。")
    for eye in eyes:
        if eye.get("segmentation_status") == "completed":
            continue
        if eye.get("localization_status") == "unsupported_category":
            reason = "當次物種範圍未啟用"
        elif eye.get("localization_status") == "failed":
            reason = "眼點定位失敗或未通過有效性檢查"
        elif eye.get("segmentation_status") == "empty_mask":
            reason = "SAM 輸出空遮罩"
        elif eye.get("segmentation_status") == "failed":
            reason = "SAM 分割或遮罩檢查失敗"
        else:
            reason = "眼部分割未執行"
        counts[reason] += 1
    if counts:
        description = "；".join(f"{reason} {count} 筆" for reason, count in counts.items())
        prefix = (
            "本次沒有產生任何眼部遮罩。"
            if not any(row.get("segmentation_status") == "completed" for row in eyes)
            else "部分眼部結果無法使用。"
        )
        st.warning(f"{prefix}{description}。逐眼原因保留在下方表格與 CSV／JSON。")


def render_result(run_dir: Path, result: dict[str, Any]) -> None:
    """Render a saved run; this path never triggers inference."""
    st.subheader("本次推論結果")
    st.caption(
        f"執行 ID：{result['run_id']} · 圖片：{result.get('input', {}).get('filename', '—')}"
    )
    status = result.get("status", "failed")
    if status == "failed":
        st.error("推論未完成。下方保留已完成階段與錯誤原因，可下載結果紀錄。")
    elif status == "no_animals":
        st.info("未找到達門檻的動物，沒有可計算的眼部距離。這不代表圖片一定沒有動物。")
    elif status == "completed_with_issues":
        st.warning("本次流程包含未完成項目，請查看定位、分割與量測的個別狀態。")
    else:
        st.success("推論執行完成；分割正確性與量測資格仍待獨立驗證。")
    _table(stage_rows(result), "尚無階段紀錄。")
    _eye_execution_summary(result)
    _animal_segmentation(run_dir, result)
    st.subheader("眼部分割與候選距離")
    st.warning(
        "所有可計算距離都只是「候選量測」。左右眼沿用 AP-10K 解剖學通道，"
        "尚未確認可見性或左右眼歸屬；重心是分割區域重心，不是瞳孔中心。"
        "距離單位為圖片像素，沒有實體尺度或誤差估計。"
    )
    _overlays(run_dir, result)
    st.caption("疊圖：左眼青色、右眼紅色；黃色十字為 HRNet 眼點，圓點為遮罩重心。")
    _eye_zoom(run_dir, result)
    st.markdown("**動物偵測與實例狀態**")
    _table(animal_rows(result), "沒有動物候選紀錄。")
    st.markdown("**定位、分割與重心**")
    _table(eye_rows(result), "沒有進入眼部流程的候選。")
    st.caption("熱圖分數與 SAM 候選分數不是經校準的眼睛正確機率。空遮罩不產生重心。")
    st.markdown("**候選距離**")
    _table(measurement_rows(result), "沒有可建立的雙眼或跨動物右眼配對。")
    _download_assets(run_dir, result)
    with st.expander("執行紀錄與重現資訊"):
        st.code(str(run_dir), language=None)
        st.json(
            {
                key: value
                for key, value in result.items()
                if key not in {"eyes", "animals", "measurements"}
            },
            expanded=False,
        )


def inference_workspace(root: Path) -> None:
    st.subheader("新圖片自動推論")
    st.write("上傳圖片 → 動物偵測 → HRNet 眼點 → SAM 眼部分割 → 重心與候選距離")
    st.caption(
        "本版可對 COCO 全部 10 類動物執行眼部流程："
        + "、".join(CATEGORIES[category] for category in COCO_ANIMAL_CATEGORIES)
        + "，採 CPU 執行。"
        "使用固定模型與 ROI，輸出動物輪廓、眼部遮罩及兩種像素距離。"
    )
    st.caption(
        "鳥類使用 AP-10K 訓練範圍外的試驗推論；其餘九類有資料集物種對應，"
        "但所有類別的眼部正確性與量測資格仍待驗證。這不表示可處理 COCO 以外的所有物種。"
    )
    config_path = Path(
        os.environ.get("AEM_INFERENCE_CONFIG", root / "configs/inference.runtime.toml")
    )
    try:
        config = pipeline.load_config(config_path)
        resources = pipeline.preflight(config)
    except Exception as exc:
        LOGGER.exception("inference_config_failed")
        st.error(f"無法載入推論設定：{exc}")
        return
    ready = all(item["status"] == "ready" for item in resources)
    with st.expander("本機執行資源", expanded=not ready):
        _table(
            [
                {"資源": item["component"], "狀態": label(item["status"]), "說明": item["detail"]}
                for item in resources
            ],
            "沒有資源檢查紀錄。",
        )
    if not ready:
        st.error("必要資源尚未就緒。請修復上方項目後再執行；不會自動下載模型。")
    uploaded = st.file_uploader(
        "上傳一張 JPG 或 PNG 圖片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        max_upload_size=max(1, config.get("max_upload_bytes", 20 * 1024 * 1024) // (1024 * 1024)),
        help="單張圖片；依 EXIF 校正方向，轉為 RGB，透明背景合成白色，保留原圖解析度。",
        key="inference_upload",
    )
    st.caption(
        f"檔案上限 {config.get('max_upload_bytes', 20 * 1024 * 1024) / (1024 * 1024):g} MiB；"
        f"像素上限 {config.get('max_image_pixels', 12000000):,}。"
    )
    conditions = (BASELINE,)
    data = uploaded.getvalue() if uploaded is not None else None
    signature = request_signature(data, conditions, config_path) if data is not None else None
    if st.button(
        "開始自動推論",
        type="primary",
        disabled=not ready or data is None,
        key="inference_run",
    ):
        st.session_state.pop("inference_last_result", None)
        try:
            with st.status("執行 CPU 推論…", expanded=True) as progress_box:

                def progress(stage: str, message: str) -> None:
                    progress_box.write(f"{STAGES.get(stage, stage)}：{label(message)}")

                directory = pipeline.run_inference(
                    data,
                    uploaded.name,
                    config_path,
                    conditions=conditions,
                    progress=progress,
                )
                result = pipeline.load_result(directory)
                progress_box.update(
                    label="執行結束，詳見下方各階段結果",
                    state="error" if result.get("status") == "failed" else "complete",
                    expanded=False,
                )
            st.session_state["inference_last_result"] = {
                "signature": signature,
                "directory": str(directory),
            }
        except Exception as exc:
            LOGGER.exception("new_image_inference_failed")
            st.error(f"無法完成這次推論：{exc}")
    previous = st.session_state.get("inference_last_result")
    if previous and previous["signature"] != signature:
        st.info("圖片或設定已改變；請重新執行。上一筆結果仍保存在原執行目錄。")
    elif previous:
        try:
            directory = Path(previous["directory"])
            render_result(directory, pipeline.load_result(directory))
        except Exception as exc:
            LOGGER.exception("new_image_result_read_failed")
            st.error(f"無法讀取這筆推論結果：{exc}")
