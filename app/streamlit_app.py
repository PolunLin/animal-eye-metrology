"""Local research-workspace entry point. Run with streamlit run."""

import csv
import hashlib
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from animal_eye_metrology.analysis import AnalysisResult, analyze
from animal_eye_metrology.annotation_view import eye_mask_workspace
from animal_eye_metrology.coco import CocoDataset, load_coco
from animal_eye_metrology.config import load_config
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.prompt_view import eye_prompt_workspace
from animal_eye_metrology.review_store import (
    CHOICES,
    SELECTION_STATUSES,
    image_eye_condition,
    initialize_image_selections,
    initialize_reviews,
    is_reviewed,
    load_reviews,
    review_key,
    save_image_selection,
    save_review,
)
from animal_eye_metrology.visualization import render_instance_overlay

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
LABELS = {
    "left_eye_visibility": "Left Eye Visible",
    "right_eye_visibility": "Right Eye Visible",
    "head_pose": "Head Pose",
    "occlusion": "Occlusion",
    "measurement_eligibility": "Eligibility",
}


@st.cache_resource(show_spinner="讀取 COCO 註記…")
def dataset_resources(
    annotations: str,
    modified_ns: int,
    size: int,
    expected_count: int,
    expected_digest: str,
) -> tuple[CocoDataset, AnalysisResult]:
    dataset = load_coco(
        Path(annotations),
        expected_animal_categories=expected_count,
        expected_sha256=expected_digest,
    )
    return dataset, analyze(dataset)


def read_sample(path: Path, dataset: CocoDataset, result: AnalysisResult) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image_id, category_id = int(row["image_id"]), int(row["category_id"])
        annotations = result.eligible_by_image.get(image_id, [])
        expected = {ann.id for ann in annotations if ann.category_id == category_id}
        if (
            len(expected) < 2
            or set(json.loads(row["target_annotation_ids"])) != expected
            or dataset.categories[category_id].name != row["sample_category"]
        ):
            raise Phase1Error("抽樣清單與目前 COCO 註記不一致，請先核對資料版本。")
    return rows


def move_image(image_ids: list[int], step: int) -> None:
    current = image_ids.index(st.session_state["image_id"])
    st.session_state["image_id"] = image_ids[max(0, min(len(image_ids) - 1, current + step))]


def review_form(
    path: Path,
    template: list[dict[str, Any]],
    current: dict[str, str],
) -> None:
    key = review_key(current)
    prefix = hashlib.sha256(f"{path}:{key}".encode()).hexdigest()[:20]
    snapshot_key = prefix + "_snapshot"
    if st.button("重新載入此實例", key=prefix + "_reload"):
        for field in [*CHOICES, "notes", "eye_resolution", "exclusion_reason", "reviewer"]:
            st.session_state.pop(prefix + "_" + field, None)
        st.session_state[snapshot_key] = dict(current)
        st.rerun()
    if snapshot_key not in st.session_state:
        st.session_state[snapshot_key] = dict(current)
    snapshot = st.session_state[snapshot_key]
    if snapshot != current:
        st.warning("這筆實例已由另一個視窗更新，請先重新載入。")
    with st.form(prefix + "_form", enter_to_submit=False):
        updates = {}
        eyes = st.columns(2)
        context = st.columns(2)
        controls = dict(zip(CHOICES, [*eyes, *context, st.container()], strict=True))
        for field, choices in CHOICES.items():
            previous = snapshot.get(field, "")
            with controls[field]:
                if previous and previous not in choices:
                    st.caption(f"既有值：{previous}；請人工確認。")
                updates[field] = st.selectbox(
                    LABELS[field],
                    choices,
                    index=choices.index(previous) if previous in choices else None,
                    placeholder="尚未審核",
                    key=prefix + "_" + field,
                )
        updates["notes"] = st.text_area(
            "Reviewer Notes", value=snapshot["notes"], key=prefix + "_notes", height=90
        )
        with st.expander("其他既有審核欄位（選填）"):
            for field, label in [
                ("eye_resolution", "Eye resolution（原始像素／量法）"),
                ("exclusion_reason", "Exclusion reason"),
                ("reviewer", "Reviewer"),
            ]:
                updates[field] = st.text_input(
                    label, value=snapshot[field], key=prefix + "_" + field
                )
        submitted = st.form_submit_button("儲存審核", type="primary", width="stretch")
    st.caption("切換影像或實例前請先儲存。未送出的修改不會自動寫入 CSV。")
    if submitted:
        if any(updates[field] is None for field in CHOICES):
            st.error("請完成五個選項；無法判斷時選 Uncertain 或 Other。")
            return
        try:
            saved = save_review(
                path,
                template,
                key,
                updates
                | {
                    "review_protocol_version": "phase1-explorer-v1-anatomical",
                },
                expected=snapshot,
            )
        except (OSError, Phase1Error) as error:
            st.error(str(error))
            return
        st.session_state[snapshot_key] = saved
        st.session_state["saved_message"] = f"已儲存 Image {key[0]} · Annotation {key[1]}"
        LOGGER.info(
            json.dumps(
                {
                    "event": "review_saved",
                    "image_id": key[0],
                    "annotation_id": key[1],
                    "csv": str(path),
                }
            )
        )
        st.rerun()


def selection_form(
    path: Path,
    candidate_ids: list[int],
    current: dict[str, str],
    visible_ids: list[int],
    navigation_context: str,
) -> None:
    image_id = int(current["image_id"])
    prefix = "selection_" + hashlib.sha256(f"{path}:{image_id}".encode()).hexdigest()[:20]
    snapshot_key = prefix + "_snapshot"
    st.subheader("Image Selection")
    st.caption(f"Image {image_id} · 目前狀態：{current['selection_status']}")
    st.caption("快速判斷這張圖是否值得進一步審核，不需逐隻標記眼睛。")
    if st.button("重新載入圖片選擇", key=prefix + "_reload"):
        st.session_state.pop(prefix + "_notes", None)
        st.session_state[snapshot_key] = dict(current)
        st.rerun()
    if snapshot_key not in st.session_state:
        st.session_state[snapshot_key] = dict(current)
    snapshot = st.session_state[snapshot_key]
    if snapshot != current:
        st.warning("這張圖已由另一視窗更新，請先重新載入圖片選擇。")
    advance = st.checkbox("儲存後自動下一張", value=True, key="auto_advance_selection")
    with st.form(prefix + "_form", enter_to_submit=False):
        selected = st.form_submit_button("Selected · 保留", type="primary", width="stretch")
        rejected = st.form_submit_button("Rejected · 排除", width="stretch")
        pending = st.form_submit_button("Pending · 待確認", width="stretch")
        notes = st.text_area(
            "Image Review Notes", value=snapshot["review_notes"], key=prefix + "_notes", height=120
        )
    status = "Selected" if selected else "Rejected" if rejected else "Pending" if pending else None
    st.caption("按狀態按鈕即保存圖片選擇與備註；不會改動既有眼部審核。")
    if status is not None:
        try:
            saved = save_image_selection(
                path, candidate_ids, image_id, status, notes, expected=snapshot
            )
        except (OSError, Phase1Error) as error:
            st.error(str(error))
            return
        st.session_state[snapshot_key] = saved
        st.session_state["saved_message"] = f"已儲存 Image {image_id}：{status}"
        if advance:
            next_id = visible_ids[(visible_ids.index(image_id) + 1) % len(visible_ids)]
            st.session_state["navigate_image"] = (navigation_context, next_id)
        LOGGER.info(
            json.dumps(
                {
                    "event": "image_selection_saved",
                    "image_id": image_id,
                    "selection_status": status,
                    "csv": str(path),
                }
            )
        )
        st.rerun()


def show_image(
    dataset: CocoDataset,
    result: AnalysisResult,
    image_directory: Path,
    image_id: int,
    annotation_id: int | None = None,
) -> None:
    options = st.columns(3)
    boxes = options[0].checkbox("Bounding Box", value=True, key="show_boxes")
    masks = options[1].checkbox("COCO Mask", value=True, key="show_masks")
    crop = annotation_id is not None and options[2].checkbox(
        "選取動物原圖裁切", value=True, key="show_crop"
    )
    if annotation_id is None:
        options[2].caption(f"{len(result.eligible_by_image[image_id])} 個動物實例")
    record = dataset.images[image_id]
    image_path = (image_directory / record.file_name).resolve()
    try:
        if not image_path.is_relative_to(image_directory):
            raise Phase1Error("Image path is outside the configured dataset.")
        with Image.open(image_path) as opened:
            original = opened.convert("RGB")
        overlay = render_instance_overlay(
            original,
            record,
            result.eligible_by_image[image_id],
            dataset,
            show_boxes=boxes,
            show_masks=masks,
            selected_id=annotation_id,
        )
    except (OSError, ValueError) as error:
        st.error(f"無法顯示影像：{error}")
        st.stop()
    annotated_tab, original_tab = st.tabs(["動物註記", "原始影像"])
    suffix = f" · 選取 #{annotation_id}" if annotation_id is not None else ""
    annotated_tab.image(overlay, width="stretch", caption=f"Image {image_id}{suffix}")
    original_tab.image(
        original, width="stretch", caption=f"原始解析度 {record.width} × {record.height}"
    )
    if crop:
        x, y, width, height = dataset.annotations[annotation_id].bbox
        bounds = (
            max(0, int(x)),
            max(0, int(y)),
            min(record.width, int(x + width + 1)),
            min(record.height, int(y + height + 1)),
        )
        if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
            st.image(
                original.crop(bounds),
                width="stretch",
                caption="依 COCO 動物 bbox 裁切原圖；放大不增加原始解析度。",
            )
    with st.expander("此圖全部非 crowd 動物實例"):
        st.dataframe(
            [
                {
                    "annotation_id": ann.id,
                    "category": dataset.categories[ann.category_id].name,
                    "selected": ann.id == annotation_id,
                }
                for ann in result.eligible_by_image[image_id]
            ],
            hide_index=True,
        )


def show_eye_condition(rows: list[dict[str, str]]) -> None:
    summary = image_eye_condition(rows)
    count = summary["qualifying_count"]
    if summary["condition"] == "Ready":
        st.success(f"已找到 {count} 隻左右眼與 Eligibility 均為 Yes 的雙眼候選。")
    elif summary["condition"] == "Pending":
        st.info(
            f"待確認：目前 {count} 隻雙眼候選，"
            f"{summary['unreviewed_count']} 隻未完成、{summary['uncertain_count']} 隻仍不確定。"
        )
    else:
        st.warning(f"目前已儲存標記中有 {count} 隻雙眼候選，尚未達到兩隻。")
    st.caption(
        "此提示依全圖所有非 crowd 動物的人工紀錄整理，不會自動改變圖片選擇。"
        "找到兩隻候選後，可先停止審核其餘動物；精確標註可行性仍待確認。"
    )


def main() -> None:
    st.set_page_config(page_title="Animal Eye · Research Workspace", page_icon="🔬", layout="wide")
    st.markdown(
        "<style>.block-container{padding-top:4rem;}h1{font-size:2rem!important;}"
        "[data-testid=stMetricValue]{font-size:1.7rem;}</style>",
        unsafe_allow_html=True,
    )
    st.caption("ANIMAL EYE · RESEARCH WORKSPACE")
    st.title("Animal Eye Metrology")
    st.caption("資料審核、眼部標註與實驗成果；保留每一步的研究依據。")
    annotation_session = None
    if session_path := os.environ.get("AEM_ANNOTATION_SESSION"):
        from animal_eye_metrology.annotation_session import read_session, verify_session

        try:
            annotation_session = read_session(Path(session_path))
            verify_session(annotation_session)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"無法開啟獨立重標工作區：{error}")
            st.info("請先執行 annotation_session --prepare，並保留準備時的設定與規範。")
            return
    workspaces = [
        "Image Selection",
        "Eye Review",
        "Eye Mask Annotation",
        "Eye Segmentation Prompts",
        "Results Explorer",
        "New Image Inference",
    ]
    if annotation_session:
        workspaces = ["Eye Mask Annotation"]
    if st.session_state.get("workspace") not in workspaces:
        st.session_state.pop("workspace", None)
    workspace = st.radio(
        "工作區",
        workspaces,
        horizontal=True,
        key="workspace",
        index=workspaces.index(os.environ.get("AEM_WORKSPACE"))
        if os.environ.get("AEM_WORKSPACE") in workspaces
        else 0,
    )
    if workspace == "Results Explorer":
        from animal_eye_metrology.results_view import results_workspace

        results_workspace(ROOT)
        return
    if workspace == "New Image Inference":
        from animal_eye_metrology.inference_view import inference_workspace

        inference_workspace(ROOT)
        return
    with st.sidebar:
        st.subheader("資料與篩選")
        if annotation_session:
            config_path = annotation_session.phase1_config
            st.caption("v2 專用重標模式 · 只顯示原圖與本輪人工草稿")
        else:
            with st.expander("資料與工作檔設定"):
                config_path = (
                    Path(
                        st.text_input(
                            "設定檔",
                            value=os.environ.get("AEM_CONFIG", str(ROOT / "configs/phase1.toml")),
                            key="config_path",
                        )
                    )
                    .expanduser()
                    .resolve()
                )
    try:
        config = load_config(config_path)
        if annotation_session:
            config = replace(config, images=annotation_session.images)
        manifest_path = config.output / "run_manifest.json"
        digest = config.expected_annotations_sha256
        if manifest_path.exists():
            digest = json.loads(manifest_path.read_text())["dataset"]["annotations_sha256"]
        stat = config.annotations.stat()
        dataset, result = dataset_resources(
            str(config.annotations),
            stat.st_mtime_ns,
            stat.st_size,
            config.expected_animal_categories,
            digest,
        )
        if workspace == "Eye Segmentation Prompts":
            eye_prompt_workspace(dataset, config, assets_directory=ROOT / "app/assets")
            return
        if workspace == "Eye Mask Annotation":
            eye_mask_workspace(
                dataset, config, annotation_session, assets_directory=ROOT / "app/assets"
            )
            return
        template = load_reviews(config.output / "eye_visibility_review.csv", result.review)
        with st.sidebar.expander("CSV 儲存位置"):
            selection_path = (
                Path(
                    st.text_input(
                        "Image Selection CSV",
                        value=os.environ.get(
                            "AEM_SELECTION_CSV", str(config.output / "image_selection.csv")
                        ),
                        key="selection_path",
                    )
                )
                .expanduser()
                .resolve()
            )
            review_path = (
                Path(
                    st.text_input(
                        "Eye Review CSV",
                        value=os.environ.get(
                            "AEM_REVIEW_CSV",
                            str(config.output / "eye_visibility_review_working.csv"),
                        ),
                        key="review_path",
                    )
                )
                .expanduser()
                .resolve()
            )
            st.caption("兩個工作檔分開保存；原始分析與抽樣 CSV 不會被覆寫。")
        protected = {
            config.output / name
            for name in (
                "eye_visibility_review.csv",
                "category_statistics.csv",
                "animal_instances.csv",
                "area_summary.csv",
                "multi_any.csv",
                "multi_same_category.csv",
                "multi_different_categories.csv",
            )
        } | {path.resolve() for path in config.output.rglob("sampled_images.csv")}
        protected |= {path.resolve() for path in config.output.rglob("eye_visibility_review.csv")}
        if review_path == selection_path:
            raise Phase1Error("Image Selection 與 Eye Review 必須使用不同的 CSV。")
        for path in (selection_path, review_path):
            if (
                path.suffix.lower() != ".csv"
                or not path.is_relative_to(config.output)
                or path in protected
            ):
                raise Phase1Error("請在 Phase 1 輸出目錄使用獨立工作檔，不能覆寫原始產物。")
        reviews = initialize_reviews(review_path, template)
        records = {review_key(row): row for row in reviews}
        candidate_ids = [row["image_id"] for row in result.candidates["multi_any"]]
        selections = initialize_image_selections(selection_path, candidate_ids)
        image_selections = {int(row["image_id"]): row for row in selections}
        selected_ids = {
            key for key, row in image_selections.items() if row["selection_status"] == "Selected"
        }
        sample_path = config.output / "visual_review_seed42/sampled_images.csv"
        sampled = read_sample(sample_path, dataset, result) if sample_path.exists() else []
    except (OSError, ValueError, KeyError) as error:
        st.error(f"無法載入工作區：{error}")
        st.info("請核對設定檔、本機資料與 CSV；工具不會下載資料或修正原檔。")
        st.stop()
    with st.sidebar:
        scopes = ["全部候選", "Seed 42 審核批次"] if sampled else ["全部候選"]
        if st.session_state.get("scope") not in scopes:
            st.session_state["scope"] = scopes[0]
        scope = st.selectbox("資料範圍", scopes, key="scope")
        sample_mode = scope == "Seed 42 審核批次"
        names = (
            sorted({row["sample_category"] for row in sampled})
            if sample_mode
            else [cat.name for cat in dataset.animal_categories.values()]
        )
        if st.session_state.get("category") not in ["全部", *names]:
            st.session_state["category"] = "全部"
        category = st.selectbox("動物類別", ["全部", *names], key="category")
        if workspace == "Image Selection":
            status = st.selectbox(
                "圖片選擇狀態", ["全部", *SELECTION_STATUSES], key="selection_filter"
            )
        else:
            status = st.selectbox("實例審核狀態", ["全部", "未完成", "已完成"], key="status")
        st.caption("類別篩選用來挑圖；Eye Review 可審核已選圖片中的全部非 crowd 動物。")
        st.divider()
        st.download_button(
            "下載 Image Selection CSV",
            data=selection_path.read_bytes(),
            file_name=selection_path.name,
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "下載 Eye Review CSV",
            data=review_path.read_bytes(),
            file_name=review_path.name,
            mime="text/csv",
            width="stretch",
        )
    if sample_mode:
        base_ids = {
            int(row["image_id"])
            for row in sampled
            if category == "全部" or row["sample_category"] == category
        }
    else:
        category_id = next(
            (cat.id for cat in dataset.animal_categories.values() if cat.name == category), None
        )
        base_ids = {
            row["image_id"]
            for row in result.candidates["multi_any"]
            if category_id is None or category_id in json.loads(row["category_ids"])
        }
    columns = st.columns(4)
    if workspace == "Image Selection":
        values = [
            len(candidate_ids),
            *[
                sum(row["selection_status"] == choice for row in selections)
                for choice in SELECTION_STATUSES
            ],
        ]
        labels = ["候選圖片", "Selected", "Rejected", "Pending"]
        image_ids = sorted(
            key
            for key in base_ids
            if status == "全部" or image_selections[key]["selection_status"] == status
        )
        scope_keys = set()
    else:
        scope_keys = {
            (key, ann.id)
            for key in base_ids & selected_ids
            for ann in result.eligible_by_image[key]
        }
        count = sum(is_reviewed(records[key]) for key in scope_keys)
        values = [len(base_ids & selected_ids), len(scope_keys), count, len(scope_keys) - count]
        labels = ["本範圍 Selected 圖片", "已選圖中動物", "已完成審核", "待確認實例"]
        visible_keys = {
            key
            for key in scope_keys
            if status == "全部" or (status == "已完成") == is_reviewed(records[key])
        }
        image_ids = sorted({key[0] for key in visible_keys})
    for column, label, value in zip(columns, labels, values, strict=True):
        column.metric(label, value, border=True)
    with st.expander("Dataset Overview · 動物類別統計"):
        st.caption(
            f"COCO：{len(dataset.images):,} 張 · {len(dataset.animal_categories)} 類動物 · "
            f"任意多動物 {len(candidate_ids)} 張 · "
            f"同類 {len(result.candidates['multi_same_category'])} 張 · "
            f"不同類 {len(result.candidates['multi_different_categories'])} 張；候選組可重疊。"
        )
        st.dataframe(result.categories, hide_index=True, width="stretch")
    if workspace == "Image Selection":
        st.caption(
            f"目前類別與範圍共有 {len(base_ids)} 張候選圖。未選擇一律保持 Pending。"
            f"已載入並保留 {sum(is_reviewed(row) for row in reviews)} 筆既有眼部審核。"
        )
    else:
        st.caption(
            "只有 Selected 圖片會進入此工作區；五項填完含 Uncertain 算完成審核，"
            "不等於量測合格。取消選圖不會刪除既有眼部資料。"
        )
    if message := st.session_state.pop("saved_message", ""):
        st.success(message)
    if not image_ids:
        if workspace == "Eye Review" and not selected_ids:
            st.info(
                "尚無 Selected 圖片。請先到 Image Selection 保留圖片，再進行眼部審核。"
                "既有人工審核仍完整保存。"
            )
        else:
            st.info("目前篩選沒有符合的圖片或實例，請調整類別、範圍或狀態。")
        return
    navigation_context = f"{workspace}:{selection_path}:{review_path}:{scope}:{category}:{status}"
    if st.session_state.get("browser_context") != navigation_context:
        st.session_state["browser_context"] = navigation_context
        st.session_state["image_id"] = image_ids[0]
    if request := st.session_state.pop("navigate_image", None):
        if request[0] == navigation_context and request[1] in image_ids:
            st.session_state["image_id"] = request[1]
    if st.session_state.get("image_id") not in image_ids:
        st.session_state["image_id"] = image_ids[0]
    eye_mode = workspace == "Eye Review"
    nav = st.columns([1, 3, 3, 1] if eye_mode else [1, 6, 1], vertical_alignment="bottom")
    image_id = nav[1].selectbox(
        "COCO Image ID", image_ids, key="image_id", format_func=lambda key: f"Image {key}"
    )
    index = image_ids.index(image_id)
    nav[0].button("← 上一張", on_click=move_image, args=(image_ids, -1), disabled=index == 0)
    nav[-1].button(
        "下一張 →", on_click=move_image, args=(image_ids, 1), disabled=index == len(image_ids) - 1
    )
    annotation_id = None
    if eye_mode:
        annotation_ids = sorted(key[1] for key in visible_keys if key[0] == image_id)
        if st.session_state.get("annotation_id") not in annotation_ids:
            st.session_state["annotation_id"] = annotation_ids[0]
        annotation_id = nav[2].selectbox(
            "Animal Instance · Annotation ID",
            annotation_ids,
            key="annotation_id",
            format_func=lambda key: (
                f"{dataset.categories[dataset.annotations[key].category_id].name} · #{key} · "
                f"{'已完成' if is_reviewed(records[(image_id, key)]) else '待審核'}"
            ),
        )
        show_eye_condition(
            [records[(image_id, ann.id)] for ann in result.eligible_by_image[image_id]]
        )
    left, right = st.columns([2.1, 1], gap="large")
    with left:
        show_image(dataset, result, config.images, image_id, annotation_id)
    with right:
        if eye_mode:
            st.subheader("Eye Review")
            current = records[(image_id, annotation_id)]
            st.caption(
                f"{current['animal_category']} · Image {image_id} · Annotation {annotation_id}"
            )
            st.caption("左右以動物本身的解剖學方向判讀；可見不等於可精確分割。")
            review_form(review_path, template, current)
        else:
            selection_form(
                selection_path,
                candidate_ids,
                image_selections[image_id],
                image_ids,
                navigation_context,
            )


if __name__ == "__main__":
    main()
