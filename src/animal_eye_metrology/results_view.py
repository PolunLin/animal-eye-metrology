"""Read-only Streamlit views for frozen eye experiments; no inference or annotation writes."""

import csv
import io
import logging
import math
import os
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from animal_eye_metrology.results_definitions import (
    DefinitionWorkspace,
    load_definition_workspace,
)
from animal_eye_metrology.results_overlay import (
    render_eye_comparison,
    render_measurement_overview,
)
from animal_eye_metrology.results_workspace import ResultBundle

LOGGER = logging.getLogger(__name__)
BASELINE = "A_r100"
SIDES = {"left": "左眼", "right": "右眼"}
CATEGORIES = {"bear": "熊", "dog": "狗", "giraffe": "長頸鹿"}
DEFINITIONS = {
    "v1": "v1 · 可見眼裂",
    "v2": "v2 · 眼裂＋眼瞼邊緣",
    "paired": "v1／v2 並列",
}
PROTOCOLS = {"v1": "eye-visible-aperture-v1", "v2": "eye-visible-eyelid-v2"}
FAILURE_CASES = {
    "狗 17207 · 左眼邊界": (
        125405,
        17207,
        "left",
        "各提示仍有眼周過度分割；既有候選重排不足以解決眼部邊界。",
    ),
    "狗 17207 · 右眼邊界": (
        125405,
        17207,
        "right",
        "即使人工點框診斷也不理想；須區分定位資訊與分割邊界能力。",
    ),
    "狗 13692 · 右眼定位": (
        125405,
        13692,
        "right",
        "原基線分割與參考無重疊；可比較縮小 ROI 與加框的影響。",
    ),
    "狗 13692 · 左眼退步": (
        125405,
        13692,
        "left",
        "加框使原本較好的結果退步；不能只展示同隻動物右眼的改善。",
    ),
    "長頸鹿 599888 · 右眼漏蓋": (
        333069,
        599888,
        "right",
        "小框可能漏蓋參考眼部，大框即使全蓋仍包含眼周；候選品質與排序需分開看。",
    ),
    "熊 588042 · 左眼退步": (
        20247,
        588042,
        "left",
        "原基線較佳，加大幾何框破壞原本良好的分割，是必須保留的反例。",
    ),
}
V2_CASE_NOTES = {
    "狗 17207 · 左眼邊界": "A_r100 對 v2 的範圍更吻合；原 v1 低 IoU 部分來自標的定義。"
    "重心誤差沒有同步改善，不能宣稱模型能力進步。",
    "狗 17207 · 右眼邊界": "A_r100 對 v2 的重疊與重心誤差較佳；保留原 v1 失敗紀錄，"
    "不能再把新版結果一概稱為邊界失敗。",
    "狗 13692 · 右眼定位": "A_r100 對 v2 仍只有少量重疊與明顯重心偏移；定位／局部範圍問題仍在。",
    "狗 13692 · 左眼退步": "A_r100 的小遮罩未涵蓋 v2 眼部區域；即使重心接近，也不代表分割完整。",
    "長頸鹿 599888 · 右眼漏蓋": "A_r100 仍分到大片頭部，對 v2 仍屬低重疊；"
    "改變眼睛定義沒有解決此誤分。",
    "熊 588042 · 左眼退步": "v2 人工邊界及重心變化較大，A_r100 的重疊下降、"
    "重心誤差增加；需區分參考定義敏感性與模型問題。",
}


def selected_definitions(workspace: DefinitionWorkspace, mode: str) -> tuple[str, ...]:
    return workspace.definitions if mode == "paired" else (mode,)


def tagged(row: dict[str, Any], definition: str) -> dict[str, Any]:
    return dict(row, reference_definition=definition, reference_protocol=PROTOCOLS[definition])


def original_image(bundle: ResultBundle, image_id: int) -> Image.Image:
    run = bundle.runs["reference"]
    hashes = run.manifest.get("files_sha256", run.manifest.get("artifacts_sha256", {}))
    matches = [
        name
        for name in hashes
        if name.startswith("frozen/images/")
        and Path(name).stem.isdigit()
        and int(Path(name).stem) == image_id
    ]
    if len(matches) != 1:
        raise ValueError(f"原圖來源必須唯一：{image_id}")
    return load_image(run.asset_path(matches[0]))


def difference(first: Any, second: Any) -> float | None:
    return second - first if first is not None and second is not None else None


def condition_label(condition: str) -> str:
    if condition == BASELINE:
        return "A_r100 · 原基線／自動點／1× ROI"
    if condition == "A_r050":
        return "A_r050 · 對照／自動點／0.5× ROI"
    if condition.startswith("auto_box_"):
        gamma = int(condition.split("_")[2][1:]) / 100
        prompt = "點＋框" if condition.endswith("point") else "只有框"
        status = "主方法未晉級" if condition == "auto_box_k050_point" else "探索"
        return f"γ={gamma:g} {prompt} · {status}"
    group, scale = condition.split("_r")
    source = {"A": "自動點", "B": "人工點", "C": "自動點＋人工框", "D": "人工點＋人工框"}[group]
    return f"{condition} · {source}／{int(scale) / 100:g}× ROI"


def method_kind(condition: str) -> str:
    if condition == BASELINE:
        return "原自動基線"
    if condition.startswith(("B_", "C_", "D_")):
        return "人工介入診斷"
    if condition == "auto_box_k050_point":
        return "自動框主方法・未晉級"
    return "自動提示對照／探索"


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def table(rows: list[dict[str, Any]], fields: dict[str, str], key: str) -> None:
    if not rows:
        st.info("此篩選沒有已保存結果。")
        return
    shown = [{label: row.get(field) for field, label in fields.items()} for row in rows]
    numeric = {
        label: st.column_config.NumberColumn(format="%.3f")
        for field, label in fields.items()
        if any(isinstance(row.get(field), float) for row in rows)
    }
    st.dataframe(shown, hide_index=True, width="stretch", column_config=numeric, key=key)


def download(rows: list[dict[str, Any]], label: str, filename: str, key: str) -> None:
    st.download_button(label, csv_bytes(rows), filename, "text/csv", key=key)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.copy()


def pick(label: str, options: list[Any], key: str, **kwargs: Any) -> Any:
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    return st.selectbox(label, options, key=key, **kwargs)


def find_eye(bundle: ResultBundle, condition: str, case: tuple[int, int, str]) -> dict[str, Any]:
    return next(
        row
        for row in bundle.eyes
        if row["condition_id"] == condition
        and (row["image_id"], row["annotation_id"], row["eye_side"]) == case
    )


def summary_view(workspace: DefinitionWorkspace, mode: str) -> None:
    bundle = workspace.v1
    st.subheader("條件比較")
    binocular = sum(r["measurement_type"] == "binocular" for r in bundle.reference_measurements)
    right_pair = sum(
        r["measurement_type"] == "right_eye_pair" for r in bundle.reference_measurements
    )
    st.caption(
        f"每列都是同一批開發圖：{len(bundle.reference_eyes)} 眼、{binocular} 筆雙眼距離、"
        f"{right_pair} 筆跨動物右眼距離。不同條件不增加獨立樣本數。"
    )
    include_manual = st.checkbox("加入人工提示診斷 B／C／D", key="results_include_manual")
    rows = []
    for condition in bundle.conditions:
        identifier = condition["condition_id"]
        if not include_manual and identifier.startswith(("B_", "C_", "D_")):
            continue
        for definition in selected_definitions(workspace, mode):
            row = next(
                row
                for row in workspace.for_definition(definition).conditions
                if row["condition_id"] == identifier
            )
            rows.append(tagged(dict(row, method=method_kind(identifier)), definition))
    table(
        rows,
        {
            "condition_id": "條件",
            "reference_definition": "參考定義",
            "method": "用途",
            "roi_factor": "ROI 倍率",
            "mean_iou_all_targets_failure_as_zero": "IoU ↑",
            "mean_dice_all_targets_failure_as_zero": "Dice ↑",
            "mean_centroid_error_px": "重心偏差 px ↓",
            "binocular_mae_px": "雙眼距離 MAE px ↓",
            "right_eye_pair_mae_px": "跨右眼距離 MAE px ↓",
            "nonempty_masks": "非空眼數",
            "eyes_expected": "預期眼數",
            "centroid_defined": "重心已定義",
            "binocular_defined": "雙眼已定義",
            "binocular_expected": "雙眼預期",
            "right_eye_pair_defined": "跨右眼已定義",
            "right_eye_pair_expected": "跨右眼預期",
        },
        "results_summary_table",
    )
    st.caption(
        "IoU／Dice 以全部目標為分母，失敗按既有規則計零；"
        "重心與距離誤差只對已定義結果計算，分母並列呈現。兩版列是相同樣本，並未增加眼數。"
        "B／C／D 沿用原 v1 人工提示，僅供診斷，非 v2 人工提示上限。"
    )
    if mode == "paired":
        with st.expander("定義變更量：v2 − v1"):
            identifiers = {r["condition_id"] for r in rows}
            fields = {
                "mean_iou_all_targets_failure_as_zero": "Δ IoU",
                "mean_dice_all_targets_failure_as_zero": "Δ Dice",
                "mean_centroid_error_px": "Δ 重心誤差 px",
                "binocular_mae_px": "Δ 雙眼 MAE px",
                "right_eye_pair_mae_px": "Δ 跨右眼 MAE px",
            }
            changes = [
                {
                    "condition_id": row["condition_id"],
                    **{k: difference(row[f"v1_{k}"], row[f"v2_{k}"]) for k in fields},
                }
                for row in workspace.paired_conditions
                if row["condition_id"] in identifiers
            ]
            table(changes, {"condition_id": "條件", **fields}, "results_summary_changes")
            st.caption("Δ IoU／Dice 正值為重疊較高；Δ 誤差負值為誤差較小。只有參考標的改變。")
    download(
        rows,
        "下載目前條件比較 CSV",
        f"condition_comparison_{mode}.csv",
        "results_summary_download",
    )
    with st.expander("原 v1 幾何自動框晉級判定"):
        st.warning("γ＝0.5 點＋框未晉級；保留 A_r100。其他 γ 為探索條件，沒有事後改選主方法。")
        st.caption("下表保留原 v1 預先指定門檻與判定，不隨定義切換重做晉級。")
        table(
            bundle.promotion,
            {
                "condition_id": "條件",
                "baseline_condition": "參考條件",
                "check_id": "門檻",
                "value": "觀察值",
                "operator": "比較",
                "threshold": "預設門檻",
                "passed": "通過",
            },
            "results_promotion_table",
        )


def shared_eye_view(
    workspace: DefinitionWorkspace,
    conditions: list[str],
    case: tuple[int, int, str],
    original: Image.Image,
) -> tuple[int, int, int, int]:
    """Keep the viewport stable across definitions and include complete saved masks."""
    extents = []
    for condition in conditions:
        row = find_eye(workspace.v1, condition, case)
        prediction = workspace.v1.prediction_for_eye(row)
        if prediction.get("roi_xyxy"):
            extents.append(prediction["roi_xyxy"])
        if row.get("predicted_mask_path"):
            mask = load_image(
                workspace.v1.asset_path(row["source_run"], row["predicted_mask_path"])
            )
            if mask.getbbox() is not None:
                extents.append(mask.getbbox())
    for definition in workspace.definitions:
        bundle = workspace.for_definition(definition)
        reference = bundle.reference_for_eye(find_eye(bundle, BASELINE, case))
        mask = load_image(bundle.asset_path("reference", reference["mask_path"]))
        if mask.getbbox() is not None:
            extents.append(mask.getbbox())
    if not extents:
        return (0, 0, original.width, original.height)
    return (
        max(0, math.floor(min(roi[0] for roi in extents)) - 3),
        max(0, math.floor(min(roi[1] for roi in extents)) - 3),
        min(original.width, math.ceil(max(roi[2] for roi in extents)) + 3),
        min(original.height, math.ceil(max(roi[3] for roi in extents)) + 3),
    )


def eye_panel(
    bundle: ResultBundle,
    row: dict[str, Any],
    original: Image.Image,
    view: tuple[int, int, int, int],
    heading: str,
) -> None:
    st.markdown(f"**{heading}**")
    prediction = bundle.prediction_for_eye(row)
    reference = bundle.reference_for_eye(row)
    reference_mask = load_image(bundle.asset_path("reference", reference["mask_path"]))
    mask_path = row.get("predicted_mask_path")
    mask = load_image(bundle.asset_path(row["source_run"], mask_path)) if mask_path else None
    st.image(
        render_eye_comparison(original, mask, reference_mask, row, prediction, view),
        width="stretch",
    )
    iou = "未定義" if row.get("iou") is None else f"{row['iou']:.3f}"
    st.caption(f"IoU {iou} · {row['prediction_status']} · 可見性：{row['visibility_status']}")


def eye_pair(
    workspace: DefinitionWorkspace, mode: str, case: tuple[int, int, str], prefix: str
) -> None:
    bundle = workspace.v1
    options = [r["condition_id"] for r in bundle.conditions if r["condition_id"] != BASELINE]
    default = "auto_box_k050_point" if prefix == "failure" else "A_r050"
    compare = pick(
        "對照條件",
        options,
        f"results_{prefix}_condition",
        index=options.index(default),
        format_func=condition_label,
    )
    conditions = [BASELINE, compare]
    definitions = selected_definitions(workspace, mode)
    selected = [
        tagged(find_eye(workspace.for_definition(tag), condition, case), tag)
        for condition in conditions
        for tag in definitions
    ]
    original = original_image(bundle, case[0])
    view = shared_eye_view(workspace, conditions, case, original)
    if mode == "paired":
        st.caption("每排固定同一模型輸出，左為 v1、右為 v2；預測 Mask、正點、框與 ROI 完全相同。")
        for condition in conditions:
            st.markdown(f"**{condition_label(condition)}**")
            for column, tag in zip(st.columns(2), definitions, strict=True):
                with column:
                    current = workspace.for_definition(tag)
                    eye_panel(
                        current,
                        find_eye(current, condition, case),
                        original,
                        view,
                        DEFINITIONS[tag],
                    )
    else:
        current = workspace.for_definition(mode)
        for column, condition in zip(st.columns(2), conditions, strict=True):
            with column:
                eye_panel(
                    current,
                    find_eye(current, condition, case),
                    original,
                    view,
                    f"{condition_label(condition)} · {DEFINITIONS[mode]}",
                )
    st.caption(
        "所有欄使用相同原圖視野與縮放，包含兩版參考與完整預測遮罩。"
        "青色：固定預測 Mask／圓形重心；綠色：該版人工參考輪廓／方形重心；"
        "黃色＋：實際 SAM 正點；粉紅：提示框；灰色：ROI。只有框條件不畫正點。"
    )
    table(
        selected,
        {
            "condition_id": "條件",
            "reference_definition": "參考定義",
            "iou": "IoU",
            "dice": "Dice",
            "predicted_cx_px": "預測重心 x",
            "predicted_cy_px": "預測重心 y",
            "reference_cx_px": "參考重心 x",
            "reference_cy_px": "參考重心 y",
            "centroid_error_px": "重心偏差 px",
            "predicted_area_px": "預測面積 px²",
            "reference_area_px": "參考面積 px²",
            "visibility_status": "可見性狀態",
            "failure_reason": "執行失敗原因",
        },
        f"results_{prefix}_metrics",
    )
    if mode == "paired":
        changes = [
            row
            for row in workspace.paired_eyes
            if row["condition_id"] in conditions
            and (row["image_id"], row["annotation_id"], row["eye_side"]) == case
        ]
        table(
            changes,
            {
                "condition_id": "條件",
                "delta_iou_v2_minus_v1": "Δ IoU（v2 − v1）",
                "delta_dice_v2_minus_v1": "Δ Dice",
                "delta_centroid_error_px_v2_minus_v1": "Δ 重心誤差 px",
            },
            f"results_{prefix}_changes",
        )
    st.caption(
        "執行失敗原因為空僅代表未記錄執行錯誤；分割品質差仍是失敗案例。"
        "重心是該版眼部區域 Mask 重心，並非瞳孔中心；定義變更不代表模型能力進步。"
    )
    if compare.startswith(("B_", "C_", "D_")):
        st.caption("本條件沿用原 v1 人工提示，僅供診斷，非 v2 人工提示上限。")
    with st.expander("固定提示與原 v1 定位／候選診斷"):
        for condition in conditions:
            row = find_eye(bundle, condition, case)
            prediction = bundle.prediction_for_eye(row)
            st.markdown(f"**{condition_label(condition)}**")
            st.json(
                {
                    "prompt_point_xy": prediction.get("point_xy"),
                    "prompt_box_xyxy": prediction.get("box_xyxy"),
                    "roi_xyxy": prediction.get("roi_xyxy"),
                    "hrnet_geometry_point_xy": prediction.get("hrnet_point_xy"),
                    "selected_candidate": prediction.get("selected_candidate"),
                    "candidate_scores": prediction.get("candidate_scores"),
                    "historical_diagnostic_reference": "v1",
                    "v1_point_in_target_mask": row.get(
                        "point_in_target_mask", row.get("geometry_point_in_target_mask")
                    ),
                    "v1_point_to_reference_mask_centroid_px": row.get(
                        "point_offset_from_mask_centroid_px",
                        row.get("geometry_point_offset_from_mask_centroid_px"),
                    ),
                    "v1_candidate_best_gt_iou_diagnostic_only": row.get("candidate_best_iou"),
                    "v1_reference_box_coverage_fraction": row.get(
                        "reference_box_coverage_fraction"
                    ),
                },
                expanded=True,
            )
        st.caption(
            "提示與候選分數保持原值；點命中、GT 最佳候選 IoU、框涵蓋率只保存了 v1 診斷，"
            "不當成 v2 分數，也沒有用於重新選擇候選。點到 Mask 重心距離不是瞳孔定位誤差。"
        )
    download(
        selected,
        "下載此眼對照 CSV",
        f"eye_{case[0]}_{case[1]}_{case[2]}_{mode}.csv",
        f"results_{prefix}_download",
    )


def eyes_view(workspace: DefinitionWorkspace, mode: str) -> None:
    bundle = workspace.v1
    st.subheader("定位、分割與重心")
    baseline_eyes = [r for r in bundle.eyes if r["condition_id"] == BASELINE]
    image_ids = sorted({r["image_id"] for r in baseline_eyes})
    image_id = pick("圖片", image_ids, "results_eye_image", format_func=lambda x: f"COCO {x}")
    animals = {
        r["annotation_id"]: r["animal_category"] for r in baseline_eyes if r["image_id"] == image_id
    }
    annotation_id = pick(
        "動物實例",
        sorted(animals),
        f"results_eye_animal_{image_id}",
        format_func=lambda x: f"{CATEGORIES.get(animals[x], animals[x])} · {x}",
    )
    side = st.radio(
        "動物自身的左右眼",
        ["left", "right"],
        format_func=SIDES.get,
        horizontal=True,
        key="results_eye_side",
    )
    eye_pair(workspace, mode, (image_id, annotation_id, side), "eye")


def distances_view(workspace: DefinitionWorkspace, mode: str) -> None:
    bundle = workspace.v1
    st.subheader("像素距離量測")
    condition = pick(
        "量測條件",
        [r["condition_id"] for r in bundle.conditions],
        "results_distance_condition",
        format_func=condition_label,
    )
    image_id = pick(
        "量測圖片",
        sorted({r["image_id"] for r in bundle.eyes}),
        "results_distance_image",
        format_func=lambda x: f"COCO {x}",
    )
    chosen_conditions = list(dict.fromkeys([BASELINE, "A_r050", condition]))
    rows = [
        tagged(row, tag)
        for identifier in chosen_conditions
        for tag in selected_definitions(workspace, mode)
        for row in workspace.for_definition(tag).distances
        if row["image_id"] == image_id and row["condition_id"] == identifier
    ]
    selected_distances = [
        row
        for row in bundle.distances
        if row["image_id"] == image_id and row["condition_id"] == condition
    ]
    eyes = [r for r in bundle.eyes if r["image_id"] == image_id and r["condition_id"] == condition]
    original = original_image(bundle, image_id)
    masks = [
        load_image(bundle.asset_path(r["source_run"], r["predicted_mask_path"]))
        if r.get("predicted_mask_path")
        else None
        for r in eyes
    ]
    st.image(
        render_measurement_overview(original, masks, eyes, selected_distances),
        caption=f"{condition_label(condition)} · 固定預測幾何，不隨參考定義改變。"
        "白線：同動物雙眼；粉紅線：跨動物右眼。",
        width="stretch",
    )
    st.caption("解剖左右依動物自身定義；所有座標與距離均在原圖中，以 px 表示，未換算真實長度。")
    table(
        rows,
        {
            "condition_id": "條件",
            "reference_definition": "參考定義",
            "measurement_type": "量測類型",
            "annotation_id_a": "動物 A",
            "annotation_id_b": "動物 B",
            "predicted_distance_px": "預測距離 px",
            "reference_distance_px": "參考距離 px",
            "signed_error_px": "帶符號誤差 px",
            "absolute_error_px": "絕對誤差 px",
            "relative_absolute_error_percent": "相對絕對誤差 %",
            "measurement_status": "數學定義狀態",
            "visibility_status": "量測資格／可見性",
        },
        "results_distances_table",
    )
    st.info("Defined 代表兩個重心可計算距離，不代表量測已通過可見性、側別或可靠性驗收。")
    st.caption("表內預測距離保持原值；兩版差異來自人工參考距離與誤差，不是重新量測模型。")
    if mode == "paired":
        table(
            [
                row
                for row in workspace.paired_distances
                if row["image_id"] == image_id and row["condition_id"] in chosen_conditions
            ],
            {
                "condition_id": "條件",
                "measurement_type": "量測類型",
                "annotation_id_a": "動物 A",
                "annotation_id_b": "動物 B",
                "v1_reference_distance_px": "v1 參考距離 px",
                "v2_reference_distance_px": "v2 參考距離 px",
                "delta_absolute_error_px_v2_minus_v1": "Δ 絕對誤差 px（v2 − v1）",
            },
            "results_distance_changes",
        )
    download(
        rows,
        "下載目前距離對照 CSV",
        f"distances_{image_id}_{mode}.csv",
        "results_distances_download",
    )


def failures_view(workspace: DefinitionWorkspace, mode: str) -> None:
    bundle = workspace.v1
    st.subheader("失敗案例與退步反例")
    st.caption("以下是既有報告保留的案例，沒有新增自動失敗門檻，也沒有因展示而改選候選。")
    name = st.selectbox("既有失敗案例", list(FAILURE_CASES), key="results_failure_case")
    image_id, annotation_id, side, note = FAILURE_CASES[name]
    available = {(r["image_id"], r["annotation_id"], r["eye_side"]) for r in bundle.eyes}
    if (image_id, annotation_id, side) not in available:
        st.info("這份結果資料沒有此既有案例。")
        return
    if mode in {"v1", "paired"}:
        st.warning(f"v1 歷史結論：{note}")
    if mode in {"v2", "paired"}:
        st.info(f"v2 定義對照：{V2_CASE_NOTES[name]}")
        if mode == "v2":
            with st.expander("保留原 v1 失敗結論"):
                st.caption(note)
    eye_pair(workspace, mode, (image_id, annotation_id, side), "failure")
    with st.expander("查看原 v1 完整條件網格（歷史結果）"):
        st.caption("下方網格沿用原 v1 人工參考；不隨上方定義切換，不是 v2 的候選分析。")
        for run_id, label in [
            ("ablation", "12 組提示／ROI"),
            ("auto_box", "6 組幾何自動框＋A 對照"),
        ]:
            relative = f"visualizations/{image_id}_{annotation_id}_{side}_grid.png"
            try:
                path = bundle.asset_path(run_id, relative)
                st.image(str(path), caption=label, width="stretch")
            except (OSError, ValueError) as error:
                st.caption(f"{label}：未保存此案例網格（{error}）")


def provenance_view(workspace: DefinitionWorkspace, mode: str) -> None:
    bundle = workspace.v1
    st.subheader("資料來源與重現")
    st.write(
        "讀取已凍結的實驗產物；本工作區不呼叫模型、不修改人工標註、不覆寫基線。"
        "下載 CSV 只產生瀏覽器下載。"
    )
    st.caption("此頁展示來源紀錄；完整產物驗證仍使用各實驗的 verify 指令。")
    runs = dict(bundle.runs)
    if workspace.v2 is not None:
        runs["reference_v2"] = workspace.v2.runs["reference"]
    if workspace.comparison is not None:
        runs["definition_comparison"] = workspace.comparison
    for run_id, run in runs.items():
        with st.expander(f"{run_id} · {run.directory.name}"):
            st.code(str(run.directory), language=None)
            manifest = run.manifest
            st.json(
                {
                    k: v
                    for k, v in manifest.items()
                    if k not in {"files_sha256", "artifacts_sha256"}
                },
                expanded=False,
            )
            hashes = manifest.get("files_sha256", manifest.get("artifacts_sha256", {}))
            st.caption(f"來源 manifest 登錄 {len(hashes)} 個檔案。")
    st.code(
        ".venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage verify\n"
        ".venv/bin/python -m animal_eye_metrology.prompt_roi_ablation --stage verify\n"
        ".venv/bin/python -m animal_eye_metrology.auto_eye_box --stage verify\n"
        + (
            ".venv/bin/python -m animal_eye_metrology.measurement_baseline "
            "--verify outputs/phase3/manual_measurement_baseline_v2"
            if workspace.v2 is not None
            else ""
        ),
        language="bash",
    )
    eyes = [
        tagged(row, tag)
        for tag in selected_definitions(workspace, mode)
        for row in workspace.for_definition(tag).eyes
    ]
    distances = [
        tagged(row, tag)
        for tag in selected_definitions(workspace, mode)
        for row in workspace.for_definition(tag).distances
    ]
    download(eyes, "下載全部逐眼結果 CSV", f"all_eye_results_{mode}.csv", "results_all_eyes")
    download(
        distances,
        "下載全部距離結果 CSV",
        f"all_distance_results_{mode}.csv",
        "results_all_distances",
    )
    if workspace.comparison is not None:
        st.caption("以下寬表固定包含兩版保存值與差值，不隨上方篩選切換。")
        download(
            workspace.paired_eyes,
            "下載兩版逐眼差異 CSV",
            "v1_v2_eye_comparison.csv",
            "results_paired_eyes",
        )
        download(
            workspace.paired_distances,
            "下載兩版距離差異 CSV",
            "v1_v2_distance_comparison.csv",
            "results_paired_distances",
        )


def results_workspace(root: Path) -> None:
    """Render before loading COCO or initializing any writable review stores."""
    with st.sidebar:
        st.subheader("成果資料")
        st.caption("唯讀 · 已保存實驗")
        with st.expander("結果來源設定"):
            config_path = (
                Path(
                    st.text_input(
                        "成果設定檔",
                        value=os.environ.get(
                            "AEM_RESULTS_CONFIG", str(root / "configs/results_definitions.toml")
                        ),
                        key="results_config_path",
                    )
                )
                .expanduser()
                .resolve()
            )
        st.info("原基線：A_r100\n\n對照：A_r050\n\nγ＝0.5 主方法未晉級")
    try:
        workspace = load_definition_workspace(config_path)
    except (OSError, ValueError, KeyError) as error:
        LOGGER.warning("results_workspace_load_failed", extra={"reason": str(error)})
        st.error(f"無法載入成果：{error}")
        st.info("請核對成果設定檔與已保存產物；此頁不會自動下載、推論或修復來源檔。")
        return
    bundle = workspace.v1
    st.header("自動眼部分析 · 成果檢視")
    st.caption("HRNet-W32 / AP-10K → SAM 2.1 Tiny → Mask 重心 → 像素距離")
    modes = list(workspace.definitions)
    if len(modes) > 1:
        modes.append("paired")
    if st.session_state.get("results_definition") not in modes:
        st.session_state.pop("results_definition", None)
    mode = st.radio(
        "參考定義",
        modes,
        format_func=DEFINITIONS.get,
        horizontal=True,
        key="results_definition",
    )
    if workspace.v2 is not None:
        st.info(
            "v1：可見眼裂；v2：加上可辨識眼瞼邊緣，排除毛髮與色塊。"
            "切換只改變人工參考與已保存評分，同一條件的預測 Mask、重心、距離不變。"
            "選「v1／v2 並列」可直接比較；A_r100 仍為原基線。"
        )
    else:
        st.caption("此設定只有原 v1 結果；使用雙定義成果設定可檢視已凍結 v2 比較。")
    originals = [r for r in bundle.eyes if r["condition_id"] == BASELINE]
    for column, label, value in zip(
        st.columns(4),
        ["開發圖片", "動物實例", "參考眼睛", "實驗條件"],
        [
            len({r["image_id"] for r in originals}),
            len({(r["image_id"], r["annotation_id"]) for r in originals}),
            len(originals),
            len(bundle.conditions),
        ],
        strict=True,
    ):
        column.metric(label, value)
    st.warning(
        "目前為開發圖的研究結果，尚未獨立泛化驗收。全部候選可見性仍為 unverified；"
        "非空 Mask 與可計算距離不代表量測合格。"
    )
    tabs = st.tabs(["總覽比較", "逐眼檢視", "距離量測", "失敗案例", "來源與驗證"])
    for tab, renderer in zip(
        tabs, [summary_view, eyes_view, distances_view, failures_view, provenance_view], strict=True
    ):
        with tab:
            try:
                renderer(workspace, mode)
            except (OSError, ValueError, KeyError, StopIteration) as error:
                LOGGER.warning(
                    "results_workspace_view_failed",
                    extra={"view": renderer.__name__, "reason": str(error)},
                )
                st.error(f"此檢視所需資料不完整或不一致：{error}")
