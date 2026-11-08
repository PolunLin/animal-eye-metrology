"""Read-only paired evaluation of the approved frozen prompt/ROI factorial experiment."""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.measurement.measurement_baseline import derive, verify
from animal_eye_metrology.measurement.metrology import build_measurements
from animal_eye_metrology.research.auto_eye_localization import point_in_mask
from animal_eye_metrology.research.eye_comparison import distance_statistics, eye_statistics

Row = dict[str, Any]
GROUPS = {
    "A": ("hrnet", "none"),
    "B": ("manual", "none"),
    "C": ("hrnet", "manual"),
    "D": ("manual", "manual"),
}
FACTORS = (0.5, 1.0, 1.5)
IDENTITY = ("image_id", "annotation_id", "eye_side")
DESCRIPTOR = ("condition_id", "group", "roi_factor", "point_source", "box_source")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def finite_mean(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return mean(defined) if defined else None


def difference(target: float | None, baseline: float | None) -> float | None:
    return target - baseline if target is not None and baseline is not None else None


def key(row: Row) -> tuple[int, int, str]:
    return int(row["image_id"]), int(row["annotation_id"]), row["eye_side"]


def distance_key(row: Row) -> tuple[int, str, int, int]:
    return (
        int(row["image_id"]),
        row["measurement_type"],
        int(row["annotation_id_a"]),
        int(row["annotation_id_b"]),
    )


def binary(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        pixels = np.array(opened)
    if pixels.ndim != 2 or not np.isin(pixels, [0, 255]).all():
        raise Phase1Error(f"Expected an original-grid binary PNG: {path}")
    return (pixels > 0).astype(np.uint8)


def safe_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise Phase1Error("Prediction artifact must remain inside the run directory.")
    return path


def validate_conditions(conditions: list[Row]) -> dict[str, Row]:
    expected = {(g, f) for g in GROUPS for f in FACTORS}
    if len(conditions) != 12 or {(r["group"], r["roi_factor"]) for r in conditions} != expected:
        raise Phase1Error("Expected the complete approved 2x2x3 design.")
    result = {}
    for row in conditions:
        if (row["point_source"], row["box_source"]) != GROUPS[row["group"]]:
            raise Phase1Error("Condition prompt sources do not match its group.")
        if row["condition_id"] in result:
            raise Phase1Error("Duplicate condition identifier.")
        result[row["condition_id"]] = row
    return result


def mask_and_candidates(out: Path, result: Row, gt: np.ndarray) -> tuple[Any, list[Row]]:
    """Reconstruct every candidate; main mask must equal the highest model-score candidate."""
    if result["status"] == "failed":
        return None, [
            {
                "candidate_index": i,
                "model_score": None,
                "is_model_selected": False,
                "is_best_iou_diagnostic": False,
                **eye_statistics(None, gt),
            }
            for i in range(3)
        ]
    if result["status"] != "passed":
        raise Phase1Error("Unknown prediction status.")
    actual = binary(safe_artifact(out, result["mask_path"]))
    logits = np.load(safe_artifact(out, result["logits_path"]), allow_pickle=False)
    scores = np.asarray(result["candidate_scores"], dtype=float)
    x0, y0, x1, y1 = result["roi_xyxy"]
    if (
        actual.shape != gt.shape
        or logits.shape != (3, y1 - y0, x1 - x0)
        or scores.shape != (3,)
        or not np.isfinite(logits).all()
        or not np.isfinite(scores).all()
    ):
        raise Phase1Error("Invalid candidate shape, scores or original-grid dimensions.")
    selected = int(np.argmax(scores))
    if selected != result["selected_candidate"]:
        raise Phase1Error("Saved selection differs from maximum model score.")
    candidates = []
    for index in range(3):
        mask = np.zeros_like(gt)
        mask[y0:y1, x0:x1] = logits[index] > 0
        if index == selected and not np.array_equal(mask, actual):
            raise Phase1Error("Selected logits do not reproduce the saved original-grid mask.")
        candidates.append(
            {
                "candidate_index": index,
                "model_score": float(scores[index]),
                "is_model_selected": index == selected,
                **eye_statistics(mask, gt),
            }
        )
    best = int(np.argmax([r["iou"] for r in candidates]))
    for row in candidates:
        row["is_best_iou_diagnostic"] = row["candidate_index"] == best
    return actual, candidates


def eye_row(
    out: Path, prediction: Row, ref: Row, gt: np.ndarray, opposite: np.ndarray
) -> tuple[Row, list[Row]]:
    roi = prediction["roi_xyxy"]
    if (
        len(roi) != 4
        or any(type(v) is not int for v in roi)
        or not (0 <= roi[0] < roi[2] <= gt.shape[1] and 0 <= roi[1] < roi[3] <= gt.shape[0])
    ):
        raise Phase1Error("Invalid ROI on the reference original grid.")
    point = prediction["point_xy"]
    if len(point) != 2 or not all(math.isfinite(v) for v in point):
        raise Phase1Error("Nonfinite or invalid prompt point.")
    predicted, candidates = mask_and_candidates(out, prediction, gt)
    metrics = eye_statistics(predicted, gt)
    x0, y0, x1, y1 = roi
    border = None
    if predicted is not None:
        crop = predicted[y0:y1, x0:x1]
        border = bool(
            crop[0, :].any() or crop[-1, :].any() or crop[:, 0].any() or crop[:, -1].any()
        )
    best = next((r for r in candidates if r["is_best_iou_diagnostic"]), None)
    identity = {field: prediction[field] for field in DESCRIPTOR + IDENTITY}
    identity.update(
        prediction_id=prediction["prediction_id"],
        case_id=prediction["case_id"],
        animal_category=ref["animal_category"],
    )
    row = {
        **identity,
        "point_x": point[0],
        "point_y": point[1],
        "point_score": prediction["point_score"],
        "point_in_target_mask": point_in_mask(point, gt),
        "point_in_opposite_mask": point_in_mask(point, opposite),
        "point_offset_from_mask_centroid_px": math.dist(
            point, (ref["centroid_x_px"], ref["centroid_y_px"])
        ),
        "roi_x0": x0,
        "roi_y0": y0,
        "roi_x1": x1,
        "roi_y1": y1,
        "roi_clipped_to_image": prediction["roi_clipped_to_image"],
        "point_outside_prompt_box": prediction["point_outside_prompt_box"],
        "reference_pixels_outside_roi": int(gt.sum()) - int(gt[y0:y1, x0:x1].sum()),
        "mask_touches_roi_boundary": border,
        "failure_reason": prediction["failure_reason"],
        "visibility_status": "unverified",
        "predicted_mask_path": prediction.get("mask_path", ""),
        **metrics,
        "area_ratio": metrics["predicted_area_px"] / metrics["reference_area_px"]
        if metrics["predicted_area_px"] is not None
        else None,
        "candidate_best_iou": best["iou"] if best else None,
        "candidate_best_iou_index": best["candidate_index"] if best else None,
        "candidate_iou_gap": difference(best["iou"], metrics["iou"]) if best else None,
    }
    return row, [{**identity, **r, "diagnostic_only": True} for r in candidates]


def measurements(eyes: list[Row], references: list[Row], descriptor: Row) -> list[Row]:
    centers = [
        {
            **{k: r[k] for k in IDENTITY},
            "annotation_version": 1,
            "centroid_x_px": r["predicted_cx_px"],
            "centroid_y_px": r["predicted_cy_px"],
        }
        for r in eyes
    ]
    predictions = {distance_key(r): r for r in build_measurements(centers)}
    rows = []
    for ref in references:
        pred = predictions[distance_key(ref)]
        rows.append(
            {
                **{k: descriptor[k] for k in DESCRIPTOR},
                **{
                    k: ref[k]
                    for k in ("image_id", "measurement_type", "annotation_id_a", "annotation_id_b")
                },
                **distance_statistics(pred["distance_px"], ref["distance_px"]),
                **{f"predicted_{k}": pred[k] for k in ("cx_a_px", "cy_a_px", "cx_b_px", "cy_b_px")},
                "visibility_status": "unverified",
            }
        )
    return rows


def summarize(eyes: list[Row], distances: list[Row], descriptor: Row) -> Row:
    errors = [r["centroid_error_px"] for r in eyes if r["centroid_error_px"] is not None]
    result = {
        **{k: descriptor[k] for k in DESCRIPTOR},
        "eyes_expected": len(eyes),
        "inference_failed": sum(r["prediction_status"] == "inference_failed" for r in eyes),
        "empty_masks": sum(r["prediction_status"] == "empty" for r in eyes),
        "nonempty_masks": sum(r["prediction_status"] == "nonempty" for r in eyes),
        "centroid_defined": len(errors),
        "mean_centroid_error_px": mean(errors) if errors else None,
        "max_centroid_error_px": max(errors) if errors else None,
        "mean_iou_all_targets_failure_as_zero": mean(r["iou"] or 0.0 for r in eyes),
        "mean_dice_all_targets_failure_as_zero": mean(r["dice"] or 0.0 for r in eyes),
        "points_in_target_mask": sum(r["point_in_target_mask"] is True for r in eyes),
        "points_in_opposite_mask": sum(r["point_in_opposite_mask"] is True for r in eyes),
        "mean_point_offset_from_mask_centroid_px": finite_mean(
            [r["point_offset_from_mask_centroid_px"] for r in eyes]
        ),
        "reference_pixels_outside_roi": sum(r["reference_pixels_outside_roi"] for r in eyes),
        "masks_touching_roi_boundary": sum(r["mask_touches_roi_boundary"] is True for r in eyes),
        "rois_clipped_to_image": sum(r["roi_clipped_to_image"] for r in eyes),
        "point_box_conflicts": sum(r["point_outside_prompt_box"] for r in eyes),
        "candidate_gap_defined": sum(r["candidate_iou_gap"] is not None for r in eyes),
        "mean_candidate_iou_gap": finite_mean([r["candidate_iou_gap"] for r in eyes]),
        "candidate_best_iou_all_targets_failure_as_zero": mean(
            r["candidate_best_iou"] or 0.0 for r in eyes
        ),
    }
    for kind in ("binocular", "right_eye_pair"):
        values = [r for r in distances if r["measurement_type"] == kind]
        errors = [r["absolute_error_px"] for r in values if r["absolute_error_px"] is not None]
        result.update(
            {
                f"{kind}_expected": len(values),
                f"{kind}_defined": len(errors),
                f"{kind}_mae_px": mean(errors) if errors else None,
                f"{kind}_rmse_px": math.sqrt(mean(e * e for e in errors)) if errors else None,
                f"{kind}_max_error_px": max(errors) if errors else None,
            }
        )
    return result


def contrasts(conditions: list[Row]) -> list[Row]:
    lookup = {(r["group"], r["roi_factor"]): r["condition_id"] for r in conditions}
    rows = []
    for factor in FACTORS:
        tag = f"r{round(factor * 100):03d}"
        for kind, target, baseline in (
            ("point", "B", "A"),
            ("point", "D", "C"),
            ("box", "C", "A"),
            ("box", "D", "B"),
        ):
            rows.append(
                {
                    "contrast_id": f"{kind}_{target}_minus_{baseline}_{tag}",
                    "contrast_type": kind,
                    "target_condition": lookup[target, factor],
                    "baseline_condition": lookup[baseline, factor],
                }
            )
    for group in GROUPS:
        for factor in (0.5, 1.5):
            tag = f"r{round(factor * 100):03d}"
            rows.append(
                {
                    "contrast_id": f"roi_{group}_{tag}_minus_r100",
                    "contrast_type": "roi",
                    "target_condition": lookup[group, factor],
                    "baseline_condition": lookup[group, 1.0],
                }
            )
    return rows


def paired(
    conditions: list[Row],
    eyes_by_condition: dict[str, list[Row]],
    distances_by_condition: dict[str, list[Row]],
) -> tuple[list[Row], list[Row], list[Row]]:
    eyes, distances, summaries = [], [], []
    for contrast in contrasts(conditions):
        base_id, target_id = contrast["baseline_condition"], contrast["target_condition"]
        base_eyes = {key(r): r for r in eyes_by_condition[base_id]}
        local_eyes = []
        for target in eyes_by_condition[target_id]:
            base = base_eyes[key(target)]
            row = {
                **contrast,
                **{k: target[k] for k in IDENTITY},
                "case_id": target["case_id"],
                "animal_category": target["animal_category"],
                "baseline_prediction_status": base["prediction_status"],
                "target_prediction_status": target["prediction_status"],
            }
            for field in (
                "iou",
                "dice",
                "centroid_error_px",
                "point_offset_from_mask_centroid_px",
                "area_ratio",
                "candidate_iou_gap",
            ):
                row.update(
                    {
                        f"baseline_{field}": base[field],
                        f"target_{field}": target[field],
                        f"delta_{field}": difference(target[field], base[field]),
                    }
                )
            for metric in ("iou", "dice"):
                row[f"delta_{metric}_failure_as_zero"] = (target[metric] or 0) - (base[metric] or 0)
            local_eyes.append(row)
        base_distances = {distance_key(r): r for r in distances_by_condition[base_id]}
        local_distances = []
        for target in distances_by_condition[target_id]:
            base = base_distances[distance_key(target)]
            row = {
                **contrast,
                **{
                    k: target[k]
                    for k in ("image_id", "measurement_type", "annotation_id_a", "annotation_id_b")
                },
            }
            for field in ("predicted_distance_px", "absolute_error_px", "signed_error_px"):
                row.update(
                    {
                        f"baseline_{field}": base[field],
                        f"target_{field}": target[field],
                        f"delta_{field}": difference(target[field], base[field]),
                    }
                )
            local_distances.append(row)
        summary = {
            **contrast,
            "eyes_expected": len(local_eyes),
            "mean_delta_iou_all_targets_failure_as_zero": mean(
                r["delta_iou_failure_as_zero"] for r in local_eyes
            ),
            "mean_delta_dice_all_targets_failure_as_zero": mean(
                r["delta_dice_failure_as_zero"] for r in local_eyes
            ),
            "iou_improved": sum(r["delta_iou_failure_as_zero"] > 0 for r in local_eyes),
            "iou_worsened": sum(r["delta_iou_failure_as_zero"] < 0 for r in local_eyes),
            "iou_unchanged": sum(r["delta_iou_failure_as_zero"] == 0 for r in local_eyes),
            "centroid_pairs_defined": sum(
                r["delta_centroid_error_px"] is not None for r in local_eyes
            ),
            "mean_delta_centroid_error_px": finite_mean(
                [r["delta_centroid_error_px"] for r in local_eyes]
            ),
        }
        for kind in ("binocular", "right_eye_pair"):
            values = [
                r["delta_absolute_error_px"]
                for r in local_distances
                if r["measurement_type"] == kind
            ]
            summary[f"{kind}_pairs_expected"] = len(values)
            summary[f"{kind}_pairs_defined"] = sum(v is not None for v in values)
            summary[f"{kind}_mean_delta_absolute_error_px"] = finite_mean(values)
        eyes.extend(local_eyes)
        distances.extend(local_distances)
        summaries.append(summary)
    return eyes, distances, summaries


def interactions(
    conditions: list[Row], eyes: dict[str, list[Row]], distances: dict[str, list[Row]]
) -> list[Row]:
    lookup = {(r["group"], r["roi_factor"]): r["condition_id"] for r in conditions}
    rows = []
    for factor in FACTORS:
        indexes = {g: {key(r): r for r in eyes[lookup[g, factor]]} for g in GROUPS}
        result = {
            "roi_factor": factor,
            "interaction": "(D-C)-(B-A)",
            "eyes_expected": len(indexes["A"]),
        }
        for field in ("iou", "dice", "centroid_error_px"):
            values = []
            for identity in indexes["A"]:
                v = {g: indexes[g][identity][field] for g in GROUPS}
                if field in ("iou", "dice"):
                    v = {g: value or 0.0 for g, value in v.items()}
                if all(value is not None for value in v.values()):
                    values.append((v["D"] - v["C"]) - (v["B"] - v["A"]))
            suffix = "_all_targets_failure_as_zero" if field in ("iou", "dice") else ""
            result[f"{field}_defined"] = len(values)
            result[f"mean_{field}_difference_in_differences{suffix}"] = finite_mean(values)
        dist_index = {g: {distance_key(r): r for r in distances[lookup[g, factor]]} for g in GROUPS}
        for kind in ("binocular", "right_eye_pair"):
            values = []
            identities = [k for k in dist_index["A"] if k[1] == kind]
            for identity in identities:
                v = {g: dist_index[g][identity]["absolute_error_px"] for g in GROUPS}
                if all(value is not None for value in v.values()):
                    values.append((v["D"] - v["C"]) - (v["B"] - v["A"]))
            result[f"{kind}_expected"] = len(identities)
            result[f"{kind}_defined"] = len(values)
            result[f"{kind}_mean_absolute_error_difference_in_differences_px"] = finite_mean(values)
        rows.append(result)
    return rows


def compute(out: Path) -> dict[str, list[Row]]:
    """Recompute all tables without changing any prediction, source or reference file."""
    conditions = read(out / "inputs/conditions.json")
    descriptors = validate_conditions(conditions)
    reference = out / "evaluation_inputs/reference"
    verify(reference)
    refs, ref_distances = derive(reference / "frozen")
    ref_index = {key(r): r for r in refs}
    if len(ref_index) != len(refs):
        raise Phase1Error("Duplicate reference eye identity.")
    masks = {k: binary(reference / r["mask_path"]) for k, r in ref_index.items()}
    predictions: dict[str, dict[tuple, Row]] = defaultdict(dict)
    for row in read(out / "predictions.json"):
        cid = row["condition_id"]
        if cid not in descriptors or any(row[k] != descriptors[cid][k] for k in DESCRIPTOR):
            raise Phase1Error("Prediction condition metadata differs from the frozen design.")
        identity = key(row)
        if identity in predictions[cid]:
            raise Phase1Error("Duplicate condition/eye prediction.")
        predictions[cid][identity] = row
    if set(predictions) != set(descriptors) or any(
        set(rows) != set(ref_index) for rows in predictions.values()
    ):
        raise Phase1Error("Each condition must retain every reference eye, including failures.")
    all_eyes, all_candidates, all_distances, summaries, image_summaries = [], [], [], [], []
    eyes_by_condition, distances_by_condition = {}, {}
    localization = {}
    for condition in conditions:
        cid = condition["condition_id"]
        eyes = []
        for identity in sorted(ref_index):
            ref = ref_index[identity]
            opposite = (identity[0], identity[1], "right" if identity[2] == "left" else "left")
            row, candidates = eye_row(
                out, predictions[cid][identity], ref, masks[identity], masks[opposite]
            )
            eyes.append(row)
            all_candidates.extend(candidates)
            loc = {
                k: row[k]
                for k in IDENTITY
                + (
                    "point_source",
                    "case_id",
                    "animal_category",
                    "point_x",
                    "point_y",
                    "point_score",
                    "point_in_target_mask",
                    "point_in_opposite_mask",
                    "point_offset_from_mask_centroid_px",
                )
            }
            loc_key = (row["point_source"], *identity)
            if loc_key in localization and loc != localization[loc_key]:
                raise Phase1Error("Point localization changed when only box/ROI should change.")
            localization[loc_key] = loc
        distances = measurements(eyes, ref_distances, condition)
        summaries.append(summarize(eyes, distances, condition))
        for image_id in sorted({r["image_id"] for r in eyes}):
            image_summaries.append(
                {
                    "image_id": image_id,
                    **summarize(
                        [r for r in eyes if r["image_id"] == image_id],
                        [r for r in distances if r["image_id"] == image_id],
                        condition,
                    ),
                }
            )
        eyes_by_condition[cid], distances_by_condition[cid] = eyes, distances
        all_eyes.extend(eyes)
        all_distances.extend(distances)
    paired_eyes, paired_distances, contrast_summary = paired(
        conditions, eyes_by_condition, distances_by_condition
    )
    return {
        "eye_metrics.csv": all_eyes,
        "candidate_metrics.csv": all_candidates,
        "measurement_comparison.csv": all_distances,
        "condition_summary.csv": summaries,
        "image_summary.csv": image_summaries,
        "localization_metrics.csv": [localization[k] for k in sorted(localization)],
        "paired_eye_deltas.csv": paired_eyes,
        "paired_distance_deltas.csv": paired_distances,
        "contrast_summary.csv": contrast_summary,
        "interaction_summary.csv": interactions(
            conditions, eyes_by_condition, distances_by_condition
        ),
    }
