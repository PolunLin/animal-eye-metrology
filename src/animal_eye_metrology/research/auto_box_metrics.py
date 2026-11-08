"""Read-only evaluation of six automatic geometric eye-box conditions."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.measurement.measurement_baseline import derive, verify
from animal_eye_metrology.research import ablation_metrics as shared

Row = dict[str, Any]
PRIMARY = "auto_box_k050_point"
BASELINES = ("A_r100", "A_r050", "A_r150", "C_r100")
DESCRIPTOR = shared.DESCRIPTOR + ("gamma", "is_primary")
GEOMETRY_FIELDS = {
    "point_x": "geometry_point_x",
    "point_y": "geometry_point_y",
    "point_score": "geometry_point_score",
    "point_in_target_mask": "geometry_point_in_target_mask",
    "point_in_opposite_mask": "geometry_point_in_opposite_mask",
    "point_offset_from_mask_centroid_px": "geometry_point_offset_from_mask_centroid_px",
}


def validate_conditions(conditions: list[Row]) -> dict[str, Row]:
    expected = {
        f"auto_box_k{round(gamma * 100):03d}_{group}": (gamma, group)
        for gamma in (0.3, 0.5, 0.7)
        for group in ("point", "only")
    }
    if len(conditions) != 6 or {r["condition_id"] for r in conditions} != set(expected):
        raise Phase1Error("Expected the six approved automatic box conditions.")
    for row in conditions:
        gamma, group = expected[row["condition_id"]]
        if any(
            row[k] != value
            for k, value in {
                "gamma": gamma,
                "group": group,
                "roi_factor": 1.0,
                "point_source": "hrnet" if group == "point" else "none",
                "box_source": "geometry",
                "is_primary": row["condition_id"] == PRIMARY,
            }.items()
        ):
            raise Phase1Error("Automatic box condition changed from approved design.")
    return {r["condition_id"]: r for r in conditions}


def box_statistics(
    box: list[float] | None, reference: np.ndarray, predicted: np.ndarray | None
) -> Row:
    """Evaluate half-open boxes against original-grid pixel centers; never clip masks."""
    if box is None:
        return dict.fromkeys(
            (
                "reference_pixels_in_box",
                "reference_pixels_outside_box",
                "reference_box_coverage_fraction",
                "reference_fully_in_box",
                "box_area_px2",
                "box_reference_area_ratio",
                "prediction_pixels_outside_box",
                "prediction_outside_box_fraction",
            )
        )
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise Phase1Error("Invalid box coordinates.")
    x0, y0, x1, y1 = box
    height, width = reference.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise Phase1Error("Box must be nonempty and within the original image.")
    xs, ys = np.arange(width) + 0.5, np.arange(height) + 0.5
    inside = ((ys >= y0) & (ys < y1))[:, None] & ((xs >= x0) & (xs < x1))[None, :]
    reference_area = int(reference.sum())
    if reference_area == 0:
        raise Phase1Error("Reference eye masks must be nonempty.")
    included = int(reference[inside].sum())
    area = (x1 - x0) * (y1 - y0)
    outside = int(predicted[~inside].sum()) if predicted is not None else None
    predicted_area = int(predicted.sum()) if predicted is not None else 0
    return {
        "reference_pixels_in_box": included,
        "reference_pixels_outside_box": reference_area - included,
        "reference_box_coverage_fraction": included / reference_area,
        "reference_fully_in_box": included == reference_area,
        "box_area_px2": area,
        "box_reference_area_ratio": area / reference_area,
        "prediction_pixels_outside_box": outside,
        "prediction_outside_box_fraction": outside / predicted_area if predicted_area else None,
    }


def eye_row(out: Path, prediction: Row, ref: Row, gt: np.ndarray, opposite: np.ndarray) -> tuple:
    """Distinguish the geometry point from the optional SAM positive point."""
    geometry = prediction["hrnet_point_xy"]
    positive = prediction["point_source"] == "hrnet"
    if (
        prediction["human_assisted"] is not False
        or (positive and (prediction["point_xy"] != geometry or prediction["point_labels"] != [1]))
        or (
            not positive
            and any(
                prediction.get(k) is not None
                for k in ("point_xy", "point_labels", "point_coords_in_roi")
            )
        )
    ):
        raise Phase1Error("Positive prompt policy differs from its condition.")
    box = prediction["box_xyxy"]
    if box is None and prediction["status"] != "failed":
        raise Phase1Error("Successful inference requires its automatic box.")
    normalized = prediction | {
        "point_xy": geometry,
        "point_score": prediction["hrnet_score"],
        "point_outside_prompt_box": bool(
            box and not (box[0] <= geometry[0] < box[2] and box[1] <= geometry[1] < box[3])
        ),
    }
    row, candidates = shared.eye_row(out, normalized, ref, gt, opposite)
    for original, renamed in GEOMETRY_FIELDS.items():
        row[renamed] = row.pop(original)
    row["geometry_point_outside_box"] = row.pop("point_outside_prompt_box")
    predicted = (
        shared.binary(shared.safe_artifact(out, prediction["mask_path"]))
        if prediction["status"] == "passed"
        else None
    )
    row.update(
        gamma=prediction["gamma"],
        is_primary=prediction["is_primary"],
        had_positive_prompt=positive,
        prompt_point_x=geometry[0] if positive else None,
        prompt_point_y=geometry[1] if positive else None,
        box_side_length_px=prediction["box_side_length_px"],
        box_clipped=prediction["box_clipped"],
        **{f"box_{k}": v for k, v in zip(("x0", "y0", "x1", "y1"), box or [None] * 4, strict=True)},
        **box_statistics(box, gt, predicted),
    )
    for candidate in candidates:
        candidate.update(gamma=prediction["gamma"], is_primary=prediction["is_primary"])
    return row, candidates


def summarize(eyes: list[Row], distances: list[Row], descriptor: Row) -> Row:
    normalized = [
        r
        | {old: r[new] for old, new in GEOMETRY_FIELDS.items()}
        | {"point_outside_prompt_box": r["geometry_point_outside_box"]}
        for r in eyes
    ]
    result = shared.summarize(normalized, distances, descriptor)
    for old, new in {
        "points_in_target_mask": "geometry_points_in_target_mask",
        "points_in_opposite_mask": "geometry_points_in_opposite_mask",
        "mean_point_offset_from_mask_centroid_px": (
            "mean_geometry_point_offset_from_mask_centroid_px"
        ),
        "point_box_conflicts": "geometry_points_outside_box",
    }.items():
        result[new] = result.pop(old)
    result.update(
        gamma=descriptor["gamma"],
        is_primary=descriptor["is_primary"],
        had_positive_prompt=descriptor["point_source"] == "hrnet",
        boxes_defined=sum(r["box_area_px2"] is not None for r in eyes),
        boxes_clipped=sum(r["box_clipped"] for r in eyes),
        references_fully_inside_box=sum(r["reference_fully_in_box"] is True for r in eyes),
        mean_reference_box_coverage_fraction=shared.finite_mean(
            [r["reference_box_coverage_fraction"] for r in eyes]
        ),
        mean_box_reference_area_ratio=shared.finite_mean(
            [r["box_reference_area_ratio"] for r in eyes]
        ),
        prediction_outside_box_defined=sum(
            r["prediction_outside_box_fraction"] is not None for r in eyes
        ),
        mean_prediction_outside_box_fraction=shared.finite_mean(
            [r["prediction_outside_box_fraction"] for r in eyes]
        ),
    )
    return result


def comparisons(conditions: list[Row]) -> list[Row]:
    triples = [
        ("baseline", c["condition_id"], baseline)
        for c in conditions
        for baseline in ("A_r100", "A_r050")
    ]
    triples += [
        ("remove_point", f"auto_box_k{g:03d}_only", f"auto_box_k{g:03d}_point")
        for g in (30, 50, 70)
    ]
    triples += [
        ("box_scale", f"auto_box_k{g:03d}_{mode}", f"auto_box_k050_{mode}")
        for mode in ("point", "only")
        for g in (30, 70)
    ]
    return [
        {
            "contrast_id": f"{target}_minus_{baseline}",
            "contrast_type": kind,
            "target_condition": target,
            "baseline_condition": baseline,
        }
        for kind, target, baseline in triples
    ]


def paired(
    conditions: list[Row], eyes: dict[str, list[Row]], distances: dict[str, list[Row]]
) -> tuple:
    eye_rows, distance_rows, summaries = [], [], []
    for descriptor in comparisons(conditions):
        target_id, base_id = descriptor["target_condition"], descriptor["baseline_condition"]
        base_eyes = {shared.key(r): r for r in eyes[base_id]}
        base_distances = {shared.distance_key(r): r for r in distances[base_id]}
        if set(base_eyes) != {shared.key(r) for r in eyes[target_id]} or set(base_distances) != {
            shared.distance_key(r) for r in distances[target_id]
        }:
            raise Phase1Error("Paired comparison identity sets differ.")
        local_eyes, local_distances = [], []
        for target in eyes[target_id]:
            base = base_eyes[shared.key(target)]
            row = descriptor | {
                **{k: target[k] for k in shared.IDENTITY},
                "case_id": target["case_id"],
                "animal_category": target["animal_category"],
                "baseline_prediction_status": base["prediction_status"],
                "target_prediction_status": target["prediction_status"],
                "new_zero_overlap": bool((base["iou"] or 0) > 0 and (target["iou"] or 0) == 0),
            }
            for field in ("iou", "dice", "centroid_error_px", "area_ratio", "candidate_iou_gap"):
                row.update(
                    {
                        f"baseline_{field}": base[field],
                        f"target_{field}": target[field],
                        f"delta_{field}": shared.difference(target[field], base[field]),
                    }
                )
            for field in ("iou", "dice"):
                row[f"delta_{field}_failure_as_zero"] = (target[field] or 0) - (base[field] or 0)
            local_eyes.append(row)
        for target in distances[target_id]:
            base = base_distances[shared.distance_key(target)]
            row = descriptor | {
                k: target[k]
                for k in ("image_id", "measurement_type", "annotation_id_a", "annotation_id_b")
            }
            for field in ("predicted_distance_px", "signed_error_px", "absolute_error_px"):
                row.update(
                    {
                        f"baseline_{field}": base[field],
                        f"target_{field}": target[field],
                        f"delta_{field}": shared.difference(target[field], base[field]),
                    }
                )
            local_distances.append(row)
        summary = descriptor | {
            "eyes_expected": len(local_eyes),
            "iou_improved": sum(r["delta_iou_failure_as_zero"] > 0 for r in local_eyes),
            "iou_worsened": sum(r["delta_iou_failure_as_zero"] < 0 for r in local_eyes),
            "iou_unchanged": sum(r["delta_iou_failure_as_zero"] == 0 for r in local_eyes),
            "new_zero_overlap": sum(r["new_zero_overlap"] for r in local_eyes),
            "mean_delta_iou_all_targets_failure_as_zero": mean(
                r["delta_iou_failure_as_zero"] for r in local_eyes
            ),
            "mean_delta_dice_all_targets_failure_as_zero": mean(
                r["delta_dice_failure_as_zero"] for r in local_eyes
            ),
            "centroid_pairs_defined": sum(
                r["delta_centroid_error_px"] is not None for r in local_eyes
            ),
            "mean_delta_centroid_error_px": shared.finite_mean(
                [r["delta_centroid_error_px"] for r in local_eyes]
            ),
        }
        for kind in ("binocular", "right_eye_pair"):
            deltas = [
                r["delta_absolute_error_px"]
                for r in local_distances
                if r["measurement_type"] == kind
            ]
            summary[f"{kind}_pairs_expected"] = len(deltas)
            summary[f"{kind}_pairs_defined"] = sum(v is not None for v in deltas)
            summary[f"{kind}_mean_delta_absolute_error_px"] = shared.finite_mean(deltas)
        eye_rows.extend(local_eyes)
        distance_rows.extend(local_distances)
        summaries.append(summary)
    return eye_rows, distance_rows, summaries


def promotion_checks(primary: Row, baseline: Row, paired_eyes: list[Row]) -> list[Row]:
    """Predeclared development gates; engineering verification is reported separately."""
    if primary["condition_id"] != PRIMARY or baseline["condition_id"] != "A_r100":
        raise Phase1Error("Promotion gates require the predeclared primary and A_r100.")
    regressions = sum(
        r["new_zero_overlap"]
        for r in paired_eyes
        if r["target_condition"] == PRIMARY and r["baseline_condition"] == "A_r100"
    )
    primary_values = {
        "no_inference_failures": primary["inference_failed"],
        "no_empty_masks": primary["empty_masks"],
        "all_eye_centroids_defined": primary["centroid_defined"],
        "all_binocular_distances_defined": primary["binocular_defined"],
        "all_right_eye_distances_defined": primary["right_eye_pair_defined"],
        "no_new_zero_overlap": regressions,
        "mean_iou_improved": primary["mean_iou_all_targets_failure_as_zero"],
        "mean_centroid_error_not_increased": primary["mean_centroid_error_px"],
        "binocular_mae_improved": primary["binocular_mae_px"],
        "right_eye_mae_improved": primary["right_eye_pair_mae_px"],
    }
    targets = {
        "no_inference_failures": ("==", 0),
        "no_empty_masks": ("==", 0),
        "all_eye_centroids_defined": ("==", primary["eyes_expected"]),
        "all_binocular_distances_defined": ("==", primary["binocular_expected"]),
        "all_right_eye_distances_defined": ("==", primary["right_eye_pair_expected"]),
        "no_new_zero_overlap": ("==", 0),
        "mean_iou_improved": (">", baseline["mean_iou_all_targets_failure_as_zero"]),
        "mean_centroid_error_not_increased": ("<=", baseline["mean_centroid_error_px"]),
        "binocular_mae_improved": ("<", baseline["binocular_mae_px"]),
        "right_eye_mae_improved": ("<", baseline["right_eye_pair_mae_px"]),
    }
    rows = []
    for check, value in primary_values.items():
        operator, target = targets[check]
        passed = False
        if value is not None and target is not None:
            passed = {
                "==": value == target,
                ">": value > target,
                "<=": value <= target,
                "<": value < target,
            }[operator]
        rows.append(
            {
                "condition_id": PRIMARY,
                "baseline_condition": "A_r100",
                "check_id": check,
                "value": value,
                "operator": operator,
                "threshold": target,
                "passed": passed,
                "scope": "development_research_gate_not_deployment",
            }
        )
    rows.append(
        {
            "condition_id": PRIMARY,
            "baseline_condition": "A_r100",
            "check_id": "all_research_gates_passed",
            "value": sum(r["passed"] for r in rows),
            "operator": "==",
            "threshold": len(rows),
            "passed": all(r["passed"] for r in rows),
            "scope": "engineering_verification_also_required",
        }
    )
    return rows


def compute(out: Path) -> dict[str, list[Row]]:
    """Recompute without writing or selecting any prediction using reference masks."""
    conditions = shared.read(out / "inputs/conditions.json")
    descriptors = validate_conditions(conditions)
    reference = out / "evaluation_inputs/reference"
    verify(reference)
    refs, ref_distances = derive(reference / "frozen")
    ref_index = {shared.key(r): r for r in refs}
    if len(ref_index) != len(refs):
        raise Phase1Error("Duplicate reference identities.")
    masks = {k: shared.binary(reference / r["mask_path"]) for k, r in ref_index.items()}
    baseline = shared.read(out / "evaluation_inputs/baseline/evaluation_values.json")
    base_eyes = {
        cid: [r for r in baseline["eye_metrics.csv"] if r["condition_id"] == cid]
        for cid in BASELINES
    }
    base_distances = {
        cid: [r for r in baseline["measurement_comparison.csv"] if r["condition_id"] == cid]
        for cid in BASELINES
    }
    base_summaries = [
        next(r for r in baseline["condition_summary.csv"] if r["condition_id"] == cid)
        for cid in BASELINES
    ]
    base_a1 = {shared.key(r): r for r in base_eyes["A_r100"]}
    if set(base_a1) != set(ref_index):
        raise Phase1Error("A_r100 identities differ from the frozen reference.")
    matches = {}
    with (out / "evaluation_inputs/animal_matches.csv").open() as handle:
        for row in csv.DictReader(handle):
            if (
                row["match_status"] != "TP"
                or float(row["confidence_threshold"]) != 0.5
                or float(row["iou_threshold"]) != 0.5
            ):
                continue
            identity = (int(row["image_id"]), row["prediction_id"])
            if identity in matches:
                raise Phase1Error("Duplicate frozen animal mapping.")
            matches[identity] = int(row["annotation_id"])
    predictions = defaultdict(dict)
    for prediction in shared.read(out / "predictions.json"):
        cid = prediction["condition_id"]
        if cid not in descriptors or any(prediction[k] != descriptors[cid][k] for k in DESCRIPTOR):
            raise Phase1Error("Prediction descriptor differs from approved condition.")
        mapping = (prediction["image_id"], prediction["prediction_id"])
        if mapping not in matches:
            raise Phase1Error("No frozen identity match for prediction.")
        normalized = prediction | {"annotation_id": matches[mapping]}
        identity = shared.key(normalized)
        if identity not in ref_index or identity in predictions[cid]:
            raise Phase1Error("Duplicate or unexpected prediction identity.")
        old = base_a1[identity]
        if (
            prediction["roi_xyxy"] != [old[k] for k in ("roi_x0", "roi_y0", "roi_x1", "roi_y1")]
            or prediction["hrnet_point_xy"] != [old["point_x"], old["point_y"]]
            or prediction["hrnet_score"] != old["point_score"]
        ):
            raise Phase1Error("Frozen A_r100 ROI or HRNet localization changed.")
        if (old["reference_cx_px"], old["reference_cy_px"], old["reference_area_px"]) != (
            ref_index[identity]["centroid_x_px"],
            ref_index[identity]["centroid_y_px"],
            int(masks[identity].sum()),
        ):
            raise Phase1Error("Baseline and new evaluation references differ.")
        normalized["roi_clipped_to_image"] = old["roi_clipped_to_image"]
        predictions[cid][identity] = normalized
    if set(predictions) != set(descriptors) or any(
        set(rows) != set(ref_index) for rows in predictions.values()
    ):
        raise Phase1Error("Every condition must retain every reference eye, including failures.")
    eyes_by_condition, distances_by_condition = dict(base_eyes), dict(base_distances)
    all_eyes, all_candidates, all_distances, summaries, image_summaries = [], [], [], [], []
    localization = {}
    for condition in conditions:
        cid = condition["condition_id"]
        eyes = []
        for identity in sorted(ref_index):
            opposite = (*identity[:2], "right" if identity[2] == "left" else "left")
            row, candidates = eye_row(
                out,
                predictions[cid][identity],
                ref_index[identity],
                masks[identity],
                masks[opposite],
            )
            eyes.append(row)
            all_candidates.extend(candidates)
            loc = {
                k: row[k]
                for k in shared.IDENTITY
                + ("prediction_id", "case_id", "animal_category", *GEOMETRY_FIELDS.values())
            }
            if identity in localization and loc != localization[identity]:
                raise Phase1Error("Geometry localization changed between conditions.")
            localization[identity] = loc
        distances = shared.measurements(eyes, ref_distances, condition)
        for row in distances:
            row.update(gamma=condition["gamma"], is_primary=condition["is_primary"])
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
    paired_eyes, paired_distances, contrasts = paired(
        conditions, eyes_by_condition, distances_by_condition
    )
    return {
        "eye_metrics.csv": all_eyes,
        "candidate_metrics.csv": all_candidates,
        "measurement_comparison.csv": all_distances,
        "condition_summary.csv": summaries,
        "image_summary.csv": image_summaries,
        "localization_metrics.csv": [localization[k] for k in sorted(localization)],
        "baseline_summary.csv": base_summaries,
        "paired_eye_deltas.csv": paired_eyes,
        "paired_distance_deltas.csv": paired_distances,
        "contrast_summary.csv": contrasts,
        "promotion_checks.csv": promotion_checks(
            next(r for r in summaries if r["condition_id"] == PRIMARY),
            next(r for r in base_summaries if r["condition_id"] == "A_r100"),
            paired_eyes,
        ),
    }
