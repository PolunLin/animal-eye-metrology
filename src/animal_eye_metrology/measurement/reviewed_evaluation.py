"""Read-only scoring against reviewed v2 references, without model or prompt changes."""

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.measurement.metrology import (
    Point,
    euclidean_distance,
    mask_centroid,
    mask_overlap,
)

Row = dict[str, Any]


def binary_mask(path: Path) -> np.ndarray:
    """Load an original-grid binary PNG; do not threshold arbitrary image intensities."""
    with Image.open(path) as opened:
        pixels = np.array(opened)
    if pixels.ndim != 2 or not np.isin(pixels, [0, 1, 255]).all():
        raise Phase1Error(f"Expected an original-grid binary mask: {path}")
    return pixels > 0


def _box_iou(first: list[float], second: list[float]) -> float:
    for box in (first, second):
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise Phase1Error("Animal matching requires finite xyxy boxes.")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise Phase1Error("Animal matching requires nonempty xyxy boxes.")
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1])
    )
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (first, second)]
    return overlap / (sum(areas) - overlap)


def match_mask_arrays(
    predictions: list[Row], targets: list[Row]
) -> tuple[dict[str, Row], list[Row]]:
    """Apply the existing preannotation identity rule to all competing animal instances.

    Each record supplies category, bbox_xyxy and mask. Predictions additionally
    supply prediction_id; targets supply annotation_id. Eye geometry is never
    consulted. The caller must include every target animal on the image, even
    when its eye references are excluded from numerical evaluation.
    """
    if len({str(r["annotation_id"]) for r in targets}) != len(targets) or len(
        {r["prediction_id"] for r in predictions}
    ) != len(predictions):
        raise Phase1Error("Duplicate animal identity in matching inputs.")
    matrix = np.zeros((len(predictions), len(targets)), dtype=float)
    boxes = matrix.copy()
    for i, prediction in enumerate(predictions):
        for j, target in enumerate(targets):
            # Validate the original grid even when categories differ.
            overlap = mask_overlap(prediction["mask"], target["mask"])
            box_overlap = _box_iou(prediction["bbox_xyxy"], target["bbox_xyxy"])
            if prediction["category"] == target["category"]:
                matrix[i, j] = overlap["iou"] or 0.0
                boxes[i, j] = box_overlap
    matches, audit = {}, []
    for j, target in enumerate(targets):
        row: Row = {"annotation_id": str(target["annotation_id"]), "matched": False}
        if not predictions:
            row["reason"] = "no_accepted_animal_masks"
        else:
            i = int(matrix[:, j].argmax())
            target_margin = matrix[i, j] - max(np.delete(matrix[:, j], i), default=0.0)
            prediction_margin = matrix[i, j] - max(np.delete(matrix[i], j), default=0.0)
            ok = bool(
                matrix[i, j] >= 0.5
                and boxes[i, j] >= 0.5
                and int(matrix[i].argmax()) == j
                and target_margin >= 0.1
                and prediction_margin >= 0.1
            )
            row.update(
                prediction_id=predictions[i]["prediction_id"],
                mask_iou=float(matrix[i, j]),
                bbox_iou=float(boxes[i, j]),
                target_margin=float(target_margin),
                prediction_margin=float(prediction_margin),
                matched=ok,
                reason="" if ok else "missing_or_ambiguous_animal_match",
            )
            if ok:
                matches[row["annotation_id"]] = row
        audit.append(row)
    return matches, audit


def match_animals(result: Row, run: Path, targets: list[Row], dataset: Any) -> tuple[dict, list]:
    """Read saved inference/COCO animal masks using the preannotation matching rule."""
    from pycocotools import mask as mask_utils

    from animal_eye_metrology.ui.annotation_overlay import _decode_mask

    predictions = []
    root = run.resolve()
    for animal in result["animals"]:
        if not animal["accepted"] or animal["mask_status"] != "completed":
            continue
        path = (root / animal["mask_path"]).resolve()
        if Path(animal["mask_path"]).is_absolute() or not path.is_relative_to(root):
            raise Phase1Error("Animal mask must remain inside its inference run.")
        predictions.append({**animal, "mask": binary_mask(path)})
    references = []
    for target in targets:
        annotation = dataset.annotations[int(target["annotation_id"])]
        image = dataset.images[int(target["image_id"])]
        x, y, width, height = json.loads(target["coco_bbox_xywh"])
        references.append(
            {
                "annotation_id": target["annotation_id"],
                "category": target["animal_category"],
                "bbox_xyxy": [x, y, x + width, y + height],
                "mask": _decode_mask(annotation, image, mask_utils),
            }
        )
    return match_mask_arrays(predictions, references)


def score_eye(reference: np.ndarray, predicted: np.ndarray | None, failure_reason: str = "") -> Row:
    """Score one eligible reference; failures stay in the IoU/Dice denominator."""
    reference_center = mask_centroid(reference)
    if reference_center is None:
        raise Phase1Error("An eligible eye reference must have a nonempty mask.")
    predicted_center = None
    if predicted is None:
        status = "inference_failed"
        overlap: Row = {"iou": 0.0, "dice": 0.0}
        reason = failure_reason or "missing_prediction"
    else:
        overlap = mask_overlap(predicted, reference)
        predicted_center = mask_centroid(predicted)
        status = "nonempty" if predicted_center is not None else "empty"
        reason = "" if predicted_center is not None else (failure_reason or "empty_mask")
    return {
        "prediction_status": status,
        "failure_reason": reason,
        "reference_area_px": int(np.count_nonzero(reference)),
        "predicted_area_px": int(np.count_nonzero(predicted)) if predicted is not None else None,
        "iou": overlap["iou"],
        "dice": overlap["dice"],
        "reference_cx_px": reference_center[0],
        "reference_cy_px": reference_center[1],
        "predicted_cx_px": predicted_center[0] if predicted_center is not None else None,
        "predicted_cy_px": predicted_center[1] if predicted_center is not None else None,
        "centroid_error_px": euclidean_distance(predicted_center, reference_center),
    }


def score_distance(
    reference_a: Point,
    reference_b: Point,
    predicted_a: Point | None,
    predicted_b: Point | None,
    failure_reason: str = "",
) -> Row:
    """Evaluate a qualified reference pair; missing predictions are never zero-filled."""
    reference = euclidean_distance(reference_a, reference_b)
    if reference is None:
        raise Phase1Error("A distance reference requires two eligible endpoints.")
    predicted = euclidean_distance(predicted_a, predicted_b)
    error = predicted - reference if predicted is not None else None
    return {
        "reference_distance_px": reference,
        "predicted_distance_px": predicted,
        "signed_error_px": error,
        "absolute_error_px": abs(error) if error is not None else None,
        "measurement_status": "defined" if predicted is not None else "unavailable",
        "failure_reason": "" if predicted is not None else (failure_reason or "missing_endpoint"),
    }


def summarize(eyes: list[Row], measurements: list[Row]) -> Row:
    """Report micro averages plus explicit denominators, without independence claims."""
    statuses = Counter(row["prediction_status"] for row in eyes)
    if statuses.keys() - {"inference_failed", "empty", "nonempty"}:
        raise Phase1Error("Unknown prediction status in evaluated eyes.")
    if {r["measurement_type"] for r in measurements} - {"binocular", "right_eye_pair"}:
        raise Phase1Error("Unknown evaluated distance type.")
    errors = [r["centroid_error_px"] for r in eyes if r["centroid_error_px"] is not None]
    result: Row = {
        "eyes_expected": len(eyes),
        "inference_failed": statuses["inference_failed"],
        "empty_masks": statuses["empty"],
        "nonempty_masks": statuses["nonempty"],
        "centroid_defined": len(errors),
        "mean_iou_all_targets_failure_as_zero": mean(r["iou"] for r in eyes) if eyes else None,
        "mean_dice_all_targets_failure_as_zero": mean(r["dice"] for r in eyes) if eyes else None,
        "mean_centroid_error_px": mean(errors) if errors else None,
        "max_centroid_error_px": max(errors) if errors else None,
    }
    for kind in ("binocular", "right_eye_pair"):
        rows = [r for r in measurements if r["measurement_type"] == kind]
        errors = [r["absolute_error_px"] for r in rows if r["absolute_error_px"] is not None]
        result.update(
            {
                f"{kind}_expected": len(rows),
                f"{kind}_defined": len(errors),
                f"{kind}_failed": len(rows) - len(errors),
                f"{kind}_mae_px": mean(errors) if errors else None,
                f"{kind}_rmse_px": math.sqrt(mean(e * e for e in errors)) if errors else None,
                f"{kind}_max_error_px": max(errors) if errors else None,
            }
        )
    return result
