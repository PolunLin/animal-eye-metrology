"""Shared predicted-eye crop geometry for inference and frozen research runs.

The formulas retain original-image coordinates, anatomical channel order,
whole-rule ROI scaling, and the original floor/ceil boundary policy.
"""

import math
from typing import Any

import numpy as np

from animal_eye_metrology.core.errors import Phase1Error

ROI_FACTORS = (0.5, 1.0, 1.5)


def eye_crops(
    pose: dict[str, Any], width: int, height: int, minimum: int = 32, factor: float = 2.0
) -> tuple[list[list[int]] | None, str]:
    """Use only predicted points/scores and native crop geometry; no GT fallback."""
    if pose["status"] != "passed":
        return None, "pose_inference_failed"
    points = np.asarray(pose["keypoints_xy"][:2], dtype=float)
    scores = np.asarray(pose["scores"][:2], dtype=float)
    if (
        points.shape != (2, 2)
        or scores.shape != (2,)
        or not (np.isfinite(points).all() and np.isfinite(scores).all())
    ):
        return None, "nonfinite_or_invalid_eye_output"
    if np.any(scores <= 0):
        return None, "nonpositive_heatmap_score"
    center, scale = (np.asarray(pose["geometry"][k]) for k in ("input_center", "input_scale"))
    if np.any(points < 0) or np.any(points >= [width, height]):
        return None, "eye_outside_image"
    if np.any(points < center - scale / 2) or np.any(points > center + scale / 2):
        return None, "eye_outside_native_animal_crop"
    separation = float(np.linalg.norm(points[0] - points[1]))
    if separation <= 0:
        return None, "coincident_eye_points"
    size = max(minimum, factor * separation)
    crops = [
        [
            max(0, math.floor(x - size / 2)),
            max(0, math.floor(y - size / 2)),
            min(width, math.ceil(x + size / 2)),
            min(height, math.ceil(y + size / 2)),
        ]
        for x, y in points
    ]
    return crops, ""


def roi_for_case(case: dict[str, Any], factor: float) -> tuple[list[int], bool]:
    """Scale the entire original ROI rule, including its minimum, about the HRNet point."""
    if factor not in ROI_FACTORS:
        raise Phase1Error("Unapproved ROI factor.")
    x, y = case["hrnet_point_xy"]
    length = factor * max(32.0, 2.0 * case["hrnet_eye_distance_px"])
    raw = [
        math.floor(x - length / 2),
        math.floor(y - length / 2),
        math.ceil(x + length / 2),
        math.ceil(y + length / 2),
    ]
    roi = [
        max(0, raw[0]),
        max(0, raw[1]),
        min(case["image_width"], raw[2]),
        min(case["image_height"], raw[3]),
    ]
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise Phase1Error("Invalid deterministic crop.")
    return roi, roi != raw
