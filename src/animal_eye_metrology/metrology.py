"""Geometry for original-image binary eye masks; coordinates are pixel centers."""

import math
from itertools import combinations
from typing import Any

import numpy as np

from animal_eye_metrology.errors import Phase1Error

Point = tuple[float, float]


def mask_overlap(first: np.ndarray, second: np.ndarray) -> dict[str, int | float | None]:
    """Binary pixel overlap; two empty masks have undefined IoU/Dice, not perfect agreement."""
    arrays = [np.asarray(first), np.asarray(second)]
    for array in arrays:
        if (
            array.ndim != 2
            or not all(array.shape)
            or array.dtype.kind not in "buif"
            or not np.isfinite(array).all()
            or not np.isin(array, [0, 1]).all()
        ):
            raise Phase1Error("Overlap requires nonzero-sized 2D binary 0/1 masks.")
    if arrays[0].shape != arrays[1].shape:
        raise Phase1Error("Overlap masks must have the same original-image dimensions.")
    a, b = [array.astype(bool) for array in arrays]
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    total = int(np.count_nonzero(a) + np.count_nonzero(b))
    return {
        "intersection_px": intersection,
        "union_px": union,
        "iou": intersection / union if union else None,
        "dice": 2 * intersection / total if total else None,
    }


def mask_centroid(mask: np.ndarray) -> Point | None:
    """Return (x,y) in pixels, or None for an empty foreground; never intensity-weight."""
    array = np.asarray(mask)
    if (
        array.ndim != 2
        or not all(array.shape)
        or array.dtype.kind not in "buif"
        or not np.isfinite(array).all()
        or not np.isin(array, [0, 1]).all()
    ):
        raise Phase1Error("Centroid requires a nonzero-sized 2D binary 0/1 mask.")
    rows, columns = np.nonzero(array)
    if not len(rows):
        return None
    return float(columns.mean(dtype=np.float64) + 0.5), float(rows.mean(dtype=np.float64) + 0.5)


def euclidean_distance(first: Point | None, second: Point | None) -> float | None:
    """Undefined inputs produce undefined distance, distinct from coincident points."""
    for point in (first, second):
        if point is not None and (len(point) != 2 or not all(math.isfinite(x) for x in point)):
            raise Phase1Error("Distance endpoints must be finite x,y coordinates.")
    if first is None or second is None:
        return None
    return math.hypot(first[0] - second[0], first[1] - second[1])


def build_measurements(centers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same animal L/R and unordered same-image pairs of different animals' R eyes."""
    index: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in centers:
        key = int(row["image_id"]), int(row["annotation_id"]), row["eye_side"]
        if key[2] not in ("left", "right") or key in index:
            raise Phase1Error("Duplicate eye identity or unsupported anatomical side.")
        if (row["centroid_x_px"] is None) != (row["centroid_y_px"] is None):
            raise Phase1Error(
                "A centroid must contain both coordinates, or both must be undefined."
            )
        index[key] = row
    images = sorted({key[0] for key in index})
    measurements = []

    def append(kind: str, first: dict[str, Any], second: dict[str, Any]) -> None:
        a = (
            None
            if first["centroid_x_px"] is None
            else (first["centroid_x_px"], first["centroid_y_px"])
        )
        b = (
            None
            if second["centroid_x_px"] is None
            else (second["centroid_x_px"], second["centroid_y_px"])
        )
        distance = euclidean_distance(a, b)
        result = {"image_id": first["image_id"], "measurement_type": kind}
        for suffix, source in (("a", first), ("b", second)):
            for field in ("annotation_id", "eye_side", "annotation_version"):
                result[f"{field}_{suffix}"] = source[field]
            result[f"cx_{suffix}_px"] = source["centroid_x_px"]
            result[f"cy_{suffix}_px"] = source["centroid_y_px"]
        result.update(
            distance_px=distance,
            measurement_status="Valid" if distance is not None else "Undefined",
            undefined_reason="" if distance is not None else "empty_mask",
        )
        measurements.append(result)

    for image_id in images:
        animals = sorted({key[1] for key in index if key[0] == image_id})
        for annotation_id in animals:
            if any((image_id, annotation_id, side) not in index for side in ("left", "right")):
                raise Phase1Error(
                    "Both left/right records are required; missing eyes are not zero."
                )
            append(
                "binocular",
                index[(image_id, annotation_id, "left")],
                index[(image_id, annotation_id, "right")],
            )
        for first, second in combinations(animals, 2):
            append(
                "right_eye_pair",
                index[(image_id, first, "right")],
                index[(image_id, second, "right")],
            )
    return measurements
