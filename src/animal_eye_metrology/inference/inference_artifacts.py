"""Candidate geometry and review artifacts for GT-free image inference."""

import csv
import io
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.measurement.metrology import euclidean_distance

SIDES = ("left", "right")
COLORS = {"left": (35, 215, 245), "right": (245, 85, 110)}
ANIMAL_COLORS = (
    (40, 220, 245),
    (255, 155, 45),
    (170, 120, 255),
    (90, 235, 135),
    (255, 100, 175),
    (245, 225, 70),
)


def _centroid(row: dict[str, Any]) -> tuple[float, float] | None:
    x, y = row.get("centroid_x_px"), row.get("centroid_y_px")
    if (x is None) != (y is None):
        raise Phase1Error("A centroid requires both coordinates or two null values.")
    if x is None:
        return None
    if not all(isinstance(value, int | float) and math.isfinite(value) for value in (x, y)):
        raise Phase1Error("A centroid must contain finite numeric coordinates.")
    if row.get("segmentation_status") != "completed":
        return None
    return float(x), float(y)


def build_measurements(eyes: list[dict[str, Any]], conditions: list[str]) -> list[dict[str, Any]]:
    """Preserve unavailable endpoints; finite distances remain unverified candidates.

    The caller provides two named eye slots per accepted animal and condition,
    including unsupported animals and failed model outputs. No ground truth,
    visibility inference, coordinate swapping, or zero filling occurs here.
    """
    if len(set(conditions)) != len(conditions):
        raise Phase1Error("Measurement conditions must be unique.")
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in eyes:
        key = row["condition_id"], row["prediction_id"], row["eye_side"]
        if key[0] not in conditions or key[2] not in SIDES or key in index:
            raise Phase1Error("Duplicate eye identity or unsupported condition/side.")
        _centroid(row)
        index[key] = row
    animals = sorted({row["prediction_id"] for row in eyes})
    expected = {
        (condition, animal, side)
        for condition in conditions
        for animal in animals
        for side in SIDES
    }
    if set(index) != expected:
        raise Phase1Error("Both eye slots are required for every animal in every condition.")
    measurements: list[dict[str, Any]] = []

    def append(kind: str, first: dict[str, Any], second: dict[str, Any]) -> None:
        a, b = _centroid(first), _centroid(second)
        distance = euclidean_distance(a, b)
        row: dict[str, Any] = {
            "condition_id": first["condition_id"],
            "measurement_type": kind,
            "distance_px": distance,
            "status": "candidate" if distance is not None else "unavailable",
            "qualification": "unverified",
            "units": "px",
            "experimental": any(
                eye.get("experimental", False) or eye.get("pose_domain") == "out_of_training_domain"
                for eye in (first, second)
            ),
        }
        reasons = []
        for suffix, eye, center in (("a", first, a), ("b", second, b)):
            row[f"prediction_id_{suffix}"] = eye["prediction_id"]
            row[f"eye_side_{suffix}"] = eye["eye_side"]
            row[f"pose_domain_{suffix}"] = eye.get("pose_domain", "not_recorded")
            row[f"cx_{suffix}_px"] = center[0] if center is not None else None
            row[f"cy_{suffix}_px"] = center[1] if center is not None else None
            if center is None:
                reason = eye.get("failure_reason") or eye.get("segmentation_status")
                if reason in (None, "", "completed"):
                    reason = "missing_centroid"
                reasons.append(f"{eye['prediction_id']}:{eye['eye_side']}:{reason}")
        row["reason"] = "; ".join(reasons)
        measurements.append(row)

    for condition in conditions:
        for animal in animals:
            append(
                "binocular", index[(condition, animal, "left")], index[(condition, animal, "right")]
            )
        for first, second in combinations(animals, 2):
            append(
                "right_to_right",
                index[(condition, first, "right")],
                index[(condition, second, "right")],
            )
    return measurements


def csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> bytes:
    """UTF-8 CSV for downloads; preserve nulls as blank and nested values as JSON."""
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys([*(fieldnames or []), *(key for row in rows for key in row)]))
    if fields:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                    if isinstance(value, list | dict)
                    else value
                    for key, value in row.items()
                }
            )
    return stream.getvalue().encode("utf-8-sig")


def _asset(run_dir: Path, relative: str) -> Path:
    root = run_dir.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise Phase1Error("Inference artifact must remain inside its run directory.")
    return path


def _mask(run_dir: Path, path: str, image_size: tuple[int, int]) -> Image.Image:
    with Image.open(_asset(run_dir, path)) as source:
        values = np.asarray(source)
        if source.size != image_size or values.ndim != 2 or not np.isin(values, (0, 1, 255)).all():
            raise Phase1Error("Inference overlay requires an original-size binary mask.")
        return Image.fromarray((values > 0).astype(np.uint8) * 255)


def _coordinates(values: Any, length: int) -> tuple[float, ...] | None:
    if values is None or all(value is None for value in values):
        return None
    if len(values) != length or not all(
        isinstance(v, int | float) and math.isfinite(v) for v in values
    ):
        raise Phase1Error("Invalid inference overlay coordinates.")
    return tuple(float(value) for value in values)


def _inner_boundary(foreground: np.ndarray, width: int = 2) -> np.ndarray:
    """Display an inward band around every component and hole, without altering the mask."""
    interior = foreground.copy()
    for _ in range(width):
        padded = np.pad(interior, 1, constant_values=False)
        interior = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return foreground & ~interior


def animal_segmentation_views(
    run_dir: Path, result: dict[str, Any], show_boxes: bool = False
) -> dict[str, Image.Image]:
    """Read saved instance masks into two explicit silhouette views, without writing files.

    The overlay uses a different display color for each accepted animal and a
    two-pixel inward contour derived from its foreground pixels. Optional white
    rectangles are detection boxes, never substitutes for missing masks. The
    instances view has a black background and contains only saved foreground;
    at overlapping pixels the last rendered instance's display color wins.
    No image coordinates, mask geometry, model outputs, or measurements change.
    """
    with Image.open(_asset(run_dir, result["input"]["image_path"])) as source:
        original = source.convert("RGB")
    overlay = np.array(original)
    instances = np.zeros_like(overlay)
    font = ImageFont.load_default(size=max(10, min(18, original.width // 45)))
    labels = []
    accepted = sorted(
        (animal for animal in result.get("animals", []) if animal.get("accepted")),
        key=lambda animal: animal["prediction_id"],
    )
    for index, animal in enumerate(accepted):
        color = ANIMAL_COLORS[index % len(ANIMAL_COLORS)]
        relative = animal.get("mask_path")
        if not relative:
            continue
        foreground = np.asarray(_mask(run_dir, relative, original.size)) > 0
        if not foreground.any():
            continue
        overlay[foreground] = np.rint(
            overlay[foreground].astype(np.float64) * 0.7 + np.asarray(color) * 0.3
        ).astype(np.uint8)
        overlay[_inner_boundary(foreground)] = color
        instances[foreground] = color
        rows, columns = np.nonzero(foreground)
        labels.append((animal, color, int(columns.min()), int(rows.min()), int(rows.max())))
    canvas = Image.fromarray(overlay)
    draw = ImageDraw.Draw(canvas)
    if show_boxes:
        for animal in accepted:
            box = _coordinates(animal.get("bbox_xyxy"), 4)
            if box is not None:
                draw.rectangle(box, outline=(245, 245, 245), width=1)
                draw.text(
                    (box[0] + 2, box[1] + 2),
                    "bbox",
                    font=font,
                    fill=(255, 255, 255),
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
    for animal, color, left, top, bottom in labels:
        text = f"{animal['prediction_id']} {animal.get('category', '')}"
        bounds = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        label_width, label_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = max(0, min(left, original.width - label_width - 4))
        # Below the visible body avoids placing labels over the face in typical poses.
        y = bottom + 3 if bottom + label_height + 8 < original.height else top - label_height - 8
        y = max(0, min(y, original.height - label_height - 4))
        draw.rectangle((x, y, x + label_width + 4, y + label_height + 4), fill=(20, 25, 30))
        draw.text((x + 2 - bounds[0], y + 2 - bounds[1]), text, font=font, fill=color)
    return {"overlay": canvas, "instances": Image.fromarray(instances)}


def render_overlays(run_dir: Path, result: dict[str, Any]) -> dict[str, str]:
    """Render saved outputs only, without modifying masks or recalculating geometry.

    Overlays retain the normalized image dimensions. A point cross is the HRNet
    prompt, an outlined circle is the mask centroid, and L/R colors are the
    unverified anatomical output channels, never an observer-left assignment.
    """
    with Image.open(_asset(run_dir, result["input"]["image_path"])) as source:
        original = source.convert("RGB")
    artifacts: dict[str, str] = {}
    font = ImageFont.load_default(size=max(10, min(18, original.width // 45)))
    for condition in result["conditions"]:
        if condition not in ("A_r100", "A_r050"):
            raise Phase1Error("Unsupported inference overlay condition.")
        canvas = original.copy()
        eyes = [row for row in result["eyes"] if row["condition_id"] == condition]
        for animal in result["animals"]:
            if animal.get("accepted") and animal.get("mask_path"):
                mask = _mask(run_dir, animal["mask_path"], original.size)
                tint = Image.blend(canvas, Image.new("RGB", original.size, (255, 185, 40)), 0.12)
                canvas = Image.composite(tint, canvas, mask)
        for eye in eyes:
            if eye.get("mask_path"):
                mask = _mask(run_dir, eye["mask_path"], original.size)
                tint = Image.blend(
                    canvas, Image.new("RGB", original.size, COLORS[eye["eye_side"]]), 0.65
                )
                canvas = Image.composite(tint, canvas, mask)
        draw = ImageDraw.Draw(canvas)
        for animal in result["animals"]:
            if not animal.get("accepted"):
                continue
            box = _coordinates(animal.get("bbox_xyxy"), 4)
            if box is not None:
                draw.rectangle(box, outline=(255, 185, 40), width=1)
                name = f"{animal['prediction_id']} {animal['category']}"
                if not animal.get("supported"):
                    name += " [unsupported]"
                draw.text(
                    (box[0] + 2, box[1] + 2),
                    name,
                    font=font,
                    fill=(255, 230, 160),
                    stroke_width=1,
                    stroke_fill=(20, 25, 30),
                )
        for measurement in result.get("measurements", []):
            if measurement["condition_id"] != condition or measurement["status"] != "candidate":
                continue
            endpoints = _coordinates(
                [
                    measurement.get(f"c{axis}_{side}_px")
                    for side in ("a", "b")
                    for axis in ("x", "y")
                ],
                4,
            )
            if endpoints is not None:
                color = (
                    (255, 255, 255)
                    if measurement["measurement_type"] == "binocular"
                    else (225, 170, 255)
                )
                draw.line(endpoints, fill=color, width=1)
        for eye in eyes:
            color = COLORS[eye["eye_side"]]
            roi = _coordinates(eye.get("roi_xyxy"), 4)
            if roi is not None:
                draw.rectangle(roi, outline=(190, 195, 210), width=1)
            point = _coordinates([eye.get("point_x_px"), eye.get("point_y_px")], 2)
            if point is not None:
                x, y = point
                draw.line((x - 3, y, x + 3, y), fill=(255, 235, 40), width=1)
                draw.line((x, y - 3, x, y + 3), fill=(255, 235, 40), width=1)
            center = _centroid(eye)
            if center is not None:
                x, y = center
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=1)
        # Use a compact label; all overlays remain on the original pixel grid.
        legend = f"{condition} | candidate / unverified | L cyan / R red | + prompt, o centroid"
        draw.text(
            (3, 3), legend, font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0)
        )
        if any(eye.get("experimental") for eye in eyes):
            draw.text(
                (3, 23),
                "EXPERIMENTAL: bird eye pose outside AP-10K training domain",
                font=font,
                fill=(255, 210, 90),
                stroke_width=1,
                stroke_fill=(0, 0, 0),
            )
        relative = f"overlay_{condition}.png"
        canvas.save(run_dir / relative)
        artifacts[f"{condition}_overlay"] = relative
    if "A_r100_overlay" in artifacts:
        artifacts["overlay"] = artifacts["A_r100_overlay"]
    return artifacts
