"""Render fixed-view diagnostic grids from saved prompt/ROI ablation predictions."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.measurement_baseline import derive
from animal_eye_metrology.verification import sha256_file

BACKGROUND = (19, 26, 38)
CARD = (28, 37, 52)
TEXT = (235, 241, 247)
SUBTEXT = (181, 194, 209)
PREDICTION = (50, 205, 237)
REFERENCE = (75, 255, 120)
POINT = (255, 225, 65)
BOX = (249, 120, 225)
ROI = (190, 199, 213)
GROUPS = {
    "A": ("HRNet point", "No box"),
    "B": ("Manual point", "No box"),
    "C": ("HRNet point", "Manual box"),
    "D": ("Manual point", "Manual box"),
}
FACTORS = (0.5, 1.0, 1.5)
FOCUS = (
    (125405, 13692, "right"),
    (125405, 17207, "left"),
    (125405, 17207, "right"),
    (333069, 599888, "right"),
)


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int = 18,
    color: tuple[int, int, int] = TEXT,
) -> None:
    draw.text(xy, text, fill=color, font=font(size))


def number(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def eye_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row["image_id"]), int(row["annotation_id"]), row["eye_side"]


def metric_text(value: Any, digits: int = 3) -> str:
    numeric = number(value)
    return "undefined" if numeric is None else f"{numeric:.{digits}f}"


def viewport(records: list[dict[str, Any]], size: tuple[int, int]) -> tuple[int, ...]:
    """A shared view uses only predicted ROIs, with a fixed 8-pixel display margin."""
    boxes = [r["roi_xyxy"] for r in records if r.get("roi_xyxy")]
    if not boxes:
        raise Phase1Error("No ROI is available for the diagnostic viewport.")
    width, height = size
    return (
        max(0, min(b[0] for b in boxes) - 8),
        max(0, min(b[1] for b in boxes) - 8),
        min(width, max(b[2] for b in boxes) + 8),
        min(height, max(b[3] for b in boxes) + 8),
    )


def boundary(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    transform: Any,
) -> None:
    """Draw exact foreground pixel-edge segments, without enlarging contour thickness."""
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask), strict=True):
        for empty, start, end in (
            (y == 0 or not mask[y - 1, x], (x, y), (x + 1, y)),
            (y + 1 == height or not mask[y + 1, x], (x, y + 1), (x + 1, y + 1)),
            (x == 0 or not mask[y, x - 1], (x, y), (x, y + 1)),
            (x + 1 == width or not mask[y, x + 1], (x + 1, y), (x + 1, y + 1)),
        ):
            if empty:
                draw.line((*transform(*start), *transform(*end)), fill=REFERENCE, width=2)


def panel(
    out: Path,
    original: Image.Image,
    ground_truth: np.ndarray,
    record: dict[str, Any],
    metrics: dict[str, Any],
    view: tuple[int, ...],
) -> Image.Image:
    result = Image.new("RGB", (340, 340), CARD)
    pixels = np.array(original)
    if record["status"] == "passed":
        predicted = np.array(Image.open(out / record["mask_path"])) > 0
        if predicted.shape != pixels.shape[:2]:
            raise Phase1Error("Prediction and original image dimensions differ.")
        pixels[predicted] = (pixels[predicted] * 0.45 + np.array(PREDICTION) * 0.55).astype(
            np.uint8
        )
    image = Image.fromarray(pixels).crop(view)
    scale = min(316 / image.width, 250 / image.height)
    scaled = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.NEAREST,
    )
    xoff, yoff = (340 - scaled.width) // 2, 12 + (250 - scaled.height) // 2
    result.paste(scaled, (xoff, yoff))
    # Use exact rendered x/y scales after integer image-size rounding.
    sx, sy = scaled.width / image.width, scaled.height / image.height

    def xy(x: float, y: float) -> tuple[float, float]:
        return xoff + (x - view[0]) * sx, yoff + (y - view[1]) * sy

    draw = ImageDraw.Draw(result)
    roi = record["roi_xyxy"]
    draw.rectangle((*xy(roi[0], roi[1]), *xy(roi[2], roi[3])), outline=ROI, width=1)
    boundary(draw, ground_truth, xy)
    box = record.get("box_xyxy")
    if box:
        draw.rectangle((*xy(box[0], box[1]), *xy(box[2], box[3])), outline=BOX, width=2)
    point = record.get("point_xy")
    if point:
        x, y = xy(*point)
        draw.line((x - 6, y, x + 6, y), fill=POINT, width=2)
        draw.line((x, y - 6, x, y + 6), fill=POINT, width=2)
    label(draw, (12, 272), f"IoU {metric_text(metrics['iou'])}", 18)
    label(draw, (12, 297), f"Centroid error {metric_text(metrics['centroid_error_px'])} px", 16)
    if record["status"] != "passed":
        label(draw, (12, 320), "Inference failed", 14, (255, 135, 135))
    elif record.get("point_outside_prompt_box"):
        label(draw, (12, 320), "Point outside prompt box (retained)", 13, BOX)
    return result


def focus_grid(
    out: Path,
    destination: Path,
    key: tuple[int, int, str],
    predictions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    references: dict,
) -> dict[str, Any]:
    selected = [r for r in predictions if eye_key(r) == key]
    metrics = {r["condition_id"]: r for r in rows if eye_key(r) == key}
    if len(selected) != 12 or len(metrics) != 12:
        raise Phase1Error("Each diagnostic eye requires all twelve conditions.")
    image_id, annotation_id, side = key
    original = Image.open(out / f"inputs/images/{image_id:012d}.jpg").convert("RGB")
    view = viewport(selected, original.size)
    gt = np.array(Image.open(references[key])) > 0
    if gt.shape != (original.height, original.width):
        raise Phase1Error("Reference and original image dimensions differ.")
    grid = Image.new("RGB", (1225, 1530), BACKGROUND)
    draw = ImageDraw.Draw(grid)
    category = "dog" if image_id == 125405 else "giraffe"
    label(draw, (18, 14), f"{category} {annotation_id} / {side} eye | COCO {image_id}", 25)
    label(
        draw, (18, 51), "Cyan: selected prediction | green: reference | yellow +: prompt point", 17
    )
    label(draw, (18, 76), "Pink: prompt box | gray: ROI | common viewport in all panels", 17)
    for column, factor in enumerate(FACTORS):
        label(draw, (190 + column * 345, 112), f"ROI {factor:g} x original size", 20)
    index = {(r["group"], float(r["roi_factor"])): r for r in selected}
    for row, (group, group_labels) in enumerate(GROUPS.items()):
        y = 150 + row * 343
        label(draw, (18, y + 18), f"Group {group}", 23)
        label(draw, (18, y + 57), group_labels[0], 18)
        label(draw, (18, y + 83), group_labels[1], 18)
        for column, factor in enumerate(FACTORS):
            record = index[(group, factor)]
            card = panel(out, original, gt, record, metrics[record["condition_id"]], view)
            grid.paste(card, (185 + column * 345, y))
    filename = f"{image_id}_{annotation_id}_{side}_grid.png"
    grid.save(destination / filename)
    return {
        "file": filename,
        "eye": list(key),
        "display_viewport_xyxy": list(view),
        "viewport_uses_reference_geometry": False,
    }


def heat_color(value: float) -> tuple[int, int, int]:
    stops = np.array([(63, 25, 88), (43, 111, 142), (50, 173, 126), (244, 226, 78)])
    position = min(1.0, max(0.0, value)) * 3
    lower = min(int(position), 2)
    return tuple(
        (stops[lower] * (1 - position + lower) + stops[lower + 1] * (position - lower)).astype(int)
    )


def heatmap(destination: Path, rows: list[dict[str, Any]]) -> None:
    conditions = [f"{group}_r{int(factor * 100):03d}" for group in GROUPS for factor in FACTORS]
    identities = sorted({eye_key(r) for r in rows})
    index = {(eye_key(r), r["condition_id"]): r for r in rows}
    if len(identities) != 12 or len(index) != 144:
        raise Phase1Error("IoU heatmap requires twelve eyes and twelve conditions.")
    canvas = Image.new("RGB", (1260, 795), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    label(draw, (18, 14), "IoU for all 12 eyes and 12 prompt / ROI conditions", 26)
    label(draw, (18, 54), "Highest model-score candidate only | same 3 development images", 18)
    label(
        draw,
        (18, 84),
        "A: HRNet | B: manual point | C: HRNet + manual box | D: manual point + box",
        17,
    )
    for column, condition in enumerate(conditions):
        group, factor = condition.split("_r")
        label(draw, (286 + column * 79, 132), f"{group} / {int(factor) / 100:g}", 15)
    for row, key in enumerate(identities):
        y = 165 + row * 43
        category = {20247: "bear", 125405: "dog", 333069: "giraffe"}[key[0]]
        label(draw, (18, y + 11), f"{category} {key[1]} / {key[2]}", 18)
        for column, condition in enumerate(conditions):
            value = number(index[(key, condition)]["iou"])
            x = 280 + column * 79
            color = (90, 90, 90) if value is None else heat_color(value)
            draw.rectangle((x, y, x + 76, y + 40), fill=color)
            ink = (12, 25, 35) if value is not None and value >= 0.6 else TEXT
            label(draw, (x + 12, y + 10), "NA" if value is None else f"{value:.3f}", 17, ink)
    for offset in range(400):
        draw.line((280 + offset, 708, 280 + offset, 729), fill=heat_color(offset / 399))
    label(draw, (277, 740), "0", 17)
    label(draw, (668, 740), "1", 17)
    label(draw, (708, 711), "Columns show group / ROI factor", 16)
    label(
        draw,
        (18, 771),
        "Missing predictions remain NA here; summary metrics retain all targets.",
        15,
    )
    canvas.save(destination / "all_12_eyes_iou_heatmap.png")


def render(output: Path) -> None:
    predictions = json.loads((output / "predictions.json").read_text())
    with (output / "eye_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    references = {}
    for row in rows:
        if row.get("reference_mask_path"):
            references[eye_key(row)] = output / row["reference_mask_path"]
    if not all(key in references for key in FOCUS):
        centers, _ = derive(output / "evaluation_inputs/reference/frozen")
        references.update(
            {eye_key(r): output / "evaluation_inputs/reference" / r["mask_path"] for r in centers}
        )
    destination = output / "visualizations"
    destination.mkdir()
    views = [focus_grid(output, destination, key, predictions, rows, references) for key in FOCUS]
    heatmap(destination, rows)
    manifest = {
        "model_inference_performed": False,
        "candidate_selection": "highest model score; no GT-based selection",
        "eye_metrics_sha256": sha256_file(output / "eye_metrics.csv"),
        "predictions_sha256": sha256_file(output / "predictions.json"),
        "focus_views": views,
        "files_sha256": {p.name: sha256_file(p) for p in sorted(destination.glob("*.png"))},
    }
    (destination / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/phase3/prompt_roi_ablation_v1")
    )
    render(parser.parse_args().output)
