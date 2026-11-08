"""Render saved automatic-eye-box results and matched frozen Group A baselines."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from animal_eye_metrology.ablation_visualization import (
    BACKGROUND,
    BOX,
    CARD,
    POINT,
    PREDICTION,
    ROI,
    TEXT,
    boundary,
    eye_key,
    heat_color,
    label,
    metric_text,
    number,
    viewport,
)
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.measurement_baseline import derive
from animal_eye_metrology.verification import sha256_file

GEOMETRY = (255, 162, 81)
BASELINES = ("A_r100", "A_r050")
NEW_CONDITIONS = tuple(
    f"auto_box_k{gamma:03d}_{mode}" for mode in ("point", "only") for gamma in (30, 50, 70)
)
CONDITIONS = BASELINES + NEW_CONDITIONS
PANEL_ORDER = (
    "A_r100",
    "auto_box_k030_point",
    "auto_box_k050_point",
    "auto_box_k070_point",
    "A_r050",
    "auto_box_k030_only",
    "auto_box_k050_only",
    "auto_box_k070_only",
)
FOCUS = (
    (125405, 13692, "right"),
    (125405, 17207, "left"),
    (125405, 17207, "right"),
    (333069, 599888, "right"),
    (20247, 588042, "left"),
    (20247, 587181, "right"),
)
CATEGORIES = {20247: "bear", 125405: "dog", 333069: "giraffe"}
Row = dict[str, Any]


def condition_label(condition: str) -> str:
    if condition in BASELINES:
        return "A baseline / ROI " + ("1x" if condition == "A_r100" else "0.5x")
    _, _, scale, mode = condition.split("_")
    gamma = int(scale[1:]) / 100
    return f"Gamma {gamma:g} / {'point + box' if mode == 'point' else 'box only'}"


def read_inputs(output: Path) -> tuple[list[Row], list[Row], dict]:
    """Join evaluation identities only after inference; never select predictions by GT."""
    with (output / "eye_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics = {(r["condition_id"], r["prediction_id"], r["eye_side"]): r for r in rows}
    predictions = []
    for record in json.loads((output / "predictions.json").read_text()):
        identity = (record["condition_id"], record["prediction_id"], record["eye_side"])
        if identity not in metrics:
            raise Phase1Error("Prediction is missing its evaluated eye identity.")
        predictions.append(
            record
            | {
                "annotation_id": int(metrics[identity]["annotation_id"]),
                "artifact_root": output,
            }
        )
    if len(predictions) != 72 or len(metrics) != 72:
        raise Phase1Error("Expected all 72 new eye results, including failures.")

    baseline = output / "evaluation_inputs/baseline"
    baseline_rows = json.loads((baseline / "evaluation_values.json").read_text())["eye_metrics.csv"]
    rows.extend(r for r in baseline_rows if r["condition_id"] in BASELINES)
    predictions.extend(
        r | {"artifact_root": baseline}
        for r in json.loads((baseline / "predictions.json").read_text())
        if r["condition_id"] in BASELINES
    )
    centers, _ = derive(output / "evaluation_inputs/reference/frozen")
    references = {
        eye_key(r): output / "evaluation_inputs/reference" / r["mask_path"] for r in centers
    }
    expected = {(condition, key) for condition in CONDITIONS for key in references}
    if (
        len(rows) != 96
        or len(predictions) != 96
        or {(r["condition_id"], eye_key(r)) for r in rows} != expected
        or {(r["condition_id"], eye_key(r)) for r in predictions} != expected
    ):
        raise Phase1Error("The comparison must contain 12 eyes and all eight conditions.")
    return predictions, rows, references


def panel(
    original: Image.Image, gt: np.ndarray, record: Row, metrics: Row, view: tuple
) -> Image.Image:
    result = Image.new("RGB", (360, 385), CARD)
    pixels = np.array(original)
    if record["status"] == "passed":
        predicted = np.array(Image.open(record["artifact_root"] / record["mask_path"])) > 0
        if predicted.shape != pixels.shape[:2]:
            raise Phase1Error("Prediction and original dimensions differ.")
        pixels[predicted] = (pixels[predicted] * 0.45 + np.array(PREDICTION) * 0.55).astype(
            np.uint8
        )
    cropped = Image.fromarray(pixels).crop(view)
    scale = min(336 / cropped.width, 250 / cropped.height)
    scaled = cropped.resize(
        (round(cropped.width * scale), round(cropped.height * scale)),
        Image.Resampling.NEAREST,
    )
    xoff = (360 - scaled.width) // 2
    yoff = 40 + (250 - scaled.height) // 2
    result.paste(scaled, (xoff, yoff))
    sx, sy = scaled.width / cropped.width, scaled.height / cropped.height

    def xy(x: float, y: float) -> tuple[float, float]:
        return xoff + (x - view[0]) * sx, yoff + (y - view[1]) * sy

    draw = ImageDraw.Draw(result)
    roi = record["roi_xyxy"]
    draw.rectangle((*xy(roi[0], roi[1]), *xy(roi[2], roi[3])), outline=ROI, width=1)
    boundary(draw, gt, xy)
    box = record.get("box_xyxy")
    if box:
        draw.rectangle((*xy(box[0], box[1]), *xy(box[2], box[3])), outline=BOX, width=2)
    point = record.get("point_xy")
    center = point or record.get("hrnet_point_xy")
    if center:
        x, y = xy(*center)
        if point is not None:
            draw.line((x - 6, y, x + 6, y), fill=POINT, width=2)
            draw.line((x, y - 6, x, y + 6), fill=POINT, width=2)
        else:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=GEOMETRY, width=2)

    label(draw, (12, 10), condition_label(record["condition_id"]), 18)
    label(draw, (12, 299), f"IoU {metric_text(metrics['iou'])}", 20)
    label(draw, (12, 325), f"Centroid error {metric_text(metrics['centroid_error_px'])} px", 17)
    coverage = number(metrics.get("reference_box_coverage_fraction"))
    coverage_label = (
        "n/a (no box)" if box is None else ("undefined" if coverage is None else f"{coverage:.1%}")
    )
    label(draw, (12, 350), f"Reference pixels in box: {coverage_label}", 16)
    if record["status"] != "passed":
        label(draw, (12, 371), "Inference failed; no mask displayed", 13, (255, 135, 135))
    return result


def focus_grid(
    output: Path,
    destination: Path,
    key: tuple[int, int, str],
    predictions: list[Row],
    rows: list[Row],
    references: dict,
) -> Row:
    selected = {r["condition_id"]: r for r in predictions if eye_key(r) == key}
    metrics = {r["condition_id"]: r for r in rows if eye_key(r) == key}
    image_id, annotation_id, side = key
    original = Image.open(output / f"inputs/images/{image_id:012d}.jpg").convert("RGB")
    # Only the fixed A_r100 ROI determines the shared viewport, never GT geometry.
    view = viewport([selected["A_r100"]], original.size)
    gt = np.array(Image.open(references[key])) > 0
    if gt.shape != (original.height, original.width):
        raise Phase1Error("Reference and original dimensions differ.")
    grid = Image.new("RGB", (1500, 935), BACKGROUND)
    draw = ImageDraw.Draw(grid)
    label(
        draw,
        (18, 14),
        f"{CATEGORIES[image_id]} {annotation_id} / {side} eye | COCO {image_id}",
        26,
    )
    label(
        draw,
        (18, 52),
        "Cyan: selected prediction | green: reference boundary | pink: automatic box | gray: ROI",
        18,
    )
    label(
        draw,
        (18, 79),
        "Yellow +: SAM positive point | orange o: HRNet geometry center only (not a SAM point)",
        18,
    )
    for index, condition in enumerate(PANEL_ORDER):
        card = panel(original, gt, selected[condition], metrics[condition], view)
        grid.paste(card, (18 + (index % 4) * 372, 116 + (index // 4) * 397))
    label(
        draw,
        (18, 916),
        "Same display viewport in every panel; all new variants use the frozen 1x ROI. "
        "Primary: gamma 0.5 + point.",
        15,
    )
    filename = f"{image_id}_{annotation_id}_{side}_grid.png"
    grid.save(destination / filename)
    return {
        "file": filename,
        "eye": list(key),
        "display_viewport_xyxy": list(view),
        "viewport_source": "frozen A_r100 ROI plus 8-pixel margin, clipped to image",
        "viewport_uses_reference_geometry": False,
    }


def heatmap(destination: Path, rows: list[Row]) -> None:
    identities = sorted({eye_key(r) for r in rows})
    index = {(eye_key(r), r["condition_id"]): r for r in rows}
    if len(identities) != 12 or len(index) != 96:
        raise Phase1Error("The IoU heatmap requires 12 eyes and eight conditions.")
    canvas = Image.new("RGB", (1120, 800), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    label(draw, (18, 14), "Automatic eye boxes vs frozen Group A | IoU for all 12 eyes", 25)
    label(
        draw,
        (18, 52),
        "P: HRNet point + automatic box | B: automatic box only | "
        "box size = max(8 px, gamma x eye distance)",
        16,
    )
    label(
        draw,
        (18, 82),
        "All box variants: fixed 1x ROI | Primary: P 0.5 | highest model-score candidate only",
        17,
    )
    headers = ("A 1x", "A 0.5x", "P 0.3", "P 0.5", "P 0.7", "B 0.3", "B 0.5", "B 0.7")
    for column, title in enumerate(headers):
        label(draw, (291 + column * 100, 132), title, 18)
    for row, key in enumerate(identities):
        y = 165 + row * 43
        label(draw, (18, y + 11), f"{CATEGORIES[key[0]]} {key[1]} / {key[2]}", 18)
        for column, condition in enumerate(CONDITIONS):
            value = number(index[(key, condition)]["iou"])
            x = 280 + column * 100
            color = (90, 90, 90) if value is None else heat_color(value)
            draw.rectangle((x, y, x + 96, y + 40), fill=color)
            ink = (12, 25, 35) if value is not None and value >= 0.6 else TEXT
            label(draw, (x + 22, y + 10), "NA" if value is None else f"{value:.3f}", 17, ink)
    for offset in range(400):
        draw.line((280 + offset, 710, 280 + offset, 731), fill=heat_color(offset / 399))
    label(draw, (277, 741), "0", 17)
    label(draw, (669, 741), "1", 17)
    label(
        draw,
        (18, 774),
        "Same three development images; failed predictions stay NA here "
        "and contribute zero to summary IoU.",
        15,
    )
    canvas.save(destination / "all_12_eyes_iou_heatmap.png")


def render(output: Path) -> None:
    predictions, rows, references = read_inputs(output)
    destination = output / "visualizations"
    destination.mkdir()
    views = [focus_grid(output, destination, key, predictions, rows, references) for key in FOCUS]
    heatmap(destination, rows)
    baseline = output / "evaluation_inputs/baseline"
    manifest = {
        "model_inference_performed": False,
        "candidate_selection": "highest model score; no GT-based selection",
        "box_only_center_marker_is_a_prompt": False,
        "eye_metrics_sha256": sha256_file(output / "eye_metrics.csv"),
        "predictions_sha256": sha256_file(output / "predictions.json"),
        "baseline_predictions_sha256": sha256_file(baseline / "predictions.json"),
        "baseline_evaluation_sha256": sha256_file(baseline / "evaluation_values.json"),
        "condition_order": list(CONDITIONS),
        "panel_order": list(PANEL_ORDER),
        "focus_views": views,
        "files_sha256": {p.name: sha256_file(p) for p in sorted(destination.glob("*.png"))},
    }
    (destination / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase3/auto_eye_box_poc_v1"))
    render(parser.parse_args().output)
