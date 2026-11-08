"""Original/overlay contact sheets from local images and official-format masks."""

import colorsys
import hashlib
import math
import textwrap
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.config import AnalysisConfig
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.data.analysis import AnalysisResult
from animal_eye_metrology.data.coco import Annotation, CocoDataset, ImageRecord

PALETTE = ("#ffb000", "#00a6ed", "#dc267f", "#43aa8b", "#9467bd", "#e76f51")


def render_measurement_overlay(
    original: Any,
    centers: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    masks: dict[tuple[int, str], Any],
    *,
    prediction_model: str | None = None,
) -> Any:
    """Original, manual masks/centers, and pixel-distance lines with an external legend."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = original.size
    animals = sorted({int(row["annotation_id"]) for row in centers})
    labels = {key: chr(65 + i) for i, key in enumerate(animals)}
    panel_masks, panel_lines = original.copy(), original.copy()
    colors = {"left": PALETTE[0], "right": PALETTE[1]}
    for row in centers:
        key = int(row["annotation_id"]), row["eye_side"]
        panel_masks.paste(colors[key[1]], (0, 0), masks[key].point(lambda value: value // 2))
    font = ImageFont.load_default(size=13)
    for row in measurements:
        if row["distance_px"] is None:
            continue
        a, b = (row["cx_a_px"], row["cy_a_px"]), (row["cx_b_px"], row["cy_b_px"])
        draw = ImageDraw.Draw(panel_lines)
        color = "#43aa8b" if row["measurement_type"] == "right_eye_pair" else "#ffb000"
        draw.line([a, b], fill="white", width=4)
        draw.line([a, b], fill=color, width=2)
    for panel in (panel_masks, panel_lines):
        draw = ImageDraw.Draw(panel)
        for row in centers:
            if row["centroid_x_px"] is None:
                continue
            x, y = row["centroid_x_px"], row["centroid_y_px"]
            draw.ellipse(
                (x - 2.5, y - 2.5, x + 2.5, y + 2.5),
                fill=colors[row["eye_side"]],
                outline="white",
                width=1,
            )
        for annotation_id in animals:
            eyes = [
                row
                for row in centers
                if int(row["annotation_id"]) == annotation_id and row["centroid_x_px"] is not None
            ]
            if eyes:
                x = sum(row["centroid_x_px"] for row in eyes) / len(eyes)
                y = min(row["centroid_y_px"] for row in eyes) - 18
                draw.text(
                    (max(1, min(width - 15, x)), max(1, y)),
                    labels[annotation_id],
                    fill="white",
                    font=font,
                    stroke_fill="black",
                    stroke_width=2,
                )
    legend_height = 24 * max(len(centers) + 1, len(measurements) + len(animals) + 2)
    page = Image.new("RGB", (width * 3, height + 64 + legend_height), "white")
    draw = ImageDraw.Draw(page)
    titles = (
        "Original image",
        "Predicted eye masks + centroids" if prediction_model else "Manual eye masks + centroids",
        "Mask-centroid distances (px)",
    )
    for i, (panel, title) in enumerate(
        zip((original, panel_masks, panel_lines), titles, strict=True)
    ):
        page.paste(panel, (i * width, 64))
        draw.text((i * width + 7, 7), title, fill="black", font=font)
        draw.text(
            (i * width + 7, 30), f"COCO Image {centers[0]['image_id']}", fill="black", font=font
        )
    footer = height + 72
    description = (
        f"{prediction_model}; v=human prompt version"
        if prediction_model
        else "Manual reference; no AI eye inference"
    )
    draw.text((7, footer), description, fill="black", font=font)
    draw.text((7, footer + 24), "L/R: animal's anatomical side", fill="black", font=font)
    draw.text((7, footer + 48), "Pixels; not physical length", fill="black", font=font)
    draw.text((7, footer + 72), "L yellow | R blue", fill="black", font=font)
    for i, row in enumerate(centers):
        point = (
            "undefined"
            if row["centroid_x_px"] is None
            else f"({row['centroid_x_px']:.3f}, {row['centroid_y_px']:.3f})"
        )
        label = labels[int(row["annotation_id"])] + "-" + row["eye_side"][0].upper()
        draw.text(
            (width + 7, footer + i * 24),
            f"{label} v{row['annotation_version']}: {point}",
            fill="black",
            font=font,
        )
    for i, annotation_id in enumerate(animals):
        draw.text(
            (2 * width + 7, footer + i * 24),
            f"{labels[annotation_id]} = Animal #{annotation_id}",
            fill="black",
            font=font,
        )
    for i, row in enumerate(measurements):
        first = labels[int(row["annotation_id_a"])] + "-" + row["eye_side_a"][0].upper()
        second = labels[int(row["annotation_id_b"])] + "-" + row["eye_side_b"][0].upper()
        value = "undefined" if row["distance_px"] is None else f"{row['distance_px']:.3f} px"
        draw.text(
            (2 * width + 7, footer + (len(animals) + i) * 24),
            f"{first} -> {second}: {value}",
            fill="black",
            font=font,
        )
    return page


def render_eye_comparison_sheets(output: Path, metrics: list[dict[str, Any]]) -> None:
    """Reuse native Pillow overlays for frozen GT/predictions; crops follow prompts, not GT."""
    import json

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from animal_eye_metrology.measurement.metrology import build_measurements

    models = ("sam2_1_tiny", "sam_vit_b", "mobilesam")
    labels = ("Frozen primary reference", "SAM 2.1 Hiera-Tiny", "SAM ViT-B", "MobileSAM ViT-T")
    folder = output / "overlays"
    folder.mkdir()
    font, small = ImageFont.load_default(size=17), ImageFont.load_default(size=14)
    for image_id in sorted({row["image_id"] for row in metrics}):
        selected = [r for r in metrics if r["image_id"] == image_id]
        keys = sorted({(r["annotation_id"], r["eye_side"]) for r in selected})
        sheet = Image.new("RGB", (1440, 120 + 300 * len(keys)), "#f3f5f7")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (18, 14),
            f"COCO {image_id} / {selected[0]['animal_category']} | "
            "Fixed human prompts, semi-automatic segmentation",
            font=font,
            fill="#17212b",
        )
        draw.text(
            (18, 42),
            "Gold = reference only; blue = prediction only; white = intersection. "
            "+ reference centroid; x prediction centroid.",
            font=small,
            fill="#17212b",
        )
        for col, label in enumerate(labels):
            draw.text((18 + col * 355, 75), label, font=font, fill="#17212b")
        for row_index, (annotation_id, side) in enumerate(keys):
            rows = [
                next(
                    r
                    for r in selected
                    if r["model"] == model
                    and r["annotation_id"] == annotation_id
                    and r["eye_side"] == side
                )
                for model in models
            ]
            case = output / "cases" / f"{image_id}_{annotation_id}_{side}"
            request = json.loads((case / "request.json").read_text())
            roi = request["roi_xyxy"]
            with Image.open(case / "input.png") as opened:
                original = opened.convert("RGB")
            crop = original.crop(roi)
            with Image.open(output / rows[0]["reference_mask_path"]) as opened:
                ref = np.array(opened.crop(roi)) > 0
            top = 108 + row_index * 300
            draw.text(
                (18, top),
                f"Animal {annotation_id} / anatomical {side} | "
                f"GT v{rows[0]['reference_annotation_version']} / "
                f"prompt v{rows[0]['prompt_version']} | ROI {roi}",
                font=small,
                fill="#17212b",
            )
            for col in range(4):
                x, y = 18 + col * 355, top + 26
                pixels = np.array(crop).copy()
                if col == 0:
                    pixels[ref] = (pixels[ref] * 0.35 + np.array([255, 176, 0]) * 0.65).astype(
                        np.uint8
                    )
                    row = rows[0]
                else:
                    row = rows[col - 1]
                    pred = np.zeros(ref.shape, dtype=bool)
                    if row["predicted_mask_path"]:
                        with Image.open(output / row["predicted_mask_path"]) as opened:
                            pred = np.array(opened.crop(roi)) > 0
                    pixels[ref & ~pred] = (255, 176, 0)
                    pixels[pred & ~ref] = (0, 166, 237)
                    pixels[pred & ref] = (255, 255, 255)
                scale = min(330 / crop.width, 188 / crop.height)
                panel = Image.fromarray(pixels).resize(
                    (round(crop.width * scale), round(crop.height * scale)),
                    Image.Resampling.NEAREST,
                )
                marker = ImageDraw.Draw(panel)
                for source, color in (("reference", "#ffb000"), ("predicted", "#00a6ed")):
                    if col == 0 and source == "predicted":
                        continue
                    cx, cy = row[f"{source}_cx_px"], row[f"{source}_cy_px"]
                    if cx is None:
                        continue
                    px, py = (
                        (cx - roi[0]) * panel.width / crop.width,
                        (cy - roi[1]) * panel.height / crop.height,
                    )
                    segments = (
                        [(px - 5, py, px + 5, py), (px, py - 5, px, py + 5)]
                        if source == "reference"
                        else [(px - 4, py - 4, px + 4, py + 4), (px - 4, py + 4, px + 4, py - 4)]
                    )
                    for width, fill in ((4, "black"), (2, color)):
                        for segment in segments:
                            marker.line(segment, fill=fill, width=width)
                sheet.paste(panel, (x, y))
                if col == 0:
                    lines = [
                        f"Reference area: {row['reference_area_px']} px",
                        "Original pixels, nearest-neighbor enlargement",
                        f"GT pixels outside ROI: {row['reference_pixels_outside_roi']}",
                    ]
                else:
                    overlap = (
                        f"IoU {row['iou']:.3f} | Dice {row['dice']:.3f}"
                        if row["iou"] is not None
                        else "IoU / Dice undefined: inference failed"
                    )
                    error = (
                        f"Center error {row['centroid_error_px']:.3f} px"
                        if row["centroid_error_px"] is not None
                        else "Center undefined"
                    )
                    lines = [
                        overlap,
                        error,
                        f"Predicted area {row['predicted_area_px']} px | "
                        f"{row['prediction_status']}",
                    ]
                for line_number, line in enumerate(lines):
                    draw.text((x, top + 220 + line_number * 19), line, font=small, fill="#17212b")
        sheet.save(folder / f"{image_id:012d}_eye_comparison.png")
        for model in models:
            centers, masks = [], {}
            for row in selected:
                if row["model"] != model:
                    continue
                centers.append(
                    {
                        "image_id": image_id,
                        "annotation_id": row["annotation_id"],
                        "eye_side": row["eye_side"],
                        "annotation_version": int(row["prompt_version"]),
                        "centroid_x_px": row["predicted_cx_px"],
                        "centroid_y_px": row["predicted_cy_px"],
                    }
                )
                if row["predicted_mask_path"]:
                    with Image.open(output / row["predicted_mask_path"]) as opened:
                        masks[(row["annotation_id"], row["eye_side"])] = opened.copy()
                else:
                    masks[(row["annotation_id"], row["eye_side"])] = Image.new("L", original.size)
            page = render_measurement_overlay(
                original, centers, build_measurements(centers), masks, prediction_model=model
            )
            page.save(folder / f"{image_id:012d}_{model}_measurements.png")


def instance_colors(count: int, *, unique: bool = False) -> list[Any]:
    if unique and count > len(PALETTE):
        return [
            tuple(
                round(channel * 255)
                for channel in colorsys.hsv_to_rgb(
                    (item * 0.61803398875) % 1, 0.8, 0.95 if item % 2 == 0 else 0.75
                )
            )
            for item in range(count)
        ]
    return [PALETTE[item % len(PALETTE)] for item in range(count)]


def render_instance_overlay(
    original: Any,
    record: ImageRecord,
    annotations: list[Annotation],
    dataset: CocoDataset,
    *,
    show_boxes: bool = True,
    show_masks: bool = True,
    selected_id: int | None = None,
    instance_labels: dict[int, str] | None = None,
) -> Any:
    """Draw supplied instance masks at original resolution; perform no inference."""
    from PIL import Image, ImageDraw, ImageFont
    from pycocotools import mask as mask_utils

    if original.size != (record.width, record.height):
        raise Phase1Error("Image dimensions differ from COCO annotations.")
    overlay = original.convert("RGB").copy()
    colors = instance_colors(len(annotations), unique=True)
    if show_masks:
        for color, ann in zip(colors, annotations, strict=True):
            mask = _decode_mask(ann, record, mask_utils)
            overlay.paste(
                color, (0, 0), Image.fromarray(mask * (95 if ann.id == selected_id else 60))
            )
    if show_boxes:
        draw = ImageDraw.Draw(overlay)
        font = ImageFont.load_default(size=14)
        for color, ann in zip(colors, annotations, strict=True):
            x, y, width, height = ann.bbox
            box = (x, y, x + width, y + height)
            if ann.id == selected_id:
                draw.rectangle(box, outline="white", width=7)
            draw.rectangle(box, outline=color, width=4 if ann.id == selected_id else 2)
            label = (
                instance_labels[ann.id]
                if instance_labels is not None and ann.id in instance_labels
                else f"{dataset.categories[ann.category_id].name} #{ann.id}"
            )
            tx = max(0, min(x, overlay.width - draw.textlength(label, font=font) - 2))
            ty = max(0, min(y, overlay.height - 18))
            draw.text((tx, ty), label, fill=color, font=font, stroke_width=1, stroke_fill="black")
    return overlay


def _decode_mask(ann: Annotation, image: ImageRecord, mask_utils: Any) -> Any:
    segmentation = ann.segmentation
    try:
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, image.height, image.width))
        elif isinstance(segmentation["counts"], list):
            rle = mask_utils.frPyObjects(segmentation, image.height, image.width)
        else:
            rle = segmentation
        decoded = mask_utils.decode(rle)
    except (ValueError, TypeError, OverflowError) as error:
        raise Phase1Error(f"Unable to decode mask for annotation {ann.id}: {error}") from error
    if decoded.shape != (image.height, image.width):
        raise Phase1Error(f"Decoded mask shape mismatch for annotation {ann.id}.")
    return decoded


def render_contact_sheets(
    dataset: CocoDataset,
    analysis: AnalysisResult,
    config: AnalysisConfig,
    output: Path,
    *,
    instance_legend: bool = False,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise Phase1Error(
            "Contact sheets require the 'visualization' extra; run uv sync --extra visualization."
        ) from error
    settings = config.contact_sheets
    image_ids = [row["image_id"] for row in analysis.candidates["multi_any"]]
    selected = image_ids[: settings.max_images] if settings.max_images else image_ids
    directory = output / "contact_sheets"
    directory.mkdir()
    font = ImageFont.load_default(size=13)
    tile_width = settings.panel_width * 2
    header_height = 100
    legend_columns = 3
    max_instances = max((len(analysis.eligible_by_image[key]) for key in selected), default=0)
    legend_height = (math.ceil(max_instances / legend_columns) * 22 + 12) if instance_legend else 0
    tile_height = settings.panel_height + header_height + legend_height
    per_page = settings.columns * settings.rows
    sources, pages = [], []
    for page_start in range(0, len(selected), per_page):
        page_ids = selected[page_start : page_start + per_page]
        page = Image.new(
            "RGB",
            (
                tile_width * settings.columns,
                tile_height * math.ceil(len(page_ids) / settings.columns),
            ),
            "white",
        )
        for index, image_id in enumerate(page_ids):
            record = dataset.images[image_id]
            path = (config.images / record.file_name).resolve()
            if not path.is_relative_to(config.images.resolve()):
                raise Phase1Error(f"Image resolves outside configured image directory: {path}")
            payload = path.read_bytes()
            try:
                with Image.open(BytesIO(payload)) as opened:
                    opened.verify()
                with Image.open(BytesIO(payload)) as opened:
                    if opened.size != (record.width, record.height):
                        raise Phase1Error(f"Image dimensions differ from annotations: {path}")
                    original = opened.convert("RGB")
            except (OSError, SyntaxError) as error:
                raise Phase1Error(f"Invalid local image {path}: {error}") from error
            thumb = ImageOps.contain(original, (settings.panel_width, settings.panel_height))
            overlay = thumb.copy()
            annotations = analysis.eligible_by_image[image_id]
            colors = instance_colors(len(annotations), unique=instance_legend)
            for ann_index, ann in enumerate(annotations):
                decoded = _decode_mask(ann, record, mask_utils)
                alpha = Image.fromarray(decoded * 90).resize(thumb.size, Image.Resampling.NEAREST)
                overlay.paste(colors[ann_index], (0, 0), alpha)
            draw = ImageDraw.Draw(overlay)
            placed_labels = []
            for ann_index, ann in enumerate(annotations):
                x, y, width, height = ann.bbox
                sx, sy = thumb.width / record.width, thumb.height / record.height
                box = (x * sx, y * sy, (x + width) * sx, (y + height) * sy)
                color = colors[ann_index]
                draw.rectangle(box, outline=color, width=2)
                label = (
                    f"#{ann.id}"
                    if instance_legend
                    else f"{dataset.categories[ann.category_id].name} #{ann.id}"
                )
                label_x = max(
                    0, min(box[0], max(0, thumb.width - draw.textlength(label, font=font)))
                )
                label_y = max(0, min(box[1], thumb.height - 16))
                if instance_legend:
                    label_width = draw.textlength(label, font=font) + 4
                    for candidate_y in [label_y, *range(0, max(1, thumb.height - 16), 18)]:
                        rect = (label_x, candidate_y, label_x + label_width, candidate_y + 16)
                        if all(
                            rect[2] <= old[0]
                            or rect[0] >= old[2]
                            or rect[3] <= old[1]
                            or rect[1] >= old[3]
                            for old in placed_labels
                        ):
                            label_y = candidate_y
                            placed_labels.append(rect)
                            break
                    draw.line((box[0], box[1], label_x, label_y), fill=color, width=1)
                draw.text(
                    (label_x, label_y),
                    label,
                    fill=color,
                    font=font,
                    stroke_width=1,
                    stroke_fill="black",
                )
            tile = Image.new("RGB", (tile_width, tile_height), "white")
            tile_draw = ImageDraw.Draw(tile)
            counts = Counter(dataset.categories[ann.category_id].name for ann in annotations)
            categories = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
            caption = (
                f"Image ID {image_id} | eligible annotated animals: {len(annotations)}\n"
                + textwrap.fill(categories, width=max(20, tile_width // 8))
                + "\niscrowd=1 excluded | eyes unreviewed"
            )
            tile_draw.multiline_text((6, 4), caption, fill="black", font=font, spacing=3)
            tile_draw.text((6, header_height - 20), "Original", fill="black", font=font)
            tile_draw.text(
                (settings.panel_width + 6, header_height - 20),
                "Animal boxes + masks",
                fill="black",
                font=font,
            )
            for panel_index, panel in enumerate((thumb, overlay)):
                tile.paste(
                    panel,
                    (
                        panel_index * settings.panel_width
                        + (settings.panel_width - panel.width) // 2,
                        header_height + (settings.panel_height - panel.height) // 2,
                    ),
                )
            if instance_legend:
                for ann_index, ann in enumerate(annotations):
                    legend_x = (ann_index % legend_columns) * (tile_width // legend_columns) + 8
                    legend_y = (
                        header_height
                        + settings.panel_height
                        + 6
                        + (ann_index // legend_columns) * 22
                    )
                    tile_draw.rectangle(
                        (legend_x, legend_y + 3, legend_x + 12, legend_y + 15),
                        fill=colors[ann_index],
                        outline="black",
                    )
                    tile_draw.text(
                        (legend_x + 18, legend_y),
                        f"{dataset.categories[ann.category_id].name} #{ann.id}",
                        fill="black",
                        font=font,
                    )
            page.paste(
                tile,
                (
                    (index % settings.columns) * tile_width,
                    (index // settings.columns) * tile_height,
                ),
            )
            sources.append(
                {
                    "image_id": image_id,
                    "file_name": record.file_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "dimensions_verified": True,
                }
            )
        relative = f"contact_sheets/page_{len(pages) + 1:04d}.png"
        page.save(output / relative)
        pages.append(relative)
    return {
        "enabled": True,
        "candidate_images": len(image_ids),
        "selected_image_ids": selected,
        "selection": "first_n_by_image_id_preview_only"
        if settings.max_images
        else "all_candidates",
        "random_seed": None,
        "eye_review_sample": False,
        "pages": pages,
        "verified_images": sources,
    }
