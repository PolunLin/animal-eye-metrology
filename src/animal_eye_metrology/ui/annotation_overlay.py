"""Original-resolution animal identity overlays for human review."""

import colorsys
from typing import Any

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.data.coco import Annotation, CocoDataset, ImageRecord

PALETTE = ("#ffb000", "#00a6ed", "#dc267f", "#43aa8b", "#9467bd", "#e76f51")


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
