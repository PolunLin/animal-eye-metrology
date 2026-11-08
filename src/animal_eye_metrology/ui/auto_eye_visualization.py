"""Render saved automatic predictions and evaluation; never changes model outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {"left": (50, 210, 240), "right": (250, 150, 60)}
BACKGROUND = (18, 25, 37)
TEXT = (233, 239, 245)


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def text_at(draw: ImageDraw.ImageDraw, xy: tuple, value: str, size: int = 18) -> None:
    draw.text(xy, value, font=font(size), fill=TEXT)


def tinted(image: Image.Image, mask: np.ndarray, color: tuple, alpha: float) -> Image.Image:
    pixels = np.array(image.convert("RGB"))
    pixels[mask] = (pixels[mask] * (1 - alpha) + np.asarray(color) * alpha).astype(np.uint8)
    return Image.fromarray(pixels)


def contour(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    interior = mask.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return mask & ~interior


def cross(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int = 4) -> None:
    draw.line((x - radius, y, x + radius, y), fill=(255, 240, 40), width=2)
    draw.line((x, y - radius, x, y + radius), fill=(255, 240, 40), width=2)


def render(output: Path) -> None:
    destination = output / "visualizations"
    destination.mkdir()
    values = read(output / "evaluation_values.json")
    eyes, measurements = values["eyes"], values["measurements"]
    segmentation = read(output / "segmentation.json")["results"]
    by_eye = {(r["prediction_id"], r["eye_side"]): r for r in segmentation}
    all_cards = []
    for image_id in sorted({r["image_id"] for r in eyes}):
        original = Image.open(output / f"inputs/images/{image_id:012d}.jpg").convert("RGB")
        overlay = original.copy()
        selected = [r for r in eyes if r["image_id"] == image_id]
        for row in selected:
            if row["predicted_mask_path"]:
                mask = np.array(Image.open(output / row["predicted_mask_path"])) > 0
                overlay = tinted(overlay, mask, COLORS[row["eye_side"]], 0.55)
        draw = ImageDraw.Draw(overlay)
        for row in selected:
            if row["point_x"] is not None:
                cross(draw, row["point_x"], row["point_y"])
            if row["predicted_cx_px"] is not None:
                x, y = row["predicted_cx_px"], row["predicted_cy_px"]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=COLORS[row["eye_side"]], width=2)
        image_measurements = [r for r in measurements if r["image_id"] == image_id]
        for row in image_measurements:
            if row["predicted_distance_px"] is not None:
                draw.line(
                    (
                        row["predicted_cx_a_px"],
                        row["predicted_cy_a_px"],
                        row["predicted_cx_b_px"],
                        row["predicted_cy_b_px"],
                    ),
                    fill=(240, 190, 255)
                    if row["measurement_type"] == "right_eye_pair"
                    else (255, 255, 255),
                    width=1,
                )
        panel = Image.new("RGB", (max(760, original.width), original.height + 180), BACKGROUND)
        panel.paste(overlay, (0, 46))
        d = ImageDraw.Draw(panel)
        text_at(d, (16, 12), f"COCO {image_id} | Automatic HRNet + SAM 2.1 Tiny", 20)
        y = original.height + 60
        for row in image_measurements:
            val = row["predicted_distance_px"]
            value = f"{val:.3f} px" if val is not None else "undefined"
            label = (
                f"animal {row['annotation_id_a']} L-R"
                if row["measurement_type"] == "binocular"
                else "between animals: R-R"
            )
            text_at(d, (16, y), f"{label}: {value}", 16)
            y += 23
        text_at(
            d,
            (16, y + 5),
            "Yellow +: HRNet | cyan: left | orange: right | visibility unverified",
            14,
        )
        panel.save(destination / f"{image_id}_measurements.png")
        for row in selected:
            result = by_eye.get((row["prediction_id"], row["eye_side"]))
            roi = result["roi_xyxy"] if result else None
            card = Image.new("RGB", (300, 370), BACKGROUND)
            cd = ImageDraw.Draw(card)
            text_at(
                cd,
                (10, 10),
                f"{row['animal_category']} {row['annotation_id']} / {row['eye_side']}",
                17,
            )
            if roi:
                # A fixed 8-pixel display margin is independent of GT geometry.
                x0, y0, x1, y1 = roi
                display_box = (
                    max(0, x0 - 8),
                    max(0, y0 - 8),
                    min(original.width, x1 + 8),
                    min(original.height, y1 + 8),
                )
                gt = np.array(Image.open(output / row["reference_mask_path"])) > 0
                view = original.copy()
                if row["predicted_mask_path"]:
                    pred = np.array(Image.open(output / row["predicted_mask_path"])) > 0
                    view = tinted(view, pred, COLORS[row["eye_side"]], 0.6)
                pixels = np.array(view)
                pixels[contour(gt)] = (50, 255, 100)
                view = Image.fromarray(pixels)
                vd = ImageDraw.Draw(view)
                vd.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(180, 180, 180), width=1)
                view = view.crop(display_box)
                scale = min(280 / view.width, 260 / view.height)
                resized = view.resize(
                    (round(view.width * scale), round(view.height * scale)),
                    Image.Resampling.NEAREST,
                )
                xoff, yoff = (300 - resized.width) // 2, 38 + (260 - resized.height) // 2
                card.paste(resized, (xoff, yoff))
                cd = ImageDraw.Draw(card)
                if row["point_x"] is not None:
                    cross(
                        cd,
                        xoff + (row["point_x"] - display_box[0]) * scale,
                        yoff + (row["point_y"] - display_box[1]) * scale,
                        6,
                    )
            else:
                text_at(cd, (15, 130), "No valid automatic ROI", 18)
            iou = row["iou"]
            offset = row["point_offset_from_mask_centroid_px"]
            text_at(cd, (10, 310), f"IoU: {iou:.3f}" if iou is not None else "IoU: missing", 18)
            text_at(
                cd,
                (10, 336),
                f"Point-centroid: {offset:.2f} px"
                if offset is not None
                else "Point-centroid: undefined",
                16,
            )
            card.save(destination / f"{image_id}_{row['annotation_id']}_{row['eye_side']}.png")
            all_cards.append(card)
    sheet = Image.new("RGB", (1220, 80 + len(all_cards) // 4 * 380), BACKGROUND)
    sd = ImageDraw.Draw(sheet)
    text_at(
        sd,
        (14, 12),
        "Automatic eye localization and segmentation | 12 fixed development targets",
        23,
    )
    text_at(
        sd,
        (14, 45),
        "Green: reference boundary | cyan/orange: predicted mask | yellow +: point | gray: ROI",
        17,
    )
    for index, card in enumerate(all_cards):
        sheet.paste(card, (5 + (index % 4) * 305, 80 + (index // 4) * 380))
    sheet.save(destination / "all_12_eyes.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/phase3/auto_eye_localization_poc_v1")
    )
    render(parser.parse_args().output)
