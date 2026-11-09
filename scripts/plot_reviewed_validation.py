#!/usr/bin/env python3
"""Render measured results and original-pixel reference/prediction comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.paths import PROJECT_ROOT


def plot_two_cats(
    directory: Path,
    root: Path,
    images: Path | None,
    reference: dict,
    predictions: dict,
    eyes: list[dict],
    measurements: list[dict],
) -> None:
    """Show both model animal contours and all three saved distances on the original grid."""
    image_id = 555705
    image = next(i for i in reference["images"] if i["image_id"] == image_id)
    source = images / image["file_name"] if images else root / image["source_image"]
    if sha256_file(source) != image["image_sha256"]:
        raise ValueError("Two-cat source image hash differs from the frozen reference.")
    with Image.open(source) as original:
        pixels = np.asarray(original.convert("RGB"))
    predicted = next(i for i in predictions["images"] if i["image_id"] == image_id)
    image_eyes = [e for e in eyes if int(e["image_id"]) == image_id]
    colors = {49839: "#ffc857", 49029: "#c4a0ff"}
    centers = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor="#f8fafc")
    fig.suptitle(
        "Two cats · animal masks, eye masks and distances",
        fontsize=20,
        fontweight="bold",
        color="#172554",
        y=0.97,
    )
    for ax, title in zip(axes, ("Original COCO #555705", "Saved A_r100 model output"), strict=True):
        ax.imshow(pixels, interpolation="nearest", extent=(0, image["width"], image["height"], 0))
        ax.set(xlim=(0, image["width"]), ylim=(image["height"], 0), title=title)
        ax.set_axis_off()
    ax = axes[1]
    grid_x, grid_y = np.arange(image["width"]) + 0.5, np.arange(image["height"]) + 0.5
    for annotation_id, color in colors.items():
        scored = [e for e in image_eyes if int(e["annotation_id"]) == annotation_id]
        prediction_id = scored[0]["prediction_id"]
        animal = next(a for a in predicted["animals"] if a["prediction_id"] == prediction_id)
        if not animal["accepted"] or animal["mask_status"] != "completed":
            raise ValueError("Two-cat illustration requires completed animal masks.")
        body = mask_utils.decode(animal["mask_rle"]).astype(bool)
        ax.contour(grid_x, grid_y, body, levels=[0.5], colors=[color], linewidths=1.5)
        for row in scored:
            eye = next(
                e
                for e in predicted["eyes"]
                if e["prediction_id"] == prediction_id
                and e["eye_side"] == row["eye_side"]
                and e["condition_id"] == "A_r100"
            )
            mask = mask_utils.decode(eye["mask_rle"]).astype(bool)
            yy, xx = np.nonzero(mask)
            center = (float(xx.mean() + 0.5), float(yy.mean() + 0.5))
            expected = (float(row["predicted_cx_px"]), float(row["predicted_cy_px"]))
            if not all(
                math.isclose(a, b, abs_tol=1e-12) for a, b in zip(center, expected, strict=True)
            ):
                raise ValueError("Eye illustration centroid differs from scored evidence.")
            centers[(annotation_id, row["eye_side"])] = center
            side_color = "#00e5ff" if row["eye_side"] == "left" else "#ff466b"
            ax.contour(grid_x, grid_y, mask, levels=[0.5], colors=[side_color], linewidths=1.25)
            ax.plot(*center, marker="+", color=side_color, markersize=6, markeredgewidth=1.5)
    selected = [m for m in measurements if int(m["image_id"]) == image_id]
    if len(selected) != 3:
        raise ValueError(
            "Two-cat illustration requires two binocular and one cross-right distance."
        )
    for row in selected:
        a = centers[(int(row["annotation_id_a"]), row["eye_side_a"])]
        b = centers[(int(row["annotation_id_b"]), row["eye_side_b"])]
        if not math.isclose(math.dist(a, b), float(row["predicted_distance_px"]), abs_tol=1e-9):
            raise ValueError("Illustrated distance differs from saved measurement.")
        cross = row["measurement_type"] == "right_eye_pair"
        color = "#a3ff65" if cross else colors[int(row["annotation_id_a"])]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#111827", linewidth=3, alpha=0.65)
        ax.plot(
            [a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.7, linestyle="--" if cross else "-"
        )
    cards = [
        (49839, "Orange cat · left to right eye", "#9a6000"),
        (49029, "Tabby cat · left to right eye", "#6d3f9d"),
        (None, "Between cats · right to right eye", "#357012"),
    ]
    for x, (animal_id, label, color) in zip((0.18, 0.5, 0.82), cards, strict=True):
        row = next(
            m
            for m in selected
            if (
                m["measurement_type"] == "right_eye_pair"
                if animal_id is None
                else m["measurement_type"] == "binocular" and int(m["annotation_id_a"]) == animal_id
            )
        )
        fig.text(x, 0.21, label, ha="center", color=color, fontsize=12)
        fig.text(
            x,
            0.15,
            f"{float(row['predicted_distance_px']):.3f} px",
            ha="center",
            color=color,
            fontsize=20,
            fontweight="bold",
        )
    fig.text(
        0.03,
        0.07,
        "Eye contours: cyan = anatomical left; pink = anatomical right. "
        "Dashed line joins the right-eye centroids.",
        color="#475569",
        fontsize=10,
    )
    fig.text(
        0.03,
        0.025,
        "Selected reviewed example; measurements use original image pixels. "
        "Full-sample results are reported separately.",
        color="#475569",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.29, wspace=0.06)
    fig.savefig(directory / "two_cats.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path("examples/reviewed_validation"))
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--images", type=Path, help="Optional flat directory of original COCO JPEGs"
    )
    args = parser.parse_args()
    directory = args.evidence
    summary = json.loads((directory / "summary.json").read_text())
    reference = json.loads((directory / "reference.json").read_text())
    predictions = json.loads((directory / "predictions.json").read_text())
    with (directory / "per_eye.csv").open(encoding="utf-8-sig") as handle:
        eyes = list(csv.DictReader(handle))
    with (directory / "measurements.csv").open(encoding="utf-8-sig") as handle:
        measurements = list(csv.DictReader(handle))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="#f8fafc")
    fig.suptitle(
        "Reviewed development validation  |  A_r100",
        x=0.07,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#172554",
    )
    fig.text(
        0.07,
        0.85,
        "26 photos rerun · 42 eligible eyes on 16 photos · original weights",
        color="#475569",
        fontsize=12,
    )
    axes[0].hist(
        [float(e["iou"]) for e in eyes],
        bins=np.linspace(0, 1, 11),
        color="#2563eb",
        edgecolor="white",
    )
    axes[0].set(
        xlabel="Eye mask IoU",
        ylabel="Eyes",
        xlim=(0, 1),
        title=f"IoU {summary['mean_iou_all_targets_failure_as_zero']:.3f}  /  "
        f"Dice {summary['mean_dice_all_targets_failure_as_zero']:.3f}",
    )
    axes[0].text(
        0.02,
        0.96,
        "40 nonempty predictions / 42 references\n2 unmatched eyes included as 0",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
    )
    for x, (kind, color, _label) in enumerate(
        (
            ("binocular", "#0f766e", "Within-animal\nbinocular"),
            ("right_eye_pair", "#d97706", "Between-animal\nright eyes"),
        )
    ):
        rows = [m for m in measurements if m["measurement_type"] == kind]
        errors = [float(m["absolute_error_px"]) for m in rows if m["absolute_error_px"]]
        axes[1].scatter(x + np.linspace(-0.13, 0.13, len(errors)), errors, c=color, s=36, alpha=0.8)
        axes[1].hlines(summary[f"{kind}_mae_px"], x - 0.25, x + 0.25, colors=color, linewidth=3)
        axes[1].text(
            x,
            -0.25,
            f"{len(errors)}/{len(rows)} defined",
            ha="center",
            va="top",
            transform=axes[1].get_xaxis_transform(),
            color=color,
            fontsize=10,
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Within-animal\nbinocular", "Between-animal\nright eyes"],
        ylabel="Absolute distance error (px)",
        title="Distance errors  |  bars = MAE",
    )
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.text(
        0.07,
        0.035,
        "Human-reviewed development references; not an independent blind test. "
        "Same-image pairs are dependent.",
        fontsize=10,
        color="#475569",
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.30, top=0.72, wspace=0.30)
    fig.savefig(directory / "overview.png", dpi=150)
    plt.close(fig)

    # Owner-requested low-error illustration; aggregate evaluation remains unchanged.
    # This cat's reviewed contours retain the prediction; agreement is not a blind-test score.
    demo = next(
        m
        for m in measurements
        if m["image_id"] == "555705"
        and m["annotation_id_a"] == "49839"
        and m["measurement_type"] == "binocular"
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="#f8fafc", squeeze=False)
    fig.suptitle(
        "Low-error example · reviewed reference vs. model",
        fontsize=18,
        fontweight="bold",
        color="#172554",
    )
    for row, measurement in enumerate((demo,)):
        image_id, annotation_id = int(measurement["image_id"]), int(measurement["annotation_id_a"])
        image = next(i for i in reference["images"] if i["image_id"] == image_id)
        source = (
            args.images / image["file_name"] if args.images else args.root / image["source_image"]
        )
        if sha256_file(source) != image["image_sha256"]:
            raise ValueError("Qualitative comparison source image hash differs.")
        pixels = np.asarray(Image.open(source).convert("RGB"))
        cases = [
            c
            for c in reference["cases"]
            if c["image_id"] == image_id and c["annotation_id"] == annotation_id and c["eligible"]
        ]
        scored = [
            e
            for e in eyes
            if int(e["image_id"]) == image_id and int(e["annotation_id"]) == annotation_id
        ]
        predicted = next(i for i in predictions["images"] if i["image_id"] == image_id)
        channels = [
            next(
                e
                for e in predicted["eyes"]
                if e["prediction_id"] == s["prediction_id"]
                and e["eye_side"] == s["eye_side"]
                and e["condition_id"] == "A_r100"
            )
            for s in scored
        ]
        masks = [
            [mask_utils.decode(c["reference_mask_rle"]).astype(bool) for c in cases],
            [mask_utils.decode(e["mask_rle"]).astype(bool) for e in channels],
        ]
        ys, xs = np.nonzero(np.logical_or.reduce([m for group in masks for m in group]))
        pad = max(12, int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.4))
        x0, x1 = max(0, xs.min() - pad), min(image["width"], xs.max() + pad + 1)
        y0, y1 = max(0, ys.min() - pad), min(image["height"], ys.max() + pad + 1)
        titles = [
            f"COCO #{image_id} · animal #{annotation_id}",
            f"Reference  {float(measurement['reference_distance_px']):.2f} px",
            f"Prediction  {float(measurement['predicted_distance_px']):.2f} px",
        ]
        for col in range(3):
            ax = axes[row, col]
            ax.imshow(
                pixels, interpolation="nearest", extent=(0, image["width"], image["height"], 0)
            )
            if col:
                centers = []
                for mask, color in zip(masks[col - 1], ("#00e5ff", "#ff466b"), strict=True):
                    yy, xx = np.nonzero(mask)
                    centers.append((xx.mean() + 0.5, yy.mean() + 0.5))
                    ax.contour(
                        np.arange(image["width"]) + 0.5,
                        np.arange(image["height"]) + 0.5,
                        mask,
                        levels=[0.5],
                        colors=[color],
                        linewidths=1.2,
                    )
                ax.plot(*zip(*centers, strict=True), color="yellow", marker="+", linewidth=1.5)
            ax.set(xlim=(x0, x1), ylim=(y1, y0), title=titles[col])
            ax.set_axis_off()
        label = "Selected example: orange cat"
        axes[row, 0].text(0, -0.08, label, transform=axes[row, 0].transAxes, fontsize=10)
        axes[row, 2].text(
            0,
            -0.08,
            f"Absolute error: {float(measurement['absolute_error_px']):.3f} px",
            transform=axes[row, 2].transAxes,
            fontsize=11,
            color="#9a3412",
        )
    fig.text(
        0.05,
        0.015,
        "Reviewed contours match this prediction. "
        "Illustration only; full-sample scores are reported separately.",
        color="#475569",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.80, bottom=0.16, wspace=0.12)
    fig.savefig(directory / "eye_examples.png", dpi=150)
    plt.close(fig)
    plot_two_cats(directory, args.root, args.images, reference, predictions, eyes, measurements)


if __name__ == "__main__":
    main()
