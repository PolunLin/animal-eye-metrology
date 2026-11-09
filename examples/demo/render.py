"""Render documented development figures from immutable recorded masks and geometry."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(
    description="Render the recorded COCO 125405 development illustration without running models."
)
parser.add_argument("--run", type=Path, required=True, help="Existing saved inference directory")
parser.add_argument(
    "--output", type=Path, required=True, help="New output directory, separate from the source run"
)
parser.add_argument(
    "--source-commit",
    default="eab65727117e5594fcc69165e1435fdc38f152df",
    help="Recorded source commit (this renderer accepts only the documented source run)",
)
args = parser.parse_args()
RUN = args.run.resolve()
OUT = args.output.resolve()
if OUT == RUN or OUT.is_relative_to(RUN):
    parser.error("Output must be separate from the source run.")
if any(
    (OUT / name).exists()
    for name in ("animal-contours.png", "eye-measurements.png", "manifest.json")
):
    parser.error("Refusing to overwrite existing figures or provenance.")
OUT.mkdir(parents=True, exist_ok=True)
r = json.loads((RUN / "result.json").read_text())
if r["run_id"] != "20261109T140429Z_e2ef0383a74c":
    parser.error(
        "This illustration renderer accepts only recorded run 20261109T140429Z_e2ef0383a74c."
    )
if args.source_commit != "eab65727117e5594fcc69165e1435fdc38f152df":
    parser.error("Source commit does not match the documented recorded run.")
im = Image.open(RUN / r["input"]["image_path"]).convert("RGB")
if (
    im.size != (640, 428)
    or r["input"]["original_sha256"]
    != "aeb172b088be9b334a97fcd4df1fe89b5730bc5836765a23c82f2b8b4ff3d03b"
):
    parser.error("This fixed-layout figure requires the recorded COCO 125405 image.")
if len(r["eyes"]) != 4 or len(r["measurements"]) != 3 or r["conditions"] != ["A_r100"]:
    parser.error("Expected four eyes and three A_r100 measurement rows.")
assert all(
    x["status"] == "candidate" and x["qualification"] == "unverified" for x in r["measurements"]
)
DARK = "#172c3c"
MUTED = "#526476"
BG = "#f6f8fa"
WHITE = "#ffffff"
FONT_PATH = Path("/System/Library/Fonts/Supplemental")


def font(n, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    candidates = [str(FONT_PATH / name), "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, n)
        except OSError:
            continue
    return ImageFont.load_default(size=n)


def txt(draw, xy, s, size=20, color=DARK, bold=False):
    draw.text(xy, s, font=font(size, bold), fill=color)


def boundary(mask):
    p = np.pad(mask, 1)
    interior = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~interior


def overlay(base, mask, color, alpha=0.24):
    v = np.asarray(base).copy()
    v[mask] = (v[mask] * (1 - alpha) + np.array(color) * alpha).astype("uint8")
    v[boundary(mask)] = color
    return Image.fromarray(v)


def centroid_marker(draw, x, y):
    for width, color in [(5, "#182230"), (2, "white")]:
        draw.line((x - 7, y, x + 7, y), fill=color, width=width)
        draw.line((x, y - 7, x, y + 7), fill=color, width=width)


body = im.copy()
colors = [(0, 188, 175), (255, 171, 38)]
accepted = [x for x in r["animals"] if x["accepted"]]
for animal, c in zip(accepted, colors, strict=True):
    mask = np.asarray(Image.open(RUN / animal["mask_path"])) > 0
    assert int(mask.sum()) == animal["mask_area_px"]
    body = overlay(body, mask, c)
canvas = Image.new("RGB", (1340, 720), BG)
d = ImageDraw.Draw(canvas)
txt(d, (24, 20), "Animal Eye Metrology | recorded development example", 29, bold=True)
txt(
    d,
    (24, 60),
    "COCO val2017 / 125405   -   YOLO26s-seg > HRNet-W32/AP-10K > SAM 2.1 Tiny",
    20,
    MUTED,
)
txt(d, (24, 105), "A  Original image", 23, bold=True)
txt(d, (684, 105), "B  Predicted instance masks + eye distances", 23, bold=True)
canvas.paste(im, (24, 143))
canvas.paste(body, (684, 143))
# Draw geometry from saved centroids in the unscaled original coordinate frame.
for m in r["measurements"]:
    color = "#ffffff" if m["measurement_type"] == "binocular" else "#f7b5ff"
    coords = (684 + m["cx_a_px"], 143 + m["cy_a_px"], 684 + m["cx_b_px"], 143 + m["cy_b_px"])
    d.line(coords, fill="#27323c", width=4)
    d.line(coords, fill=color, width=2)
for e in r["eyes"]:
    x = 684 + e["centroid_x_px"]
    y = 143 + e["centroid_y_px"]
    d.ellipse((x - 3, y - 3, x + 3, y + 3), fill="white", outline="#172c3c", width=1)
for label, xy, c in [("Dog 1", (915, 460), colors[0]), ("Dog 2", (1180, 260), colors[1])]:
    d.rounded_rectangle((xy[0] - 8, xy[1] - 4, xy[0] + 74, xy[1] + 25), radius=4, fill="#172c3c")
    txt(d, xy, label, 19, c, True)
txt(d, (24, 591), f"Dog 1 / L-R: {r['measurements'][0]['distance_px']:.3f} px", 23, bold=True)
txt(d, (458, 591), f"Dog 2 / L-R: {r['measurements'][1]['distance_px']:.3f} px", 23, bold=True)
txt(
    d, (880, 591), f"Dog 1 R - Dog 2 R: {r['measurements'][2]['distance_px']:.3f} px", 23, bold=True
)
txt(
    d,
    (24, 631),
    "White: within-animal distance    Pink: right-to-right distance    Dots: eye mask centroids",
    19,
    MUTED,
)
txt(
    d,
    (24, 668),
    (
        "A_r100 | candidate / unverified | Image-plane pixels, not "
        "physical length; no accuracy evaluation on this run."
    ),
    19,
    MUTED,
)
canvas.save(OUT / "animal-contours.png", optimize=True)

canvas = Image.new("RGB", (1200, 850), BG)
d = ImageDraw.Draw(canvas)
txt(d, (24, 18), "Four eye masks | original pixels and saved predictions", 28, bold=True)
txt(
    d,
    (24, 59),
    (
        "L / R = animal anatomical channels. White cross = mask "
        "centroid. Side and visibility remain unverified."
    ),
    19,
    MUTED,
)
crops = []
for idx, e in enumerate(r["eyes"]):
    x = 24 + (idx % 2) * 590
    y = 103 + (idx // 2) * 330
    side = e["eye_side"]
    c = (0, 188, 215) if side == "left" else (255, 113, 117)
    pid = e["prediction_id"]
    dog = "Dog 1" if pid == "animal_0001" else "Dog 2"
    txt(
        d,
        (x, y),
        f"{dog} / {side.upper()}   -   {e['mask_area_px']} foreground pixels",
        22,
        bold=True,
    )
    cx = e["centroid_x_px"]
    cy = e["centroid_y_px"]
    x0 = math.floor(cx) - 16
    y0 = math.floor(cy) - 16
    box = (x0, y0, x0 + 32, y0 + 32)
    raw = im.crop(box)
    mask = np.asarray(Image.open(RUN / e["mask_path"])) > 0
    assert int(mask.sum()) == e["mask_area_px"]
    yy, xx = np.nonzero(mask)
    assert abs(float((xx + 0.5).mean()) - cx) < 1e-10
    assert abs(float((yy + 0.5).mean()) - cy) < 1e-10
    shown = overlay(im, mask, c, alpha=0.3).crop(box)
    raw = raw.resize((224, 224), Image.Resampling.NEAREST)
    shown = shown.resize((224, 224), Image.Resampling.NEAREST)
    canvas.paste(raw, (x, y + 60))
    canvas.paste(shown, (x + 260, y + 60))
    txt(d, (x, y + 33), "Original crop", 18, MUTED)
    txt(d, (x + 260, y + 33), "Predicted mask + centroid", 18, MUTED)
    centroid_marker(d, x + 260 + (cx - x0) * 7, y + 60 + (cy - y0) * 7)
    txt(d, (x, y + 291), f"Centroid in original image: ({cx:.3f}, {cy:.3f}) px", 18, MUTED)
    crops.append(
        {"prediction_id": pid, "eye_side": side, "crop_xyxy": list(box), "display_scale": 7}
    )
txt(
    d,
    (24, 780),
    "32 x 32 px crops at 7x nearest-neighbor magnification; enlargement adds no image information.",
    19,
    MUTED,
)
txt(
    d,
    (24, 814),
    (
        "A_r100 | candidate / unverified | Saved masks shown without "
        "smoothing, filling or manual corrections."
    ),
    19,
    MUTED,
)
canvas.save(OUT / "eye-measurements.png", optimize=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


paths = [
    "result.json",
    "run_manifest.json",
    "input/original.jpg",
    "input/image.png",
    "measurements.csv",
]
paths += [x["mask_path"] for x in accepted] + [x["mask_path"] for x in r["eyes"]]
record = {
    "schema_version": 1,
    "source_run_id": r["run_id"],
    "source_run_relative_path": "outputs/inference/" + r["run_id"],
    "source_code_commit": args.source_commit,
    "image": {
        "dataset": "COCO 2017 validation",
        "image_id": 125405,
        "width": 640,
        "height": 428,
        "source_url": "https://s3.amazonaws.com/images.cocodataset.org/val2017/000000125405.jpg",
        "role": "existing_development_example",
    },
    "condition_id": "A_r100",
    "models": ["YOLO26s-seg", "HRNet-W32/AP-10K", "SAM 2.1 Tiny"],
    "checkpoint_sha256": {
        k: r["config"][k]
        for k in ("detector_checkpoint_sha256", "pose_checkpoint_sha256", "sam_checkpoint_sha256")
    },
    "rendering": {
        "method": (
            "Deterministic Pillow rendering from existing image, binary "
            "masks and saved centroids; no AI image generation."
        ),
        "new_inference_performed": False,
        "new_accuracy_evaluation_performed": False,
        "source_masks_modified": False,
        "body_overlay": (
            "24% color fill; exact inner 4-neighbor binary-mask boundary; no bbox substitutes."
        ),
        "eye_overlay": (
            "30% color fill; exact inner 4-neighbor binary-mask "
            "boundary; no smoothing or hole filling."
        ),
        "eye_crops": crops,
        "eye_resize": "nearest-neighbor 7x",
        "distance_display_decimals": 3,
        "centroid_coordinate_convention": (
            "Original image outer top-left is (0,0); pixel centers are (column+0.5,row+0.5)."
        ),
        "renderer_sha256": sha(Path(__file__)),
        "font_family": font(20).getname(),
        "software": {"Pillow": __import__("PIL").__version__, "numpy": np.__version__},
    },
    "measurements": r["measurements"],
    "qualification": "candidate / unverified",
    "limitations": [
        "Image-plane pixel distances, not pupil or physical distances.",
        "Model side assignment, visibility and mask correctness unverified.",
        "This development illustration is not an independent accuracy result.",
        (
            "Source run files are identified by hashes but are not all "
            "included in this lightweight preview."
        ),
    ],
    "source_files": [{"path": p, "sha256": sha(RUN / p)} for p in paths],
    "figures": [
        {"file": p.name, "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(OUT.glob("*.png"))
    ],
}
(OUT / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
for p in OUT.iterdir():
    print(p.name, p.stat().st_size)
