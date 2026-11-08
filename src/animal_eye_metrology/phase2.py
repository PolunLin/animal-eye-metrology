"""Local selected-image YOLO segmentation POC; no training or changes to Phase 1 data."""

import argparse
import contextlib
import csv
import io
import json
import logging
import math
import os
import platform
import random
import sys
import tempfile
import time
import tomllib
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from animal_eye_metrology.analysis import analyze
from animal_eye_metrology.coco import Annotation, CocoDataset, ImageRecord, load_coco
from animal_eye_metrology.config import AnalysisConfig, load_config
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.review_store import load_image_selections
from animal_eye_metrology.verification import sha256_file
from animal_eye_metrology.visualization import render_instance_overlay

LOGGER = logging.getLogger(__name__)


def read_config(path: Path) -> tuple[dict[str, Any], AnalysisConfig]:
    path = path.expanduser().resolve()
    with path.open("rb") as handle:
        settings = tomllib.load(handle)
    allowed = {
        "data": {"phase1_config", "selection_csv", "feasibility_csv", "expected_selected_images"},
        "model": {"checkpoint", "source", "sha256"},
        "inference": {
            "device",
            "image_size",
            "confidence_floor",
            "nms_iou",
            "max_detections",
            "diagnostic_confidence",
            "diagnostic_iou",
            "seed",
            "threads",
            "warmup_runs",
        },
        "output": {"directory"},
    }
    if set(settings) != set(allowed) or any(
        not isinstance(settings[key], dict) or set(settings[key]) != fields
        for key, fields in allowed.items()
    ):
        raise Phase1Error("Phase 2 configuration sections/fields differ from configs/phase2.toml.")
    for section, keys in (
        ("data", ("phase1_config", "selection_csv", "feasibility_csv")),
        ("model", ("checkpoint",)),
        ("output", ("directory",)),
    ):
        for key in keys:
            value = settings[section][key]
            if not isinstance(value, str) or not value:
                raise Phase1Error(f"Invalid path: {section}.{key}")
            settings[section][key] = (path.parent / Path(value).expanduser()).resolve()
    digest = settings["model"]["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise Phase1Error("A pinned checkpoint SHA-256 is required.")
    inference = settings["inference"]
    if inference["device"] not in ("cpu", "mps"):
        raise Phase1Error("Phase 2 device must be cpu or explicitly requested mps.")
    for key in ("image_size", "max_detections", "threads", "warmup_runs", "seed"):
        value = inference[key]
        minimum = 0 if key in ("warmup_runs", "seed") else 1
        if type(value) is not int or value < minimum:
            raise Phase1Error(f"Invalid inference integer: {key}")
    for key in ("confidence_floor", "nms_iou", "diagnostic_confidence", "diagnostic_iou"):
        value = inference[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise Phase1Error(f"Invalid inference threshold: {key}")
    if inference["confidence_floor"] > inference["diagnostic_confidence"]:
        raise Phase1Error("Confidence floor cannot exceed the diagnostic confidence.")
    if not 1 <= inference["max_detections"] <= 100:
        raise Phase1Error("POC max_detections must be between 1 and evaluator cap 100.")
    count = settings["data"]["expected_selected_images"]
    if type(count) is not int or count < 1:
        raise Phase1Error("Invalid expected_selected_images.")
    settings["source"] = path
    return settings, load_config(settings["data"]["phase1_config"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def candidate_recovery(
    feasibility: list[dict[str, str]],
    matches: list[dict[str, Any]],
    image_ids: list[int],
    confidence: float,
    iou: float,
) -> list[dict[str, Any]]:
    """Report official TP recovery of pre-existing eligible IDs, without eye inference."""
    recovered = {
        (int(row["image_id"]), int(row["annotation_id"]))
        for row in matches
        if row["match_status"] == "TP" and row["annotation_id"] != ""
    }
    rows = []
    for row in feasibility:
        image_id = int(row["image_id"])
        if image_id not in image_ids:
            continue
        eligible = json.loads(row["eligible_annotation_ids"])
        found = [key for key in eligible if (image_id, key) in recovered]
        rows.append(
            {
                "image_id": image_id,
                "animal_category": row["animal_category"],
                "eligible_animal_count": len(eligible),
                "eligible_annotation_ids": eligible,
                "recovered_eligible_animals": len(found),
                "recovered_annotation_ids": found,
                "missed_annotation_ids": [key for key in eligible if key not in found],
                "at_least_two_recovered": "Yes" if len(found) >= 2 else "No",
                "confidence_threshold": confidence,
                "mask_iou_threshold": iou,
            }
        )
    return rows


def comparison_overlay(
    original: Any,
    record: ImageRecord,
    dataset: CocoDataset,
    predictions: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    path: Path,
    confidence: float,
    iou: float,
    *,
    model_label: str = "YOLO11n-seg",
) -> None:
    from PIL import Image, ImageDraw, ImageFont
    from pycocotools import mask as mask_utils

    truth = [
        ann
        for ann in dataset.annotations.values()
        if ann.image_id == record.id and ann.category_id in dataset.animal_categories
    ]
    gt_status = {
        int(row["annotation_id"]): row["match_status"] for row in matches if row["annotation_id"]
    }
    predicted_status = {
        row["prediction_id"]: row["match_status"] for row in matches if row["prediction_id"]
    }
    gt_labels = {
        ann.id: f"{dataset.categories[ann.category_id].name} GT#{ann.id} "
        f"{'crowd-ignore' if ann.iscrowd else gt_status.get(ann.id, 'FN')}"
        for ann in truth
    }
    plotted = [p for p in predictions if p["score"] >= confidence]
    pred_annotations, pred_labels = [], {}
    for index, prediction in enumerate(plotted, start=1):
        pred_annotations.append(
            Annotation(
                index,
                record.id,
                prediction["category_id"],
                0,
                tuple(prediction["bbox"]),
                float(mask_utils.area(prediction["segmentation"])),
                prediction["segmentation"],
            )
        )
        native_id = prediction["prediction_id"].rsplit("_", 1)[1]
        pred_labels[index] = (
            f"{dataset.categories[prediction['category_id']].name} P{native_id} "
            f"{prediction['score']:.3f} {predicted_status[prediction['prediction_id']]}"
        )
    panels = [
        original,
        render_instance_overlay(original, record, truth, dataset, instance_labels=gt_labels),
        render_instance_overlay(
            original, record, pred_annotations, dataset, instance_labels=pred_labels
        ),
    ]
    page = Image.new("RGB", (record.width * 3, record.height + 60), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=15)
    titles = [
        "Original",
        "COCO animal GT: GT#annotation_id",
        f"{model_label} score >= {confidence}",
    ]
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        page.paste(panel, (index * record.width, 60))
        draw.text((index * record.width + 5, 5), title, fill="black", font=font)
        draw.text(
            (index * record.width + 5, 30),
            f"Image {record.id} | TP / FP / FN at mask IoU {iou}",
            fill="black",
            font=font,
        )
    page.save(path)


def run(settings: dict[str, Any], data: AnalysisConfig, *, limit: int | None = None) -> Path:
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO

    from animal_eye_metrology.segmentation import (
        MATCH_FIELDS,
        animal_class_mapping,
        coco_predictions,
        evaluate_predictions,
        normalize_predictions,
    )

    output = settings["output"]["directory"]
    if not output.is_relative_to(data.output.parent / "phase2"):
        raise Phase1Error("Phase 2 output must be inside the existing outputs/phase2 directory.")
    if output.exists():
        raise Phase1Error("Output already exists; choose a new run directory to preserve results.")
    checkpoint = settings["model"]["checkpoint"]
    model_name = checkpoint.stem.replace("yolo", "YOLO", 1)
    if not checkpoint.is_file() or sha256_file(checkpoint) != settings["model"]["sha256"]:
        raise Phase1Error(
            "Pinned local segmentation checkpoint missing or hash differs; no download."
        )
    phase1_manifest = data.output / "run_manifest.json"
    earlier = json.loads(phase1_manifest.read_text())
    dataset = load_coco(
        data.annotations,
        expected_animal_categories=data.expected_animal_categories,
        expected_sha256=earlier["dataset"]["annotations_sha256"],
    )
    analysis = analyze(dataset)
    selections = load_image_selections(
        settings["data"]["selection_csv"], [r["image_id"] for r in analysis.candidates["multi_any"]]
    )
    selected_ids = sorted(
        int(row["image_id"]) for row in selections if row["selection_status"] == "Selected"
    )
    if len(selected_ids) != settings["data"]["expected_selected_images"]:
        raise Phase1Error("Selected image count differs from the approved POC scope.")
    image_ids = selected_ids[:limit] if limit is not None else selected_ids
    feasibility = read_csv(settings["data"]["feasibility_csv"])
    candidate_ids = [int(row["image_id"]) for row in feasibility]
    if len(candidate_ids) != len(set(candidate_ids)) or not set(candidate_ids) <= set(selected_ids):
        raise Phase1Error("Feasibility images must be unique Selected images.")
    for row in feasibility:
        valid = {ann.id for ann in analysis.eligible_by_image[int(row["image_id"])]}
        ids = json.loads(row["eligible_annotation_ids"])
        if (
            len(ids) != len(set(ids))
            or not set(ids) <= valid
            or len(ids) != int(row["eligible_animal_count"])
            or len(ids) < 2
        ):
            raise Phase1Error("Feasibility annotation IDs differ from COCO metadata.")
    protected = [path for path in data.output.rglob("*") if path.is_file()]
    protected += [
        data.annotations,
        settings["source"],
        data.source,
        checkpoint,
        Path("pyproject.toml").resolve(),
        Path("uv.lock").resolve(),
    ]
    protected += [data.images / dataset.images[key].file_name for key in image_ids]
    source_hashes = {str(path): sha256_file(path) for path in protected}
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output.parent / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from ultralytics import YOLO

    if model_name == "YOLO26s-seg" and version("ultralytics") != "8.4.0":
        raise Phase1Error("This YOLO26s POC requires the isolated Ultralytics 8.4.0 package.")
    inference = settings["inference"]
    torch.set_num_threads(inference["threads"])
    random.seed(inference["seed"])
    np.random.seed(inference["seed"])
    torch.manual_seed(inference["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    if inference["device"] == "mps" and not torch.backends.mps.is_available():
        raise Phase1Error("MPS unavailable; use the explicit CPU reference configuration.")

    started = time.perf_counter()
    model = YOLO(str(checkpoint), task="segment")
    model_load_seconds = time.perf_counter() - started
    if model.task != "segment":
        raise Phase1Error("Checkpoint is not an instance segmentation model.")

    def enforce_threads(_: Any) -> None:
        # select_device resets threads during predictor setup, before this callback.
        torch.set_num_threads(inference["threads"])

    model.add_callback("on_predict_start", enforce_threads)
    mapping = animal_class_mapping(model.names, dataset)
    predict_args = {
        "device": inference["device"],
        "imgsz": inference["image_size"],
        "conf": inference["confidence_floor"],
        "iou": inference["nms_iou"],
        "max_det": inference["max_detections"],
        "retina_masks": True,
        "rect": False,
        "half": False,
        "augment": False,
        "save": False,
        "verbose": False,
        "batch": 1,
        "agnostic_nms": False,
    }

    def sync() -> None:
        if inference["device"] == "mps":
            torch.mps.synchronize()

    first_path = data.images / dataset.images[image_ids[0]].file_name
    with Image.open(first_path) as image:
        warmup_image = image.convert("RGB")
    warmup_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(inference["warmup_runs"]):
            model.predict(source=warmup_image, **predict_args)
    sync()
    warmup_seconds = time.perf_counter() - warmup_started
    predictions, image_records = [], []
    with tempfile.TemporaryDirectory(prefix=".phase2-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        (stage / "overlays").mkdir()
        for image_id in image_ids:
            record = dataset.images[image_id]
            path = (data.images / record.file_name).resolve()
            if not path.is_relative_to(data.images):
                raise Phase1Error("Image path is outside configured data directory.")
            sync()
            started = time.perf_counter()
            with Image.open(path) as image:
                original = image.convert("RGB")
            if original.size != (record.width, record.height):
                raise Phase1Error("Original image dimensions differ from COCO.")
            with torch.inference_mode():
                result = model.predict(source=original, **predict_args)[0]
            if torch.get_num_threads() != inference["threads"]:
                raise Phase1Error("Actual CPU threads differ from the configured budget.")
            if result.orig_shape != (record.height, record.width):
                raise Phase1Error("YOLO result does not reference the expected original shape.")
            boxes = result.boxes.data.cpu().numpy()
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None
            normalized = normalize_predictions(record, boxes, masks, mapping)
            sync()
            elapsed = time.perf_counter() - started
            predictions.extend(normalized)
            image_records.append(
                {
                    "image_id": image_id,
                    "status": "success",
                    "file_name": record.file_name,
                    "width": record.width,
                    "height": record.height,
                    "sha256": sha256_file(path),
                    "animal_predictions": len(normalized),
                    "all_class_predictions": len(boxes),
                    "latency_seconds": elapsed,
                    "ultralytics_stage_ms": result.speed,
                }
            )
            LOGGER.info(json.dumps({"event": "image_inferred", **image_records[-1]}))
        with contextlib.redirect_stdout(io.StringIO()):
            truth = COCO(str(data.annotations))
        metrics, matches = evaluate_predictions(
            truth,
            predictions,
            image_ids,
            sorted(dataset.animal_categories),
            confidence=inference["diagnostic_confidence"],
            iou=inference["diagnostic_iou"],
        )
        recovery = candidate_recovery(
            feasibility,
            matches,
            image_ids,
            inference["diagnostic_confidence"],
            inference["diagnostic_iou"],
        )
        demo = [
            {
                "image_id": row["image_id"],
                "eligible_animal_count": row["eligible_animal_count"],
                "recovered_eligible_animals": row["recovered_eligible_animals"],
            }
            for row in recovery
        ]
        metrics["demonstration_candidates"] = {
            "images": demo,
            "at_least_two_recovered": sum(row["recovered_eligible_animals"] >= 2 for row in demo),
            "total_images": len(demo),
            "interpretation": "animal segmentation recovery only; no eye inference or measurement",
        }
        times = [row["latency_seconds"] for row in image_records]
        metrics["timing"] = {
            "model_load_seconds": model_load_seconds,
            "warmup_runs": inference["warmup_runs"],
            "warmup_seconds": warmup_seconds,
            "actual_threads": torch.get_num_threads(),
            "samples": len(times),
            "median_seconds": float(np.median(times)),
            "p95_seconds": float(np.percentile(times, 95)),
            "total_seconds": sum(times),
            "scope": "read/decode + inference + CPU transfer + original-coordinate validation/RLE; "
            "excludes hash, evaluation, overlays and disk export; one pass, not throughput",
        }
        write_json(stage / "predictions_coco.json", coco_predictions(predictions))
        write_json(stage / "prediction_instances.json", predictions)
        write_json(stage / "metrics.json", metrics)
        write_json(stage / "image_results.json", image_records)
        if recovery:
            with (stage / "candidate_recovery.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(recovery[0]))
                writer.writeheader()
                writer.writerows(
                    {
                        key: json.dumps(value) if isinstance(value, list) else value
                        for key, value in row.items()
                    }
                    for row in recovery
                )
        with (stage / "instance_matches.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS)
            writer.writeheader()
            writer.writerows(matches)
        for image_id in image_ids:
            record = dataset.images[image_id]
            with Image.open(data.images / record.file_name) as opened:
                comparison_overlay(
                    opened.convert("RGB"),
                    record,
                    dataset,
                    [p for p in predictions if p["image_id"] == image_id],
                    [r for r in matches if r["image_id"] == image_id],
                    stage / "overlays" / f"{image_id:012d}.png",
                    inference["diagnostic_confidence"],
                    inference["diagnostic_iou"],
                    model_label=model_name,
                )
        if any(sha256_file(Path(path)) != digest for path, digest in source_hashes.items()):
            raise Phase1Error("A protected source changed during inference; outputs not published.")
        root = Path(__file__).resolve().parents[2]
        manifest = {
            "status": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": {
                "declared_version": "COCO 2017 val2017",
                "source": str(data.annotations),
                "sha256": dataset.source_sha256,
                "independent_authenticity_verified": False,
            },
            "selected_image_ids": image_ids,
            "approved_selected_ids": selected_ids,
            "scope": "full_15_image_POC" if limit is None else "limited_smoke_test",
            "model": {
                "name": model_name,
                "head": type(model.model.model[-1]).__name__,
                "end2end": bool(getattr(model.model.model[-1], "end2end", False)),
                "checkpoint": str(checkpoint),
                "source": settings["model"]["source"],
                "sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "license": "AGPL-3.0 or Enterprise",
                "license_url": "https://www.ultralytics.com/license",
                "upstream_reference_digest_verified": False,
                "data_provenance": "official COCO pretrained; train/val recipe separation; "
                "checkpoint training history not independently audited; "
                "val2017 is not an untouched independent test",
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "torchvision",
                        "ultralytics",
                        "numpy",
                        "Pillow",
                        "pycocotools",
                    )
                },
            },
            "configuration": settings,
            "predict_args": predict_args,
            "class_mapping": {
                str(k): {"name": model.names[k], "category_id": v} for k, v in mapping.items()
            },
            "mask_coordinates": "retina_masks=True: native binary masks already at original HxW; "
            "no polygon approximation or application-side resizing",
            "evaluation": {
                "api": "pycocotools.COCOeval(segm)",
                "categories": "10 animal classes",
                "crowd": "official ignore retained",
                "iou_thresholds": ".50:.05:.95",
                "recall_thresholds": "0:.01:1",
                "max_dets": [1, 10, 100],
                "diagnostic": metrics["diagnostic"]["matching"],
            },
            "training": False,
            "eye_analysis": False,
            "actual_torch_threads": torch.get_num_threads(),
            "source_files_unchanged": True,
            "input_sha256": source_hashes,
            "source_sha256": {
                str(path.relative_to(root)): sha256_file(path)
                for path in [
                    root / "pyproject.toml",
                    root / "uv.lock",
                    root / "docs/phase2_instance_segmentation.md",
                    *sorted(Path(__file__).parent.glob("*.py")),
                ]
            },
            "images": image_records,
            "execution": {"argv": sys.argv, "working_directory": str(Path.cwd())},
            "artifacts_sha256": {
                str(path.relative_to(stage)): sha256_file(path)
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
        }
        # Paths are retained as strings in the record, never encoded as Python objects.
        manifest["configuration"] = json.loads(json.dumps(settings, default=str))
        write_json(stage / "run_manifest.json", manifest)
        stage.rename(output)
    LOGGER.info(
        json.dumps(
            {
                "event": "phase2_complete",
                "output": str(output),
                "images": len(image_ids),
                "predictions": len(predictions),
            }
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2.toml"))
    parser.add_argument("--limit", type=int, help="First N selected IDs for an explicit smoke test")
    parser.add_argument("--output", type=Path, help="New directory under outputs/phase2")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        settings, data = read_config(args.config)
        if args.output is not None:
            settings["output"]["directory"] = args.output.expanduser().resolve()
        if (
            args.limit is not None
            and not 1 <= args.limit <= settings["data"]["expected_selected_images"]
        ):
            raise Phase1Error("Smoke-test limit must be within the selected image count.")
        run(settings, data, limit=args.limit)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        LOGGER.error(json.dumps({"event": "phase2_failed", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
