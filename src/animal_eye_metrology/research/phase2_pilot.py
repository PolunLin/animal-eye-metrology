"""Fixed shared sample, sequential CPU inference and transparent runtime extrapolation."""

import argparse
import contextlib
import csv
import io
import json
import logging
import os
import platform
import random
import resource
import sys
import time
import tomllib
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.config import load_config
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.data.analysis import analyze
from animal_eye_metrology.data.coco import load_coco
from animal_eye_metrology.paths import module_source
from animal_eye_metrology.research.phase2 import read_csv, write_json

LOGGER = logging.getLogger(__name__)


def sample_ids(population: list[int], size: int, seed: int) -> list[int]:
    """Draw without replacement from sorted IDs; return the random draw order."""
    if len(set(population)) != len(population) or not 1 <= size <= len(population):
        raise Phase1Error("Invalid sampling population or size.")
    return random.Random(seed).sample(sorted(population), size)


def clip_box_roundoff(boxes: Any, width: int, height: int) -> tuple[Any, int]:
    """Clip only <=0.0001 px float32 resize roundoff; reject real coordinate errors."""
    import numpy as np

    coords = boxes[:, :4]
    limits = np.array([width, height, width, height], dtype=np.float64)
    if (
        not np.isfinite(coords).all()
        or np.any(coords < -0.0001)
        or np.any(coords > limits + 0.0001)
    ):
        raise Phase1Error("Box exceeds original bounds beyond float32 roundoff tolerance.")
    clipped = boxes.copy()
    clipped[:, :4] = np.clip(coords, 0, limits)
    return clipped, int(np.any(clipped[:, :4] != coords, axis=1).sum())


def forecast(rows: list[dict[str, Any]], population: dict[str, int]) -> dict[str, Any]:
    """Estimate read/decode/infer/RLE duration, with explicit population weighting."""
    if not rows or any(float(row["latency_seconds"]) <= 0 for row in rows):
        raise Phase1Error("Runtime forecast requires positive measured durations.")
    total = sum(float(row["latency_seconds"]) for row in rows)
    groups = {}
    weighted = 0.0
    for name in ("animal", "background"):
        times = [float(r["latency_seconds"]) for r in rows if r["group"] == name]
        groups[name] = {"sample_count": len(times), "mean_seconds": None}
        if times:
            groups[name]["mean_seconds"] = sum(times) / len(times)
            weighted += population[name] * sum(times) / len(times)
        elif population[name]:
            weighted = float("nan")
    return {
        "linear_seconds": sum(population.values()) * total / len(rows),
        "population_weighted_seconds": weighted if weighted == weighted else None,
        "strata": groups,
        "scope": "read/decode + native predict + original-coordinate validation and animal RLE; "
        "excludes setup, hashing, CSV/JSON export, COCO evaluation and overlays",
        "uncertainty": "one pass on one machine; no confidence interval; thermal/load sensitive",
    }


def settings_at(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open("rb") as handle:
        cfg = tomllib.load(handle)
    for key in ("phase1_config", "output"):
        cfg[key] = (path.parent / cfg[key]).resolve()
    for model in cfg["models"]:
        model["checkpoint"] = (path.parent / model["checkpoint"]).resolve()
        if Path(model["name"]).name != model["name"]:
            raise Phase1Error("Model name must be a directory basename.")
    if cfg["sample_size"] != 100 or cfg["seed"] != 42:
        raise Phase1Error("This approved pilot requires exactly 100 images and seed 42.")
    if type(cfg["threads"]) is not int or cfg["threads"] < 1:
        raise Phase1Error("CPU thread count must be positive.")
    if type(cfg["warmup_runs"]) is not int or cfg["warmup_runs"] < 1:
        raise Phase1Error("At least one warmup is required.")
    if not 0 < cfg["confidence_floor"] <= 1 or not 1 <= cfg["max_detections"] <= 100:
        raise Phase1Error("Invalid confidence or detection limit.")
    data = load_config(cfg["phase1_config"])
    if not cfg["output"].is_relative_to(data.output.parent / "phase2"):
        raise Phase1Error("Pilot artifacts must stay inside outputs/phase2.")
    cfg["source"] = path
    return cfg


def prepare(cfg: dict[str, Any]) -> None:
    data = load_config(cfg["phase1_config"])
    earlier = json.loads((data.output / "run_manifest.json").read_text())
    dataset = load_coco(data.annotations, expected_sha256=earlier["dataset"]["annotations_sha256"])
    if len(dataset.images) != 5000:
        raise Phase1Error("Expected the complete COCO val2017 population of 5,000 images.")
    analysis = analyze(dataset)
    animals = analysis.eligible_by_image
    drawn = sample_ids(list(dataset.images), cfg["sample_size"], cfg["seed"])
    ranks = {key: index + 1 for index, key in enumerate(drawn)}
    rows = []
    for key in sorted(drawn):
        record = dataset.images[key]
        image_path = data.images / record.file_name
        rows.append(
            {
                "image_id": key,
                "draw_rank": ranks[key],
                "file_name": record.file_name,
                "width": record.width,
                "height": record.height,
                "animal_count": len(animals.get(key, [])),
                "group": "animal" if animals.get(key) else "background",
                "sha256": sha256_file(image_path),
            }
        )
    output = cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "sample_manifest.json"
    sample_path = output / "sampled_images.csv"
    if manifest_path.exists() or sample_path.exists():
        raise Phase1Error("Shared sample already exists; refusing to replace it.")
    with sample_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    positive = sum(bool(animals.get(key)) for key in dataset.images)
    write_json(
        manifest_path,
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": "COCO 2017 val2017",
            "annotations_sha256": dataset.source_sha256,
            "population": {"animal": positive, "background": len(dataset.images) - positive},
            "sample_size": len(rows),
            "seed": cfg["seed"],
            "method": "Python random.Random(42).sample(sorted(all_5000_image_ids), 100); "
            "uniform without replacement; inference in sorted image_id order",
            "sample_groups": {
                g: sum(r["group"] == g for r in rows) for g in ("animal", "background")
            },
            "animal_group_definition": "at least one non-crowd animal annotation; "
            "crowd-only images remain in the other group",
            "multi_animal_images": sum(r["animal_count"] >= 2 for r in rows),
            "sampled_images_sha256": sha256_file(sample_path),
            "pilot_config_sha256": sha256_file(cfg["source"]),
        },
    )
    print(json.dumps({"event": "sample_prepared", "images": len(rows), "output": str(output)}))


def run(cfg: dict[str, Any], name: str) -> None:
    # Heavy imports follow per-process cache configuration; no checkpoint downloads.
    model_cfg = next((m for m in cfg["models"] if m["name"] == name), None)
    if model_cfg is None:
        raise Phase1Error(f"Unknown pilot model: {name}")
    output = cfg["output"] / name
    output.mkdir(exist_ok=False)
    for env, subdir in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib")):
        cache = cfg["output"] / "cache" / name / subdir
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(cache)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import numpy as np
    import torch
    from PIL import Image
    from pycocotools.coco import COCO

    from animal_eye_metrology.models.segmentation import (
        MATCH_FIELDS,
        animal_class_mapping,
        coco_predictions,
        evaluate_predictions,
        normalize_predictions,
    )

    setup_start = time.perf_counter()
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = json.loads((cfg["output"] / "sample_manifest.json").read_text())
    sample_path = cfg["output"] / "sampled_images.csv"
    if (
        sha256_file(sample_path) != manifest["sampled_images_sha256"]
        or sha256_file(cfg["source"]) != manifest["pilot_config_sha256"]
    ):
        raise Phase1Error("Shared sample or configuration changed after preparation.")
    rows = read_csv(sample_path)
    ids = [int(row["image_id"]) for row in rows]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise Phase1Error("Pilot requires 100 distinct shared images.")
    data = load_config(cfg["phase1_config"])
    dataset = load_coco(data.annotations, expected_sha256=manifest["annotations_sha256"])
    for row in rows:
        if sha256_file(data.images / row["file_name"]) != row["sha256"]:
            raise Phase1Error("Sample source image hash differs from preparation.")
    checkpoint = model_cfg["checkpoint"]
    if sha256_file(checkpoint) != model_cfg["sha256"]:
        raise Phase1Error("Local checkpoint differs from pinned hash.")
    package = model_cfg["backend"]
    if version(package) != model_cfg["package_version"]:
        raise Phase1Error(f"{name} requires {package}=={model_cfg['package_version']}.")
    started = time.perf_counter()
    roundoff_clips = 0
    if package == "ultralytics":
        from ultralytics import YOLO

        model = YOLO(str(checkpoint), task="segment")
        if model.task != "segment":
            raise Phase1Error("Expected an instance segmentation checkpoint.")
        mapping = animal_class_mapping(model.names, dataset)
        head = model.model.model[-1]
        native = {
            "input": "640x640 letterbox",
            "head": type(head).__name__,
            "end2end": bool(getattr(head, "end2end", False)),
            "nms_iou": 0.7,
            "retina_masks": True,
        }

        def predict(original: Any) -> tuple[np.ndarray, Any]:
            result = model.predict(
                source=original,
                device="cpu",
                imgsz=640,
                conf=cfg["confidence_floor"],
                iou=0.7,
                max_det=cfg["max_detections"],
                retina_masks=True,
                rect=False,
                half=False,
                augment=False,
                save=False,
                verbose=False,
                batch=1,
                agnostic_nms=False,
            )[0]
            if result.orig_shape != (original.height, original.width):
                raise Phase1Error("Native YOLO output is not in original image coordinates.")
            return (
                result.boxes.data.cpu().numpy(),
                result.masks.data.cpu().numpy() if result.masks is not None else None,
            )

    else:
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )
        from torchvision.transforms.functional import to_tensor

        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            box_score_thresh=cfg["confidence_floor"],
            box_nms_thresh=0.5,
            box_detections_per_img=cfg["max_detections"],
            min_size=800,
            max_size=1333,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.cpu().eval()
        mapping = animal_class_mapping(
            dict(enumerate(MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"])), dataset
        )
        native = {
            "input": "short side 800, long side <=1333; native aspect ratio",
            "box_nms_iou": 0.5,
            "head": "MaskRCNN",
            "end2end": False,
        }

        def predict(original: Any) -> tuple[np.ndarray, Any]:
            nonlocal roundoff_clips
            result = model([to_tensor(original)])[0]
            boxes = (
                torch.cat(
                    (result["boxes"], result["scores"][:, None], result["labels"][:, None]), dim=1
                )
                .cpu()
                .numpy()
            )
            boxes, clipped_count = clip_box_roundoff(boxes, original.width, original.height)
            roundoff_clips += clipped_count
            return boxes, result["masks"][:, 0].cpu().numpy()

    load_seconds = time.perf_counter() - started
    first = dataset.images[ids[0]]
    with Image.open(data.images / first.file_name) as opened:
        warmup_image = opened.convert("RGB")
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(cfg["warmup_runs"]):
            predict(warmup_image)
            # Ultralytics resets CPU threads when its predictor is initialized.
            # Restore the shared budget before subsequent warmups and timed images.
            torch.set_num_threads(cfg["threads"])
    warmup_seconds = time.perf_counter() - started
    roundoff_clips = 0  # Count only the 100 timed images, not warmup predictions.
    setup_seconds = time.perf_counter() - setup_start
    records, predictions = [], []
    inference_start = time.perf_counter()

    # Each helper call releases full-resolution mask tensors before the next image.
    def infer_one(row: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        record = dataset.images[int(row["image_id"])]
        start = time.perf_counter()
        with Image.open(data.images / record.file_name) as opened:
            original = opened.convert("RGB")
        if original.size != (record.width, record.height):
            raise Phase1Error("Image dimensions differ from COCO metadata.")
        native_start = time.perf_counter()
        boxes, masks = predict(original)
        if torch.get_num_threads() != cfg["threads"]:
            raise Phase1Error("Backend changed the shared CPU thread budget during timing.")
        native_seconds = time.perf_counter() - native_start
        normalized = normalize_predictions(record, boxes, masks, mapping)
        return normalized, {
            "image_id": record.id,
            "status": "success",
            "group": row["group"],
            "animal_count_gt": int(row["animal_count"]),
            "animal_predictions": len(normalized),
            "all_class_predictions": len(boxes),
            "latency_seconds": time.perf_counter() - start,
            "native_predict_seconds": native_seconds,
        }

    with (output / "progress.jsonl").open("x", encoding="utf-8") as progress:
        with torch.inference_mode():
            for index, row in enumerate(rows, start=1):
                normalized, record = infer_one(row)
                predictions.extend(normalized)
                records.append(record)
                progress.write(json.dumps(record) + "\n")
                progress.flush()
                if index % 10 == 0:
                    LOGGER.info(json.dumps({"model": name, "completed": index, "total": 100}))
    inference_wall_seconds = time.perf_counter() - inference_start
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(data.annotations))
    metrics, matches = evaluate_predictions(
        truth, predictions, ids, sorted(dataset.animal_categories)
    )
    metrics["scope"] = "fixed_100_random_compatibility_and_timing_pilot_not_full_benchmark"
    evaluation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    write_json(output / "prediction_instances.json", predictions)
    write_json(output / "predictions_coco.json", coco_predictions(predictions))
    write_json(output / "image_results.json", records)
    write_json(output / "metrics.json", metrics)
    (output / "pilot_runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    with (output / "instance_matches.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS)
        writer.writeheader()
        writer.writerows(matches)
    export_seconds = time.perf_counter() - started
    times = [r["latency_seconds"] for r in records]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "status": "complete",
        "model": name,
        "label": model_cfg["label"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "images_completed": len(records),
        "image_ids": ids,
        "sample_sha256": sha256_file(sample_path),
        "annotations_sha256": dataset.source_sha256,
        "source_sha256": {
            path.name: sha256_file(path)
            for path in (
                cfg["source"],
                Path(__file__),
                module_source("segmentation.py"),
                module_source("paths.py"),
                module_source("_compat.py"),
            )
        },
        "checkpoint": {
            "file": checkpoint.name,
            "sha256": model_cfg["sha256"],
            "bytes": checkpoint.stat().st_size,
            "source": model_cfg["source"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": platform.processor(),
            "packages": {
                p: version(p)
                for p in ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "pycocotools")
            },
        },
        "runtime": {
            "device": "cpu",
            "precision": "FP32",
            "batch": 1,
            "threads": torch.get_num_threads(),
            "interop_threads": 1,
            "seed": cfg["seed"],
            "deterministic_algorithms_warn_only": True,
            "warmup_runs": cfg["warmup_runs"],
            "native": native,
            "confidence_floor": cfg["confidence_floor"],
            "max_detections_all_classes": cfg["max_detections"],
            "mask_threshold": 0.5,
            "box_roundoff_clipped_predictions": roundoff_clips,
            "torchvision_box_roundoff_tolerance_pixels": 0.0001,
        },
        "timing": {
            "setup_seconds": setup_seconds,
            "model_load_seconds": load_seconds,
            "warmup_seconds": warmup_seconds,
            "sum_pipeline_seconds": sum(times),
            "inference_loop_wall_seconds": inference_wall_seconds,
            "mean_seconds": float(np.mean(times)),
            "median_seconds": float(np.median(times)),
            "p95_seconds": float(np.percentile(times, 95)),
            "evaluation_seconds": evaluation_seconds,
            "export_seconds": export_seconds,
            "peak_process_rss_mib": rss / (1024**2 if sys.platform == "darwin" else 1024),
        },
        "forecast_5000": forecast(records, manifest["population"]),
        "interpretation": "compatibility and timing only; COCO sample AP is a pipeline check; "
        "not an independent held-out or full-dataset accuracy conclusion",
    }
    write_json(output / "run_manifest.json", result)
    print(
        json.dumps(
            {
                "event": "pilot_complete",
                "model": name,
                "timing": result["timing"],
                "forecast_5000": result["forecast_5000"],
            }
        )
    )


def summarize(cfg: dict[str, Any]) -> None:
    results = []
    sample_hash = sha256_file(cfg["output"] / "sampled_images.csv")
    ids = [int(row["image_id"]) for row in read_csv(cfg["output"] / "sampled_images.csv")]
    for model in cfg["models"]:
        result = json.loads((cfg["output"] / model["name"] / "run_manifest.json").read_text())
        if (
            result["status"] != "complete"
            or result["images_completed"] != 100
            or result["sample_sha256"] != sample_hash
            or result["image_ids"] != ids
        ):
            raise Phase1Error("Incomplete model run or inconsistent shared sample.")
        results.append(
            {
                "model": model["label"],
                "images_completed": result["images_completed"],
                **result["timing"],
                **{
                    f"forecast_5000_{k}": v
                    for k, v in result["forecast_5000"].items()
                    if k.endswith("seconds")
                },
            }
        )
    target = cfg["output"] / "comparison.csv"
    if target.exists():
        raise Phase1Error("Comparison already exists; refusing to overwrite results.")
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json(cfg["output"] / "comparison.json", results)
    print(json.dumps(results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_pilot.toml"))
    parser.add_argument("--model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        cfg = settings_at(args.config)
        if args.action == "prepare":
            prepare(cfg)
        elif args.action == "run":
            if not args.model:
                raise Phase1Error("run requires --model")
            run(cfg, args.model)
        else:
            summarize(cfg)
    except (OSError, ValueError, Phase1Error, RuntimeError) as error:
        LOGGER.exception("Pilot failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
