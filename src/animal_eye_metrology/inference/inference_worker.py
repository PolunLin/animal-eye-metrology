"""Isolated CPU model workers for uploaded images, without annotations or manual prompts."""

import argparse
import json
import logging
import math
import os
import platform
import random
import re
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.inference.inference_capabilities import (
    COCO_ANIMAL_CATEGORIES,
    capability_policy,
    category_capability,
    validate_category_scope,
)

LOGGER = logging.getLogger(__name__)


def _save_json(path: Path, value: Any) -> None:
    """Refuse to replace the artifacts of an already executed stage."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _event(name: str, **details: Any) -> None:
    LOGGER.info(json.dumps({"event": name, **details}, allow_nan=False))


def _common(request: dict[str, Any]) -> tuple[Path, Path, Path]:
    if request.get("seed") != 42 or request.get("threads") != 4:
        raise Phase1Error("The approved CPU policy requires seed 42 and four threads.")
    image, output, checkpoint = (Path(request[key]) for key in ("image", "output", "checkpoint"))
    if not image.is_file() or not output.is_dir() or not checkpoint.is_file():
        raise Phase1Error(
            "Input image, existing output directory, and local checkpoint are required."
        )
    if sha256_file(checkpoint) != request["checkpoint_sha256"]:
        raise Phase1Error("Model checkpoint SHA-256 differs from configuration.")
    return image, output, checkpoint


def _configure_torch() -> Any:
    # Only the subprocess imports model runtimes; Streamlit and unit tests stay lightweight.
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    return torch


def detection_records(
    boxes: np.ndarray,
    masks: np.ndarray | None,
    names: dict[int, str],
    width: int,
    height: int,
    output: Path,
    threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    """Retain below-threshold/unsupported animals; bad instance masks do not invent boxes."""
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise Phase1Error("Detector returned invalid box, confidence, or class arrays.")
    if any(int(label) != label or int(label) not in names for label in boxes[:, 5]):
        raise Phase1Error("Detector returned an unknown or fractional class index.")
    if np.any((boxes[:, 4] < 0) | (boxes[:, 4] > 1)):
        raise Phase1Error("Detector confidence is outside [0, 1].")
    masks_valid = (
        masks is not None
        and masks.shape == (len(boxes), height, width)
        and np.isfinite(masks).all()
        and np.isin(masks, (0, 1)).all()
    )
    directory = output / "animal_masks"
    directory.mkdir(exist_ok=True)
    results: list[dict[str, Any]] = []
    nonanimal_count = 0
    for index, box in enumerate(boxes):
        class_id = int(box[5])
        category = names[class_id]
        if category not in COCO_ANIMAL_CATEGORIES:
            nonanimal_count += 1
            continue
        x0, y0, x1, y1, score = map(float, box[:5])
        valid_box = 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        accepted = score >= threshold and valid_box
        capability = category_capability(category)
        supported = capability["supported"]
        record: dict[str, Any] = {
            "prediction_id": f"animal_{index + 1:04d}",
            "category": category,
            "class_id": class_id,
            "score": score,
            "bbox_xyxy": [x0, y0, x1, y1],
            "accepted": accepted,
            **capability,
            "status": "completed" if supported else "unsupported_category",
            "failure_reason": "" if supported else capability["capability_reason"],
            "mask_path": None,
            "mask_status": "not_run",
            "mask_failure_reason": "",
            "mask_area_px": None,
        }
        if not valid_box:
            record.update(status="failed", failure_reason="invalid_animal_box")
        elif not accepted:
            record.update(status="below_threshold", failure_reason="animal_score_below_threshold")
        else:
            try:
                if not masks_valid:
                    raise Phase1Error("missing_or_invalid_instance_masks")
                mask = masks[index].astype(np.uint8) * 255
                path = directory / f"{record['prediction_id']}.png"
                if path.exists():
                    raise Phase1Error("Instance mask artifact already exists.")
                Image.fromarray(mask).save(path)
                area = int(np.count_nonzero(mask))
                record.update(
                    mask_path=str(path.relative_to(output)),
                    mask_status="completed" if area else "empty_mask",
                    mask_failure_reason="" if area else "empty_instance_mask",
                    mask_area_px=area,
                )
            except Exception as exc:
                record.update(
                    mask_status="failed", mask_failure_reason=f"{type(exc).__name__}: {exc}"
                )
        results.append(record)
    return results, nonanimal_count


def detection(request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    image_path, output, checkpoint = _common(request)
    expected = {
        "package_version": "8.4.0",
        "image_size": 640,
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "max_detections": 100,
        "animal_score_threshold": 0.5,
    }
    if any(request.get(key) != value for key, value in expected.items()):
        raise Phase1Error("Detector settings differ from the approved YOLO26s baseline.")
    validate_category_scope(request.get("supported_categories"))
    if version("ultralytics") != "8.4.0":
        raise Phase1Error(
            "YOLO26s requires the existing isolated Ultralytics 8.4.0 target directory."
        )
    for key, relative in (("YOLO_CONFIG_DIR", "yolo_cache"), ("MPLCONFIGDIR", "mpl_cache")):
        directory = output / relative
        directory.mkdir(exist_ok=True)
        os.environ[key] = str(directory)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    torch = _configure_torch()
    from ultralytics import YOLO

    torch.use_deterministic_algorithms(True, warn_only=True)
    load_started = time.perf_counter()
    model = YOLO(str(checkpoint), task="segment")
    load_seconds = time.perf_counter() - load_started
    if model.task != "segment":
        raise Phase1Error("Checkpoint is not an instance segmentation model.")

    def enforce_threads(_: Any) -> None:
        torch.set_num_threads(4)

    model.add_callback("on_predict_start", enforce_threads)
    with Image.open(image_path) as image:
        original = image.convert("RGB")
    predict_started = time.perf_counter()
    with torch.inference_mode():
        predictions = model.predict(
            source=original,
            device="cpu",
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=100,
            retina_masks=True,
            rect=False,
            half=False,
            augment=False,
            save=False,
            verbose=False,
            batch=1,
            agnostic_nms=False,
        )
    inference_seconds = time.perf_counter() - predict_started
    if len(predictions) != 1 or torch.get_num_threads() != 4:
        raise Phase1Error("Unexpected detector result count or CPU thread count.")
    result = predictions[0]
    if tuple(result.orig_shape) != (original.height, original.width):
        raise Phase1Error("Detector coordinates do not match the original image dimensions.")
    boxes = result.boxes.data.cpu().numpy()
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    results, nonanimal_count = detection_records(
        boxes, masks, model.names, original.width, original.height, output, 0.5
    )
    value = {
        "status": "completed",
        "results": results,
        "all_class_predictions": len(boxes),
        "nonanimal_predictions": nonanimal_count,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "image_sha256": sha256_file(image_path),
        "model": "YOLO26s-seg",
        "capability_policy": capability_policy(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": version("ultralytics"),
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "stage_wall_seconds": time.perf_counter() - started,
            "warmup_runs": 0,
        },
        "cpu_threads": 4,
        "seed": 42,
        "device": "cpu",
    }
    _save_json(output / "detection.json", value)
    _event(
        "detection_completed", animals=len(results), accepted=sum(r["accepted"] for r in results)
    )
    return value


def segment_case(
    predictor: Any, image: np.ndarray, case: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Apply the original point-only argmax-quality SAM rule to one frozen request."""
    started = time.perf_counter()
    record = {
        **case,
        "status": "failed",
        "failure_reason": "",
        "mask_path": None,
        "candidate_scores": None,
        "selected_candidate": None,
        "inference_seconds": None,
    }
    try:
        if not re.fullmatch(r"animal_[0-9]{4}", case["prediction_id"]):
            raise Phase1Error("Invalid prediction identifier.")
        if case["eye_side"] not in ("left", "right"):
            raise Phase1Error("Invalid anatomical eye side.")
        if case["condition_id"] not in ("A_r100", "A_r050"):
            raise Phase1Error("Only A_r100 and the A_r050 comparison are supported.")
        roi = case["roi_xyxy"]
        if len(roi) != 4 or any(type(value) is not int for value in roi):
            raise Phase1Error("ROI must contain four integer pixel edges.")
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
            raise Phase1Error("ROI is outside the original image.")
        point = case["point_xy"]
        if len(point) != 2 or not all(math.isfinite(v) for v in point):
            raise Phase1Error("Nonfinite or invalid eye point.")
        if not (x0 <= point[0] < x1 and y0 <= point[1] < y1):
            raise Phase1Error("Eye point is outside its ROI; no prompt repair is allowed.")
        local_point = np.asarray([point], dtype=np.float32) - [x0, y0]
        predictor.set_image(image[y0:y1, x0:x1].copy())
        logits, scores, low_res = predictor.predict(
            point_coords=local_point,
            point_labels=np.array([1]),
            box=None,
            multimask_output=True,
            return_logits=True,
        )
        if (
            logits.shape != (3, y1 - y0, x1 - x0)
            or scores.shape != (3,)
            or not all(np.isfinite(value).all() for value in (logits, scores, low_res))
        ):
            raise Phase1Error("SAM returned invalid logits/scores.")
        selected = int(np.argmax(scores))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y0:y1, x0:x1] = (logits[selected] > 0).astype(np.uint8) * 255
        directory = output / "eye_masks"
        directory.mkdir(exist_ok=True)
        path = directory / f"{case['prediction_id']}_{case['eye_side']}_{case['condition_id']}.png"
        if path.exists():
            raise Phase1Error("Eye mask artifact already exists.")
        Image.fromarray(mask).save(path)
        area = int(np.count_nonzero(mask))
        record.update(
            status="completed" if area else "empty_mask",
            failure_reason="" if area else "empty_eye_mask",
            mask_path=str(path.relative_to(output)),
            mask_area_px=area,
            candidate_scores=scores.tolist(),
            selected_candidate=selected,
        )
    except Exception as exc:
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Eye segmentation failed for %s", case.get("prediction_id"))
    record["inference_seconds"] = time.perf_counter() - started
    return record


def segmentation(request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    image_path, output, checkpoint = _common(request)
    from animal_eye_metrology.models.sam_runtime import load_predictor, settings, validate_source

    runtime = settings(Path(request["sam_runtime"]))
    spec = runtime["models"]["sam2_1_tiny"]
    if checkpoint.resolve() != (runtime["weights_directory"] / spec["checkpoint"]).resolve():
        raise Phase1Error("SAM checkpoint differs from the approved runtime.")
    source = validate_source(runtime, spec)
    torch = _configure_torch()
    load_started = time.perf_counter()
    predictor = load_predictor("sam2_1_tiny", checkpoint)
    load_seconds = time.perf_counter() - load_started
    with Image.open(image_path) as image_file:
        image = np.array(image_file.convert("RGB"))
    with torch.inference_mode():
        results = [segment_case(predictor, image, case, output) for case in request["cases"]]
    value = {
        "status": "completed"
        if all(r["status"] == "completed" for r in results)
        else "completed_with_issues",
        "results": results,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "image_sha256": sha256_file(image_path),
        "model": "SAM 2.1 Tiny",
        "sam_source": source,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "SAM-2": version("SAM-2"),
        },
        "timing": {
            "model_load_seconds": load_seconds,
            "stage_wall_seconds": time.perf_counter() - started,
        },
        "cpu_threads": 4,
        "seed": 42,
        "device": "cpu",
        "point_only": True,
        "threshold": 0,
        "postprocessing": "none",
        "dtype": "float32",
        "selection": "highest_model_quality_score",
    }
    _save_json(output / "segmentation.json", value)
    _event("segmentation_completed", eyes=len(results), status=value["status"])
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("detection", "segmentation"))
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    request: dict[str, Any] = {}
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        (detection if args.stage == "detection" else segmentation)(request)
        return 0
    except Exception as exc:
        error = {
            "stage": args.stage,
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
        LOGGER.exception(json.dumps({"event": "worker_failed", **error}))
        output = Path(request.get("output", ""))
        if request.get("output") and output.is_dir():
            error_path = output / f"{args.stage}_error.json"
            if not error_path.exists():
                _save_json(error_path, error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
