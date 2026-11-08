"""Single-image CPU inference, isolated from frozen research references and annotations."""

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
import time
import tomllib
import uuid
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.execution_lock import inference_execution_lock
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.inference.inference_artifacts import (
    build_measurements,
    csv_bytes,
    render_overlays,
)
from animal_eye_metrology.inference.inference_capabilities import (
    capability_policy,
    category_capability,
    validate_category_scope,
)
from animal_eye_metrology.measurement.eye_geometry import eye_crops, roi_for_case
from animal_eye_metrology.measurement.metrology import mask_centroid
from animal_eye_metrology.paths import (
    PACKAGE_ROOT,
    PROJECT_ROOT,
    SOURCE_ROOT,
    module_source,
    package_sources,
)

LOGGER = logging.getLogger(__name__)
ROOT = PROJECT_ROOT
STAGES = ("detection", "localization", "segmentation", "measurement")
PATH_FIELDS = (
    "output_directory",
    "python",
    "pose_python",
    "detector_source",
    "detector_checkpoint",
    "pose_checkpoint",
    "pose_source_manifest",
    "sam_runtime",
    "sam_checkpoint",
)


def load_config(path: Path) -> dict[str, Any]:
    """Resolve configurable paths without dereferencing virtual-environment interpreters."""
    path = path.resolve()
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in PATH_FIELDS:
        cfg[key] = Path(os.path.abspath(path.parent / cfg[key]))
    cfg["source"] = path
    validate_category_scope(cfg["supported_categories"])
    if (
        cfg["device"] != "cpu"
        or cfg["seed"] != 42
        or cfg["threads"] != 4
        or cfg["animal_score_threshold"] != 0.5
        or cfg["image_size"] != 640
        or cfg["confidence_floor"] != 0.001
        or cfg["nms_iou"] != 0.7
        or cfg["max_detections"] != 100
    ):
        raise ValueError("Configuration changes the approved CPU/model/category/threshold scope.")
    for key in ("max_upload_bytes", "max_image_pixels", "stage_timeout_seconds"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    return cfg


def preflight(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Lightweight local readiness check; workers verify package/source/weight identity again."""
    rows = []
    for field in PATH_FIELDS[1:]:
        path = cfg[field]
        ok = path.is_dir() if field == "detector_source" else path.is_file()
        reason = "Local file available; runtime validation occurs during execution."
        if field in ("python", "pose_python"):
            ok = ok and os.access(path, os.X_OK)
        if ok and field.endswith("_checkpoint"):
            ok = sha256_file(path) == cfg[field + "_sha256"]
            reason = "Checkpoint SHA-256 verified." if ok else "Checkpoint SHA-256 mismatch."
        if not path.exists():
            reason = f"Missing local dependency: {path}"
        rows.append({"component": field, "status": "ready" if ok else "missing", "detail": reason})
    return rows


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_result(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def normalize_image(data: bytes, cfg: dict[str, Any]) -> tuple[Image.Image, str]:
    """Decode JPEG/PNG, apply EXIF orientation, composite transparency on white; no resizing."""
    if not data or len(data) > cfg["max_upload_bytes"]:
        raise ValueError("圖片為空或超過上傳大小限制。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ("JPEG", "PNG"):
                    raise ValueError("僅接受實際內容為 JPEG 或 PNG 的圖片。")
                if source.width * source.height > cfg["max_image_pixels"]:
                    raise ValueError("圖片像素數超過限制；請先提供較小的原圖。")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("請提供單張靜態圖片。")
                suffix = ".jpg" if source.format == "JPEG" else ".png"
                image = ImageOps.exif_transpose(source)
                if "A" in image.getbands() or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", image.size, "white")
                    image = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    image = image.convert("RGB")
                image.load()
                return image, suffix
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("無法解碼圖片；請確認檔案是完整且可開啟的 JPEG／PNG 圖片。") from error


def _run_worker(
    cfg: dict[str, Any],
    run_dir: Path,
    stage: str,
    request: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    request_path = run_dir / f"{stage}_request.json"
    _save(request_path, request)
    if stage == "localization":
        command = [
            str(cfg["pose_python"]),
            str(module_source("eye_pose_worker.py")),
            "--request",
            str(request_path),
        ]
        expected = run_dir / "pose/pose_results.json"
    else:
        command = [
            str(cfg["python"]),
            "-m",
            "animal_eye_metrology.inference.inference_worker",
            "--stage",
            stage,
            "--request",
            str(request_path),
        ]
        expected = run_dir / f"{stage}.json"
    source_paths = [str(SOURCE_ROOT)]
    if stage == "detection":
        source_paths.insert(0, str(cfg["detector_source"]))
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(source_paths),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(run_dir / "cache/matplotlib"),
        "YOLO_CONFIG_DIR": str(run_dir / "cache/ultralytics"),
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    (run_dir / "cache/matplotlib").mkdir(parents=True, exist_ok=True)
    (run_dir / "cache/ultralytics").mkdir(parents=True, exist_ok=True)
    result["execution"].append(
        {"stage": stage, "command": command, "pythonpath": source_paths, "log": f"{stage}.log"}
    )
    with (run_dir / f"{stage}.log").open("w") as log:
        try:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=ROOT,
                timeout=cfg["stage_timeout_seconds"],
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{stage}_timeout; see {stage}.log") from error
    if process.returncode:
        raise RuntimeError(f"{stage}_process_exit_{process.returncode}; see {stage}.log")
    return json.loads(expected.read_text(encoding="utf-8"))


def _eye_slots(animals: list[dict[str, Any]], conditions: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "prediction_id": animal["prediction_id"],
            "category": animal["category"],
            **{
                key: animal.get(key, category_capability(animal["category"])[key])
                for key in ("pose_domain", "experimental", "capability_reason")
            },
            "condition_id": condition,
            "eye_side": side,
            "localization_status": "pending" if animal["supported"] else "unsupported_category",
            "segmentation_status": "skipped",
            "failure_reason": "" if animal["supported"] else "unsupported_category",
            "point_x_px": None,
            "point_y_px": None,
            "point_score": None,
            "roi_xyxy": None,
            "mask_path": None,
            "mask_area_px": None,
            "centroid_x_px": None,
            "centroid_y_px": None,
            "visibility_status": "unverified",
            "side_assignment_status": "unverified",
            "measurement_qualification": "unverified",
        }
        for animal in animals
        if animal["accepted"]
        for condition in conditions
        for side in ("left", "right")
    ]


def _apply_pose(
    eyes: list[dict[str, Any]],
    poses: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    index = {pose["prediction_id"]: pose for pose in poses}
    if len(index) != len(poses):
        raise ValueError("Duplicate pose prediction identity.")
    requests = []
    for eye in eyes:
        if eye["localization_status"] == "unsupported_category":
            continue
        pose = index.get(eye["prediction_id"], {"status": "failed"})
        try:
            _, reason = eye_crops(pose, width, height)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            reason = f"invalid_pose_result:{error}"
        if reason:
            eye.update(localization_status="failed", failure_reason=reason)
            continue
        side_index = 0 if eye["eye_side"] == "left" else 1
        point = pose["keypoints_xy"][side_index]
        distance = float(
            np.linalg.norm(
                np.asarray(pose["keypoints_xy"][:2])[0] - np.asarray(pose["keypoints_xy"][:2])[1]
            )
        )
        roi, clipped = roi_for_case(
            {
                "hrnet_point_xy": point,
                "hrnet_eye_distance_px": distance,
                "image_width": width,
                "image_height": height,
            },
            1.0,
        )
        eye.update(
            localization_status="completed",
            point_x_px=point[0],
            point_y_px=point[1],
            point_score=pose["scores"][side_index],
            roi_xyxy=roi,
            roi_clipped=clipped,
            segmentation_status="pending",
            hrnet_eye_distance_px=distance,
        )
        requests.append(
            {
                "prediction_id": eye["prediction_id"],
                "eye_side": eye["eye_side"],
                "condition_id": eye["condition_id"],
                "point_xy": point,
                "roi_xyxy": roi,
                "pose_domain": eye["pose_domain"],
                "experimental": eye["experimental"],
                "capability_reason": eye["capability_reason"],
            }
        )
    return requests


def _apply_segmentation(
    eyes: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    run_dir: Path,
    width: int,
    height: int,
) -> None:
    index = {(r["prediction_id"], r["eye_side"], r["condition_id"]): r for r in predictions}
    if len(index) != len(predictions):
        raise ValueError("Duplicate segmentation identity.")
    for eye in eyes:
        if eye["segmentation_status"] != "pending":
            continue
        row = index.get((eye["prediction_id"], eye["eye_side"], eye["condition_id"]))
        if row is None:
            eye.update(segmentation_status="failed", failure_reason="missing_segmentation_output")
            continue
        eye["segmentation_status"] = row["status"]
        eye["failure_reason"] = row.get("failure_reason", "")
        for key in ("candidate_scores", "selected_candidate", "inference_seconds"):
            eye[key] = row.get(key)
        if row["status"] not in ("completed", "empty_mask"):
            continue
        try:
            path = (run_dir / row["mask_path"]).resolve()
            if not path.is_relative_to(run_dir.resolve()):
                raise ValueError("Mask path escapes run directory.")
            with Image.open(path) as image:
                mask = np.asarray(image)
            if mask.shape != (height, width) or not np.isin(mask, [0, 255]).all():
                raise ValueError("Mask is not an original-resolution binary 0/255 image.")
            binary = mask != 0
            centroid = mask_centroid(binary)
            eye.update(
                mask_path=row["mask_path"],
                mask_area_px=int(binary.sum()),
                centroid_x_px=centroid[0] if centroid else None,
                centroid_y_px=centroid[1] if centroid else None,
                segmentation_status="completed" if centroid else "empty_mask",
                failure_reason="" if centroid else "empty_mask",
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            eye.update(segmentation_status="failed", failure_reason=f"invalid_mask:{error}")


def run_inference(
    image_bytes: bytes,
    filename: str,
    config_path: Path,
    conditions: tuple[str, ...] = ("A_r100",),
    progress: Callable[[str, str], None] | None = None,
) -> Path:
    """Serialize API/UI/CLI runs sharing an output root; reject invalid inputs before locking."""
    cfg = load_config(config_path)
    normalize_image(image_bytes, cfg)
    with inference_execution_lock(cfg["output_directory"]):
        return _run_inference(image_bytes, filename, config_path, conditions, progress)


def _run_inference(
    image_bytes: bytes,
    filename: str,
    config_path: Path,
    conditions: tuple[str, ...] = ("A_r100",),
    progress: Callable[[str, str], None] | None = None,
) -> Path:
    """Run once, write unique artifacts, and retain all failure slots. No reference inputs."""
    cfg = load_config(config_path)
    if conditions != ("A_r100",):
        raise ValueError("The delivery uses the fixed A_r100 baseline.")
    image, suffix = normalize_image(image_bytes, cfg)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    run_dir = cfg["output_directory"] / run_id
    (run_dir / "input").mkdir(parents=True)
    (run_dir / f"input/original{suffix}").write_bytes(image_bytes)
    image_path = run_dir / "input/image.png"
    image.save(image_path)
    (run_dir / "config.toml").write_bytes(config_path.read_bytes())
    source_files = package_sources()
    launch_source_hashes = {
        p.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(p) for p in source_files
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "baseline": "A_r100",
        "conditions": list(conditions),
        "capability_policy": capability_policy(),
        "input": {
            "filename": Path(filename).name,
            "width": image.width,
            "height": image.height,
            "image_path": "input/image.png",
            "original_path": f"input/original{suffix}",
            "original_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "normalized_sha256": sha256_file(image_path),
            "normalization": "EXIF transpose; RGB; alpha on white; no resize",
        },
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in cfg.items()
        },
        "stages": {stage: {"status": "not_run", "reason": "", "seconds": 0.0} for stage in STAGES},
        "animals": [],
        "eyes": [],
        "measurements": [],
        "artifacts": {},
        "execution": [],
        "limitations": [
            "No ground truth, IoU/Dice or measurement-error evaluation on this upload.",
            "Model anatomical channels, visibility and measurement eligibility unverified.",
            "Distances are candidate mask-centroid distances in image pixels, "
            "not pupil or physical distances.",
            "Successful execution and nonempty masks do not establish correct eye boundaries.",
            "Bird eye inference transfers a mammal-trained AP-10K model outside its training "
            "domain; all bird eye outputs and their measurements are experimental and unverified.",
        ],
    }
    started = time.perf_counter()

    def update(stage: str, status: str, reason: str = "", seconds: float = 0.0) -> None:
        result["stages"][stage] = {"status": status, "reason": reason, "seconds": seconds}
        _save(run_dir / "result.json", result)
        LOGGER.info(
            json.dumps(
                {"run_id": run_id, "stage": stage, "status": status, "reason": reason},
                ensure_ascii=False,
            )
        )
        if progress:
            progress(stage, status)

    common = {"image": str(image_path), "output": str(run_dir), "seed": 42, "threads": 4}
    stage = "detection"
    stage_started = time.perf_counter()
    try:
        readiness = preflight(cfg)
        result["preflight"] = readiness
        missing = [r["component"] for r in readiness if r["status"] != "ready"]
        if missing:
            raise RuntimeError("Missing or changed dependencies: " + ", ".join(missing))
        update(stage, "running")
        request = common | {
            key: cfg[key]
            for key in (
                "image_size",
                "confidence_floor",
                "nms_iou",
                "max_detections",
                "animal_score_threshold",
                "supported_categories",
            )
        }
        request.update(
            checkpoint=str(cfg["detector_checkpoint"]),
            checkpoint_sha256=cfg["detector_checkpoint_sha256"],
            package_version="8.4.0",
        )
        detected = _run_worker(cfg, run_dir, stage, request, result)
        result["animals"] = detected["results"]
        ids = [animal["prediction_id"] for animal in result["animals"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate detector prediction identity.")
        result["eyes"] = _eye_slots(result["animals"], conditions)
        accepted = [a for a in result["animals"] if a["accepted"]]
        supported = [a for a in accepted if a["supported"]]
        detection_issues = any(
            a["status"] == "failed" or (a["accepted"] and a.get("mask_status") != "completed")
            for a in result["animals"]
        )
        update(
            stage,
            "completed_with_issues"
            if detection_issues
            else "completed"
            if accepted
            else "no_animals",
            "Inspect animal instance failures." if detection_issues else "",
            seconds=time.perf_counter() - stage_started,
        )
        if not supported:
            reason = "no_animals_above_threshold" if not accepted else "no_supported_animals"
            for skipped in ("localization", "segmentation"):
                update(skipped, "skipped", reason)
        else:
            stage = "localization"
            stage_started = time.perf_counter()
            update(stage, "running")
            pose_request = {
                "checkpoint": str(cfg["pose_checkpoint"]),
                "checkpoint_sha256": cfg["pose_checkpoint_sha256"],
                "output": str(run_dir / "pose"),
                "seed": 42,
                "threads": 4,
                "mmpose_commit": cfg["mmpose_commit"],
                "compatibility_probe": False,
                "mmpose_source_manifest": str(cfg["pose_source_manifest"]),
                "mmpose_source_manifest_sha256": sha256_file(cfg["pose_source_manifest"]),
                "cases": [
                    {
                        "image_id": 0,
                        "prediction_id": a["prediction_id"],
                        "image": str(image_path),
                        "image_sha256": sha256_file(image_path),
                        "bbox_xyxy": a["bbox_xyxy"],
                    }
                    for a in supported
                ],
            }
            pose = _run_worker(cfg, run_dir, stage, pose_request, result)
            cases = _apply_pose(result["eyes"], pose["results"], image.width, image.height)
            failures = any(e["localization_status"] == "failed" for e in result["eyes"])
            update(
                stage,
                "completed_with_issues" if failures else "completed",
                "invalid_or_missing_eye_pair" if failures else "",
                time.perf_counter() - stage_started,
            )
            stage = "segmentation"
            stage_started = time.perf_counter()
            if cases:
                update(stage, "running")
                request = common | {
                    "cases": cases,
                    "checkpoint": str(cfg["sam_checkpoint"]),
                    "checkpoint_sha256": cfg["sam_checkpoint_sha256"],
                    "sam_runtime": str(cfg["sam_runtime"]),
                }
                segmented = _run_worker(cfg, run_dir, stage, request, result)
                _apply_segmentation(
                    result["eyes"], segmented["results"], run_dir, image.width, image.height
                )
                failures = any(e["segmentation_status"] != "completed" for e in result["eyes"])
                update(
                    stage,
                    "completed_with_issues" if failures else "completed",
                    "Some eye slots unavailable; inspect per-eye reasons." if failures else "",
                    time.perf_counter() - stage_started,
                )
            else:
                update(stage, "skipped", "no_valid_eye_pairs")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        RuntimeError,
        Phase1Error,
    ) as error:
        update(stage, "failed", str(error), time.perf_counter() - stage_started)
        result["fatal_error"] = str(error)
        for eye in result["eyes"]:
            if eye["localization_status"] == "pending":
                eye.update(localization_status="failed", failure_reason=str(error))
            if eye["segmentation_status"] == "pending":
                eye.update(segmentation_status="failed", failure_reason=str(error))
        for following in STAGES[STAGES.index(stage) + 1 : -1]:
            update(following, "skipped", f"upstream_{stage}_failed")

    # Geometric availability is independent of successful stage execution or biological validity.
    stage_started = time.perf_counter()
    result["measurements"] = build_measurements(result["eyes"], list(conditions))
    available = sum(r["status"] == "candidate" for r in result["measurements"])
    update(
        "measurement",
        ("completed" if available == len(result["measurements"]) else "completed_with_issues")
        if available
        else "unavailable",
        "candidate_measurements_only" if available else "no_valid_mask_centroid_pairs",
        time.perf_counter() - stage_started,
    )
    if "fatal_error" in result:
        result["status"] = "failed"
    elif any(e["segmentation_status"] != "completed" for e in result["eyes"]) or any(
        a["status"] == "failed" or (a["accepted"] and a.get("mask_status") != "completed")
        for a in result["animals"]
    ):
        result["status"] = "completed_with_issues"
    elif not any(a["accepted"] for a in result["animals"]):
        result["status"] = "no_animals"
    else:
        result["status"] = "completed"
    export_fields = {
        "animals": [
            "prediction_id",
            "category",
            "score",
            "accepted",
            "supported",
            "status",
            "failure_reason",
            "mask_status",
            "mask_failure_reason",
        ],
        "eyes": [
            "condition_id",
            "prediction_id",
            "eye_side",
            "localization_status",
            "segmentation_status",
            "mask_area_px",
            "centroid_x_px",
            "centroid_y_px",
            "failure_reason",
            "visibility_status",
            "side_assignment_status",
        ],
        "measurements": [
            "condition_id",
            "measurement_type",
            "prediction_id_a",
            "prediction_id_b",
            "distance_px",
            "status",
            "qualification",
            "reason",
        ],
    }
    for key in ("animals", "eyes", "measurements"):
        (run_dir / f"{key}.csv").write_bytes(csv_bytes(result[key], export_fields[key]))
        result["artifacts"][key + "_csv"] = f"{key}.csv"
    result["artifacts"].update(render_overlays(run_dir, result))
    result["elapsed_seconds"] = time.perf_counter() - started
    result["finished_at_utc"] = datetime.now(UTC).isoformat()
    _save(run_dir / "result.json", result)
    end_source_hashes = {
        p.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(p) for p in package_sources()
    }
    _save(
        run_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "config_sha256": sha256_file(run_dir / "config.toml"),
            "source_sha256": launch_source_hashes,
            "source_sha256_at_finish": end_source_hashes,
            "source_unchanged_during_run": launch_source_hashes == end_source_hashes,
            "files_sha256": {
                str(p.relative_to(run_dir)): sha256_file(p)
                for p in sorted(run_dir.rglob("*"))
                if p.is_file() and "cache" not in p.relative_to(run_dir).parts
            },
            "research_reference_used": False,
            "measurement_qualification": "unverified",
            "capability_policy": capability_policy(),
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get("AEM_INFERENCE_CONFIG", ROOT / "configs/inference.runtime.toml")
        ),
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.preflight:
        checks = preflight(load_config(args.config))
        print(json.dumps(checks, indent=2))
        raise SystemExit(0 if all(r["status"] == "ready" for r in checks) else 1)
    if not args.image:
        parser.error("--image is required unless --preflight is used")
    output = run_inference(
        args.image.read_bytes(),
        args.image.name,
        args.config,
        ("A_r100",),
    )
    print(output)
    raise SystemExit(1 if load_result(output)["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
