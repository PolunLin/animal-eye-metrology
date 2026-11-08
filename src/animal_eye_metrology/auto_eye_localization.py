"""Frozen animal detections -> official HRNet eye points -> SAM -> independent evaluation."""

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.eye_comparison import distance_statistics, eye_statistics
from animal_eye_metrology.eye_geometry import eye_crops as eye_crops
from animal_eye_metrology.eye_poc import load_predictor, settings, validate_source
from animal_eye_metrology.measurement_baseline import derive, verify, write_csv
from animal_eye_metrology.metrology import build_measurements
from animal_eye_metrology.verification import sha256_file

LOGGER = logging.getLogger(__name__)
SIDES = ("left", "right")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_json(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = tomllib.loads(path.read_text())
    for key in (
        "output_directory",
        "pose_python",
        "images_directory",
        "predictions",
        "animal_matches",
        "primary_reference",
        "sam_runtime",
        "pose_checkpoint",
        "pose_download_manifest",
        "pose_source_manifest",
    ):
        # Resolving the venv Python symlink would bypass its isolated site-packages.
        joined = path.parent / cfg[key]
        cfg[key] = Path(os.path.abspath(joined)) if key == "pose_python" else joined.resolve()
    cfg["source"] = path
    if cfg["image_ids"] != [20247, 125405, 333069] or cfg["expected_predictions"] != 6:
        raise Phase1Error("Approved scope is three images and six frozen predictions.")
    if (cfg["roi_min_size"], cfg["roi_eye_distance_factor"], cfg["seed"], cfg["threads"]) != (
        32,
        2.0,
        42,
        4,
    ) or cfg["animal_score_threshold"] != 0.5:
        raise Phase1Error("Approved prompt or execution policy differs.")
    return cfg


def xywh_to_xyxy(box: list[float]) -> list[float]:
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise Phase1Error("Invalid animal box.")
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise Phase1Error("Animal box must have positive area.")
    return [x, y, x + w, y + h]


def prepare(cfg: dict[str, Any]) -> None:
    out = cfg["output_directory"]
    inputs = out / "inputs"
    inputs.mkdir()
    (inputs / "images").mkdir()
    predictions = [
        row
        for row in read_json(cfg["predictions"])
        if row["image_id"] in cfg["image_ids"] and row["score"] >= cfg["animal_score_threshold"]
    ]
    predictions.sort(key=lambda row: (row["image_id"], row["prediction_id"]))
    if len(predictions) != cfg["expected_predictions"]:
        raise Phase1Error("Frozen animal prediction count changed.")
    save_json(inputs / "animal_predictions.json", predictions)
    cases = []
    for image_id in cfg["image_ids"]:
        name = f"{image_id:012d}.jpg"
        shutil.copyfile(cfg["images_directory"] / name, inputs / "images" / name)
    for row in predictions:
        relative = f"images/{row['image_id']:012d}.jpg"
        cases.append(
            {
                "image_id": row["image_id"],
                "prediction_id": row["prediction_id"],
                "image": relative,
                "image_sha256": sha256_file(inputs / relative),
                "bbox_xyxy": xywh_to_xyxy(row["bbox"]),
            }
        )
    save_json(inputs / "pose_cases.json", cases)
    save_json(
        inputs / "manifest.json",
        {
            "dataset": "COCO 2017 validation; same three purpose-selected development images",
            "images": cfg["image_ids"],
            "predictions": len(predictions),
            "source_predictions_sha256": sha256_file(cfg["predictions"]),
            "model_inputs_include_eye_ground_truth": False,
            "files_sha256": {
                str(p.relative_to(inputs)): sha256_file(p)
                for p in sorted(inputs.rglob("*"))
                if p.is_file()
            },
        },
    )
    LOGGER.info("Frozen three images and %d predicted animal boxes", len(cases))


def check_inputs(out: Path) -> None:
    for rel, expected in read_json(out / "inputs/manifest.json")["files_sha256"].items():
        if sha256_file(out / "inputs" / rel) != expected:
            raise Phase1Error(f"Frozen input changed: {rel}")


def pose_stage(cfg: dict[str, Any], *, compatibility: bool) -> None:
    out, inputs = cfg["output_directory"], cfg["output_directory"] / "inputs"
    check_inputs(out)
    worker = Path(__file__).with_name("eye_pose_worker.py")
    cases = read_json(inputs / "pose_cases.json")
    if compatibility:
        attempt = 1 + len(list((out / "compatibility").glob("pose_attempt_*.json")))
        request_path = out / "compatibility" / f"pose_attempt_{attempt:02d}.json"
        result_dir = out / "compatibility" / f"probe_{attempt:02d}"
        cases = [r for r in cases if r["prediction_id"] == "image_20247_prediction_001"]
    else:
        passed = read_json(out / "compatibility/passed.json")
        if passed["worker_sha256"] != sha256_file(worker):
            raise Phase1Error("Pose worker changed since Stage A; repeat compatibility validation.")
        if passed["result_sha256"] != sha256_file(out / passed["result"]):
            raise Phase1Error("Stage A compatibility evidence changed.")
        request_path, result_dir = inputs / "pose_b_request.json", out / "pose"
    download = read_json(cfg["pose_download_manifest"])
    source_manifest = cfg["pose_source_manifest"]
    if sha256_file(cfg["pose_checkpoint"]) != download["sha256"]:
        raise Phase1Error("Downloaded checkpoint digest changed.")
    if not compatibility and download["sha256"] != passed["checkpoint_sha256"]:
        raise Phase1Error("Checkpoint differs from the validated Stage A checkpoint.")
    save_json(
        request_path,
        {
            "checkpoint": os.path.relpath(cfg["pose_checkpoint"], request_path.parent),
            "checkpoint_sha256": download["sha256"],
            "output": os.path.relpath(result_dir, request_path.parent),
            "cases": [
                r | {"image": os.path.relpath(inputs / r["image"], request_path.parent)}
                for r in cases
            ],
            "seed": 42,
            "threads": 4,
            "mmpose_commit": cfg["mmpose_commit"],
            "compatibility_probe": compatibility,
            "mmpose_source_manifest": os.path.relpath(source_manifest, request_path.parent),
            "mmpose_source_manifest_sha256": sha256_file(source_manifest),
        },
    )
    command = [str(cfg["pose_python"]), str(worker), "--request", str(request_path)]
    environment = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": str(out / "mpl_cache"),
    }
    with request_path.with_suffix(".log").open("x") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    if proc.returncode:
        raise Phase1Error(f"Pose process failed; see {request_path.with_suffix('.log')}")
    result = read_json(result_dir / "pose_results.json")
    if compatibility:
        if result["status"] != "passed" or len(result["results"]) != 1:
            raise Phase1Error("Stage A did not pass.")
        save_json(
            out / "compatibility/passed.json",
            {
                "status": "passed",
                "worker_sha256": sha256_file(worker),
                "result": str((result_dir / "pose_results.json").relative_to(out)),
                "result_sha256": sha256_file(result_dir / "pose_results.json"),
                "checkpoint_sha256": download["sha256"],
                "command": command,
            },
        )
    LOGGER.info("Pose stage %s completed", "A" if compatibility else "B")


def segment(cfg: dict[str, Any]) -> None:
    import torch

    stage_started = time.perf_counter()
    out = cfg["output_directory"]
    check_inputs(out)
    runtime = settings(cfg["sam_runtime"])
    spec = runtime["models"]["sam2_1_tiny"]
    source = validate_source(runtime, spec)
    checkpoint = runtime["weights_directory"] / spec["checkpoint"]
    expected = read_json(runtime["weights_directory"] / "download_manifest.json")[checkpoint.name]
    if sha256_file(checkpoint) != expected["sha256"]:
        raise Phase1Error("SAM checkpoint changed.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    started = time.perf_counter()
    predictor = load_predictor("sam2_1_tiny", checkpoint)
    load_seconds = time.perf_counter() - started
    (out / "eyes").mkdir()
    outputs = []
    for pose in read_json(out / "pose/pose_results.json")["results"]:
        image = np.array(
            Image.open(out / f"inputs/images/{pose['image_id']:012d}.jpg").convert("RGB")
        )
        roi_started = time.perf_counter()
        crops, failure = eye_crops(pose, image.shape[1], image.shape[0])
        roi_seconds = time.perf_counter() - roi_started
        for index, side in enumerate(SIDES):
            name = f"{pose['prediction_id']}_{side}"
            record = {
                "image_id": pose["image_id"],
                "prediction_id": pose["prediction_id"],
                "eye_side": side,
                "prompt_source": "hrnet_ap10k",
                "visibility_status": "unverified",
                "status": "failed",
                "failure_reason": failure,
                "case": name,
                "point_xy": pose["keypoints_xy"][index] if pose["status"] == "passed" else None,
                "point_score": pose["scores"][index] if pose["status"] == "passed" else None,
                "roi_xyxy": crops[index] if crops else None,
                "animal_pair_roi_seconds": roi_seconds,
            }
            if crops:
                directory = out / "eyes" / name
                directory.mkdir()
                x0, y0, x1, y1 = crops[index]
                point = np.asarray([record["point_xy"]], dtype=np.float32) - [x0, y0]
                try:
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        predictor.set_image(image[y0:y1, x0:x1].copy())
                        logits, scores, low_res = predictor.predict(
                            point_coords=point,
                            point_labels=np.array([1]),
                            box=None,
                            multimask_output=True,
                            return_logits=True,
                        )
                    elapsed = time.perf_counter() - t0
                    if (
                        logits.shape != (3, y1 - y0, x1 - x0)
                        or scores.shape != (3,)
                        or not all(np.isfinite(a).all() for a in (logits, scores, low_res))
                    ):
                        raise Phase1Error("SAM returned invalid logits/scores.")
                    selected = int(np.argmax(scores))
                    output_started = time.perf_counter()
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask[y0:y1, x0:x1] = (logits[selected] > 0).astype(np.uint8) * 255
                    Image.fromarray(mask).save(directory / "mask.png")
                    np.save(directory / "logits.npy", logits, allow_pickle=False)
                    record.update(
                        status="passed",
                        failure_reason="",
                        candidate_scores=scores.tolist(),
                        selected_candidate=selected,
                        inference_seconds=elapsed,
                        mask_path=str((directory / "mask.png").relative_to(out)),
                        logits_path=str((directory / "logits.npy").relative_to(out)),
                        output_seconds=time.perf_counter() - output_started,
                    )
                except Exception as exc:
                    LOGGER.exception("SAM failed for %s", name)
                    record["failure_reason"] = f"{type(exc).__name__}: {exc}"
                save_json(directory / "result.json", record)
            outputs.append(record)
    save_json(
        out / "segmentation.json",
        {
            "sam_source": source,
            "checkpoint_sha256": expected["sha256"],
            "model_load_seconds": load_seconds,
            "results": outputs,
            "point_only": True,
            "threshold": 0,
            "postprocessing": "none",
            "cpu_threads": 4,
            "seed": 42,
            "dtype": "float32",
            "stage_wall_seconds": time.perf_counter() - stage_started,
        },
    )
    LOGGER.info("Saved %d eye outcomes", len(outputs))


def point_in_mask(point: list[float] | None, mask: np.ndarray) -> bool | None:
    if point is None or not all(math.isfinite(v) for v in point):
        return None
    x, y = point
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        return False
    return bool(mask[math.floor(y), math.floor(x)])


def evaluate_saved(out: Path) -> tuple[list[dict], list[dict], dict]:
    """Evaluation-only identity matching and eye GT access, after predictions are frozen."""
    reference = out / "evaluation_inputs/reference"
    verify(reference)
    ref_centers, ref_distances = derive(reference / "frozen")
    with (out / "evaluation_inputs/animal_matches.csv").open() as handle:
        matches = list(csv.DictReader(handle))
    poses = read_json(out / "pose/pose_results.json")["results"]
    outcomes = read_json(out / "segmentation.json")["results"]
    mapping = {
        (int(r["image_id"]), int(r["annotation_id"])): r["prediction_id"]
        for r in matches
        if r["match_status"] == "TP"
        and float(r["confidence_threshold"]) == 0.5
        and float(r["iou_threshold"]) == 0.5
    }
    by_eye = {(r["prediction_id"], r["eye_side"]): r for r in outcomes}
    refs = {(r["image_id"], r["annotation_id"], r["eye_side"]): r for r in ref_centers}
    rows, centers = [], []
    for ref in ref_centers:
        image_id, aid, side = int(ref["image_id"]), int(ref["annotation_id"]), ref["eye_side"]
        prediction_id = mapping.get((image_id, aid), "")
        result = by_eye.get((prediction_id, side))
        gt = np.array(Image.open(reference / ref["mask_path"])) // 255
        opposite_ref = refs[(image_id, aid, "right" if side == "left" else "left")]
        opposite = np.array(Image.open(reference / opposite_ref["mask_path"])) // 255
        predicted = None
        if result and result["status"] == "passed":
            predicted = np.array(Image.open(out / result["mask_path"])) // 255
        metrics = eye_statistics(predicted, gt)
        point = result["point_xy"] if result else None
        offset = math.dist(point, (ref["centroid_x_px"], ref["centroid_y_px"])) if point else None
        roi = result["roi_xyxy"] if result else None
        outside = None
        if roi:
            x0, y0, x1, y1 = roi
            outside = int(gt.sum() - gt[y0:y1, x0:x1].sum())
        rows.append(
            {
                "image_id": image_id,
                "annotation_id": aid,
                "prediction_id": prediction_id,
                "animal_category": ref["animal_category"],
                "eye_side": side,
                "point_x": point[0] if point else None,
                "point_y": point[1] if point else None,
                "point_score": result["point_score"] if result else None,
                "point_in_target_mask": point_in_mask(point, gt),
                "point_in_opposite_mask": point_in_mask(point, opposite),
                "point_offset_from_mask_centroid_px": offset,
                "reference_pixels_outside_roi": outside,
                "failure_reason": result["failure_reason"] if result else "animal_not_recovered",
                "visibility_status": "unverified",
                "predicted_mask_path": result.get("mask_path", "") if result else "",
                "reference_mask_path": str((reference / ref["mask_path"]).relative_to(out)),
                **metrics,
            }
        )
        centers.append(
            {
                "image_id": image_id,
                "annotation_id": aid,
                "eye_side": side,
                "annotation_version": 1,
                "centroid_x_px": metrics["predicted_cx_px"],
                "centroid_y_px": metrics["predicted_cy_px"],
            }
        )
    distances = build_measurements(centers)
    pairs = {
        (r["image_id"], r["measurement_type"], r["annotation_id_a"], r["annotation_id_b"]): r
        for r in distances
    }
    measurements = []
    for ref in ref_distances:
        pred = pairs[
            (
                ref["image_id"],
                ref["measurement_type"],
                ref["annotation_id_a"],
                ref["annotation_id_b"],
            )
        ]
        measurements.append(
            {
                **{
                    k: ref[k]
                    for k in ("image_id", "measurement_type", "annotation_id_a", "annotation_id_b")
                },
                **distance_statistics(pred["distance_px"], ref["distance_px"]),
                **{f"predicted_{k}": pred[k] for k in ("cx_a_px", "cy_a_px", "cx_b_px", "cy_b_px")},
                "visibility_status": "unverified",
            }
        )
    used_ids = {r["prediction_id"] for r in rows if r["prediction_id"]}
    summary = {
        "reference_eyes": len(rows),
        "predicted_animals": len(poses),
        "matched_reference_animals": len(used_ids),
        "unmatched_prediction_ids": [
            r["prediction_id"] for r in poses if r["prediction_id"] not in used_ids
        ],
        "pose_failed": sum(r["status"] != "passed" for r in poses),
        "eye_inference_failed": sum(r["prediction_status"] == "inference_failed" for r in rows),
        "empty_masks": sum(r["prediction_status"] == "empty" for r in rows),
        "points_in_target_mask": sum(r["point_in_target_mask"] is True for r in rows),
        "points_in_opposite_mask": sum(r["point_in_opposite_mask"] is True for r in rows),
        "mean_iou_all_targets_failure_as_zero": mean(r["iou"] or 0.0 for r in rows),
        "mean_dice_all_targets_failure_as_zero": mean(r["dice"] or 0.0 for r in rows),
    }
    for field in ("point_offset_from_mask_centroid_px", "centroid_error_px"):
        values = [r[field] for r in rows if r[field] is not None]
        summary[f"{field}_defined"] = len(values)
        summary[f"{field}_mean"] = mean(values) if values else None
        summary[f"{field}_max"] = max(values) if values else None
    for kind in ("binocular", "right_eye_pair"):
        group = [r for r in measurements if r["measurement_type"] == kind]
        errors = [r["absolute_error_px"] for r in group if r["absolute_error_px"] is not None]
        summary[kind] = {
            "expected": len(group),
            "defined": len(errors),
            "mae_px": mean(errors) if errors else None,
            "rmse_px": math.sqrt(mean(e * e for e in errors)) if errors else None,
            "max_error_px": max(errors) if errors else None,
        }
    return rows, measurements, summary


def evaluate(cfg: dict[str, Any]) -> None:
    out = cfg["output_directory"]
    reference_status = verify(cfg["primary_reference"])
    if reference_status["snapshot_sha256"] != cfg["primary_snapshot_sha256"]:
        raise Phase1Error("Primary eye reference differs from the approved snapshot.")
    evaluation = out / "evaluation_inputs"
    evaluation.mkdir()
    shutil.copytree(cfg["primary_reference"], evaluation / "reference")
    shutil.copyfile(cfg["animal_matches"], evaluation / "animal_matches.csv")
    rows, measurements, summary = evaluate_saved(out)
    write_csv(out / "eye_metrics.csv", rows)
    write_csv(out / "measurement_comparison.csv", measurements)
    save_json(out / "summary.json", summary)
    save_json(out / "evaluation_values.json", {"eyes": rows, "measurements": measurements})
    LOGGER.info("Evaluation summary: %s", json.dumps(summary))


def finalize(cfg: dict[str, Any]) -> None:
    out = cfg["output_directory"]
    provenance = out / "provenance"
    provenance.mkdir()
    for path in (
        Path(__file__),
        Path(__file__).with_name("eye_pose_worker.py"),
        cfg["source"],
        Path(__file__).with_name("metrology.py"),
        Path(__file__).with_name("eye_comparison.py"),
        Path(__file__).with_name("auto_eye_visualization.py"),
        Path(__file__).with_name("eye_poc.py"),
        Path(__file__).with_name("measurement_baseline.py"),
        cfg["pose_source_manifest"],
        cfg["source"].with_name("eye_pose_lock_macos_arm64.txt"),
        cfg["source"].with_name("eye_pose_requirements.txt"),
        cfg["sam_runtime"],
    ):
        shutil.copyfile(path, provenance / path.name)
    save_json(
        out / "run_manifest.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "dataset": "COCO 2017 val2017, same purpose-selected development images",
            "image_ids": cfg["image_ids"],
            "seed": 42,
            "trained": False,
            "primary_snapshot_sha256": cfg["primary_snapshot_sha256"],
            "visibility_eligibility_automated": False,
            "files_sha256": {
                str(p.relative_to(out)): sha256_file(p)
                for p in sorted(out.rglob("*"))
                if p.is_file() and "mpl_cache" not in p.parts and "__pycache__" not in p.parts
            },
        },
    )


def verify_run(out: Path) -> dict[str, Any]:
    manifest = read_json(out / "run_manifest.json")
    for relative, expected in manifest["files_sha256"].items():
        if sha256_file(out / relative) != expected:
            raise Phase1Error(f"Run artifact changed: {relative}")
    for eye in read_json(out / "segmentation.json")["results"]:
        if eye["status"] != "passed":
            continue
        logits = np.load(out / eye["logits_path"], allow_pickle=False)
        actual = np.array(Image.open(out / eye["mask_path"]))
        reconstructed = np.zeros_like(actual)
        x0, y0, x1, y1 = eye["roi_xyxy"]
        selected = int(np.argmax(eye["candidate_scores"]))
        if selected != eye["selected_candidate"]:
            raise Phase1Error("Candidate selection changed.")
        reconstructed[y0:y1, x0:x1] = (logits[selected] > 0) * np.uint8(255)
        if not np.array_equal(actual, reconstructed):
            raise Phase1Error("Saved logits do not reproduce the full-image mask.")
    eyes, distances, summary = evaluate_saved(out)
    if summary != read_json(out / "summary.json") or {
        "eyes": eyes,
        "measurements": distances,
    } != read_json(out / "evaluation_values.json"):
        raise Phase1Error("Saved evaluation did not reproduce.")
    return {
        "status": "passed",
        "files_checked": len(manifest["files_sha256"]),
        "eyes_recomputed": len(eyes),
        "distances_recomputed": len(distances),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/auto_eye_localization.toml"))
    parser.add_argument(
        "--stage",
        choices=("prepare", "compatibility", "pose", "sam", "evaluate", "finalize", "verify"),
        required=True,
    )
    args = parser.parse_args()
    cfg = config(args.config)
    started = time.perf_counter()
    LOGGER.info(
        json.dumps(
            {
                "event": "stage_started",
                "stage": args.stage,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
        )
    )
    action = {
        "prepare": prepare,
        "compatibility": lambda c: pose_stage(c, compatibility=True),
        "pose": lambda c: pose_stage(c, compatibility=False),
        "sam": segment,
        "evaluate": evaluate,
        "finalize": finalize,
        "verify": lambda c: print(json.dumps(verify_run(c["output_directory"]))),
    }[args.stage]
    action(cfg)
    LOGGER.info(
        json.dumps(
            {
                "event": "stage_completed",
                "stage": args.stage,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
