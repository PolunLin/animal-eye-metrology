"""Approved paired prompt/ROI experiment with frozen predictions and separate evaluation."""

import argparse
import csv
import json
import logging
import math
import platform
import random
import shutil
import time
import tomllib
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.measurement.eye_geometry import ROI_FACTORS as FACTORS
from animal_eye_metrology.measurement.eye_geometry import roi_for_case as roi_for_case
from animal_eye_metrology.measurement.measurement_baseline import verify, write_csv
from animal_eye_metrology.paths import module_source
from animal_eye_metrology.research.auto_eye_localization import read_json, save_json, verify_run
from animal_eye_metrology.research.eye_comparison import verify_comparison
from animal_eye_metrology.research.eye_poc import load_predictor, settings, validate_source

LOGGER = logging.getLogger(__name__)
GROUPS = {
    "A": ("hrnet", "none"),
    "B": ("manual", "none"),
    "C": ("hrnet", "manual"),
    "D": ("manual", "manual"),
}


def event(name: str, **details: Any) -> None:
    LOGGER.info(json.dumps({"event": name, **details}, allow_nan=False))


def config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = tomllib.loads(path.read_text())
    for field in (
        "output_directory",
        "automatic_run",
        "manual_run",
        "primary_reference",
        "sam_runtime",
    ):
        cfg[field] = (path.parent / cfg[field]).resolve()
    cfg["source"] = path
    if (
        cfg["image_ids"] != [20247, 125405, 333069]
        or cfg["expected_eyes"] != 12
        or tuple(cfg["roi_factors"]) != FACTORS
        or cfg["seed"] != 42
        or cfg["threads"] != 4
    ):
        raise Phase1Error("Configuration differs from the approved twelve-condition scope.")
    return cfg


def conditions() -> list[dict[str, Any]]:
    return [
        dict(
            condition_id=f"{group}_r{round(factor * 100):03d}",
            group=group,
            point_source=point,
            box_source=box,
            roi_factor=factor,
        )
        for group, (point, box) in GROUPS.items()
        for factor in FACTORS
    ]


def request_for_case(case: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    roi, clipped = roi_for_case(case, condition["roi_factor"])
    point = case[f"{condition['point_source']}_point_xy"]
    box = case["manual_box_xyxy"] if condition["box_source"] == "manual" else None
    if not (roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3]):
        raise Phase1Error("A frozen point is outside its ROI; do not repair it.")
    if box and not (roi[0] <= box[0] < box[2] <= roi[2] and roi[1] <= box[1] < box[3] <= roi[3]):
        raise Phase1Error("A frozen box is outside its ROI; do not alter it.")
    conflict = bool(box and not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]))
    # Match the original automatic runner's float32 global coordinate cast before subtraction.
    local_point = np.asarray([point], dtype=np.float32) - roi[:2]
    local_box = np.asarray(box, dtype=np.float32) - roi[:2] * 2 if box else None
    return {
        **condition,
        **{
            k: case[k]
            for k in (
                "case_id",
                "image_id",
                "annotation_id",
                "prediction_id",
                "eye_side",
                "image_width",
                "image_height",
                "image_path",
                "image_sha256",
            )
        },
        "point_xy": point,
        "box_xyxy": box,
        "roi_xyxy": roi,
        "point_score": case["hrnet_score"] if condition["point_source"] == "hrnet" else None,
        "point_coords_in_roi": local_point.tolist(),
        "box_coords_in_roi": local_box.tolist() if local_box is not None else None,
        "roi_clipped_to_image": clipped,
        "point_outside_prompt_box": conflict,
        "reused_baseline": condition["condition_id"] == "A_r100",
        "visibility_status": "unverified",
        "human_assisted": condition["group"] != "A",
    }


def hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not any(s in p.parts for s in ("__pycache__", "mpl_cache"))
    }


def check_hashes(root: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        if sha256_file(root / relative) != expected:
            raise Phase1Error(f"Frozen artifact changed: {relative}")


def prepare(cfg: dict[str, Any]) -> None:
    out, automatic, manual = (cfg[k] for k in ("output_directory", "automatic_run", "manual_run"))
    validations = {"automatic": verify_run(automatic), "manual": verify_comparison(manual)}
    inputs = out / "inputs"
    inputs.mkdir(parents=True)
    for name in ("images", "manual_requests", "reused"):
        (inputs / name).mkdir()
    if sha256_file(manual / "inputs/prompts.csv") != cfg["manual_prompts_sha256"]:
        raise Phase1Error("Manual prompt snapshot differs from the approved source.")
    shutil.copyfile(manual / "inputs/prompts.csv", inputs / "manual_prompts.csv")
    shutil.copyfile(automatic / "pose/pose_results.json", inputs / "pose_results.json")
    shutil.copyfile(automatic / "segmentation.json", inputs / "baseline_segmentation.json")
    shutil.copyfile(
        automatic / "evaluation_inputs/animal_matches.csv", inputs / "animal_matches.csv"
    )
    for name, source in (
        ("automatic_run_manifest.json", automatic),
        ("manual_run_manifest.json", manual),
    ):
        shutil.copyfile(source / "run_manifest.json", inputs / name)
    with (inputs / "animal_matches.csv").open() as handle:
        pairs = [
            r
            for r in csv.DictReader(handle)
            if r["match_status"] == "TP"
            and float(r["confidence_threshold"]) == 0.5
            and float(r["iou_threshold"]) == 0.5
        ]
    mapping = {r["prediction_id"]: int(r["annotation_id"]) for r in pairs}
    pose_rows = read_json(inputs / "pose_results.json")["results"]
    if len(pose_rows) != 6 or any(r["status"] != "passed" for r in pose_rows):
        raise Phase1Error("Expected six valid frozen animal poses.")
    by_pose = {r["prediction_id"]: r for r in pose_rows}
    baseline = read_json(inputs / "baseline_segmentation.json")
    if len(baseline["results"]) != cfg["expected_eyes"]:
        raise Phase1Error("Automatic baseline eye count differs.")
    for image_id in cfg["image_ids"]:
        name = f"{image_id:012d}.jpg"
        shutil.copyfile(automatic / "inputs/images" / name, inputs / "images" / name)
    cases = []
    for row in baseline["results"]:
        pid, side, image_id = row["prediction_id"], row["eye_side"], row["image_id"]
        aid = mapping[pid]
        case_id = f"{image_id}_{aid}_{side}"
        request_path = manual / "cases" / case_id / "request.json"
        req = read_json(request_path)
        image_path = inputs / "images" / f"{image_id:012d}.jpg"
        image = Image.open(image_path)
        if row["status"] != "passed" or sha256_file(image_path) != req["source_image_sha256"]:
            raise Phase1Error("Baseline outcome or source image changed.")
        case = dict(
            case_id=case_id,
            image_id=image_id,
            annotation_id=aid,
            prediction_id=pid,
            eye_side=side,
            image_width=image.width,
            image_height=image.height,
            image_path=str(image_path.relative_to(out)),
            image_sha256=sha256_file(image_path),
            hrnet_point_xy=row["point_xy"],
            hrnet_score=row["point_score"],
            hrnet_eye_distance_px=math.dist(*by_pose[pid]["keypoints_xy"][:2]),
            manual_point_xy=req["point_xy"],
            manual_box_xyxy=req["box_xyxy"],
            manual_prompt_version=int(req["prompt_record"]["prompt_version"]),
        )
        if roi_for_case(case, 1.0)[0] != row["roi_xyxy"]:
            raise Phase1Error("Reconstructed baseline ROI differs.")
        shutil.copyfile(request_path, inputs / "manual_requests" / f"{case_id}.json")
        reused = inputs / "reused" / case_id
        reused.mkdir()
        for kind in ("mask", "logits"):
            shutil.copyfile(
                automatic / row[f"{kind}_path"],
                reused / f"{kind}{'.png' if kind == 'mask' else '.npy'}",
            )
        save_json(reused / "result.json", row)
        cases.append(case)
    cases.sort(key=lambda r: (r["image_id"], r["annotation_id"], r["eye_side"]))
    groups = conditions()
    requests = [request_for_case(case, cond) for case in cases for cond in groups]
    if len({(r["condition_id"], r["case_id"]) for r in requests}) != 144:
        raise Phase1Error("Incomplete paired experiment matrix.")
    save_json(inputs / "cases.json", cases)
    save_json(inputs / "conditions.json", groups)
    save_json(inputs / "requests.json", requests)
    save_json(
        inputs / "manifest.json",
        {
            "dataset": "COCO 2017 val2017; same three purpose-selected development images",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_validations": validations,
            "conditions": 12,
            "eyes_per_condition": 12,
            "eye_ground_truth_used_to_generate_prompts": False,
            "human_prompts_used_in_diagnostic_arms": True,
            "manual_prompts_sha256": cfg["manual_prompts_sha256"],
            "files_sha256": hash_tree(inputs),
        },
    )
    event(
        "inputs_frozen",
        conditions=12,
        eye_conditions=144,
        prompt_conflicts=sum(r["point_outside_prompt_box"] for r in requests),
        image_clipped_conditions=sum(r["roi_clipped_to_image"] for r in requests),
    )


def selected_mask(
    logits: np.ndarray, scores: list[float], shape: tuple[int, int], roi: list[int]
) -> tuple[np.ndarray, int]:
    x0, y0, x1, y1 = roi
    if (
        logits.shape != (3, y1 - y0, x1 - x0)
        or np.asarray(scores).shape != (3,)
        or not np.isfinite(logits).all()
        or not np.isfinite(scores).all()
    ):
        raise Phase1Error("Invalid native logits or candidate scores.")
    selected = int(np.argmax(scores))
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = (logits[selected] > 0).astype(np.uint8) * 255
    return mask, selected


def run_predictions(cfg: dict[str, Any]) -> None:
    import torch

    out = cfg["output_directory"]
    check_hashes(out / "inputs", read_json(out / "inputs/manifest.json")["files_sha256"])
    runtime = settings(cfg["sam_runtime"])
    spec = runtime["models"]["sam2_1_tiny"]
    source = validate_source(runtime, spec)
    checkpoint = runtime["weights_directory"] / spec["checkpoint"]
    checkpoint_sha = sha256_file(checkpoint)
    baseline = read_json(out / "inputs/baseline_segmentation.json")
    if source != baseline["sam_source"] or checkpoint_sha != baseline["checkpoint_sha256"]:
        raise Phase1Error("SAM source/checkpoint differs from the frozen baseline.")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    start = time.perf_counter()
    predictor = load_predictor("sam2_1_tiny", checkpoint)
    load_seconds = time.perf_counter() - start
    if next(predictor.model.parameters()).dtype != torch.float32:
        raise Phase1Error("Expected FP32 SAM.")
    (out / "cases").mkdir()
    requests = sorted(
        read_json(out / "inputs/requests.json"),
        key=lambda r: (r["case_id"], r["roi_factor"], r["group"]),
    )
    outcomes, encodings = [], []
    cache_key = None
    image_id = None
    for request in requests:
        destination = out / "cases" / request["condition_id"] / request["case_id"]
        destination.mkdir(parents=True)
        result = request | {"status": "failed", "failure_reason": ""}
        try:
            if request["reused_baseline"]:
                reused = out / "inputs/reused" / request["case_id"]
                previous = read_json(reused / "result.json")
                for field in ("point_xy", "roi_xyxy"):
                    if previous[field] != request[field]:
                        raise Phase1Error("Baseline prompt geometry differs.")
                logits = np.load(reused / "logits.npy", allow_pickle=False)
                scores = previous["candidate_scores"]
                decode_seconds = None
            else:
                if image_id != request["image_id"]:
                    image_path = out / request["image_path"]
                    if sha256_file(image_path) != request["image_sha256"]:
                        raise Phase1Error("Frozen image changed.")
                    loaded_image = np.array(Image.open(image_path).convert("RGB"))
                    image, image_id = loaded_image, request["image_id"]
                key = (image_id, tuple(request["roi_xyxy"]))
                x0, y0, x1, y1 = request["roi_xyxy"]
                with torch.inference_mode():
                    if key != cache_key:
                        t0 = time.perf_counter()
                        predictor.set_image(image[y0:y1, x0:x1].copy())
                        encodings.append(
                            {
                                "case_id": request["case_id"],
                                "roi_factor": request["roi_factor"],
                                "seconds": time.perf_counter() - t0,
                            }
                        )
                        cache_key = key
                    t0 = time.perf_counter()
                    logits, scores, low_res = predictor.predict(
                        point_coords=np.asarray(request["point_coords_in_roi"]),
                        point_labels=np.array([1]),
                        box=np.asarray(request["box_coords_in_roi"])
                        if request["box_coords_in_roi"] is not None
                        else None,
                        mask_input=None,
                        multimask_output=True,
                        return_logits=True,
                    )
                    decode_seconds = time.perf_counter() - t0
                    if not np.isfinite(low_res).all():
                        raise Phase1Error("Nonfinite low-resolution SAM output.")
                scores = scores.tolist()
            mask, selected = selected_mask(
                logits,
                scores,
                (request["image_height"], request["image_width"]),
                request["roi_xyxy"],
            )
            if request["reused_baseline"] and not np.array_equal(
                mask, np.array(Image.open(reused / "mask.png"))
            ):
                raise Phase1Error("Reused logits do not reproduce the baseline mask.")
            np.save(destination / "logits.npy", logits, allow_pickle=False)
            Image.fromarray(mask).save(destination / "mask.png")
            result.update(
                status="passed",
                candidate_scores=scores,
                selected_candidate=selected,
                decode_seconds=decode_seconds,
                mask_path=str((destination / "mask.png").relative_to(out)),
                logits_path=str((destination / "logits.npy").relative_to(out)),
            )
        except Exception as exc:
            LOGGER.exception(
                "Prediction failed: %s %s", request["condition_id"], request["case_id"]
            )
            result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        save_json(destination / "result.json", result)
        outcomes.append(result)
        if len(outcomes) % 12 == 0:
            event("prediction_progress", completed=len(outcomes), expected=144)
    save_json(out / "predictions.json", outcomes)
    save_json(
        out / "runtime.json",
        {
            "sam_source": source,
            "checkpoint_sha256": checkpoint_sha,
            "model_load_seconds": load_seconds,
            "wall_seconds": time.perf_counter() - start,
            "image_encodings": encodings,
            "new_inference_cases": sum(not r["reused_baseline"] for r in outcomes),
            "reused_cases": sum(r["reused_baseline"] for r in outcomes),
            "seed": 42,
            "threads": 4,
            "device": "cpu",
            "dtype": "float32",
            "mask_input": None,
            "threshold": 0,
            "postprocessing": "none",
            "platform": platform.platform(),
            "environment": {d.metadata["Name"]: d.version for d in distributions()},
        },
    )
    save_json(
        out / "predictions_manifest.json",
        {
            "predictions_sha256": sha256_file(out / "predictions.json"),
            "case_files_sha256": hash_tree(out / "cases"),
        },
    )


def check_predictions(out: Path) -> None:
    manifest = read_json(out / "predictions_manifest.json")
    if sha256_file(out / "predictions.json") != manifest["predictions_sha256"]:
        raise Phase1Error("Predictions changed after inference.")
    check_hashes(out / "cases", manifest["case_files_sha256"])


def evaluate(cfg: dict[str, Any]) -> None:
    from animal_eye_metrology.research.ablation_metrics import compute

    out = cfg["output_directory"]
    check_predictions(out)
    if verify(cfg["primary_reference"])["snapshot_sha256"] != cfg["primary_snapshot_sha256"]:
        raise Phase1Error("Primary reference snapshot differs.")
    (out / "evaluation_inputs").mkdir()
    shutil.copytree(cfg["primary_reference"], out / "evaluation_inputs/reference")
    tables = compute(out)
    for name, rows in tables.items():
        write_csv(out / name, rows)
    save_json(out / "evaluation_values.json", tables)
    event("evaluation_completed", tables={k: len(v) for k, v in tables.items()})


def finalize(cfg: dict[str, Any]) -> None:
    out = cfg["output_directory"]
    provenance = out / "provenance"
    provenance.mkdir()
    for path in [cfg["source"], cfg["sam_runtime"]] + [
        module_source(name)
        for name in (
            "paths.py",
            "_compat.py",
            "prompt_roi_ablation.py",
            "ablation_metrics.py",
            "ablation_visualization.py",
            "eye_poc.py",
            "eye_comparison.py",
            "metrology.py",
            "measurement_baseline.py",
            "auto_eye_localization.py",
        )
    ]:
        shutil.copyfile(path, provenance / path.name)
    save_json(
        out / "run_manifest.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "conditions": 12,
            "image_ids": cfg["image_ids"],
            "independent_images": 3,
            "trained": False,
            "source_config_sha256": sha256_file(cfg["source"]),
            "primary_snapshot_sha256": cfg["primary_snapshot_sha256"],
            "files_sha256": hash_tree(out),
        },
    )


def verify_experiment(out: Path) -> dict[str, Any]:
    from animal_eye_metrology.research.ablation_metrics import compute

    manifest = read_json(out / "run_manifest.json")
    check_hashes(out, manifest["files_sha256"])
    check_hashes(out / "inputs", read_json(out / "inputs/manifest.json")["files_sha256"])
    check_predictions(out)
    cases = read_json(out / "inputs/cases.json")
    expected = {
        (c["condition_id"], r["case_id"]): request_for_case(r, c)
        for r in cases
        for c in conditions()
    }
    predictions = read_json(out / "predictions.json")
    if len(predictions) != 144 or {(r["condition_id"], r["case_id"]) for r in predictions} != set(
        expected
    ):
        raise Phase1Error("Incomplete condition matrix.")
    masks = 0
    for row in predictions:
        request = expected[(row["condition_id"], row["case_id"])]
        if any(row[k] != v for k, v in request.items()):
            raise Phase1Error("Prompt or ROI changed from the frozen factorial design.")
        if row["status"] != "passed":
            continue
        logits = np.load(out / row["logits_path"], allow_pickle=False)
        reconstructed, index = selected_mask(
            logits,
            row["candidate_scores"],
            (row["image_height"], row["image_width"]),
            row["roi_xyxy"],
        )
        if row["selected_candidate"] != index or not np.array_equal(
            reconstructed, np.array(Image.open(out / row["mask_path"]))
        ):
            raise Phase1Error("Saved candidate scores/logits do not reproduce selected mask.")
        masks += 1
    tables = compute(out)
    if tables != read_json(out / "evaluation_values.json"):
        raise Phase1Error("Evaluation did not reproduce.")
    for name, rows in tables.items():
        with (out / name).open() as handle:
            actual = list(csv.DictReader(handle))
        if actual != [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]:
            raise Phase1Error(f"CSV did not reproduce: {name}")
    return {
        "status": "passed",
        "files_checked": len(manifest["files_sha256"]),
        "conditions_verified": 12,
        "masks_reconstructed": masks,
        "eye_rows_recomputed": len(tables["eye_metrics.csv"]),
        "distance_rows_recomputed": len(tables["measurement_comparison.csv"]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prompt_roi_ablation.toml"))
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "run", "evaluate", "finalize", "verify")
    )
    args = parser.parse_args()
    cfg = config(args.config)
    start = time.perf_counter()
    event("stage_started", stage=args.stage, timestamp_utc=datetime.now(UTC).isoformat())
    action = {
        "prepare": prepare,
        "run": run_predictions,
        "evaluate": evaluate,
        "finalize": finalize,
        "verify": lambda c: print(json.dumps(verify_experiment(c["output_directory"]))),
    }[args.stage]
    action(cfg)
    event("stage_completed", stage=args.stage, wall_seconds=time.perf_counter() - start)


if __name__ == "__main__":
    main()
