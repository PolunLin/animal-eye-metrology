"""Approved six-condition geometric eye-box experiment with isolated model inputs."""

import argparse
import csv
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

from animal_eye_metrology.auto_eye_localization import read_json, save_json
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.eye_poc import load_predictor, settings, validate_source
from animal_eye_metrology.measurement_baseline import verify, write_csv
from animal_eye_metrology.prompt_roi_ablation import (
    check_hashes,
    check_predictions,
    event,
    hash_tree,
    selected_mask,
)
from animal_eye_metrology.verification import sha256_file

LOGGER = logging.getLogger(__name__)
GAMMAS = (0.3, 0.5, 0.7)
PRIMARY = "auto_box_k050_point"
BASELINES = ("A_r100", "A_r050", "A_r150", "C_r100")
CASE_FIELDS = {
    "case_id",
    "image_id",
    "prediction_id",
    "eye_side",
    "image_width",
    "image_height",
    "image_path",
    "image_sha256",
    "hrnet_point_xy",
    "hrnet_score",
    "hrnet_eye_distance_px",
    "roi_xyxy",
}


def config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = tomllib.loads(path.read_text())
    for field in (
        "output_directory",
        "automatic_run",
        "ablation_run",
        "primary_reference",
        "sam_runtime",
        "proposal",
    ):
        cfg[field] = (path.parent / cfg[field]).resolve()
    cfg["source"] = path
    if (
        cfg["image_ids"] != [20247, 125405, 333069]
        or tuple(cfg["gammas"]) != GAMMAS
        or cfg["minimum_box_side_px"] != 8.0
        or cfg["roi_factor"] != 1.0
        or cfg["primary_condition"] != PRIMARY
        or cfg["expected_eyes"] != 12
        or cfg["seed"] != 42
        or cfg["threads"] != 4
    ):
        raise Phase1Error("Configuration differs from the approved six-condition design.")
    return cfg


def conditions() -> list[dict[str, Any]]:
    return [
        dict(
            condition_id=f"auto_box_k{round(gamma * 100):03d}_{mode}",
            group=mode,
            point_source="hrnet" if mode == "point" else "none",
            box_source="geometry",
            roi_factor=1.0,
            gamma=gamma,
            is_primary=gamma == 0.5 and mode == "point",
        )
        for gamma in GAMMAS
        for mode in ("point", "only")
    ]


def request_for_case(case: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    """Derive a float box using predictions only; preserve geometry failures as requests."""
    if set(case) != CASE_FIELDS or condition not in conditions():
        raise Phase1Error("Unapproved case fields or condition; manual inputs are prohibited.")
    result = {
        **case,
        **condition,
        "geometry_status": "failed",
        "geometry_failure_reason": "",
        "point_xy": None,
        "point_coords_in_roi": None,
        "point_labels": None,
        "raw_box_xyxy": None,
        "box_xyxy": None,
        "box_coords_in_roi": None,
        "box_side_length_px": None,
        "box_clipped": False,
        "human_assisted": False,
        "visibility_status": "unverified",
    }
    try:
        d = case["hrnet_eye_distance_px"]
        if d is None or not math.isfinite(d) or d <= 0:
            raise Phase1Error("Invalid frozen eye separation.")
        point = np.asarray(case["hrnet_point_xy"], dtype=np.float64)
        roi = case["roi_xyxy"]
        if (
            point.shape != (2,)
            or not np.isfinite(point).all()
            or not math.isfinite(case["hrnet_score"])
            or len(roi) != 4
            or any(type(v) is not int for v in roi)
            or not (
                0 <= roi[0] < roi[2] <= case["image_width"]
                and 0 <= roi[1] < roi[3] <= case["image_height"]
            )
            or not (roi[0] <= point[0] < roi[2] and roi[1] <= point[1] < roi[3])
        ):
            raise Phase1Error("Invalid frozen point or original-grid ROI.")
        length = max(8.0, condition["gamma"] * d)
        raw = np.concatenate((point - length / 2, point + length / 2))
        clipped = np.array(
            [max(raw[0], roi[0]), max(raw[1], roi[1]), min(raw[2], roi[2]), min(raw[3], roi[3])]
        )
        box = clipped.astype(np.float32)
        if not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
            raise Phase1Error("Invalid generated box.")
        result.update(
            geometry_status="passed",
            raw_box_xyxy=raw.tolist(),
            box_xyxy=box.tolist(),
            box_coords_in_roi=(box - roi[:2] * 2).tolist(),
            box_side_length_px=length,
            box_clipped=not np.array_equal(raw, clipped),
        )
        if condition["group"] == "point":
            result.update(
                point_xy=case["hrnet_point_xy"],
                point_labels=[1],
                point_coords_in_roi=(point.astype(np.float32)[None] - roi[:2]).tolist(),
            )
    except (Phase1Error, TypeError, ValueError) as exc:
        result["geometry_failure_reason"] = f"{type(exc).__name__}: {exc}"
        # Keep malformed geometry as a strict-JSON failed record, never NaN/Infinity.
        for field in ("hrnet_eye_distance_px", "hrnet_point_xy", "hrnet_score"):
            value = result[field]
            if isinstance(value, list):
                result[field] = [
                    None if isinstance(v, float) and not math.isfinite(v) else v for v in value
                ]
            elif isinstance(value, float) and not math.isfinite(value):
                result[field] = None
    return result


def build_cases(out: Path) -> list[dict[str, Any]]:
    inputs = out / "inputs"
    poses = read_json(inputs / "pose_results.json")["results"]
    pose_index = {r["prediction_id"]: r for r in poses}
    sources = {r["prediction_id"]: r for r in read_json(inputs / "pose_cases.json")}
    baseline = read_json(inputs / "baseline_segmentation.json")["results"]
    if (
        len(poses) != 6
        or len(baseline) != 12
        or any(r["status"] != "passed" for r in poses + baseline)
    ):
        raise Phase1Error("Expected six frozen successful poses and twelve baseline eyes.")
    cases = []
    for row in baseline:
        pid, side = row["prediction_id"], row["eye_side"]
        pose, source = pose_index[pid], sources[pid]
        points = np.asarray(pose["keypoints_xy"][:2], dtype=float)
        point = points[0 if side == "left" else 1].tolist()
        if point != row["point_xy"] or source["image_id"] != row["image_id"]:
            raise Phase1Error("Frozen pose and baseline identity/point differ.")
        image_path = inputs / source["image"]
        if sha256_file(image_path) != source["image_sha256"]:
            raise Phase1Error("Frozen image differs.")
        with Image.open(image_path) as opened:
            width, height = opened.size
        cases.append(
            dict(
                case_id=f"{pid}_{side}",
                image_id=row["image_id"],
                prediction_id=pid,
                eye_side=side,
                image_width=width,
                image_height=height,
                image_path=str(image_path.relative_to(out)),
                image_sha256=source["image_sha256"],
                hrnet_point_xy=point,
                hrnet_score=row["point_score"],
                hrnet_eye_distance_px=float(np.linalg.norm(points[0] - points[1])),
                roi_xyxy=row["roi_xyxy"],
            )
        )
    return sorted(cases, key=lambda r: r["case_id"])


def check_inputs(out: Path) -> list[dict[str, Any]]:
    check_hashes(out / "inputs", read_json(out / "inputs/manifest.json")["files_sha256"])
    cases = read_json(out / "inputs/cases.json")
    if cases != build_cases(out) or read_json(out / "inputs/conditions.json") != conditions():
        raise Phase1Error("Frozen case geometry or design changed.")
    requests = [request_for_case(c, cond) for c in cases for cond in conditions()]
    if requests != read_json(out / "inputs/requests.json") or len(requests) != 72:
        raise Phase1Error("Incomplete or modified experiment matrix.")
    return requests


def prepare(cfg: dict[str, Any]) -> None:
    out, source = cfg["output_directory"], cfg["automatic_run"]
    manifest = read_json(source / "run_manifest.json")
    check_hashes(source, manifest["files_sha256"])
    inputs = out / "inputs"
    inputs.mkdir(parents=True)
    shutil.copytree(source / "inputs/images", inputs / "images")
    for origin, target in (
        ("inputs/pose_cases.json", "pose_cases.json"),
        ("inputs/animal_predictions.json", "animal_predictions.json"),
        ("pose/pose_results.json", "pose_results.json"),
        ("segmentation.json", "baseline_segmentation.json"),
    ):
        shutil.copyfile(source / origin, inputs / target)
    cases = build_cases(out)
    if sorted({r["image_id"] for r in cases}) != cfg["image_ids"]:
        raise Phase1Error("Images differ from the approved scope.")
    requests = [request_for_case(c, cond) for c in cases for cond in conditions()]
    save_json(inputs / "cases.json", cases)
    save_json(inputs / "conditions.json", conditions())
    save_json(inputs / "requests.json", requests)
    save_json(
        inputs / "manifest.json",
        {
            "dataset": "COCO 2017 validation; same three development images",
            "independent_images": 3,
            "eye_conditions": 72,
            "trained": False,
            "model_inputs_include_manual_prompts_or_eye_gt": False,
            "eye_distance_dtype": "float64, identical to original eye_crops",
            "roi_origin": "frozen original automatic segmentation; identical to A_r100",
            "source_manifest_sha256": sha256_file(source / "run_manifest.json"),
            "files_sha256": hash_tree(inputs),
        },
    )
    event(
        "inputs_frozen",
        conditions=6,
        eye_conditions=len(requests),
        geometry_failures=sum(r["geometry_status"] != "passed" for r in requests),
    )


def run_predictions(cfg: dict[str, Any]) -> None:
    import torch

    out = cfg["output_directory"]
    requests = check_inputs(out)
    runtime = settings(cfg["sam_runtime"])
    spec = runtime["models"]["sam2_1_tiny"]
    source = validate_source(runtime, spec)
    checkpoint = runtime["weights_directory"] / spec["checkpoint"]
    baseline = read_json(out / "inputs/baseline_segmentation.json")
    checkpoint_sha = sha256_file(checkpoint)
    if source != baseline["sam_source"] or checkpoint_sha != baseline["checkpoint_sha256"]:
        raise Phase1Error("SAM differs from the frozen automatic baseline.")
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
    outcomes, encodings = [], []
    cache_key, image_id = None, None
    for request in requests:
        destination = out / "cases" / request["condition_id"] / request["case_id"]
        destination.mkdir(parents=True)
        result = request | {"status": "failed", "failure_reason": ""}
        try:
            if request["geometry_status"] != "passed":
                raise Phase1Error(request["geometry_failure_reason"])
            if image_id != request["image_id"]:
                path = out / request["image_path"]
                if sha256_file(path) != request["image_sha256"]:
                    raise Phase1Error("Frozen image changed.")
                with Image.open(path) as opened:
                    loaded = np.array(opened.convert("RGB"))
                if loaded.shape[:2] != (request["image_height"], request["image_width"]):
                    raise Phase1Error("Frozen image dimensions differ.")
                image, image_id = loaded, request["image_id"]
            key = (image_id, tuple(request["roi_xyxy"]))
            x0, y0, x1, y1 = request["roi_xyxy"]
            with torch.inference_mode():
                if key != cache_key:
                    t0 = time.perf_counter()
                    predictor.set_image(image[y0:y1, x0:x1].copy())
                    encodings.append(
                        {"case_id": request["case_id"], "seconds": time.perf_counter() - t0}
                    )
                    cache_key = key
                t0 = time.perf_counter()
                logits, scores, low_res = predictor.predict(
                    point_coords=np.asarray(request["point_coords_in_roi"])
                    if request["point_coords_in_roi"] is not None
                    else None,
                    point_labels=np.asarray(request["point_labels"])
                    if request["point_labels"] is not None
                    else None,
                    box=np.asarray(request["box_coords_in_roi"]),
                    mask_input=None,
                    multimask_output=True,
                    return_logits=True,
                )
                decode_seconds = time.perf_counter() - t0
                if not np.isfinite(low_res).all():
                    raise Phase1Error("Nonfinite low-resolution logits.")
            scores = scores.tolist()
            mask, selected = selected_mask(
                logits,
                scores,
                (request["image_height"], request["image_width"]),
                request["roi_xyxy"],
            )
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
            cache_key = None
            LOGGER.exception(
                "Prediction failed: %s %s", request["condition_id"], request["case_id"]
            )
            result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        save_json(destination / "result.json", result)
        outcomes.append(result)
        if len(outcomes) % 12 == 0:
            event("prediction_progress", completed=len(outcomes), expected=72)
    save_json(out / "predictions.json", outcomes)
    save_json(
        out / "runtime.json",
        dict(
            sam_source=source,
            checkpoint_sha256=checkpoint_sha,
            model_load_seconds=load_seconds,
            wall_seconds=time.perf_counter() - start,
            image_encodings=encodings,
            new_inference_cases=len(outcomes),
            seed=42,
            threads=4,
            device="cpu",
            dtype="float32",
            mask_input=None,
            threshold=0,
            postprocessing="none",
            platform=platform.platform(),
            environment={d.metadata["Name"]: d.version for d in distributions()},
        ),
    )
    save_json(
        out / "predictions_manifest.json",
        {
            "predictions_sha256": sha256_file(out / "predictions.json"),
            "case_files_sha256": hash_tree(out / "cases"),
        },
    )


def evaluate(cfg: dict[str, Any]) -> None:
    from animal_eye_metrology.auto_box_metrics import compute

    out, source = cfg["output_directory"], cfg["ablation_run"]
    check_predictions(out)
    check_hashes(source, read_json(source / "run_manifest.json")["files_sha256"])
    if verify(cfg["primary_reference"])["snapshot_sha256"] != cfg["primary_snapshot_sha256"]:
        raise Phase1Error("Reference snapshot differs.")
    folder = out / "evaluation_inputs"
    folder.mkdir()
    shutil.copytree(cfg["primary_reference"], folder / "reference")
    shutil.copyfile(
        cfg["automatic_run"] / "evaluation_inputs/animal_matches.csv", folder / "animal_matches.csv"
    )
    baseline = folder / "baseline"
    baseline.mkdir()
    for name in ("evaluation_values.json", "run_manifest.json"):
        shutil.copyfile(source / name, baseline / name)
    rows = [r for r in read_json(source / "predictions.json") if r["condition_id"] in BASELINES]
    for row in rows:
        for field in ("mask_path", "logits_path"):
            if field in row:
                dest = baseline / row[field]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / row[field], dest)
    save_json(baseline / "predictions.json", rows)
    originals = {r["case_id"]: r for r in read_json(out / "inputs/cases.json")}
    for row in rows:
        if row["condition_id"] == "A_r100":
            case = originals[f"{row['prediction_id']}_{row['eye_side']}"]
            if case["roi_xyxy"] != row["roi_xyxy"] or case["hrnet_point_xy"] != row["point_xy"]:
                raise Phase1Error("New experiment differs from A_r100 ROI or localization.")
    save_json(folder / "manifest.json", {"files_sha256": hash_tree(folder)})
    tables = compute(out)
    for name, table in tables.items():
        write_csv(out / name, table)
    save_json(out / "evaluation_values.json", tables)
    event("evaluation_completed", tables={k: len(v) for k, v in tables.items()})


def finalize(cfg: dict[str, Any]) -> None:
    out = cfg["output_directory"]
    provenance = out / "provenance"
    provenance.mkdir()
    paths = [cfg["source"], cfg["sam_runtime"], cfg["proposal"]]
    paths += [
        Path(__file__).with_name(n)
        for n in (
            "auto_eye_box.py",
            "auto_box_metrics.py",
            "auto_box_visualization.py",
            "ablation_metrics.py",
            "prompt_roi_ablation.py",
            "auto_eye_localization.py",
            "eye_poc.py",
            "eye_comparison.py",
            "metrology.py",
            "measurement_baseline.py",
        )
    ]
    for path in paths:
        shutil.copyfile(path, provenance / path.name)
    save_json(
        out / "run_manifest.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "conditions": 6,
            "independent_images": 3,
            "primary_condition": PRIMARY,
            "trained": False,
            "source_config_sha256": sha256_file(cfg["source"]),
            "primary_snapshot_sha256": cfg["primary_snapshot_sha256"],
            "files_sha256": hash_tree(out),
        },
    )


def verify_experiment(out: Path) -> dict[str, Any]:
    from animal_eye_metrology.auto_box_metrics import compute

    manifest = read_json(out / "run_manifest.json")
    check_hashes(out, manifest["files_sha256"])
    requests = check_inputs(out)
    check_predictions(out)
    check_hashes(
        out / "evaluation_inputs",
        read_json(out / "evaluation_inputs/manifest.json")["files_sha256"],
    )
    expected = {(r["condition_id"], r["case_id"]): r for r in requests}
    predictions = read_json(out / "predictions.json")
    if len(predictions) != 72 or {(r["condition_id"], r["case_id"]) for r in predictions} != set(
        expected
    ):
        raise Phase1Error("Incomplete predictions.")
    reconstructed = 0
    for row in predictions:
        request = expected[(row["condition_id"], row["case_id"])]
        if any(row[k] != v for k, v in request.items()):
            raise Phase1Error("Prediction inputs differ from frozen geometry.")
        if row["status"] == "passed":
            mask, index = selected_mask(
                np.load(out / row["logits_path"], allow_pickle=False),
                row["candidate_scores"],
                (row["image_height"], row["image_width"]),
                row["roi_xyxy"],
            )
            if index != row["selected_candidate"] or not np.array_equal(
                mask, np.array(Image.open(out / row["mask_path"]))
            ):
                raise Phase1Error("Saved mask is not the unmodified highest-score candidate.")
            reconstructed += 1
    tables = compute(out)
    if tables != read_json(out / "evaluation_values.json"):
        raise Phase1Error("Evaluation did not reproduce.")
    for name, rows in tables.items():
        with (out / name).open() as handle:
            actual = list(csv.DictReader(handle))
        if actual != [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows]:
            raise Phase1Error(f"CSV did not reproduce: {name}")
    return dict(
        status="passed",
        files_checked=len(manifest["files_sha256"]),
        conditions_verified=6,
        masks_reconstructed=reconstructed,
        eye_rows_recomputed=len(tables["eye_metrics.csv"]),
        distance_rows_recomputed=len(tables["measurement_comparison.csv"]),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/auto_eye_box.toml"))
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "run", "evaluate", "finalize", "verify")
    )
    args = parser.parse_args()
    cfg = config(args.config)
    start = time.perf_counter()
    event("stage_started", stage=args.stage, timestamp_utc=datetime.now(UTC).isoformat())
    if args.stage == "verify":
        import json

        print(json.dumps(verify_experiment(cfg["output_directory"])))
    else:
        {"prepare": prepare, "run": run_predictions, "evaluate": evaluate, "finalize": finalize}[
            args.stage
        ](cfg)
    event("stage_completed", stage=args.stage, wall_seconds=time.perf_counter() - start)


if __name__ == "__main__":
    main()
