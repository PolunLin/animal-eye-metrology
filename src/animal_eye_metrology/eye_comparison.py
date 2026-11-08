"""Fixed human prompts, three pretrained models and frozen mask-centroid references."""

import argparse
import fcntl
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image

from animal_eye_metrology.coco import load_coco
from animal_eye_metrology.config import load_config
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.eye_annotations import load_targets
from animal_eye_metrology.eye_poc import settings as runtime_settings
from animal_eye_metrology.eye_prompts import (
    FIELDS,
    crop_for_prompt,
    latest_prompt,
    load_prompts,
)
from animal_eye_metrology.measurement_baseline import derive, safe_path, verify, write_csv
from animal_eye_metrology.metrology import build_measurements, mask_centroid, mask_overlap
from animal_eye_metrology.phase2 import read_csv, write_json
from animal_eye_metrology.verification import sha256_file

LOGGER = logging.getLogger(__name__)
MODELS = ("sam2_1_tiny", "sam_vit_b", "mobilesam")


def eye_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row["image_id"]), int(row["annotation_id"]), row["eye_side"]


def case_name(key: tuple[int, int, str]) -> str:
    return f"{key[0]}_{key[1]}_{key[2]}"


def eye_statistics(predicted: np.ndarray | None, reference: np.ndarray) -> dict[str, Any]:
    """Missing inference is distinct from an empty prediction and is never a zero distance."""
    center_ref = mask_centroid(reference)
    if center_ref is None:
        raise Phase1Error("This fixed evaluation requires nonempty reference masks.")
    base = {
        "reference_area_px": int(reference.sum()),
        "reference_cx_px": center_ref[0],
        "reference_cy_px": center_ref[1],
    }
    if predicted is None:
        return base | {
            "prediction_status": "inference_failed",
            "predicted_area_px": None,
            "intersection_px": None,
            "union_px": None,
            "iou": None,
            "dice": None,
            "predicted_cx_px": None,
            "predicted_cy_px": None,
            "centroid_dx_px": None,
            "centroid_dy_px": None,
            "centroid_error_px": None,
        }
    overlap = mask_overlap(predicted, reference)
    center_pred = mask_centroid(predicted)
    dx = center_pred[0] - center_ref[0] if center_pred else None
    dy = center_pred[1] - center_ref[1] if center_pred else None
    return (
        base
        | overlap
        | {
            "prediction_status": "nonempty" if center_pred else "empty",
            "predicted_area_px": int(predicted.sum()),
            "predicted_cx_px": center_pred[0] if center_pred else None,
            "predicted_cy_px": center_pred[1] if center_pred else None,
            "centroid_dx_px": dx,
            "centroid_dy_px": dy,
            "centroid_error_px": math.hypot(dx, dy) if center_pred else None,
        }
    )


def distance_statistics(predicted: float | None, reference: float) -> dict[str, Any]:
    delta = predicted - reference if predicted is not None else None
    return {
        "predicted_distance_px": predicted,
        "reference_distance_px": reference,
        "signed_error_px": delta,
        "absolute_error_px": abs(delta) if delta is not None else None,
        "relative_absolute_error_percent": 100 * abs(delta) / reference
        if delta is not None and reference
        else None,
        "measurement_status": "Defined" if predicted is not None else "Undefined",
    }


def read_settings(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    with path.open("rb") as handle:
        cfg = tomllib.load(handle)
    for key in ("runtime_config", "primary_reference", "repeat_reference", "output_directory"):
        cfg[key] = (path.parent / cfg[key]).resolve()
    cfg["source"] = path
    runtime = runtime_settings(cfg["runtime_config"])
    out = cfg["output_directory"]
    if out == runtime["output_directory"] or not out.is_relative_to(runtime["output_directory"]):
        raise Phase1Error("Use a new comparison subdirectory of the approved POC output.")
    for reference in (
        cfg["primary_reference"],
        cfg["repeat_reference"],
        runtime["prompts_directory"],
    ):
        if out.is_relative_to(reference) or reference.is_relative_to(out):
            raise Phase1Error("Comparison outputs must be separate from human/reference inputs.")
    if (cfg["expected_images"], cfg["expected_instances"], cfg["expected_eyes"]) != (3, 6, 12):
        raise Phase1Error(
            "This approved comparison requires three images, six animals, twelve eyes."
        )
    return cfg, runtime


def freeze_inputs(cfg: dict[str, Any], runtime: dict[str, Any], out: Path) -> dict[str, Any]:
    inputs = out / "inputs"
    inputs.mkdir()
    data = load_config(runtime["phase1_config"])
    phase1 = json.loads((data.output / "run_manifest.json").read_text())
    dataset = load_coco(data.annotations, expected_sha256=phase1["dataset"]["annotations_sha256"])
    targets = load_targets(runtime["proposal_csv"], dataset)
    keys = [
        (int(t["image_id"]), int(t["annotation_id"]), side)
        for t in targets
        for side in ("left", "right")
    ]
    if len(targets) != 6 or len({k[0] for k in keys}) != 3 or len(set(keys)) != 12:
        raise Phase1Error("Prompt target scope differs from the approved comparison.")
    for tag in ("primary", "repeat"):
        source = cfg[f"{tag}_reference"]
        result = verify(source)
        if result["snapshot_sha256"] != cfg[f"{tag}_snapshot_sha256"]:
            raise Phase1Error("Reference snapshot differs from the approved fixed version.")
        shutil.copytree(source, inputs / f"reference_{tag}")
        verify(inputs / f"reference_{tag}")
        snapshot = json.loads((source / "frozen/snapshot_manifest.json").read_text())
        ref_centers, _ = derive(source / "frozen")
        ref_keys = {eye_key(row) for row in ref_centers}
        if snapshot["dataset_annotations_sha256"] != dataset.source_sha256:
            raise Phase1Error("Reference dataset digest differs from the original image source.")
        if tag == "primary" and ref_keys != set(keys):
            raise Phase1Error("Primary reference does not match the twelve approved eyes.")
        if tag == "repeat" and not (
            len(ref_keys) == 4
            and ref_keys.issubset(keys)
            and len({k[0] for k in ref_keys}) == 1
            and len({k[:2] for k in ref_keys}) == 2
        ):
            raise Phase1Error("Repeat reference must contain the four repeated eyes only.")
    with (runtime["prompts_directory"] / ".prompts.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_prompts(runtime["prompts_directory"], targets)
        selected = [latest_prompt(rows, key) for key in keys]
        if any(row is None for row in selected):
            raise Phase1Error("All twelve human prompts must be saved before inference.")
        write_csv(inputs / "prompts.csv", selected, FIELDS)
        prompt_source_sha = sha256_file(runtime["prompts_directory"] / "prompts.csv")
    shutil.copyfile(runtime["proposal_csv"], inputs / "proposed_instances.csv")
    (out / "cases").mkdir()
    for row in selected:
        key = eye_key(row)
        target = next(
            t for t in targets if (int(t["image_id"]), int(t["annotation_id"])) == key[:2]
        )
        image_path = data.images / target["file_name"]
        if sha256_file(image_path) != row["image_sha256"]:
            raise Phase1Error("Original image differs from the frozen human prompt.")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if image.size != (int(target["image_width"]), int(target["image_height"])):
            raise Phase1Error("Original-image dimensions differ from the prompt target.")
        case = out / "cases" / case_name(key)
        case.mkdir()
        image.save(case / "input.png")
        write_json(
            case / "request.json",
            {
                "kind": "fixed_prompt_eye_segmentation_POC_prediction",
                "prompt_source": "manual_original_image",
                "prompt_record": row,
                "image_id": key[0],
                "annotation_id": key[1],
                "eye_side": key[2],
                "source_image_sha256": row["image_sha256"],
                "saved_input_sha256": sha256_file(case / "input.png"),
                "point_xy": [float(row[k]) for k in ("point_x", "point_y")],
                "box_xyxy": [float(row[k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1")],
                "roi_xyxy": list(crop_for_prompt(row, image.width, image.height)),
            },
        )
    # Paths in this snapshot are explicit run-time inputs, not paths relative to a moved config.
    lines = [
        f"{k} = {json.dumps(str(runtime[k]))}"
        for k in (
            "phase1_config",
            "proposal_csv",
            "prompts_directory",
            "output_directory",
            "weights_directory",
        )
    ]
    lines += [f"seed = {runtime['seed']}", f"threads = {runtime['threads']}", 'device = "cpu"']
    for model in MODELS:
        lines.append(f"\n[models.{model}]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in runtime["models"][model].items()]
    (inputs / "runtime.toml").write_text("\n".join(lines) + "\n")
    provenance = out / "provenance"
    provenance.mkdir()
    for source in (
        Path(__file__),
        Path(__file__).with_name("eye_poc.py"),
        Path(__file__).with_name("eye_prompts.py"),
        Path(__file__).with_name("metrology.py"),
        Path(__file__).with_name("visualization.py"),
        cfg["source"],
        runtime["source"],
    ):
        shutil.copyfile(source, provenance / source.name)
    frozen_files = [p for p in inputs.rglob("*") if p.is_file()]
    frozen_files += [p for p in (out / "cases").rglob("*") if p.is_file()]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": data.dataset_name,
        "dataset_annotations_sha256": dataset.source_sha256,
        "images": 3,
        "animals": 6,
        "eyes": 12,
        "models": list(MODELS),
        "selection": "all twelve approved eyes; latest saved human prompt per anatomical identity",
        "prompt_source_csv_sha256": prompt_source_sha,
        "source_prompt_history_rows": len(rows),
        "primary_snapshot_sha256": cfg["primary_snapshot_sha256"],
        "repeat_snapshot_sha256": cfg["repeat_snapshot_sha256"],
        "case_order": [case_name(eye_key(row)) for row in selected],
        "files_sha256": {str(p.relative_to(out)): sha256_file(p) for p in frozen_files},
        "seed": 42,
        "sample_random_seed": None,
        "trained": False,
        "model_inputs_include_eye_ground_truth": False,
        "reference_roles": {
            "primary": "all twelve eyes",
            "repeat": "four dog eyes, sensitivity only",
        },
    }
    write_json(out / "input_manifest.json", manifest)
    return manifest


def run_predictions(out: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = []
    for model in MODELS:
        for name in inputs["case_order"]:
            case = out / "cases" / name
            LOGGER.info(
                json.dumps({"event": "eye_prediction_started", "model": model, "case": name})
            )
            started = time.perf_counter()
            with (case / f"{model}.log").open("x") as log:
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "animal_eye_metrology.eye_poc",
                            "--config",
                            str(out / "inputs/runtime.toml"),
                            "--worker",
                            model,
                            "--directory",
                            str(case),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=600,
                    )
                    code, reason = (
                        result.returncode,
                        "" if result.returncode == 0 else "worker_failed",
                    )
                except subprocess.TimeoutExpired:
                    code, reason = None, "worker_timeout"
            record = {
                "model": model,
                "case": name,
                "returncode": code,
                "status": "passed" if code == 0 else "failed",
                "failure_reason": reason,
                "subprocess_wall_seconds": time.perf_counter() - started,
            }
            outcomes.append(record)
            LOGGER.info(json.dumps({"event": "eye_prediction_finished", **record}))
    write_json(out / "prediction_outcomes.json", outcomes)
    return outcomes


def evaluate(out: Path, reference_tag: str) -> tuple[list[dict], list[dict]]:
    frozen = out / "inputs" / f"reference_{reference_tag}"
    verify(frozen)
    reference_centers, reference_distances = derive(frozen / "frozen")
    index = {eye_key(r): r for r in reference_centers}
    prompts = load_prompts(out / "inputs", read_csv(out / "inputs/proposed_instances.csv"))
    outcomes = {
        (r["model"], r["case"]): r
        for r in json.loads((out / "prediction_outcomes.json").read_text())
    }
    eye_rows, measurement_rows = [], []
    for model in MODELS:
        centers = []
        for prompt in prompts:
            key = eye_key(prompt)
            if key not in index:
                continue
            ref = index[key]
            with Image.open(frozen / ref["mask_path"]) as opened:
                mask_ref = np.array(opened) // 255
            case = out / "cases" / case_name(key)
            request = json.loads((case / "request.json").read_text())
            outcome = outcomes[(model, case.name)]
            mask_path = case / model / "mask.png"
            predicted = None
            if outcome["status"] == "passed":
                result = json.loads((case / model / "result.json").read_text())
                if result["input_request_sha256"] != sha256_file(case / "request.json") or result[
                    "mask_sha256"
                ] != sha256_file(mask_path):
                    raise Phase1Error("Saved prediction request/mask digest differs.")
                with Image.open(mask_path) as opened:
                    if opened.mode != "L" or opened.size != (mask_ref.shape[1], mask_ref.shape[0]):
                        raise Phase1Error("Prediction encoding or original dimensions differ.")
                    actual = np.array(opened)
                if not np.isin(actual, [0, 255]).all():
                    raise Phase1Error("Prediction is not a binary 0/255 mask.")
                predicted = actual // 255
            metrics = eye_statistics(predicted, mask_ref)
            x0, y0, x1, y1 = request["roi_xyxy"]
            outside = int(mask_ref.sum() - mask_ref[y0:y1, x0:x1].sum())
            eye_rows.append(
                {
                    "model": model,
                    "reference_set": reference_tag,
                    "image_id": key[0],
                    "annotation_id": key[1],
                    "animal_category": ref["animal_category"],
                    "eye_side": key[2],
                    "prompt_version": prompt["prompt_version"],
                    "reference_annotation_version": ref["annotation_version"],
                    "reference_snapshot_sha256": ref["snapshot_sha256"],
                    "predicted_mask_path": str(mask_path.relative_to(out))
                    if predicted is not None
                    else "",
                    "reference_mask_path": str((frozen / ref["mask_path"]).relative_to(out)),
                    "failure_reason": outcome["failure_reason"],
                    "reference_pixels_outside_roi": outside,
                    **metrics,
                }
            )
            centers.append(
                {
                    "image_id": key[0],
                    "annotation_id": key[1],
                    "eye_side": key[2],
                    # The measurement builder carries versions without interpreting their source.
                    "annotation_version": int(prompt["prompt_version"]),
                    "centroid_x_px": metrics["predicted_cx_px"],
                    "centroid_y_px": metrics["predicted_cy_px"],
                }
            )
        predicted_distances = build_measurements(centers)
        by_pair = {
            (r["image_id"], r["measurement_type"], r["annotation_id_a"], r["annotation_id_b"]): r
            for r in predicted_distances
        }
        for ref in reference_distances:
            key = (
                ref["image_id"],
                ref["measurement_type"],
                ref["annotation_id_a"],
                ref["annotation_id_b"],
            )
            pred = by_pair[key]
            row = {
                "model": model,
                "reference_set": reference_tag,
                "image_id": key[0],
                "measurement_type": key[1],
                "annotation_id_a": key[2],
                "annotation_id_b": key[3],
                "eye_side_a": ref["eye_side_a"],
                "eye_side_b": ref["eye_side_b"],
                "reference_snapshot_sha256": ref["snapshot_sha256"],
                **distance_statistics(pred["distance_px"], ref["distance_px"]),
            }
            for suffix in ("a", "b"):
                row[f"prompt_version_{suffix}"] = pred[f"annotation_version_{suffix}"]
                row[f"reference_annotation_version_{suffix}"] = ref[f"annotation_version_{suffix}"]
                for axis in ("cx", "cy"):
                    row[f"predicted_{axis}_{suffix}_px"] = pred[f"{axis}_{suffix}_px"]
                    row[f"reference_{axis}_{suffix}_px"] = ref[f"{axis}_{suffix}_px"]
            measurement_rows.append(row)
    return eye_rows, measurement_rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(eyes: list[dict], measurements: list[dict]) -> list[dict]:
    rows = []
    for model in MODELS:
        selected = [r for r in eyes if r["model"] == model]
        failed = sum(r["prediction_status"] == "inference_failed" for r in selected)
        centers = [r["centroid_error_px"] for r in selected if r["centroid_error_px"] is not None]
        ious = [r["iou"] or 0.0 for r in selected]
        dices = [r["dice"] or 0.0 for r in selected]
        row = {
            "model": model,
            "eyes_expected": len(selected),
            "inference_failed": failed,
            "empty_predictions": sum(r["prediction_status"] == "empty" for r in selected),
            "defined_centers": len(centers),
            "mean_iou_all_targets_failure_as_zero": mean(ious),
            "median_iou_all_targets_failure_as_zero": median(ious) if ious else None,
            "min_iou_all_targets_failure_as_zero": min(ious) if ious else None,
            "mean_dice_all_targets_failure_as_zero": mean(dices),
            "median_dice_all_targets_failure_as_zero": median(dices) if dices else None,
            "mean_centroid_error_px": mean(centers),
            "median_centroid_error_px": median(centers) if centers else None,
            "max_centroid_error_px": max(centers) if centers else None,
            "images_expected": len({r["image_id"] for r in selected}),
            "geometrically_complete_images": sum(
                all(
                    r["centroid_error_px"] is not None
                    for r in selected
                    if r["image_id"] == image_id
                )
                for image_id in {r["image_id"] for r in selected}
            ),
        }
        for kind in ("binocular", "right_eye_pair"):
            group = [
                r for r in measurements if r["model"] == model and r["measurement_type"] == kind
            ]
            errors = [r["absolute_error_px"] for r in group if r["absolute_error_px"] is not None]
            row[f"{kind}_expected"] = len(group)
            row[f"{kind}_defined"] = len(errors)
            row[f"{kind}_mae_px"] = mean(errors)
            row[f"{kind}_median_absolute_error_px"] = median(errors) if errors else None
            row[f"{kind}_mean_signed_error_px"] = mean(
                [r["signed_error_px"] for r in group if r["signed_error_px"] is not None]
            )
            row[f"{kind}_rmse_px"] = math.sqrt(mean([v * v for v in errors])) if errors else None
            row[f"{kind}_max_absolute_error_px"] = max(errors) if errors else None
            row[f"{kind}_mean_absolute_percentage_error"] = mean(
                [
                    r["relative_absolute_error_percent"]
                    for r in group
                    if r["relative_absolute_error_percent"] is not None
                ]
            )
        rows.append(row)
    return rows


def compare_saved(out: Path) -> dict[str, list[dict]]:
    eyes, distances = evaluate(out, "primary")
    repeat_eyes, repeat_distances = evaluate(out, "repeat")
    if (len(eyes), len(distances), len(repeat_eyes), len(repeat_distances)) != (36, 27, 12, 9):
        raise Phase1Error("Comparison rows differ from the approved complete/sensitivity scopes.")
    image_summary = []
    for image_id in sorted({r["image_id"] for r in eyes}):
        group = [r for r in eyes if r["image_id"] == image_id]
        for summary in summarize(group, [r for r in distances if r["image_id"] == image_id]):
            image_summary.append(
                {"image_id": image_id, "animal_category": group[0]["animal_category"], **summary}
            )
    return {
        "eye_metrics.csv": eyes,
        "measurement_comparison.csv": distances,
        "model_summary.csv": summarize(eyes, distances),
        "image_summary.csv": image_summary,
        "sensitivity_eye_metrics.csv": repeat_eyes,
        "sensitivity_measurements.csv": repeat_distances,
        "sensitivity_summary.csv": summarize(repeat_eyes, repeat_distances),
    }


def verify_comparison(out: Path) -> dict[str, Any]:
    manifest = json.loads((out / "run_manifest.json").read_text())
    if manifest["status"] not in ("complete", "complete_with_prediction_failures"):
        raise Phase1Error("Comparison is not complete.")
    for relative, digest in manifest["artifacts_sha256"].items():
        if sha256_file(safe_path(out, relative)) != digest:
            raise Phase1Error(f"Comparison artifact digest differs: {relative}")
    # Every published metric can be recomputed without live prompts, models, or GT work directories.
    computed = compare_saved(out)
    for name, rows in computed.items():
        expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]
        if read_csv(out / name) != expected:
            raise Phase1Error(f"Recomputed metrics differ: {name}")
    native_masks = 0
    for outcome in json.loads((out / "prediction_outcomes.json").read_text()):
        if outcome["status"] != "passed":
            continue
        case = out / "cases" / outcome["case"]
        request = json.loads((case / "request.json").read_text())
        logits = np.load(case / outcome["model"] / "roi_logits.npy", allow_pickle=False)
        with Image.open(case / outcome["model"] / "mask.png") as opened:
            actual = np.array(opened)
        expected = np.zeros_like(actual)
        x0, y0, x1, y1 = request["roi_xyxy"]
        expected[y0:y1, x0:x1] = (logits > 0).astype(np.uint8) * 255
        if not np.array_equal(expected, actual):
            raise Phase1Error("Native logits differ from the saved original-grid prediction.")
        native_masks += 1
    return {
        "status": "passed",
        "primary_eye_rows": len(computed["eye_metrics.csv"]),
        "primary_distance_rows": len(computed["measurement_comparison.csv"]),
        "native_masks_verified": native_masks,
        "model_inference_repeated": False,
        "uses_live_human_data": False,
    }


def run(cfg: dict[str, Any], runtime: dict[str, Any]) -> Path:
    out = cfg["output_directory"]
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "run_manifest.json",
        {"status": "running", "started_at_utc": datetime.now(UTC).isoformat()},
    )
    started = time.perf_counter()
    inputs = freeze_inputs(cfg, runtime, out)
    outcomes = run_predictions(out, inputs)
    computed = compare_saved(out)
    for name, rows in computed.items():
        write_csv(out / name, rows)
    from animal_eye_metrology.visualization import render_eye_comparison_sheets

    render_eye_comparison_sheets(out, computed["eye_metrics.csv"])
    source_files = (
        Path(__file__),
        Path(__file__).with_name("eye_poc.py"),
        Path(__file__).with_name("metrology.py"),
        Path(__file__).with_name("visualization.py"),
    )
    for source in source_files:
        if sha256_file(source) != sha256_file(out / "provenance" / source.name):
            raise Phase1Error("Experiment source changed while inference was running.")
    write_json(
        out / "run_manifest.json",
        {
            "status": "complete"
            if all(r["status"] == "passed" for r in outcomes)
            else "complete_with_prediction_failures",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "input_manifest_sha256": sha256_file(out / "input_manifest.json"),
            "configuration": json.loads(json.dumps(cfg, default=str)),
            "runtime_configuration": json.loads(json.dumps(runtime, default=str)),
            "execution": {
                "argv": sys.argv,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "installed_packages": {
                d.metadata["Name"]: d.version for d in distributions() if d.metadata.get("Name")
            },
            "metric_scope": (
                "human visible-aperture mask reference deviations; "
                "development POC, not physical accuracy"
            ),
            "failure_policy": (
                "failed inference has undefined raw IoU/center/distance; "
                "all-target mask means explicitly score failures zero"
            ),
            "timing_policy": (
                "one process and model load per eye/model; "
                "sequential CPU FP32; no warmup or repetition"
            ),
            "primary_summary": computed["model_summary.csv"],
            "sensitivity_summary": computed["sensitivity_summary.csv"],
            "artifacts_sha256": {
                str(p.relative_to(out)): sha256_file(p)
                for p in out.rglob("*")
                if p.is_file() and p != out / "run_manifest.json"
            },
        },
    )
    LOGGER.info(json.dumps(verify_comparison(out)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/eye_segmentation_comparison.toml")
    )
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.verify:
            print(json.dumps(verify_comparison(args.verify.resolve())))
        else:
            print(run(*read_settings(args.config)))
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error(str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
