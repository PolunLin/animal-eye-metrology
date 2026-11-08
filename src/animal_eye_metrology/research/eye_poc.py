"""Sequential CPU point/box compatibility checks; synthetic probes are not eye evaluations."""

import argparse
import json
import logging
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from importlib.metadata import distribution, distributions
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from animal_eye_metrology.annotation.eye_annotations import load_targets
from animal_eye_metrology.annotation.eye_prompts import crop_for_prompt, latest_prompt, load_prompts
from animal_eye_metrology.core.config import load_config
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.data.coco import load_coco
from animal_eye_metrology.models.sam_runtime import load_predictor, validate_source
from animal_eye_metrology.paths import module_source
from animal_eye_metrology.research.phase2 import write_json

LOGGER = logging.getLogger(__name__)


def settings(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        cfg = tomllib.load(handle)
    cfg["source"] = path.resolve()
    for key in (
        "phase1_config",
        "proposal_csv",
        "prompts_directory",
        "output_directory",
        "weights_directory",
    ):
        cfg[key] = (path.resolve().parent / cfg[key]).resolve()
    if cfg["device"] != "cpu" or cfg["threads"] != 4 or cfg["seed"] != 42:
        raise Phase1Error("This compatibility check requires CPU, four threads and seed 42.")
    if set(cfg["models"]) != {"sam2_1_tiny", "sam_vit_b", "mobilesam"}:
        raise Phase1Error("Use the three approved models.")
    return cfg


def worker(cfg: dict[str, Any], directory: Path, name: str) -> None:
    started = time.perf_counter()
    import torch

    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    spec = cfg["models"][name]
    package = distribution(spec["distribution"])
    verified_source = validate_source(cfg, spec)
    checkpoint = cfg["weights_directory"] / spec["checkpoint"]
    download = json.loads((cfg["weights_directory"] / "download_manifest.json").read_text())
    checkpoint_digest = sha256_file(checkpoint)
    if checkpoint_digest != download[checkpoint.name]["sha256"]:
        raise Phase1Error("Checkpoint changed after download.")
    request = json.loads((directory / "request.json").read_text())
    image_path = directory / "input.png"
    if sha256_file(image_path) != request["saved_input_sha256"]:
        raise Phase1Error("Frozen input image changed.")
    with Image.open(image_path) as opened:
        image = np.array(opened.convert("RGB"))
    x0, y0, x1, y1 = request["roi_xyxy"]
    crop = image[y0:y1, x0:x1].copy()
    point = np.array([request["point_xy"]], dtype=np.float32) - np.array(
        [[x0, y0]], dtype=np.float32
    )
    box = np.array(request["box_xyxy"], dtype=np.float32) - np.array(
        [x0, y0, x0, y0], dtype=np.float32
    )
    t0 = time.perf_counter()
    predictor = load_predictor(name, checkpoint)
    load_seconds = time.perf_counter() - t0
    with torch.inference_mode():
        t0 = time.perf_counter()
        predictor.set_image(crop)
        encoder_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        logits, scores, low_res = predictor.predict(
            point_coords=point,
            point_labels=np.array([1], dtype=np.int32),
            box=box,
            multimask_output=True,
            return_logits=True,
        )
        decode_seconds = time.perf_counter() - t0
    logits, scores = np.asarray(logits), np.asarray(scores)
    if (
        logits.ndim != 3
        or logits.shape[1:] != crop.shape[:2]
        or scores.shape != (len(logits),)
        or not len(scores)
        or not np.isfinite(logits).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(low_res).all()
    ):
        raise Phase1Error("Predictor produced invalid masks, scores or dimensions.")
    selected = int(np.argmax(scores))
    full = np.zeros(image.shape[:2], dtype=np.uint8)
    full[y0:y1, x0:x1] = (logits[selected] > 0).astype(np.uint8) * 255
    model_directory = directory / name
    model_directory.mkdir()
    Image.fromarray(full).save(model_directory / "mask.png")
    np.save(model_directory / "roi_logits.npy", logits[selected], allow_pickle=False)
    overlay = Image.fromarray(image)
    overlay.paste("#00a6ed", (0, 0), Image.fromarray(full // 2))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(request["box_xyxy"], outline="#ffb000", width=1)
    px, py = request["point_xy"]
    draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill="#ffb000")
    overlay.save(model_directory / "overlay.png")
    write_json(
        model_directory / "result.json",
        {
            "status": "passed",
            "scope": "package/load/point+box/CPU-forward/native-grid-output",
            "kind": request["kind"],
            "eye_accuracy_evaluated": False,
            "model": name,
            "source_commit": spec["commit"],
            "source_verification": verified_source,
            "distribution_version": package.version,
            "checkpoint_sha256": checkpoint_digest,
            "input_request_sha256": sha256_file(directory / "request.json"),
            "load_seconds": load_seconds,
            "encoder_including_preprocess_seconds": encoder_seconds,
            "decode_including_native_restore_seconds": decode_seconds,
            "worker_seconds_including_import_hash_export": time.perf_counter() - started,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_platform_unit": "bytes on macOS",
            "device": "cpu",
            "dtype": "float32",
            "actual_threads": torch.get_num_threads(),
            "seed": cfg["seed"],
            "batch_size": 1,
            "scores": scores.tolist(),
            "selected_candidate": selected,
            "logit_threshold": 0.0,
            "mask_shape_hw": list(full.shape),
            "mask_area_px": int(np.count_nonzero(full)),
            "mask_sha256": sha256_file(model_directory / "mask.png"),
            "postprocessing": "none",
            "prompt_coordinate_policy": (
                "continuous image x/y; crop translation only; official prompt encoding unchanged"
            ),
            "measurement_coordinate_policy": (
                "if measured later: foreground centers at column+.5,row+.5"
            ),
        },
    )


def run(cfg: dict[str, Any], kind: str) -> Path:
    # Manual inputs are frozen before any model loads. Synthetic probes never get manual labels.
    if kind == "synthetic":
        image = Image.new("RGB", (128, 96), "#c9d7e1")
        draw = ImageDraw.Draw(image)
        draw.rectangle((38, 28, 86, 68), fill="#b83439")
        draw.ellipse((5, 5, 24, 24), fill="#2f8665")
        request = {
            "kind": "synthetic_compatibility_probe_not_research_GT",
            "prompt_source": "invented_QA_coordinates",
            "image_id": None,
            "point_xy": [62.5, 48.5],
            "box_xyxy": [36.0, 26.0, 89.0, 71.0],
            "roi_xyxy": [0, 0, 128, 96],
        }
    else:
        data = load_config(cfg["phase1_config"])
        phase1 = json.loads((data.output / "run_manifest.json").read_text())
        dataset = load_coco(
            data.annotations, expected_sha256=phase1["dataset"]["annotations_sha256"]
        )
        targets = load_targets(cfg["proposal_csv"], dataset)
        rows = load_prompts(cfg["prompts_directory"], targets)
        eye = cfg["smoke_eye"]
        key = eye["image_id"], eye["annotation_id"], eye["eye_side"]
        row = latest_prompt(rows, key)
        if row is None:
            raise Phase1Error(
                "The first eye needs a human-saved point+box prompt before real-eye inference."
            )
        target = next(
            t for t in targets if (int(t["image_id"]), int(t["annotation_id"])) == key[:2]
        )
        path = data.images / target["file_name"]
        if sha256_file(path) != row["image_sha256"]:
            raise Phase1Error("Prompt image source differs.")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        request = {
            "kind": "single_eye_manual_prompt_compatibility_not_full_POC",
            "prompt_source": "manual_original_image",
            "prompt_record": row,
            "source_image_sha256": row["image_sha256"],
            "image_id": key[0],
            "annotation_id": key[1],
            "eye_side": key[2],
            "point_xy": [float(row[k]) for k in ("point_x", "point_y")],
            "box_xyxy": [float(row[k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1")],
            "roi_xyxy": list(crop_for_prompt(row, image.width, image.height)),
        }
    revision = cfg.get("run_revision", 1)
    if type(revision) is not int or revision < 1:
        raise Phase1Error("Run revision must be a positive integer.")
    directory = cfg["output_directory"] / f"compatibility_{kind}_v{revision}"
    directory.mkdir(parents=True, exist_ok=False)
    image.save(directory / "input.png")
    request["saved_input_sha256"] = sha256_file(directory / "input.png")
    write_json(directory / "request.json", request)
    results = []
    for name in cfg["models"]:
        LOGGER.info(
            json.dumps({"event": "compatibility_model_started", "model": name, "kind": kind})
        )
        with (directory / f"{name}.log").open("x") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "animal_eye_metrology.research.eye_poc",
                    "--config",
                    str(cfg["source"]),
                    "--worker",
                    name,
                    "--directory",
                    str(directory),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=600,
            )
        result_path = directory / name / "result.json"
        result = (
            json.loads(result_path.read_text())
            if completed.returncode == 0 and result_path.exists()
            else {
                "status": "failed",
                "model": name,
                "returncode": completed.returncode,
                "log": f"{name}.log",
                "eye_accuracy_evaluated": False,
            }
        )
        results.append(result)
        LOGGER.info(json.dumps({"event": "compatibility_model_finished", **result}))
    provenance = directory / "provenance"
    provenance.mkdir()
    source_files = (
        Path(__file__),
        module_source("eye_prompts.py"),
        module_source("paths.py"),
        module_source("_compat.py"),
        cfg["source"],
    )
    for source in source_files:
        shutil.copyfile(source, provenance / source.name)
    for source, name in (
        (cfg["weights_directory"] / "download_manifest.json", "weight_download_manifest.json"),
        (cfg["output_directory"] / "vendor/source_manifest.json", "official_source_manifest.json"),
    ):
        if source.is_file():
            shutil.copyfile(source, provenance / name)
    vendor_manifest = cfg["output_directory"] / "vendor/source_manifest.json"
    if vendor_manifest.is_file():
        for name, source in json.loads(vendor_manifest.read_text()).items():
            license_path = Path(source["directory"]) / "LICENSE"
            if license_path.is_file():
                shutil.copyfile(license_path, provenance / f"{name}_LICENSE")
    write_json(
        directory / "run_manifest.json",
        {
            "status": "passed" if all(r["status"] == "passed" for r in results) else "has_failures",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "kind": request["kind"],
            "eye_accuracy_evaluated": False,
            "ground_truth_masks_read": False,
            "results": results,
            "execution": {
                "argv": sys.argv,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "installed_packages": {
                d.metadata["Name"]: d.version for d in distributions() if d.metadata.get("Name")
            },
            "configuration": json.loads(json.dumps(cfg, default=str)),
            "source_sha256": {p.name: sha256_file(p) for p in source_files},
            "artifacts_sha256": {
                str(p.relative_to(directory)): sha256_file(p)
                for p in directory.rglob("*")
                if p.is_file()
            },
        },
    )
    return directory


def verify_run(directory: Path) -> dict[str, Any]:
    """Validate saved artifacts and native-logit restoration; no inference or GT access."""
    manifest = json.loads((directory / "run_manifest.json").read_text())
    for relative, digest in manifest["artifacts_sha256"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != digest:
            raise Phase1Error(f"Compatibility artifact digest differs: {relative}")
    request = json.loads((directory / "request.json").read_text())
    with Image.open(directory / "input.png") as image:
        width, height = image.size
    x0, y0, x1, y1 = request["roi_xyxy"]
    validated = 0
    for result in manifest["results"]:
        if result["status"] != "passed":
            continue
        model_dir = directory / result["model"]
        with Image.open(model_dir / "mask.png") as opened:
            if opened.mode != "L" or opened.size != (width, height):
                raise Phase1Error("Saved mask has invalid dimensions or encoding.")
            actual = np.array(opened)
        logits = np.load(model_dir / "roi_logits.npy", allow_pickle=False)
        expected = np.zeros((height, width), dtype=np.uint8)
        expected[y0:y1, x0:x1] = (logits > 0).astype(np.uint8) * 255
        if not np.array_equal(actual, expected):
            raise Phase1Error("Saved native logits do not reproduce the original-grid mask.")
        validated += 1
    return {
        "status": "passed",
        "model_artifacts_validated": validated,
        "models_with_failed_inference": sum(r["status"] != "passed" for r in manifest["results"]),
        "model_inference_repeated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eye_segmentation_poc.toml"))
    parser.add_argument("--kind", choices=("synthetic", "manual"), default="synthetic")
    parser.add_argument("--worker", choices=("sam2_1_tiny", "sam_vit_b", "mobilesam"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.verify:
            print(json.dumps(verify_run(args.verify.resolve())))
            return 0
        cfg = settings(args.config)
        if args.worker:
            worker(cfg, args.directory, args.worker)
        else:
            directory = run(cfg, args.kind)
            print(directory)
            if json.loads((directory / "run_manifest.json").read_text())["status"] != "passed":
                return 1
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        LOGGER.error(str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
