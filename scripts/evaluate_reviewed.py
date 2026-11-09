#!/usr/bin/env python3
"""Freeze reviewed references, run unchanged inference, and replay portable evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.inference.inference_artifacts import csv_bytes
from animal_eye_metrology.inference.inference_pipeline import (
    load_config,
    load_result,
    normalize_image,
    preflight,
    run_inference,
)
from animal_eye_metrology.measurement.reviewed_evaluation import (
    binary_mask,
    match_mask_arrays,
    score_distance,
    score_eye,
    summarize,
)
from animal_eye_metrology.paths import PROJECT_ROOT

Row = dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def encode(mask: np.ndarray) -> Row:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}


def decode(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    mask = mask_utils.decode(rle).astype(bool)
    if mask.shape != (height, width):
        raise ValueError("Mask differs from the original image grid.")
    return mask


def protected_hashes(root: Path, config: Row) -> dict[str, str]:
    files = [
        config["source"],
        config["pose_source_manifest"],
        config["sam_runtime"],
        *(config[key] for key in ("detector_checkpoint", "pose_checkpoint", "sam_checkpoint")),
        *sorted((root / "src/animal_eye_metrology/inference").glob("*.py")),
        *sorted((root / "src/animal_eye_metrology/models").glob("*.py")),
        root / "src/animal_eye_metrology/measurement/metrology.py",
        root / "src/animal_eye_metrology/measurement/eye_geometry.py",
    ]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def portable_result(run: Path, result: Row) -> Row:
    """Export exact binary predictions; no new prompts, resizing, smoothing or GT inputs."""
    exported = {key: result[key] for key in ("run_id", "status", "baseline", "stages")}
    for section in ("animals", "eyes"):
        exported[section] = []
        for original in result[section]:
            row = dict(original)
            relative = row.pop("mask_path", None)
            row["mask_rle"] = None
            row["mask_sha256"] = None
            if relative:
                path = (run / relative).resolve()
                if Path(relative).is_absolute() or not path.is_relative_to(run.resolve()):
                    raise ValueError("Prediction mask escapes its inference run.")
                row["mask_rle"] = encode(binary_mask(path))
                row["mask_sha256"] = sha256_file(path)
            exported[section].append(row)
    return exported


def infer(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    reference = json.loads(args.reference.read_text())
    config = load_config(args.config)
    checks = preflight(config)
    if any(check["status"] != "ready" for check in checks):
        raise ValueError(f"Inference prerequisites failed: {checks}")
    before = protected_hashes(root, config)
    source_hashes = reference["source_artifacts_sha256"]
    available_sources = {p: digest for p, digest in source_hashes.items() if (root / p).is_file()}
    unavailable_sources = sorted(set(source_hashes) - set(available_sources))
    if unavailable_sources and args.images is None:
        raise ValueError(
            "Original workspace sources are missing; use --images for portable replay."
        )
    for relative, expected in available_sources.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"Reference source changed since snapshot: {relative}")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest: Row = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reference_sha256": sha256_file(args.reference),
        "baseline": "A_r100",
        "device": config["device"],
        "seed": config["seed"],
        "threads": config["threads"],
        "config_sha256": sha256_file(args.config),
        "model_sha256": {
            key: config[key + "_sha256"]
            for key in ("detector_checkpoint", "pose_checkpoint", "sam_checkpoint")
        },
        "protected_sha256_before": before,
        "images": [],
        "fresh_inference": True,
        "reference_used_as_model_input": False,
        "original_source_files_verified": len(available_sources),
        "original_source_files_unavailable": unavailable_sources,
    }
    predictions: Row = {
        "schema_version": 1,
        "reference_sha256": manifest["reference_sha256"],
        "images": [],
    }
    write_json(args.output / "run_manifest.json", manifest)
    for index, image in enumerate(reference["images"]):
        source = (args.images / image["file_name"]) if args.images else root / image["source_image"]
        if sha256_file(source) != image["image_sha256"]:
            raise ValueError(f"Input image hash changed: {source}")
        normalized, _ = normalize_image(source.read_bytes(), config)
        with Image.open(source) as original:
            if not np.array_equal(np.asarray(normalized), np.asarray(original.convert("RGB"))):
                raise ValueError("Normalization changes the reviewed reference pixel coordinates.")
        print(
            json.dumps(
                {
                    "image_id": image["image_id"],
                    "index": index + 1,
                    "total": len(reference["images"]),
                    "stage": "inference",
                }
            ),
            flush=True,
        )
        run = run_inference(source.read_bytes(), source.name, args.config)
        result = load_result(run)
        if result["input"]["original_sha256"] != image["image_sha256"]:
            raise ValueError("Inference input does not match frozen reference.")
        predictions["images"].append(
            {
                "image_id": image["image_id"],
                "image_sha256": image["image_sha256"],
                **portable_result(run, result),
            }
        )
        manifest["images"].append(
            {
                "image_id": image["image_id"],
                "run": run.relative_to(root).as_posix(),
                "status": result["status"],
                "result_sha256": sha256_file(run / "result.json"),
            }
        )
        write_json(args.output / "predictions.json", predictions)
        write_json(args.output / "run_manifest.json", manifest)
    after = protected_hashes(root, config)
    manifest["protected_sha256_after"] = after
    manifest["models_and_inference_unchanged"] = before == after
    changed = [
        relative
        for relative, digest in available_sources.items()
        if sha256_file(root / relative) != digest
    ]
    manifest["reference_sources_unchanged"] = None if unavailable_sources else not changed
    manifest["changed_reference_sources"] = changed
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    manifest["predictions_sha256"] = sha256_file(args.output / "predictions.json")
    manifest["status"] = "completed" if before == after and not changed else "integrity_failed"
    write_json(args.output / "run_manifest.json", manifest)
    if manifest["status"] != "completed":
        raise ValueError("Protected source integrity failed; inspect run_manifest.json.")


def image_scores(image: Row, cases: list[Row], prediction: Row) -> tuple[list, list, list]:
    height, width = image["height"], image["width"]
    case_keys = [(int(c["image_id"]), int(c["annotation_id"]), c["eye_side"]) for c in cases]
    target_ids = {int(t["annotation_id"]) for t in image["targets"]}
    if len(set(case_keys)) != len(case_keys):
        raise ValueError("Duplicate reference eye identity.")
    if any(
        image_id != int(image["image_id"])
        or annotation_id not in target_ids
        or side not in ("left", "right")
        for image_id, annotation_id, side in case_keys
    ):
        raise ValueError("Reference eye identity differs from its image/animal scope.")
    targets = []
    for target in image["targets"]:
        bbox = target["coco_bbox_xywh"]
        x, y, w, h = json.loads(bbox) if isinstance(bbox, str) else bbox
        targets.append(
            {
                "annotation_id": target["annotation_id"],
                "category": target["animal_category"],
                "bbox_xyxy": [x, y, x + w, y + h],
                "mask": decode(target["segmentation"], height, width),
            }
        )
    animals = [
        {**a, "mask": decode(a["mask_rle"], height, width)}
        for a in prediction["animals"]
        if a["accepted"] and a["mask_status"] == "completed"
    ]
    matches, audit = match_mask_arrays(animals, targets)
    prediction_eyes = {}
    seen_prediction_eyes = set()
    for eye in prediction["eyes"]:
        key = eye["condition_id"], eye["prediction_id"], eye["eye_side"]
        if key in seen_prediction_eyes:
            raise ValueError("Duplicate prediction eye identity within a condition.")
        if eye["eye_side"] not in ("left", "right"):
            raise ValueError("Unknown prediction eye side.")
        seen_prediction_eyes.add(key)
        if eye["condition_id"] == "A_r100":
            prediction_eyes[(eye["prediction_id"], eye["eye_side"])] = eye
    eyes = []
    for case in cases:
        if not case["eligible"]:
            continue
        match = matches.get(str(case["annotation_id"]))
        eye = prediction_eyes.get((match["prediction_id"], case["eye_side"])) if match else None
        mask, reason = None, "missing_or_ambiguous_animal_match"
        if eye:
            reason = eye["failure_reason"] or eye["segmentation_status"]
            if eye["segmentation_status"] in ("completed", "empty_mask") and eye["mask_rle"]:
                mask = decode(eye["mask_rle"], height, width)
        elif match:
            reason = "missing_eye_prediction"
        row = {
            key: case[key]
            for key in ("image_id", "annotation_id", "eye_side", "annotation_version")
        }
        row.update(
            batch=image["batch"],
            category=next(
                t["category"] for t in targets if int(t["annotation_id"]) == case["annotation_id"]
            ),
            prediction_id=match["prediction_id"] if match else None,
            **score_eye(decode(case["reference_mask_rle"], height, width), mask, reason),
        )
        eyes.append(row)
    index = {(e["annotation_id"], e["eye_side"]): e for e in eyes}
    pairs = [
        ("binocular", index[(ann, "left")], index[(ann, "right")])
        for ann in sorted({e["annotation_id"] for e in eyes})
        if (ann, "left") in index and (ann, "right") in index
    ]
    pairs += [
        ("right_eye_pair", a, b)
        for a, b in combinations(
            sorted((e for e in eyes if e["eye_side"] == "right"), key=lambda e: e["annotation_id"]),
            2,
        )
    ]
    measurements = []
    for kind, a, b in pairs:

        def center(row: Row, prefix: str) -> tuple | None:
            x, y = row[prefix + "_cx_px"], row[prefix + "_cy_px"]
            return None if x is None else (x, y)

        metrics = score_distance(
            center(a, "reference"),
            center(b, "reference"),
            center(a, "predicted"),
            center(b, "predicted"),
            ";".join(r["failure_reason"] for r in (a, b) if r["failure_reason"]),
        )
        measurements.append(
            {
                "image_id": image["image_id"],
                "batch": image["batch"],
                "measurement_type": kind,
                "annotation_id_a": a["annotation_id"],
                "annotation_id_b": b["annotation_id"],
                "eye_side_a": a["eye_side"],
                "eye_side_b": b["eye_side"],
                **metrics,
            }
        )
    return eyes, measurements, [{"image_id": image["image_id"], **r} for r in audit]


def score(args: argparse.Namespace) -> None:
    reference = json.loads(args.reference.read_text())
    predictions = json.loads(args.predictions.read_text())
    if predictions["reference_sha256"] != sha256_file(args.reference):
        raise ValueError("Predictions belong to another reference snapshot.")
    reference_ids = [int(im["image_id"]) for im in reference["images"]]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Duplicate reference image identity.")
    case_keys = [
        (int(c["image_id"]), int(c["annotation_id"]), c["eye_side"]) for c in reference["cases"]
    ]
    if len(set(case_keys)) != len(case_keys):
        raise ValueError("Duplicate reference eye identity.")
    if any(key[0] not in reference_ids for key in case_keys):
        raise ValueError("Reference eye identity belongs to an unknown image.")
    images = {im["image_id"]: im for im in predictions["images"]}
    if len(images) != len(predictions["images"]) or set(images) != {
        im["image_id"] for im in reference["images"]
    }:
        raise ValueError("Prediction image identities differ from the complete frozen scope.")
    eyes, measurements, matches, image_rows = [], [], [], []
    for image in reference["images"]:
        prediction = images[image["image_id"]]
        if (
            prediction["image_sha256"] != image["image_sha256"]
            or prediction["baseline"] != "A_r100"
        ):
            raise ValueError("Prediction image or baseline mismatch.")
        cases = [c for c in reference["cases"] if c["image_id"] == image["image_id"]]
        e, m, a = image_scores(image, cases, prediction)
        eyes.extend(e)
        measurements.extend(m)
        matches.extend(a)
        image_rows.append(
            {
                "image_id": image["image_id"],
                "batch": image["batch"],
                "run_id": prediction["run_id"],
                "run_status": prediction["status"],
                "expected_eye_slots": len(cases),
                "reviewed_eye_slots": sum(c["human_reviewed"] for c in cases),
                **summarize(e, m),
            }
        )
    summary = summarize(eyes, measurements)
    summary.update(
        reference_sha256=sha256_file(args.reference),
        predictions_sha256=sha256_file(args.predictions),
        reference_scope=reference["summary"],
        images_inferred=len(images),
        images_with_eligible_eyes=sum(r["eyes_expected"] > 0 for r in image_rows),
        prediction_failures=dict(Counter(e["failure_reason"] for e in eyes if e["failure_reason"])),
        independent_blind_test=False,
        averaging="Eye/pair micro means; same-image eyes and pairs are dependent.",
    )
    for kind in ("binocular", "right_eye_pair"):
        available = [r[f"{kind}_mae_px"] for r in image_rows if r[f"{kind}_mae_px"] is not None]
        summary[f"{kind}_images_with_defined_pairs"] = len(available)
        summary[f"{kind}_image_macro_mae_px"] = mean(available) if available else None
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", summary)
    write_json(
        args.output / "by_category.json",
        {
            category: summarize([e for e in eyes if e["category"] == category], [])
            for category in sorted({e["category"] for e in eyes})
        },
    )
    for filename, rows in (
        ("per_eye.csv", eyes),
        ("measurements.csv", measurements),
        ("matches.csv", matches),
        ("images.csv", image_rows),
        (
            "reviews.csv",
            [{k: v for k, v in c.items() if k != "reference_mask_rle"} for c in reference["cases"]],
        ),
    ):
        (args.output / filename).write_bytes(csv_bytes(rows))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--root", type=Path, default=PROJECT_ROOT)
    freeze.add_argument("--output", type=Path, required=True)
    inference = sub.add_parser("infer")
    inference.add_argument("--root", type=Path, default=PROJECT_ROOT)
    inference.add_argument("--reference", type=Path, required=True)
    inference.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/inference.runtime.toml"
    )
    inference.add_argument(
        "--images", type=Path, help="Optional flat directory of original COCO JPEGs"
    )
    inference.add_argument("--output", type=Path, required=True)
    replay = sub.add_parser("score")
    replay.add_argument("--reference", type=Path, required=True)
    replay.add_argument("--predictions", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "freeze":
        from animal_eye_metrology.annotation.reviewed_snapshot import freeze_reviewed_reference

        reference = freeze_reviewed_reference(args.root, args.output)
        print(json.dumps(reference["summary"], ensure_ascii=False, indent=2))
    elif args.operation == "infer":
        infer(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
