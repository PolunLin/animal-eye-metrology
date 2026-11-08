"""Freeze reviewed manual eye masks and derive a reproducible pixel-distance reference."""

import argparse
import csv
import fcntl
import json
import logging
import platform
import shutil
import sys
import tempfile
import tomllib
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from animal_eye_metrology.coco import load_coco
from animal_eye_metrology.config import AnalysisConfig, load_config
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.eye_annotations import (
    FIELDS,
    PROTOCOL_DOCUMENTS,
    V2_PROTOCOL_VERSION,
    latest_annotation,
    load_annotations,
    load_geometry,
    load_targets,
    validate_protocol_version,
)
from animal_eye_metrology.metrology import build_measurements, mask_centroid
from animal_eye_metrology.phase2 import read_csv, write_json
from animal_eye_metrology.verification import sha256_file
from animal_eye_metrology.visualization import render_measurement_overlay

LOGGER = logging.getLogger(__name__)
COORDINATES = "top-left image edge (0,0); pixel centers (column+.5,row+.5)"
DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"


def write_csv(
    path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    result = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not result.is_relative_to(root.resolve()):
        raise Phase1Error("Snapshot paths must remain inside the frozen directory.")
    return result


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise Phase1Error(f"Refusing to replace snapshot file: {target.name}")
    shutil.copyfile(source, target)


def read_settings(path: Path) -> tuple[dict[str, Any], AnalysisConfig]:
    path = path.resolve()
    with path.open("rb") as handle:
        cfg = tomllib.load(handle)
    allowed = {
        "data": {"phase1_config", "proposal_csv", "ground_truth_directory"},
        "snapshot": {"protocol_version", "expected_images", "expected_instances", "expected_eyes"},
        "output": {"directory"},
    }
    if set(cfg) != set(allowed) or any(set(cfg[k]) != fields for k, fields in allowed.items()):
        raise Phase1Error("Baseline configuration sections/fields are invalid.")
    for section, fields in (("data", allowed["data"]), ("output", allowed["output"])):
        for field in fields:
            cfg[section][field] = (path.parent / cfg[section][field]).resolve()
    scope = cfg["snapshot"]
    validate_protocol_version(scope["protocol_version"])
    for field in ("expected_images", "expected_instances", "expected_eyes"):
        if type(scope[field]) is not int or scope[field] < 1:
            raise Phase1Error(f"Invalid baseline count: {field}")
    if scope["expected_eyes"] != 2 * scope["expected_instances"]:
        raise Phase1Error("The approved baseline requires both eyes for each instance.")
    data = load_config(cfg["data"]["phase1_config"])
    output = cfg["output"]["directory"]
    if not output.is_relative_to(data.output.parent / "phase3") or output.is_relative_to(
        cfg["data"]["ground_truth_directory"]
    ):
        raise Phase1Error("Use an independent outputs/phase3 directory, outside live Ground Truth.")
    cfg["source"] = path
    return cfg, data


def freeze(cfg: dict[str, Any], data: AnalysisConfig, destination: Path) -> dict[str, Any]:
    phase1 = json.loads((data.output / "run_manifest.json").read_text())
    dataset = load_coco(data.annotations, expected_sha256=phase1["dataset"]["annotations_sha256"])
    proposal = cfg["data"]["proposal_csv"]
    targets = load_targets(proposal, dataset)
    source = cfg["data"]["ground_truth_directory"]
    scope = cfg["snapshot"]
    validate_protocol_version(scope["protocol_version"])
    if (
        len(targets) != scope["expected_instances"]
        or len({r["image_id"] for r in targets}) != scope["expected_images"]
    ):
        raise Phase1Error("Trial identities differ from the approved baseline scope.")
    destination.mkdir()
    selected = []
    input_hashes = {}
    with (source / ".annotations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source_manifest = json.loads((source / "source_manifest.json").read_text())
        if (
            source_manifest["protocol_version"] != scope["protocol_version"]
            or source_manifest["dataset_sha256"] != dataset.source_sha256
            or source_manifest["coordinate_convention"] != COORDINATES
        ):
            raise Phase1Error("GT protocol, dataset hash or coordinate convention differs.")
        expected_targets = [
            {
                k: t[k]
                for k in (
                    "image_id",
                    "annotation_id",
                    "file_name",
                    "image_width",
                    "image_height",
                    "image_sha256",
                )
            }
            for t in targets
        ]
        if source_manifest["targets"] != expected_targets:
            raise Phase1Error("Live source manifest differs from the trial proposal.")
        rows = load_annotations(source, targets, protocol_version=scope["protocol_version"])
        for target in targets:
            for side in ("left", "right"):
                key = int(target["image_id"]), int(target["annotation_id"]), side
                row = latest_annotation(rows, key)
                if row is None or not (
                    row["mask_review_status"] == "Accepted"
                    and row["measurement_quality"] == "Pass"
                    and row["visibility"] == "Visible"
                    and row["side_confirmed"] == "Yes"
                    and row["reviewer"].strip()
                    and row["mask_path"]
                    and row["protocol_version"] == scope["protocol_version"]
                    and not load_geometry(source, row)["active"]
                ):
                    raise Phase1Error(
                        f"Latest eye {key} is not fully reviewed; no history fallback."
                    )
                with Image.open(source / row["mask_path"]) as mask:
                    if mask.getbbox() is None:
                        raise Phase1Error(f"Accepted eye {key} contains an empty mask.")
                selected.append(dict(row))
                revision = f"revisions/{key[0]}_{key[1]}_{key[2]}_v{row['annotation_version']}.json"
                for relative in (row["mask_path"], revision):
                    original = safe_path(source, relative)
                    input_hashes[str(original)] = sha256_file(original)
                    copy_file(original, safe_path(destination, relative))
        for path, relative in (
            (source / "source_manifest.json", "source_manifest.json"),
            (proposal, "proposed_instances.csv"),
        ):
            input_hashes[str(path)] = sha256_file(path)
            copy_file(path, destination / relative)
        if scope["protocol_version"] == V2_PROTOCOL_VERSION:
            prepared_protocol = source.parent / "protocol.md"
            preparation_path = source.parent / "preparation_manifest.json"
            if not prepared_protocol.is_file() or not preparation_path.is_file():
                raise Phase1Error("Prepared v2 annotation protocol and manifest are required.")
            preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
            if preparation["protocol_version"] != V2_PROTOCOL_VERSION or preparation[
                "files_sha256"
            ]["protocol.md"] != sha256_file(prepared_protocol):
                raise Phase1Error("Prepared v2 annotation protocol differs from its manifest.")
            input_hashes[str(preparation_path)] = sha256_file(preparation_path)
            copy_file(preparation_path, destination / "preparation_manifest.json")
            input_hashes[str(prepared_protocol)] = sha256_file(prepared_protocol)
            copy_file(prepared_protocol, destination / "protocol.md")
        input_hashes[str(source / "eye_mask_annotations.csv")] = sha256_file(
            source / "eye_mask_annotations.csv"
        )
        for image_id in sorted({int(t["image_id"]) for t in targets}):
            record = dataset.images[image_id]
            image_path = (data.images / record.file_name).resolve()
            target = next(t for t in targets if int(t["image_id"]) == image_id)
            if (
                not image_path.is_relative_to(data.images)
                or sha256_file(image_path) != target["image_sha256"]
            ):
                raise Phase1Error("Original image differs from the reviewed trial image.")
            with Image.open(image_path) as image:
                image.load()
                if image.size != (record.width, record.height):
                    raise Phase1Error("Original image dimensions differ from COCO.")
            input_hashes[str(image_path)] = target["image_sha256"]
            copy_file(image_path, safe_path(destination, "images/" + record.file_name))
        write_csv(destination / "eye_mask_annotations.csv", selected, FIELDS)
        if any(sha256_file(Path(path)) != digest for path, digest in input_hashes.items()):
            raise Phase1Error("A selected source changed during snapshot capture.")
    if len(selected) != scope["expected_eyes"]:
        raise Phase1Error("Selected eye count differs from the approved scope.")
    snapshot = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "kind": "frozen_manual_mask_reference",
        "dataset": data.dataset_name,
        "dataset_annotations_sha256": dataset.source_sha256,
        "dataset_authenticity_independently_verified": False,
        "protocol_version": scope["protocol_version"],
        "coordinate_convention": COORDINATES,
        "image_count": scope["expected_images"],
        "instance_count": len(targets),
        "eye_count": len(selected),
        "source_history_rows": len(rows),
        "selection": "Latest version per image_id/annotation_id/side; all must be Accepted/Pass; "
        "no automatic relabeling or historical fallback",
        "random_seed": None,
        "selected_versions": [
            {
                k: row[k]
                for k in (
                    "image_id",
                    "annotation_id",
                    "eye_side",
                    "annotation_version",
                    "mask_path",
                )
            }
            for row in selected
        ],
        "source_input_sha256": input_hashes,
        "files_sha256": {
            str(path.relative_to(destination)): sha256_file(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
    }
    write_json(destination / "snapshot_manifest.json", snapshot)
    LOGGER.info(
        json.dumps(
            {
                "event": "manual_masks_frozen",
                "eyes": len(selected),
                "source_history_rows": len(rows),
            }
        )
    )
    return snapshot


def derive(frozen: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot = json.loads((frozen / "snapshot_manifest.json").read_text())
    protocol_version = validate_protocol_version(snapshot["protocol_version"])
    if snapshot["coordinate_convention"] != COORDINATES:
        raise Phase1Error("Unsupported frozen coordinate convention.")
    for relative, digest in snapshot["files_sha256"].items():
        if sha256_file(safe_path(frozen, relative)) != digest:
            raise Phase1Error(f"Frozen input hash differs: {relative}")
    targets = read_csv(frozen / "proposed_instances.csv")
    rows = load_annotations(frozen, targets, protocol_version=protocol_version)
    if len(rows) != snapshot["eye_count"]:
        raise Phase1Error("Frozen CSV must contain only the selected eye versions.")
    target_index = {(int(t["image_id"]), int(t["annotation_id"])): t for t in targets}
    centers = []
    snapshot_sha256 = sha256_file(frozen / "snapshot_manifest.json")
    for row in sorted(
        rows, key=lambda r: (int(r["image_id"]), int(r["annotation_id"]), r["eye_side"])
    ):
        target = target_index[(int(row["image_id"]), int(row["annotation_id"]))]
        with Image.open(safe_path(frozen, row["mask_path"])) as opened:
            binary = np.array(opened, dtype=np.uint8) // 255
        center = mask_centroid(binary)
        centers.append(
            {
                "image_id": int(row["image_id"]),
                "annotation_id": int(row["annotation_id"]),
                "animal_category": target["animal_category"],
                "eye_side": row["eye_side"],
                "annotation_version": int(row["annotation_version"]),
                "protocol_version": row["protocol_version"],
                "mask_path": "frozen/" + row["mask_path"],
                "mask_sha256": sha256_file(safe_path(frozen, row["mask_path"])),
                "mask_area_px": int(binary.sum()),
                "centroid_x_px": center[0] if center else None,
                "centroid_y_px": center[1] if center else None,
                "center_status": "Defined" if center else "Undefined",
                "snapshot_sha256": snapshot_sha256,
            }
        )
    measurements = [
        row | {"snapshot_sha256": snapshot_sha256} for row in build_measurements(centers)
    ]
    return centers, measurements


def verify(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "run_manifest.json").read_text())
    frozen = directory / "frozen"
    if sha256_file(frozen / "snapshot_manifest.json") != manifest["snapshot_sha256"]:
        raise Phase1Error("Frozen manifest changed after baseline creation.")
    for relative, digest in manifest["artifacts_sha256"].items():
        if sha256_file(safe_path(directory, relative)) != digest:
            raise Phase1Error(f"Baseline artifact hash differs: {relative}")
    centers, measurements = derive(frozen)
    for name, computed in (("eye_centers.csv", centers), ("measurements.csv", measurements)):
        stored = read_csv(directory / name)
        expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in computed]
        if stored != expected:
            raise Phase1Error(f"Recomputed values differ from frozen baseline: {name}")
    return {
        "status": "passed",
        "eyes_recomputed": len(centers),
        "measurements_recomputed": len(measurements),
        "uses_live_ground_truth": False,
        "snapshot_sha256": manifest["snapshot_sha256"],
    }


def run(cfg: dict[str, Any], data: AnalysisConfig) -> Path:
    output = cfg["output"]["directory"]
    if output.exists():
        raise Phase1Error("Baseline already exists; refusing to overwrite a frozen reference.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".manual-baseline-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        snapshot = freeze(cfg, data, stage / "frozen")
        centers, measurements = derive(stage / "frozen")
        write_csv(stage / "eye_centers.csv", centers)
        write_csv(stage / "measurements.csv", measurements)
        (stage / "overlays").mkdir()
        targets = read_csv(stage / "frozen/proposed_instances.csv")
        for image_id in sorted({row["image_id"] for row in centers}):
            selected = [row for row in centers if row["image_id"] == image_id]
            target = next(t for t in targets if int(t["image_id"]) == image_id)
            masks = {}
            for row in selected:
                with Image.open(stage / row["mask_path"]) as opened:
                    masks[(row["annotation_id"], row["eye_side"])] = opened.copy()
            with Image.open(stage / "frozen/images" / target["file_name"]) as opened:
                page = render_measurement_overlay(
                    opened.convert("RGB"),
                    selected,
                    [r for r in measurements if r["image_id"] == image_id],
                    masks,
                )
            page.save(stage / "overlays" / f"{image_id:012d}.png")
        summary = {
            "reference_type": (
                "human_visible_eye_and_eyelid_region_mask_centroid_reference_not_physical_ground_truth"
                if snapshot["protocol_version"] == V2_PROTOCOL_VERSION
                else "human_visible_eye_mask_centroid_reference_not_physical_ground_truth"
            ),
            "images": snapshot["image_count"],
            "instances": snapshot["instance_count"],
            "eyes": len(centers),
            "measurements": len(measurements),
            "measurement_types": dict(Counter(r["measurement_type"] for r in measurements)),
            "undefined_measurements": sum(r["distance_px"] is None for r in measurements),
            "units": "original_image_pixels",
            "coordinate_convention": COORDINATES,
            "mask_area_range_px": [
                min(r["mask_area_px"] for r in centers),
                max(r["mask_area_px"] for r in centers),
            ],
            "model_inference": False,
            "independent_landmark_GT": False,
            "repeat_annotation_uncertainty_estimated": False,
            "physical_calibration": False,
        }
        write_json(stage / "summary.json", summary)
        source_files = [
            Path(__file__),
            Path(__file__).with_name("metrology.py"),
            Path(__file__).with_name("visualization.py"),
            cfg["source"],
        ]
        for source in source_files:
            copy_file(source, stage / "provenance" / source.name)
        protocol_document = PROTOCOL_DOCUMENTS[snapshot["protocol_version"]]
        documentation = (
            (protocol_document,)
            if snapshot["protocol_version"] == V2_PROTOCOL_VERSION
            else (protocol_document, "phase3b_annotation_tool.md")
        )
        for name in documentation:
            source = DOCS_ROOT / name
            if snapshot["protocol_version"] == V2_PROTOCOL_VERSION and name == protocol_document:
                source = stage / "frozen/protocol.md"
            if source.is_file():
                copy_file(source, stage / "provenance" / name)
            elif name == protocol_document:
                raise Phase1Error(f"Approved protocol document is missing: {name}")
        manifest = {
            "status": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "snapshot_sha256": sha256_file(stage / "frozen/snapshot_manifest.json"),
            "summary": summary,
            "configuration": json.loads(json.dumps(cfg, default=str)),
            "execution": {"argv": sys.argv, "cwd": str(Path.cwd())},
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": version("numpy"),
                "Pillow": version("Pillow"),
            },
            "calculation": {
                "precision": "float64; no intermediate rounding",
                "centroid": "mean(column+.5), mean(row+.5) over foreground pixels",
                "distance": "Euclidean; same-image anatomical identities only",
            },
            "artifacts_sha256": {
                str(p.relative_to(stage)): sha256_file(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        write_json(stage / "run_manifest.json", manifest)
        result = verify(stage)
        stage.rename(output)
    LOGGER.info(
        json.dumps(
            {
                "event": "manual_measurement_baseline_complete",
                "output": str(output),
                **summary,
                "recomputation": result["status"],
            }
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/measurement_baseline.toml"))
    parser.add_argument(
        "--verify", type=Path, help="Read-only recomputation from a frozen baseline"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.verify is not None:
            print(json.dumps(verify(args.verify.resolve())))
        else:
            cfg, data = read_settings(args.config)
            print(run(cfg, data))
    except (OSError, ValueError, KeyError) as error:
        LOGGER.error(json.dumps({"event": "manual_baseline_failed", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
