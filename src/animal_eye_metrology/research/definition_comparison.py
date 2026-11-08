"""Evaluate unchanged saved model masks against separately frozen v1/v2 human references."""

import argparse
import json
import logging
import math
import platform
import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image

from animal_eye_metrology.annotation.eye_annotations import PROTOCOL_VERSION, V2_PROTOCOL_VERSION
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.measurement.measurement_baseline import (
    derive,
    verify,
    write_csv,
)
from animal_eye_metrology.measurement.metrology import build_measurements, mask_overlap
from animal_eye_metrology.paths import module_source
from animal_eye_metrology.research.eye_comparison import distance_statistics, eye_statistics
from animal_eye_metrology.research.phase2 import read_csv, write_json
from animal_eye_metrology.ui.results_workspace import ResultBundle, ResultRun, load_results_bundle

LOGGER = logging.getLogger(__name__)
EYE_KEY = ("image_id", "annotation_id", "eye_side")
DISTANCE_KEY = ("image_id", "measurement_type", "annotation_id_a", "annotation_id_b")
COUNT_FIELDS = ("expected_images", "expected_instances", "expected_eyes")
QUALIFICATION = {
    "visibility": "Visible",
    "side_confirmed": "Yes",
    "mask_review_status": "Accepted",
    "measurement_quality": "Pass",
}
Row = dict[str, Any]


@dataclass(frozen=True)
class ComparisonInputs:
    bundle: ResultBundle
    v2: ResultRun
    v1_eyes: list[Row]
    v1_distances: list[Row]
    v2_eyes: list[Row]
    v2_distances: list[Row]


def key(row: Row, fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(
        int(row[field])
        if field.endswith("_id") or field.startswith("annotation_id")
        else row[field]
        for field in fields
    )


def index(rows: list[Row], fields: tuple[str, ...]) -> dict[tuple[Any, ...], Row]:
    result = {key(row, fields): row for row in rows}
    if len(result) != len(rows):
        raise Phase1Error("Duplicate comparison identity; no rows may be silently discarded.")
    return result


def read_settings(path: Path) -> Row:
    path = path.expanduser().resolve()
    with path.open("rb") as handle:
        cfg = tomllib.load(handle)
    paths = {"results_config", "v2_reference", "output_directory"}
    if set(cfg) != paths | set(COUNT_FIELDS):
        raise Phase1Error("Definition comparison configuration fields are invalid.")
    if tuple(cfg[field] for field in COUNT_FIELDS) != (3, 6, 12) or any(
        type(cfg[field]) is not int for field in COUNT_FIELDS
    ):
        raise Phase1Error(
            "This approved comparison requires three images, six animals, twelve eyes."
        )
    for field in paths:
        if not isinstance(cfg[field], str) or not cfg[field]:
            raise Phase1Error(f"Invalid comparison path: {field}")
        cfg[field] = (path.parent / cfg[field]).resolve()
    cfg["source"] = path
    return cfg


def _separate_output(cfg: Row, bundle: ResultBundle) -> None:
    output = cfg["output_directory"]
    for source in [cfg["v2_reference"], *(run.directory for run in bundle.runs.values())]:
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise Phase1Error("Comparison output must be separate from every input run.")
    if output.exists():
        raise Phase1Error("Comparison output already exists; refusing to overwrite.")


def _frozen_reference(directory: Path, protocol: str) -> tuple[ResultRun, list[Row], list[Row]]:
    verify(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    run = ResultRun(protocol, directory, manifest, {}, [])
    snapshot_path = run.asset_path("frozen/snapshot_manifest.json")
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot["protocol_version"] != protocol:
        raise Phase1Error(f"Expected reference protocol {protocol}.")
    rows = read_csv(run.asset_path("frozen/eye_mask_annotations.csv"))
    for row in rows:
        if row["protocol_version"] != protocol or any(
            row[k] != v for k, v in QUALIFICATION.items()
        ):
            raise Phase1Error(
                "Reference has unresolved or nonmeasurable eyes; retain their status."
            )
    eyes, distances = derive(directory / "frozen")
    if any(row["center_status"] != "Defined" or row["mask_area_px"] <= 0 for row in eyes):
        raise Phase1Error(
            "A frozen reference contains an undefined or empty eye; do not fill zero."
        )
    return run, eyes, distances


def validate_scope(inputs: ComparisonInputs, cfg: Row) -> None:
    """Keep all approved identities and the same original images in both definitions."""
    v1 = inputs.bundle.runs["reference"]
    v2 = inputs.v2
    snapshots = [
        json.loads(run.asset_path("frozen/snapshot_manifest.json").read_text()) for run in (v1, v2)
    ]
    for field in ("dataset_annotations_sha256", "coordinate_convention"):
        if snapshots[0][field] != snapshots[1][field]:
            raise Phase1Error(f"Reference source or coordinate convention differs: {field}")
    keys1, keys2 = index(inputs.v1_eyes, EYE_KEY), index(inputs.v2_eyes, EYE_KEY)
    if keys1.keys() != keys2.keys():
        raise Phase1Error(
            "v2 eye identities differ from the fixed v1 sample; no subset evaluation."
        )
    counts = (len({k[0] for k in keys1}), len({k[:2] for k in keys1}), len(keys1))
    if counts != tuple(cfg[field] for field in COUNT_FIELDS):
        raise Phase1Error("Reference counts differ from the approved scope.")
    if (
        index(inputs.v1_distances, DISTANCE_KEY).keys()
        != index(inputs.v2_distances, DISTANCE_KEY).keys()
    ):
        raise Phase1Error("v1/v2 measurement identities differ.")
    for run in (v1, v2):
        for row in inputs.v1_eyes if run is v1 else inputs.v2_eyes:
            run.asset_path(row["mask_path"])
    image_hashes = []
    for run, snapshot in zip((v1, v2), snapshots, strict=True):
        hashes = {
            name: digest
            for name, digest in snapshot["files_sha256"].items()
            if name.startswith("images/")
        }
        for name in hashes:
            run.asset_path("frozen/" + name)
        image_hashes.append(hashes)
    if len(image_hashes[0]) != cfg["expected_images"] or image_hashes[0] != image_hashes[1]:
        raise Phase1Error("v1/v2 frozen original images differ.")


def load_inputs(cfg: Row) -> ComparisonInputs:
    bundle = load_results_bundle(cfg["results_config"])
    _separate_output(cfg, bundle)
    _, eyes1, distances1 = _frozen_reference(bundle.runs["reference"].directory, PROTOCOL_VERSION)
    v2, eyes2, distances2 = _frozen_reference(cfg["v2_reference"], V2_PROTOCOL_VERSION)
    inputs = ComparisonInputs(bundle, v2, eyes1, distances1, eyes2, distances2)
    validate_scope(inputs, cfg)
    return inputs


def check_ready(cfg: Row) -> Row:
    """Report pending human work without creating a reference or result directory."""
    bundle = load_results_bundle(cfg["results_config"])
    _separate_output(cfg, bundle)
    if not (cfg["v2_reference"] / "run_manifest.json").is_file():
        return {
            "status": "awaiting_v2_reference",
            "ready": False,
            "expected_images": cfg["expected_images"],
            "expected_eyes": cfg["expected_eyes"],
            "reason": "尚無完整 v2 人工凍結參考。請先完成重標與審核；"
            "Uncertain／Partial 不自動轉 Pass。",
            "reference_directory": str(cfg["v2_reference"]),
            "model_inference": False,
            "result_created": False,
        }
    inputs = load_inputs(cfg)
    return {
        "status": "ready",
        "ready": True,
        "eyes": len(inputs.v2_eyes),
        "conditions": len(inputs.bundle.conditions),
        "model_inference": False,
        "result_created": False,
    }


def _binary(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        array = np.asarray(opened)
    if array.ndim != 2 or not np.isin(array, [0, 255]).all():
        raise Phase1Error("Expected original-grid binary 0/255 mask.")
    return (array > 0).astype(np.uint8)


def _same(saved: Any, computed: Any, label: str) -> None:
    equal = saved == computed
    if isinstance(saved, int | float) and isinstance(computed, int | float):
        equal = math.isclose(saved, computed, abs_tol=1e-12, rel_tol=1e-12)
    if not equal:
        raise Phase1Error(f"Frozen v1/prediction values changed: {label}")


def _prefixed(version_name: str, row: Row) -> Row:
    return {f"{version_name}_{field}": value for field, value in row.items()}


def _difference(first: float | None, second: float | None) -> float | None:
    return first - second if first is not None and second is not None else None


def _summary(eyes: list[Row], distances: list[Row]) -> Row:
    centers = [row["centroid_error_px"] for row in eyes if row["centroid_error_px"] is not None]
    result = {
        "eyes_expected": len(eyes),
        "inference_failed": sum(row["prediction_status"] == "inference_failed" for row in eyes),
        "empty_masks": sum(row["prediction_status"] == "empty" for row in eyes),
        "centroid_defined": len(centers),
        "mean_iou_all_targets_failure_as_zero": mean(row["iou"] or 0.0 for row in eyes),
        "mean_dice_all_targets_failure_as_zero": mean(row["dice"] or 0.0 for row in eyes),
        "mean_centroid_error_px": mean(centers) if centers else None,
    }
    for kind in ("binocular", "right_eye_pair"):
        rows = [row for row in distances if row["measurement_type"] == kind]
        errors = [row["absolute_error_px"] for row in rows if row["absolute_error_px"] is not None]
        result.update(
            {
                f"{kind}_expected": len(rows),
                f"{kind}_defined": len(errors),
                f"{kind}_mae_px": mean(errors) if errors else None,
                f"{kind}_rmse_px": math.sqrt(mean(e * e for e in errors)) if errors else None,
            }
        )
    return result


def evaluate(inputs: ComparisonInputs) -> dict[str, list[Row]]:
    """Re-evaluate the selected frozen masks, never select another candidate or run a model."""
    bundle = inputs.bundle
    references = {"v1": index(inputs.v1_eyes, EYE_KEY), "v2": index(inputs.v2_eyes, EYE_KEY)}
    reference_distances = {
        "v1": index(inputs.v1_distances, DISTANCE_KEY),
        "v2": index(inputs.v2_distances, DISTANCE_KEY),
    }
    masks = {
        tag: {k: _binary(run.asset_path(row["mask_path"])) for k, row in refs.items()}
        for tag, run, refs in (
            ("v1", bundle.runs["reference"], references["v1"]),
            ("v2", inputs.v2, references["v2"]),
        )
    }
    rows_out: dict[str, list[Row]] = {
        name: []
        for name in (
            "eye_comparison.csv",
            "distance_comparison.csv",
            "condition_summary.csv",
            "reference_eye_changes.csv",
            "reference_distance_changes.csv",
        )
    }
    for k, old in references["v1"].items():
        new = references["v2"][k]
        overlap = mask_overlap(masks["v1"][k], masks["v2"][k])
        rows_out["reference_eye_changes.csv"].append(
            {
                **{f: old[f] for f in EYE_KEY},
                "v1_annotation_version": old["annotation_version"],
                "v2_annotation_version": new["annotation_version"],
                "v1_area_px": old["mask_area_px"],
                "v2_area_px": new["mask_area_px"],
                **{
                    f"{tag}_{field}": ref[field]
                    for tag, ref in (("v1", old), ("v2", new))
                    for field in ("centroid_x_px", "centroid_y_px")
                },
                "centroid_shift_px": math.dist(
                    (old["centroid_x_px"], old["centroid_y_px"]),
                    (new["centroid_x_px"], new["centroid_y_px"]),
                ),
                "definition_overlap_iou": overlap["iou"],
                "definition_overlap_dice": overlap["dice"],
            }
        )
    for k, old in reference_distances["v1"].items():
        new = reference_distances["v2"][k]
        change = new["distance_px"] - old["distance_px"]
        rows_out["reference_distance_changes.csv"].append(
            {
                **{f: old[f] for f in DISTANCE_KEY},
                "v1_reference_distance_px": old["distance_px"],
                "v2_reference_distance_px": new["distance_px"],
                "signed_change_px": change,
                "absolute_change_px": abs(change),
            }
        )
    for condition in bundle.conditions:
        identifier = condition["condition_id"]
        descriptor = {
            "condition_id": identifier,
            "method_role": condition["method_role"],
            "original_point_source": condition.get("point_source"),
            "original_box_source": condition.get("box_source"),
            "human_prompts_remain_from_v1_experiment": condition["method_role"]
            == "human_assisted_diagnostic",
        }
        evaluated = {"v1": [], "v2": []}
        for saved in [row for row in bundle.eyes if row["condition_id"] == identifier]:
            k = key(saved, EYE_KEY)
            predicted = (
                _binary(bundle.asset_path(saved["source_run"], saved["predicted_mask_path"]))
                if saved.get("predicted_mask_path")
                else None
            )
            per_definition = {tag: eye_statistics(predicted, masks[tag][k]) for tag in ("v1", "v2")}
            for field, value in per_definition["v1"].items():
                _same(saved[field], value, f"{identifier}/{k}/{field}")
            for tag in ("v1", "v2"):
                evaluated[tag].append({**{f: saved[f] for f in EYE_KEY}, **per_definition[tag]})
            rows_out["eye_comparison.csv"].append(
                {
                    **descriptor,
                    **{f: saved[f] for f in EYE_KEY},
                    "source_run": saved["source_run"],
                    "prediction_mask_path": saved.get("predicted_mask_path"),
                    "failure_reason": saved.get("failure_reason", ""),
                    "visibility_status": saved["visibility_status"],
                    **_prefixed("v1", per_definition["v1"]),
                    **_prefixed("v2", per_definition["v2"]),
                    **{
                        f"delta_{f}_v2_minus_v1": _difference(
                            per_definition["v2"][f], per_definition["v1"][f]
                        )
                        for f in ("iou", "dice", "centroid_error_px")
                    },
                }
            )
        centers = [
            {
                **{f: row[f] for f in EYE_KEY},
                "annotation_version": 1,
                "centroid_x_px": row["predicted_cx_px"],
                "centroid_y_px": row["predicted_cy_px"],
            }
            for row in evaluated["v1"]
        ]
        distances = index(build_measurements(centers), DISTANCE_KEY)
        saved_distances = index(
            [row for row in bundle.distances if row["condition_id"] == identifier], DISTANCE_KEY
        )
        measured: dict[str, list[Row]] = {"v1": [], "v2": []}
        for k, prediction in distances.items():
            per_definition = {
                tag: distance_statistics(
                    prediction["distance_px"], reference_distances[tag][k]["distance_px"]
                )
                for tag in ("v1", "v2")
            }
            saved = saved_distances[k]
            for field, value in per_definition["v1"].items():
                _same(saved[field], value, f"{identifier}/{k}/{field}")
            for tag in ("v1", "v2"):
                measured[tag].append(
                    {"measurement_type": prediction["measurement_type"], **per_definition[tag]}
                )
            rows_out["distance_comparison.csv"].append(
                {
                    **descriptor,
                    **{f: prediction[f] for f in DISTANCE_KEY},
                    "visibility_status": saved["visibility_status"],
                    **_prefixed("v1", per_definition["v1"]),
                    **_prefixed("v2", per_definition["v2"]),
                    "delta_absolute_error_px_v2_minus_v1": _difference(
                        per_definition["v2"]["absolute_error_px"],
                        per_definition["v1"]["absolute_error_px"],
                    ),
                }
            )
        summaries = {tag: _summary(evaluated[tag], measured[tag]) for tag in ("v1", "v2")}
        for field, value in summaries["v1"].items():
            _same(condition[field], value, f"{identifier}/summary/{field}")
        rows_out["condition_summary.csv"].append(
            {
                **descriptor,
                **_prefixed("v1", summaries["v1"]),
                **_prefixed("v2", summaries["v2"]),
                "interpretation": "target_definition_change_not_model_improvement",
            }
        )
    return rows_out


def compare(cfg: Row) -> Path:
    inputs = load_inputs(cfg)
    tables = evaluate(inputs)
    output = cfg["output_directory"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".definition-comparison-", dir=output.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        for filename, rows in tables.items():
            write_csv(stage / filename, rows)
        write_json(stage / "evaluation_values.json", tables)
        provenance = stage / "provenance"
        provenance.mkdir()
        for source in (
            cfg["source"],
            cfg["results_config"],
            Path(__file__),
            module_source("paths.py"),
            module_source("_compat.py"),
            module_source("metrology.py"),
            module_source("eye_comparison.py"),
            module_source("measurement_baseline.py"),
        ):
            shutil.copyfile(source, provenance / source.name)
        source_manifests = {}
        for name, run in {**inputs.bundle.runs, "reference_v2": inputs.v2}.items():
            source = run.directory / "run_manifest.json"
            if json.loads(source.read_text()) != run.manifest:
                raise Phase1Error(f"Source manifest changed during comparison: {name}")
            shutil.copyfile(source, provenance / f"{name}_run_manifest.json")
            source_manifests[name] = {
                "directory": str(run.directory),
                "manifest_sha256": sha256_file(source),
            }
        manifest = {
            "status": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "definitions": {"v1": PROTOCOL_VERSION, "v2": V2_PROTOCOL_VERSION},
            "reference_snapshots": {
                "v1": inputs.bundle.runs["reference"].manifest["snapshot_sha256"],
                "v2": inputs.v2.manifest["snapshot_sha256"],
            },
            "source_runs": source_manifests,
            "scope": {
                "images": cfg["expected_images"],
                "instances": cfg["expected_instances"],
                "eyes": cfg["expected_eyes"],
                "conditions": len(inputs.bundle.conditions),
            },
            "model_inference": False,
            "model_training": False,
            "candidate_selection_changed": False,
            "baseline_promoted": False,
            "new_human_prompts": False,
            "random_seed": None,
            "measurement": "original-image px; region-mask centroid distance, "
            "not pupil or physical distance",
            "interpretation": "Differences reflect a changed reference target; "
            "not model improvement or independent generalization.",
            "execution": {"argv": sys.argv, "cwd": str(Path.cwd())},
            "environment": {
                "python": platform.python_version(),
                "numpy": version("numpy"),
                "Pillow": version("Pillow"),
            },
            "files_sha256": {
                str(p.relative_to(stage)): sha256_file(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        write_json(stage / "run_manifest.json", manifest)
        if output.exists():
            raise Phase1Error("Comparison output appeared during execution; refusing overwrite.")
        stage.rename(output)
    LOGGER.info(json.dumps({"event": "definition_comparison_complete", "output": str(output)}))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/definition_comparison_v2.toml")
    )
    parser.add_argument(
        "--check", action="store_true", help="Read-only readiness check; no output files"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        cfg = read_settings(args.config)
        status = check_ready(cfg)
        if args.check or not status["ready"]:
            print(json.dumps(status, ensure_ascii=False))
            return 0 if status["ready"] else 2
        print(compare(cfg))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "invalid_inputs",
                    "ready": False,
                    "reason": str(error),
                    "result_created": False,
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
