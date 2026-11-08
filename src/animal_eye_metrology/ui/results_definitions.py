"""Read saved v1/v2 scores without changing predictions or reevaluating models."""

from __future__ import annotations

import csv
import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from animal_eye_metrology.ui.results_workspace import (
    DISTANCE_KEY,
    EYE_KEY,
    ResultBundle,
    ResultRun,
    ResultsWorkspaceError,
    _index,
    _json,
    _key,
    _reference_csv,
    _rows,
    _safe_path,
    load_results_bundle,
)

Row = dict[str, Any]
PROTOCOLS = {"v1": "eye-visible-aperture-v1", "v2": "eye-visible-eyelid-v2"}
WRAPPER_FIELDS = {
    "base_results_config",
    "comparison_run",
    "v2_reference_run",
    "v2_snapshot_sha256",
}
TABLES = {
    "condition_summary.csv",
    "eye_comparison.csv",
    "distance_comparison.csv",
    "reference_eye_changes.csv",
    "reference_distance_changes.csv",
}
EYE_METRICS = {
    "reference_area_px",
    "reference_cx_px",
    "reference_cy_px",
    "intersection_px",
    "union_px",
    "iou",
    "dice",
    "prediction_status",
    "predicted_area_px",
    "predicted_cx_px",
    "predicted_cy_px",
    "centroid_dx_px",
    "centroid_dy_px",
    "centroid_error_px",
}
DISTANCE_METRICS = {
    "predicted_distance_px",
    "reference_distance_px",
    "signed_error_px",
    "absolute_error_px",
    "relative_absolute_error_percent",
    "measurement_status",
}
SUMMARY_METRICS = {
    "eyes_expected",
    "inference_failed",
    "empty_masks",
    "centroid_defined",
    "mean_iou_all_targets_failure_as_zero",
    "mean_dice_all_targets_failure_as_zero",
    "mean_centroid_error_px",
    "binocular_expected",
    "binocular_defined",
    "binocular_mae_px",
    "binocular_rmse_px",
    "right_eye_pair_expected",
    "right_eye_pair_defined",
    "right_eye_pair_mae_px",
    "right_eye_pair_rmse_px",
}


@dataclass(frozen=True)
class DefinitionWorkspace:
    """Two views of the same predictions, with the original baseline retained."""

    v1: ResultBundle
    v2: ResultBundle | None
    comparison: ResultRun | None
    paired_conditions: list[Row]
    paired_eyes: list[Row]
    paired_distances: list[Row]

    @property
    def definitions(self) -> tuple[str, ...]:
        return ("v1", "v2") if self.v2 is not None else ("v1",)

    def for_definition(self, definition: str) -> ResultBundle:
        if definition == "v1":
            return self.v1
        if definition == "v2" and self.v2 is not None:
            return self.v2
        raise ResultsWorkspaceError(f"此成果未提供定義：{definition}")


def _same(first: Any, second: Any, label: str) -> None:
    equal = first == second
    if isinstance(first, bool) or isinstance(second, bool):
        equal = type(first) is type(second) and first == second
    if (
        isinstance(first, int | float)
        and not isinstance(first, bool)
        and isinstance(second, int | float)
        and not isinstance(second, bool)
    ):
        equal = math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)
    if not equal:
        raise ResultsWorkspaceError(f"兩版成果不一致：{label}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(name: str, directory: Path) -> ResultRun:
    manifest = _json(_safe_path(directory, "run_manifest.json"))
    if not isinstance(manifest, dict):
        raise ResultsWorkspaceError(f"Manifest 格式錯誤：{name}")
    return ResultRun(name, directory, manifest, {}, [])


def _check_csv(run: ResultRun, filename: str, rows: list[Row]) -> None:
    with run.asset_path(filename).open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    _same(len(saved), len(rows), f"{filename} CSV/JSON rows")
    for actual, expected in zip(saved, rows, strict=True):
        _same(
            actual,
            {key: "" if value is None else str(value) for key, value in expected.items()},
            f"{filename} CSV/JSON values",
        )


def _snapshot(run: ResultRun, definition: str) -> Row:
    path = run.asset_path("frozen/snapshot_manifest.json")
    _same(_digest(path), run.manifest["snapshot_sha256"], f"{definition} snapshot digest")
    snapshot = _json(path)
    _same(snapshot["protocol_version"], PROTOCOLS[definition], f"{definition} protocol")
    return snapshot


def _historical_field(field: str) -> bool:
    return (
        field.startswith("reference")
        or "_reference" in field
        or "in_target_mask" in field
        or "in_opposite_mask" in field
        or "offset_from_mask_centroid" in field
        or ("candidate" in field and ("iou" in field or "gap" in field))
        or field
        in {
            "area_ratio",
            "max_centroid_error_px",
            "binocular_max_error_px",
            "right_eye_pair_max_error_px",
        }
    )


def _metadata(row: Row) -> Row:
    historical = {key: value for key, value in row.items() if _historical_field(key)}
    result = {key: value for key, value in row.items() if key not in historical}
    result["historical_v1_diagnostics"] = historical
    return result


def _project(
    rows: list[Row], paired: list[Row], fields: tuple[str, ...], metrics: set[str]
) -> list[Row]:
    original = _index(rows, fields, "original v1")
    changes = _index(paired, fields, "paired definitions")
    _same(set(original), set(changes), "comparison identities / denominator")
    projected = []
    for identity, old in original.items():
        pair = changes[identity]
        old_metrics = {name[3:]: value for name, value in pair.items() if name.startswith("v1_")}
        new_metrics = {name[3:]: value for name, value in pair.items() if name.startswith("v2_")}
        _same(set(old_metrics), metrics, "v1 metric fields")
        _same(set(new_metrics), metrics, "v2 metric fields")
        for name, value in old_metrics.items():
            if name not in old:
                raise ResultsWorkspaceError(f"v1 原結果缺少比較欄位：{name}")
            _same(old[name], value, f"{identity} / v1 {name}")
            if name.startswith("predicted_") or name in {
                "prediction_status",
                "measurement_status",
                "eyes_expected",
                "inference_failed",
                "empty_masks",
                "centroid_defined",
                "binocular_expected",
                "binocular_defined",
                "right_eye_pair_expected",
                "right_eye_pair_defined",
            }:
                _same(value, new_metrics[name], f"{identity} / unchanged prediction {name}")
        if "method_role" in pair:
            _same(pair["method_role"], old["method_role"], f"{identity} / method role")
        for name in ("visibility_status", "failure_reason", "source_run"):
            if name in pair:
                _same(pair[name], old.get(name, ""), f"{identity} / {name}")
        if "prediction_mask_path" in pair:
            _same(
                pair["prediction_mask_path"] or None,
                old.get("predicted_mask_path") or None,
                f"{identity} / prediction mask path",
            )
        projected.append({**_metadata(old), **new_metrics, "reference_definition": "v2"})
    return projected


def _verify_reference(
    base: ResultBundle, v2: ResultRun, v2_eyes: list[Row], v2_distances: list[Row]
) -> None:
    snapshots = [_snapshot(base.runs["reference"], "v1"), _snapshot(v2, "v2")]
    for name in ("dataset_annotations_sha256", "coordinate_convention"):
        _same(snapshots[0][name], snapshots[1][name], name)
    image_hashes = []
    for run, snapshot in zip((base.runs["reference"], v2), snapshots, strict=True):
        images = {
            name: value
            for name, value in snapshot["files_sha256"].items()
            if name.startswith("images/")
        }
        image_hashes.append(images)
        for name, expected in images.items():
            _same(_digest(run.asset_path("frozen/" + name)), expected, name)
    _same(image_hashes[0], image_hashes[1], "frozen original images")
    _same(len(image_hashes[0]), len(base.image_ids), "original image count")
    for before, after, fields in (
        (base.reference_eyes, v2_eyes, EYE_KEY),
        (base.reference_measurements, v2_distances, DISTANCE_KEY),
    ):
        _same(
            set(_index(before, fields, "v1 reference")),
            set(_index(after, fields, "v2 reference")),
            "v1/v2 reference identities",
        )
    for run, eyes in ((base.runs["reference"], base.reference_eyes), (v2, v2_eyes)):
        for row in eyes:
            run.asset_path(row["mask_path"])
            _same(row["snapshot_sha256"], run.manifest["snapshot_sha256"], "reference eye snapshot")
    qualification = _reference_csv(v2, "frozen/eye_mask_annotations.csv")
    _same(
        set(_index(qualification, EYE_KEY, "qualified v2 eyes")),
        set(_index(v2_eyes, EYE_KEY, "v2 centers")),
        "v2 qualification identities",
    )
    for row in qualification:
        for name, value in {
            "protocol_version": PROTOCOLS["v2"],
            "visibility": "Visible",
            "mask_review_status": "Accepted",
            "measurement_quality": "Pass",
            "side_confirmed": "Yes",
        }.items():
            _same(row[name], value, f"v2 qualification {name}")
        if not str(row.get("reviewer", "")).strip():
            raise ResultsWorkspaceError("v2 凍結參考缺少人工 Reviewer。")


def _load_paired(config_path: Path, config: Row) -> DefinitionWorkspace:
    if set(config) != WRAPPER_FIELDS or any(
        not isinstance(value, str) or not value for value in config.values()
    ):
        raise ResultsWorkspaceError("兩版成果設定欄位不完整或不合法。")
    resolve = lambda name: (config_path.parent / config[name]).resolve()  # noqa: E731
    base = load_results_bundle(resolve("base_results_config"))
    v2 = _run("reference_v2", resolve("v2_reference_run"))
    comparison = _run("definition_comparison", resolve("comparison_run"))
    _same(v2.manifest["snapshot_sha256"], config["v2_snapshot_sha256"], "pinned v2 snapshot")
    manifest = comparison.manifest
    _same(manifest["status"], "complete", "comparison status")
    _same(manifest["definitions"], PROTOCOLS, "comparison protocols")
    for name in (
        "model_inference",
        "model_training",
        "candidate_selection_changed",
        "baseline_promoted",
        "new_human_prompts",
    ):
        _same(manifest[name], False, name)
    _same(
        manifest["reference_snapshots"],
        {
            "v1": base.runs["reference"].manifest["snapshot_sha256"],
            "v2": v2.manifest["snapshot_sha256"],
        },
        "comparison snapshots",
    )
    sources = {**base.runs, "reference_v2": v2}
    _same(set(manifest["source_runs"]), set(sources), "comparison sources")
    for name, run in sources.items():
        # Historical absolute directories are provenance, not portable runtime paths.
        _same(
            manifest["source_runs"][name]["manifest_sha256"],
            _digest(run.directory / "run_manifest.json"),
            f"source manifest {name}",
        )
    values = _json(comparison.asset_path("evaluation_values.json"))
    _same(set(values), TABLES, "comparison table names")
    values = {name: _rows(rows, name) for name, rows in values.items()}
    comparison = ResultRun(comparison.key, comparison.directory, manifest, values, [])
    # Check downloadable copies too; the UI must not expose a different CSV than its JSON.
    for name in TABLES:
        _check_csv(comparison, name, values[name])
    v2_eyes = _reference_csv(v2, "eye_centers.csv")
    v2_distances = _reference_csv(v2, "measurements.csv")
    v2 = ResultRun(
        v2.key,
        v2.directory,
        v2.manifest,
        {"eye_centers.csv": v2_eyes, "measurements.csv": v2_distances},
        [],
    )
    _verify_reference(base, v2, v2_eyes, v2_distances)
    _same(
        manifest["scope"],
        {
            "images": len(base.image_ids),
            "instances": len(
                {(row["image_id"], row["annotation_id"]) for row in base.reference_eyes}
            ),
            "eyes": len(base.reference_eyes),
            "conditions": len(base.conditions),
        },
        "comparison scope",
    )
    roles = {row["condition_id"]: row["method_role"] for row in base.conditions}
    descriptors = {row["condition_id"]: row for row in base.conditions}
    for name in ("condition_summary.csv", "eye_comparison.csv", "distance_comparison.csv"):
        for row in values[name]:
            original = descriptors[row["condition_id"]]
            for field, expected in {
                "method_role": original["method_role"],
                "original_point_source": original.get("point_source"),
                "original_box_source": original.get("box_source"),
                "human_prompts_remain_from_v1_experiment": original["method_role"]
                == "human_assisted_diagnostic",
            }.items():
                _same(row[field], expected, f"{name} / {field}")
    eye_source = [dict(row, method_role=roles[row["condition_id"]]) for row in base.eyes]
    distance_source = [dict(row, method_role=roles[row["condition_id"]]) for row in base.distances]
    conditions = _project(
        base.conditions, values["condition_summary.csv"], ("condition_id",), SUMMARY_METRICS
    )
    eyes = _project(
        eye_source, values["eye_comparison.csv"], ("condition_id", *EYE_KEY), EYE_METRICS
    )
    distances = _project(
        distance_source,
        values["distance_comparison.csv"],
        ("condition_id", *DISTANCE_KEY),
        DISTANCE_METRICS,
    )
    references = _index(v2_eyes, EYE_KEY, "v2 reference")
    reference_distances = _index(v2_distances, DISTANCE_KEY, "v2 measurements")
    for row in eyes:
        ref = references[_key(row, EYE_KEY)]
        row["reference_mask_path"] = ref["mask_path"]
        for field, ref_field in (
            ("reference_area_px", "mask_area_px"),
            ("reference_cx_px", "centroid_x_px"),
            ("reference_cy_px", "centroid_y_px"),
        ):
            _same(row[field], ref[ref_field], f"v2 eye {field}")
        if row.get("predicted_mask_path"):
            base.asset_path(row["source_run"], row["predicted_mask_path"])
    for row in distances:
        _same(
            row["reference_distance_px"],
            reference_distances[_key(row, DISTANCE_KEY)]["distance_px"],
            "v2 reference distance",
        )
    for name, expected, fields in (
        ("reference_eye_changes.csv", base.reference_eyes, EYE_KEY),
        ("reference_distance_changes.csv", base.reference_measurements, DISTANCE_KEY),
    ):
        _same(set(_index(values[name], fields, name)), set(_index(expected, fields, name)), name)
    projected = ResultBundle(
        config_path,
        {**base.runs, "reference_v1": base.runs["reference"], "reference": v2},
        conditions,
        eyes,
        distances,
        [_metadata(row) for row in base.localization],
        v2_eyes,
        v2_distances,
        [dict(row, reference_definition="v1") for row in base.promotion],
    )
    return DefinitionWorkspace(
        base,
        projected,
        comparison,
        values["condition_summary.csv"],
        values["eye_comparison.csv"],
        values["distance_comparison.csv"],
    )


def load_definition_workspace(config_path: Path) -> DefinitionWorkspace:
    """Load either the original v1 config or an explicit pinned two-definition wrapper."""
    config_path = config_path.expanduser().resolve()
    try:
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        if "base_results_config" not in config and "comparison_run" not in config:
            return DefinitionWorkspace(load_results_bundle(config_path), None, None, [], [], [])
        return _load_paired(config_path, config)
    except ResultsWorkspaceError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ResultsWorkspaceError(f"無法載入兩版成果：{error}") from error
