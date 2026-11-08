"""Read-only, provenance-preserving adapters for frozen eye experiment results.

This module never imports inference runners, evaluates masks, or writes artifacts.
The A_r100 display uses original automatic results; its reused ablation entries
are checked before supplementing them with existing ROI/candidate diagnostics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Row = dict[str, Any]
BASELINE = "A_r100"
COMPARISON = "A_r050"
EYE_KEY = ("image_id", "annotation_id", "eye_side")
DISTANCE_KEY = ("image_id", "measurement_type", "annotation_id_a", "annotation_id_b")


class ResultsWorkspaceError(ValueError):
    """Missing, inconsistent, or unsafe saved results cannot be displayed."""


def _safe_path(directory: Path, relative: str) -> Path:
    candidate = (directory / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(directory.resolve()):
        raise ResultsWorkspaceError(f"Artifact path escapes its run directory: {relative}")
    if not candidate.is_file():
        raise ResultsWorkspaceError(f"找不到已保存的結果檔案：{candidate}")
    return candidate


@dataclass(frozen=True)
class ResultRun:
    key: str
    directory: Path
    manifest: Row
    tables: dict[str, list[Row]]
    predictions: list[Row]

    def asset_path(self, relative: str) -> Path:
        """Resolve only files recorded by this run and check their saved digest."""
        path = _safe_path(self.directory, relative)
        hashes = self.manifest.get("files_sha256", self.manifest.get("artifacts_sha256", {}))
        expected = hashes.get(relative)
        if not isinstance(expected, str):
            raise ResultsWorkspaceError(f"產物不在已保存 manifest 中：{self.key}/{relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ResultsWorkspaceError(f"產物雜湊不符：{self.key}/{relative}")
        return path


@dataclass(frozen=True)
class ResultBundle:
    config_path: Path
    runs: dict[str, ResultRun]
    conditions: list[Row]
    eyes: list[Row]
    measurements: list[Row]
    localization: list[Row]
    reference_eyes: list[Row]
    reference_measurements: list[Row]
    promotion: list[Row]

    @property
    def distances(self) -> list[Row]:
        return self.measurements

    @property
    def image_ids(self) -> list[int]:
        return sorted({int(row["image_id"]) for row in self.reference_eyes})

    def run_for_condition(self, condition_id: str) -> ResultRun:
        if condition_id == BASELINE:
            return self.runs["automatic"]
        for row in self.conditions:
            if row["condition_id"] == condition_id:
                return self.runs[row["source_run"]]
        raise ResultsWorkspaceError(f"未知條件：{condition_id}")

    def asset_path(self, source_id: str, relative: str) -> Path:
        try:
            return self.runs[source_id].asset_path(relative)
        except KeyError as exc:
            raise ResultsWorkspaceError(f"未知產物來源：{source_id}") from exc

    def visualizations(self, source_id: str) -> list[Path]:
        run = self.runs[source_id]
        hashes = run.manifest.get("files_sha256", run.manifest.get("artifacts_sha256", {}))
        return [
            run.asset_path(relative)
            for relative in sorted(hashes)
            if relative.startswith(("visualizations/", "overlays/")) and relative.endswith(".png")
        ]

    def reference_for_eye(self, row: Row) -> Row:
        key = _key(row, EYE_KEY)
        return _lookup(self.reference_eyes, EYE_KEY, key, "reference eye")

    def prediction_for_eye(self, row: Row) -> Row:
        run = self.run_for_condition(str(row["condition_id"]))
        fields = ("image_id", "prediction_id", "eye_side")
        if row["condition_id"] != BASELINE:
            fields = ("condition_id", *fields)
        return _lookup(run.predictions, fields, _key(row, fields), "prediction")


def _key(row: Row, fields: tuple[str, ...]) -> tuple[Any, ...]:
    try:
        return tuple(row[field] for field in fields)
    except KeyError as exc:
        raise ResultsWorkspaceError(f"結果缺少必要欄位：{exc.args[0]}") from exc


def _index(rows: list[Row], fields: tuple[str, ...], label: str) -> dict[tuple[Any, ...], Row]:
    result: dict[tuple[Any, ...], Row] = {}
    for row in rows:
        key = _key(row, fields)
        if key in result:
            raise ResultsWorkspaceError(f"重複結果鍵：{label} {key}")
        result[key] = row
    return result


def _lookup(rows: list[Row], fields: tuple[str, ...], key: tuple[Any, ...], label: str) -> Row:
    try:
        return _index(rows, fields, label)[key]
    except KeyError as exc:
        raise ResultsWorkspaceError(f"找不到對應結果：{label} {key}") from exc


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResultsWorkspaceError(f"無法讀取結果 JSON：{path} ({exc})") from exc


def _rows(value: Any, label: str) -> list[Row]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ResultsWorkspaceError(f"結果表格格式錯誤：{label}")
    return value


def _reference_csv(run: ResultRun, filename: str) -> list[Row]:
    with run.asset_path(filename).open(encoding="utf-8", newline="") as handle:
        rows: list[Row] = list(csv.DictReader(handle))
    for row in rows:
        for field, value in row.items():
            if value == "":
                row[field] = None
            elif field.startswith(("image_id", "annotation_id", "annotation_version")):
                row[field] = int(value)
            elif field.endswith("_px"):
                row[field] = float(value)
    return rows


def _load_run(key: str, directory: Path, snapshot: str) -> ResultRun:
    manifest = _json(_safe_path(directory, "run_manifest.json"))
    if not isinstance(manifest, dict):
        raise ResultsWorkspaceError(f"Manifest 格式錯誤：{directory}")
    actual = manifest.get("primary_snapshot_sha256", manifest.get("snapshot_sha256"))
    if actual != snapshot:
        raise ResultsWorkspaceError(f"人工參考版本不一致：{key}")
    run = ResultRun(key, directory, manifest, {}, [])
    if key == "reference":
        tables = {
            name: _reference_csv(run, name) for name in ("eye_centers.csv", "measurements.csv")
        }
        return ResultRun(key, directory, manifest, tables, [])
    values = _json(run.asset_path("evaluation_values.json"))
    if not isinstance(values, dict):
        raise ResultsWorkspaceError(f"評估資料格式錯誤：{key}")
    tables = {name: _rows(rows, name) for name, rows in values.items()}
    if key == "automatic":
        tables = {
            "eye_metrics.csv": tables["eyes"],
            "measurement_comparison.csv": tables["measurements"],
        }
        predictions = _rows(_json(run.asset_path("segmentation.json"))["results"], "segmentation")
    else:
        predictions = _rows(_json(run.asset_path("predictions.json")), "predictions")
    return ResultRun(key, directory, manifest, tables, predictions)


def _check_reused(original: list[Row], reused: list[Row], fields: tuple[str, ...]) -> None:
    original_by_key = _index(original, fields, "original A_r100")
    reused_by_key = _index(reused, fields, "reused A_r100")
    if original_by_key.keys() != reused_by_key.keys():
        raise ResultsWorkspaceError("A_r100 原結果與重用結果的樣本集合不同。")
    # Paths/case IDs differ between copied runs. Shared metric/status values may not.
    for key, row in original_by_key.items():
        other = reused_by_key[key]
        for name in row.keys() & other.keys():
            if name.endswith("_path") or name == "case_id":
                continue
            a, b = row[name], other[name]
            equal = a == b
            if isinstance(a, (float, int)) and isinstance(b, (float, int)):
                equal = math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            if not equal:
                raise ResultsWorkspaceError(f"A_r100 原結果與重用結果不一致：{key} / {name}")


def _condition_role(row: Row) -> str:
    condition_id = row["condition_id"]
    if condition_id == BASELINE:
        return "original_baseline"
    if condition_id == COMPARISON:
        return "roi_comparison"
    if row.get("point_source") in {"manual", "human"} or row.get("box_source") in {
        "manual",
        "human",
    }:
        return "human_assisted_diagnostic"
    if condition_id == "auto_box_k050_point":
        return "primary_not_promoted"
    return "automatic_exploratory"


def _merge_table(runs: dict[str, ResultRun], filename: str, fields: tuple[str, ...]) -> list[Row]:
    automatic = runs["automatic"].tables[filename]
    ablation = runs["ablation"].tables[filename]
    reused = [row for row in ablation if row["condition_id"] == BASELINE]
    _check_reused(automatic, reused, fields)
    reused_index = _index(reused, fields, "reused baseline")
    rows = [
        {
            **reused_index[_key(row, fields)],
            **row,
            "condition_id": BASELINE,
            "source_run": "automatic",
        }
        for row in automatic
    ]
    rows += [
        dict(row, source_run="ablation") for row in ablation if row["condition_id"] != BASELINE
    ]
    rows += [dict(row, source_run="auto_box") for row in runs["auto_box"].tables[filename]]
    _index(rows, ("condition_id", *fields), filename)
    return rows


def load_results_bundle(config_path: Path) -> ResultBundle:
    """Load saved values and verify source hashes, reference version, and A reuse.

    Paths are relative to the TOML file, so the workspace can move without
    consulting the absolute execution paths in historical manifests.
    """
    config_path = config_path.expanduser().resolve()
    try:
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        keys = {"automatic", "ablation", "auto_box", "reference"}
        expected = {f"{key}_run" for key in keys} | {"primary_snapshot_sha256"}
        if set(config) != expected or any(
            not isinstance(value, str) or not value for value in config.values()
        ):
            raise ResultsWorkspaceError(
                "成果檢視設定需包含四個 run 路徑與 primary_snapshot_sha256。"
            )
        runs = {
            key: _load_run(
                key,
                (config_path.parent / config[f"{key}_run"]).resolve(),
                config["primary_snapshot_sha256"],
            )
            for key in ("automatic", "ablation", "auto_box", "reference")
        }
        conditions = [
            dict(row, source_run=source, method_role=_condition_role(row))
            for source in ("ablation", "auto_box")
            for row in runs[source].tables["condition_summary.csv"]
        ]
        condition_index = _index(conditions, ("condition_id",), "conditions")
        if (BASELINE,) not in condition_index or (COMPARISON,) not in condition_index:
            raise ResultsWorkspaceError("缺少 A_r100 原基線或 A_r050 對照，無法載入比較工作區。")
        conditions.sort(
            key=lambda row: (
                row["condition_id"] != BASELINE,
                row["condition_id"] != COMPARISON,
                row["condition_id"],
            )
        )
        eyes = _merge_table(runs, "eye_metrics.csv", EYE_KEY)
        distances = _merge_table(runs, "measurement_comparison.csv", DISTANCE_KEY)
        for name, rows, count_field in (
            ("eyes", eyes, "eyes_expected"),
            ("distances", distances, None),
        ):
            if {row["condition_id"] for row in rows} != {row["condition_id"] for row in conditions}:
                raise ResultsWorkspaceError(f"{name} 與彙總的條件集合不同。")
            for condition in conditions:
                expected_count = (
                    condition[count_field]
                    if count_field
                    else condition["binocular_expected"] + condition["right_eye_pair_expected"]
                )
                if (
                    sum(row["condition_id"] == condition["condition_id"] for row in rows)
                    != expected_count
                ):
                    raise ResultsWorkspaceError(
                        f"{name} 筆數與既有分母不一致：{condition['condition_id']}"
                    )
        reference_eyes = runs["reference"].tables["eye_centers.csv"]
        reference_measurements = runs["reference"].tables["measurements.csv"]
        reference_eye_index = _index(reference_eyes, EYE_KEY, "reference eyes")
        reference_distance_index = _index(
            reference_measurements, DISTANCE_KEY, "reference distances"
        )
        for condition in conditions:
            condition_id = condition["condition_id"]
            if {
                _key(row, EYE_KEY) for row in eyes if row["condition_id"] == condition_id
            } != reference_eye_index.keys():
                raise ResultsWorkspaceError(f"眼部樣本與人工參考不一致：{condition_id}")
            if {
                _key(row, DISTANCE_KEY) for row in distances if row["condition_id"] == condition_id
            } != reference_distance_index.keys():
                raise ResultsWorkspaceError(f"距離樣本與人工參考不一致：{condition_id}")
        localization = [
            dict(row, source_run="ablation")
            for row in runs["ablation"].tables["localization_metrics.csv"]
        ]
        promotion = [
            dict(row, source_run="auto_box")
            for row in runs["auto_box"].tables["promotion_checks.csv"]
        ]
        return ResultBundle(
            config_path,
            runs,
            conditions,
            eyes,
            distances,
            localization,
            reference_eyes,
            reference_measurements,
            promotion,
        )
    except ResultsWorkspaceError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ResultsWorkspaceError(f"無法載入已保存結果：{exc}") from exc
