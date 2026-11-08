"""Synthetic frozen references exercise definition switching and provenance guards."""

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL.Image")

from test_definition_comparison import comparison_config  # noqa: E402, F401 - fixture

from animal_eye_metrology import definition_comparison as comparison  # noqa: E402
from animal_eye_metrology.results_definitions import load_definition_workspace  # noqa: E402
from animal_eye_metrology.results_workspace import (  # noqa: E402
    ResultsWorkspaceError,
    load_results_bundle,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(directory: Path) -> None:
    path = directory / "run_manifest.json"
    manifest = json.loads(path.read_text())
    key = "files_sha256" if "files_sha256" in manifest else "artifacts_sha256"
    manifest[key] = {
        str(p.relative_to(directory)): _digest(p)
        for p in directory.rglob("*")
        if p.is_file() and p != path
    }
    path.write_text(json.dumps(manifest))


@pytest.fixture
def definition_package(comparison_config: Path) -> Path:  # noqa: F811 - fixture dependency
    for name in ("automatic", "ablation", "auto_box"):
        path = comparison_config.parent / name / "evaluation_values.json"
        data = json.loads(path.read_text())
        for row in data["eyes" if name == "automatic" else "eye_metrics.csv"]:
            row.update(
                point_in_target_mask=True,
                candidate_best_iou=0.75,
                reference_box_coverage_fraction=0.9,
                point_score=0.98,
            )
        path.write_text(json.dumps(data))
        _seal(path.parent)
    comparison.compare(comparison.read_settings(comparison_config))
    config = comparison_config.parent / "definitions.toml"
    snapshot = json.loads((config.parent / "v2/run_manifest.json").read_text())["snapshot_sha256"]
    config.write_text(
        'base_results_config = "results.toml"\ncomparison_run = "comparison"\n'
        f'v2_reference_run = "v2"\nv2_snapshot_sha256 = "{snapshot}"\n'
    )
    return config


def _change_comparison(config: Path, change: Callable[[dict], None]) -> None:
    directory = config.parent / "comparison"
    path = directory / "evaluation_values.json"
    values = json.loads(path.read_text())
    change(values)
    path.write_text(json.dumps(values))
    for name, rows in values.items():
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (directory / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    _seal(directory)


def test_switches_references_without_changing_predictions_or_writing(definition_package: Path):
    before = {p: p.read_bytes() for p in definition_package.parent.rglob("*") if p.is_file()}
    workspace = load_definition_workspace(definition_package)
    original = load_results_bundle(definition_package.parent / "results.toml")
    assert workspace.definitions == ("v1", "v2")
    assert workspace.for_definition("v1") == original
    v2 = workspace.for_definition("v2")
    assert workspace.v1.eyes[0]["iou"] == 1.0
    assert v2.eyes[0]["iou"] == pytest.approx(2 / 3)
    assert workspace.v1.eyes[0]["reference_cx_px"] + 0.5 == v2.eyes[0]["reference_cx_px"]
    assert [r["condition_id"] for r in v2.conditions[:2]] == ["A_r100", "A_r050"]
    assert v2.runs["reference"].directory.name == "v2"
    assert v2.runs["reference_v1"] == workspace.v1.runs["reference"]
    for first, second in zip(workspace.v1.eyes, v2.eyes, strict=True):
        for field in ("predicted_mask_path", "predicted_cx_px", "predicted_cy_px", "prediction_id"):
            assert first[field] == second[field]
        assert workspace.v1.prediction_for_eye(first) == v2.prediction_for_eye(second)
        assert "point_in_target_mask" not in second
        assert "candidate_best_iou" not in second
        assert "reference_box_coverage_fraction" not in second
        assert second["historical_v1_diagnostics"]["point_in_target_mask"] is True
        assert second["point_score"] == 0.98
        assert v2.reference_for_eye(second)["protocol_version"] == "eye-visible-eyelid-v2"
    assert {p: p.read_bytes() for p in before} == before


def test_keeps_missing_empty_and_unverified_results_in_original_denominators(
    definition_package: Path,
):
    workspace = load_definition_workspace(definition_package)
    v2 = workspace.for_definition("v2")
    missing = next(row for row in v2.eyes if row["prediction_status"] == "inference_failed")
    assert missing["predicted_cx_px"] is None
    assert missing["centroid_error_px"] is None
    assert missing["iou"] is None  # Only the saved aggregate treats this failure as zero.
    assert any(row["prediction_status"] == "empty" for row in v2.eyes)
    summary = next(row for row in v2.conditions if row["condition_id"] == "A_r050")
    assert (
        summary["eyes_expected"],
        summary["centroid_defined"],
        summary["binocular_defined"],
    ) == (12, 11, 5)
    assert len(workspace.paired_eyes) == 48
    assert len(workspace.paired_distances) == 36
    assert any(row["absolute_error_px"] is None for row in v2.distances)
    assert all(row["visibility_status"] == "unverified" for row in v2.distances)
    assert (
        next(row for row in v2.conditions if row["condition_id"] == "auto_box_k050_point")[
            "method_role"
        ]
        == "primary_not_promoted"
    )


def test_legacy_config_and_config_relative_paths_remain_supported(
    definition_package: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(definition_package.parent.parent)
    workspace = load_definition_workspace(definition_package)
    assert workspace.v2.runs["reference"].directory == definition_package.parent / "v2"
    legacy = load_definition_workspace(definition_package.parent / "results.toml")
    assert legacy.definitions == ("v1",)
    assert legacy.v2 is None and legacy.comparison is None and legacy.paired_eyes == []
    with pytest.raises(ResultsWorkspaceError, match="未提供"):
        legacy.for_definition("v2")


@pytest.mark.parametrize(
    "target",
    ["comparison_artifact", "source_manifest", "pinned_snapshot", "protocol", "missing_v2"],
)
def test_rejects_missing_or_modified_sources_without_fallback(
    definition_package: Path, target: str
):
    directory = definition_package.parent
    if target == "comparison_artifact":
        path = directory / "comparison/evaluation_values.json"
        path.write_text(path.read_text() + " ")
    elif target == "source_manifest":
        path = directory / "automatic/run_manifest.json"
        data = json.loads(path.read_text())
        data["changed"] = True
        path.write_text(json.dumps(data))
    elif target == "pinned_snapshot":
        text = definition_package.read_text()
        snapshot = json.loads((directory / "v2/run_manifest.json").read_text())["snapshot_sha256"]
        definition_package.write_text(text.replace(snapshot, "0" * 64))
    elif target == "missing_v2":
        definition_package.write_text(
            definition_package.read_text().replace(
                'v2_reference_run = "v2"', 'v2_reference_run = "missing"'
            )
        )
    else:
        path = directory / "comparison/run_manifest.json"
        data = json.loads(path.read_text())
        data["definitions"]["v2"] = "unapproved-target"
        path.write_text(json.dumps(data))
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(ResultsWorkspaceError):
        load_definition_workspace(definition_package)
    assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_row",
        "duplicate_row",
        "v1_metric",
        "predicted_center",
        "predicted_distance",
        "mask_path",
        "metric_schema",
        "manual_role",
    ],
)
def test_rejects_resealed_inconsistent_comparison(definition_package: Path, corruption: str):
    def mutate(values: dict) -> None:
        eyes = values["eye_comparison.csv"]
        if corruption == "missing_row":
            eyes.pop()
        elif corruption == "duplicate_row":
            eyes.append(dict(eyes[0]))
        elif corruption == "v1_metric":
            eyes[0]["v1_iou"] = 0.1
        elif corruption == "predicted_center":
            eyes[0]["v2_predicted_cx_px"] += 1
        elif corruption == "predicted_distance":
            values["distance_comparison.csv"][0]["v2_predicted_distance_px"] += 1
        elif corruption == "mask_path":
            eyes[0]["prediction_mask_path"] = "another-mask.png"
        elif corruption == "metric_schema":
            for row in eyes:
                del row["v1_iou"], row["v2_iou"]
        else:
            row = next(r for r in values["condition_summary.csv"] if r["condition_id"] == "B_r100")
            row["human_prompts_remain_from_v1_experiment"] = False

    _change_comparison(definition_package, mutate)
    with pytest.raises(ResultsWorkspaceError):
        load_definition_workspace(definition_package)


def test_rejects_csv_json_disagreement_even_when_each_file_is_resealed(definition_package: Path):
    directory = definition_package.parent / "comparison"
    path = directory / "evaluation_values.json"
    values = json.loads(path.read_text())
    values["eye_comparison.csv"][0]["v2_iou"] = 0.123
    path.write_text(json.dumps(values))
    _seal(directory)
    with pytest.raises(ResultsWorkspaceError, match="CSV/JSON"):
        load_definition_workspace(definition_package)
