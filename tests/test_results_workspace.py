"""Synthetic read-only result packages; no personal annotations or model runs."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from animal_eye_metrology.results_workspace import ResultsWorkspaceError, load_results_bundle

SNAPSHOT = "a" * 64


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _seal(directory: Path, *, reference: bool = False) -> None:
    hashes = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    }
    _save(
        directory / "run_manifest.json",
        {
            "snapshot_sha256" if reference else "primary_snapshot_sha256": SNAPSHOT,
            "artifacts_sha256" if reference else "files_sha256": hashes,
        },
    )


def _csv_file(path: Path, rows: list[dict]) -> None:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(output.getvalue(), encoding="utf-8")


@pytest.fixture
def result_package(tmp_path: Path) -> Path:
    eyes = [
        {
            "image_id": 1,
            "annotation_id": 2,
            "prediction_id": "p1",
            "eye_side": side,
            "predicted_mask_path": f"eyes/{side}.png",
            "iou": 0.5,
            "predicted_cx_px": x,
            "predicted_cy_px": 4.5,
            "prediction_status": "nonempty",
            "visibility_status": "unverified",
        }
        for side, x in (("left", 2.5), ("right", 7.5))
    ]
    distance = {
        "image_id": 1,
        "measurement_type": "binocular",
        "annotation_id_a": 2,
        "annotation_id_b": 2,
        "predicted_distance_px": 5.0,
        "reference_distance_px": 5.5,
        "measurement_status": "Defined",
        "visibility_status": "unverified",
    }
    auto = tmp_path / "automatic"
    _save(auto / "evaluation_values.json", {"eyes": eyes, "measurements": [distance]})
    _save(auto / "segmentation.json", {"results": [dict(row, point_xy=[3.0, 4.0]) for row in eyes]})
    _seal(auto)
    for source, condition_ids in (
        ("ablation", ["A_r100", "A_r050", "B_r100"]),
        ("auto_box", ["auto_box_k050_point", "auto_box_k030_only"]),
    ):
        summaries = [
            {
                "condition_id": condition,
                "point_source": "manual" if condition.startswith("B_") else "hrnet",
                "box_source": "geometry" if source == "auto_box" else "none",
                "eyes_expected": 2,
                "binocular_expected": 1,
                "right_eye_pair_expected": 0,
            }
            for condition in condition_ids
        ]
        all_eyes = [
            dict(
                row,
                condition_id=condition,
                predicted_mask_path=f"cases/{condition}/{row['eye_side']}.png",
            )
            for condition in condition_ids
            for row in eyes
        ]
        predictions = [
            dict(row, point_xy=None if row["condition_id"].endswith("only") else [3.0, 4.0])
            for row in all_eyes
        ]
        _save(
            tmp_path / source / "evaluation_values.json",
            {
                "condition_summary.csv": summaries,
                "eye_metrics.csv": all_eyes,
                "measurement_comparison.csv": [
                    dict(distance, condition_id=condition) for condition in condition_ids
                ],
                "localization_metrics.csv": [{"point_source": "hrnet", "point_score": 0.9}],
                "promotion_checks.csv": [{"condition_id": "auto_box_k050_point", "passed": False}],
            },
        )
        _save(tmp_path / source / "predictions.json", predictions)
        _seal(tmp_path / source)
    reference = tmp_path / "reference"
    reference.mkdir()
    _csv_file(
        reference / "eye_centers.csv",
        [
            {
                "image_id": 1,
                "annotation_id": 2,
                "eye_side": side,
                "mask_path": f"frozen/{side}.png",
                "centroid_x_px": x,
                "centroid_y_px": 4.5,
                "snapshot_sha256": SNAPSHOT,
            }
            for side, x in (("left", 2.5), ("right", 8.0))
        ],
    )
    _csv_file(reference / "measurements.csv", [{**distance, "distance_px": 5.5}])
    _seal(reference, reference=True)
    config = tmp_path / "results.toml"
    config.write_text(
        "\n".join(
            f'{source}_run = "{source}"'
            for source in ("automatic", "ablation", "auto_box", "reference")
        )
        + f'\nprimary_snapshot_sha256 = "{SNAPSHOT}"\n',
        encoding="utf-8",
    )
    return config


def _change_table(config: Path, source: str, table: str, change: object) -> None:
    path = config.parent / source / "evaluation_values.json"
    data = json.loads(path.read_text())
    change(data[table])
    _save(path, data)
    _seal(path.parent)


def test_preserves_original_baseline_and_diagnostic_roles_without_writing(result_package: Path):
    before = {p: p.read_bytes() for p in result_package.parent.rglob("*") if p.is_file()}
    bundle = load_results_bundle(result_package)
    assert [row["condition_id"] for row in bundle.conditions[:2]] == ["A_r100", "A_r050"]
    assert len(bundle.eyes) == 10  # five conditions; the reused A baseline is not counted twice
    baseline = bundle.eyes[0]
    assert baseline["source_run"] == "automatic"
    assert baseline["predicted_mask_path"] == "eyes/left.png"
    assert bundle.prediction_for_eye(baseline)["point_xy"] == [3.0, 4.0]
    assert bundle.reference_for_eye(baseline)["centroid_x_px"] == 2.5
    roles = {row["condition_id"]: row["method_role"] for row in bundle.conditions}
    assert roles["B_r100"] == "human_assisted_diagnostic"
    assert roles["auto_box_k050_point"] == "primary_not_promoted"
    only_box = next(row for row in bundle.eyes if row["condition_id"] == "auto_box_k030_only")
    assert bundle.prediction_for_eye(only_box)["point_xy"] is None
    assert all(row["visibility_status"] == "unverified" for row in bundle.distances)
    assert {p: p.read_bytes() for p in result_package.parent.rglob("*") if p.is_file()} == before


def test_rejects_changed_reused_baseline_metric(result_package: Path):
    _change_table(
        result_package, "ablation", "eye_metrics.csv", lambda rows: rows[0].update(iou=0.9)
    )
    with pytest.raises(ResultsWorkspaceError, match="A_r100.*不一致"):
        load_results_bundle(result_package)


def test_keeps_failures_in_denominator_and_rejects_missing_rows(result_package: Path):
    _change_table(
        result_package,
        "auto_box",
        "eye_metrics.csv",
        lambda rows: rows[0].update(prediction_status="failed", iou=None),
    )
    bundle = load_results_bundle(result_package)
    assert len(bundle.eyes) == 10
    assert any(row["prediction_status"] == "failed" and row["iou"] is None for row in bundle.eyes)
    _change_table(result_package, "auto_box", "eye_metrics.csv", lambda rows: rows.pop())
    with pytest.raises(ResultsWorkspaceError, match="分母不一致"):
        load_results_bundle(result_package)


def test_rejects_duplicate_eye_and_wrong_reference_membership(result_package: Path):
    _change_table(
        result_package, "auto_box", "eye_metrics.csv", lambda rows: rows[1].update(eye_side="left")
    )
    with pytest.raises(ResultsWorkspaceError, match="重複結果鍵"):
        load_results_bundle(result_package)
    _change_table(
        result_package,
        "auto_box",
        "eye_metrics.csv",
        lambda rows: rows[1].update(eye_side="right", annotation_id=999),
    )
    with pytest.raises(ResultsWorkspaceError, match="樣本與人工參考不一致"):
        load_results_bundle(result_package)


def test_rejects_reference_snapshot_mismatch(result_package: Path):
    path = result_package.parent / "reference/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["snapshot_sha256"] = "b" * 64
    _save(path, manifest)
    with pytest.raises(ResultsWorkspaceError, match="人工參考版本不一致"):
        load_results_bundle(result_package)


def test_detects_modified_artifact_missing_file_and_path_escape(result_package: Path):
    bundle = load_results_bundle(result_package)
    with pytest.raises(ResultsWorkspaceError, match="escapes"):
        bundle.asset_path("automatic", "../results.toml")
    with pytest.raises(ResultsWorkspaceError, match="找不到"):
        bundle.asset_path("automatic", "missing.png")
    path = result_package.parent / "automatic/evaluation_values.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ResultsWorkspaceError, match="雜湊不符"):
        load_results_bundle(result_package)


def test_config_paths_resolve_independently_of_cwd(
    result_package: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(result_package.parent.parent)
    bundle = load_results_bundle(result_package)
    assert bundle.runs["reference"].directory == result_package.parent / "reference"
