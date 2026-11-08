"""Definition browsing keeps saved predictions fixed; all references are synthetic."""

import csv
import io
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("PIL.Image")
pytest.importorskip("numpy")
pytest.importorskip("pycocotools.mask")

from streamlit.testing.v1 import AppTest  # noqa: E402
from test_definition_comparison import (  # noqa: E402
    _hashes,
    comparison_config,  # noqa: F401 - imported pytest fixture
)

from animal_eye_metrology.core.verification import sha256_file  # noqa: E402
from animal_eye_metrology.research import definition_comparison as comparison  # noqa: E402
from animal_eye_metrology.ui import results_view  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def rendered_definitions(comparison_config: Path) -> Path:  # noqa: F811
    """Add display metadata before freezing the synthetic comparison sources."""
    root = comparison_config.parent
    for source in ("automatic", "ablation", "auto_box"):
        directory = root / source
        values = json.loads((directory / "evaluation_values.json").read_text())
        eye_key = "eyes" if source == "automatic" else "eye_metrics.csv"
        distance_key = "measurements" if source == "automatic" else "measurement_comparison.csv"
        for eye in values[eye_key]:
            eye.update(
                point_in_target_mask=True,
                point_offset_from_mask_centroid_px=0.123,
                candidate_best_iou=0.999,
                reference_box_coverage_fraction=0.75,
            )
        eyes = {
            (
                row.get("condition_id", "A_r100"),
                row["image_id"],
                row["annotation_id"],
                row["eye_side"],
            ): row
            for row in values[eye_key]
        }
        for distance in values[distance_key]:
            condition = distance.get("condition_id", "A_r100")
            side_a = "left" if distance["measurement_type"] == "binocular" else "right"
            first = eyes[(condition, distance["image_id"], distance["annotation_id_a"], side_a)]
            second = eyes[(condition, distance["image_id"], distance["annotation_id_b"], "right")]
            for suffix, eye in (("a", first), ("b", second)):
                distance[f"predicted_cx_{suffix}_px"] = eye["predicted_cx_px"]
                distance[f"predicted_cy_{suffix}_px"] = eye["predicted_cy_px"]
        _write_json(directory / "evaluation_values.json", values)
        filename = "segmentation.json" if source == "automatic" else "predictions.json"
        raw = json.loads((directory / filename).read_text())
        predictions = raw["results"] if source == "automatic" else raw
        for prediction in predictions:
            prediction["roi_xyxy"] = [0, 0, 64, 32]
            prediction["box_xyxy"] = None
        _write_json(directory / filename, raw)
        manifest = json.loads((directory / "run_manifest.json").read_text())
        manifest["files_sha256"] = _hashes(directory, {"run_manifest.json"})
        _write_json(directory / "run_manifest.json", manifest)
    comparison.compare(comparison.read_settings(comparison_config))
    config = root / "definitions_workspace.toml"
    snapshot = sha256_file(root / "v2/frozen/snapshot_manifest.json")
    config.write_text(
        'base_results_config = "results.toml"\n'
        'comparison_run = "comparison"\n'
        'v2_reference_run = "v2"\n'
        f'v2_snapshot_sha256 = "{snapshot}"\n',
        encoding="utf-8",
    )
    return config


def _app(config: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.delenv("AEM_ANNOTATION_SESSION", raising=False)
    monkeypatch.setenv("AEM_WORKSPACE", "Results Explorer")
    monkeypatch.setenv("AEM_RESULTS_CONFIG", str(config))
    monkeypatch.setenv("AEM_CONFIG", str(config.parent / "missing-phase1.toml"))
    return AppTest.from_file(str(APP), default_timeout=30).run()


def _table(at: AppTest, column: str):
    return next(element.value for element in at.dataframe if column in element.value.columns)


def _files(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_switching_definition_changes_reference_metrics_only(
    rendered_definitions: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _files(rendered_definitions.parent)
    at = _app(rendered_definitions, monkeypatch)
    assert not at.exception
    assert not at.error
    assert at.radio(key="results_definition").value == "v1"
    summary = _table(at, "用途")
    assert summary.iloc[0]["條件"] == "A_r100"
    assert summary.iloc[0]["IoU ↑"] == pytest.approx(1)
    old = _table(at, "執行失敗原因").copy()

    at.radio(key="results_definition").set_value("v2").run()
    assert not at.exception
    assert not at.error
    summary = _table(at, "用途")
    assert summary.iloc[0]["條件"] == "A_r100"
    assert summary.iloc[0]["IoU ↑"] == pytest.approx(2 / 3)
    assert summary.iloc[0]["重心偏差 px ↓"] == pytest.approx(0.5)
    new = _table(at, "執行失敗原因")
    for field in ("預測重心 x", "預測重心 y", "預測面積 px²"):
        assert old[field].equals(new[field])
    assert new.iloc[0]["參考重心 x"] - old.iloc[0]["參考重心 x"] == pytest.approx(0.5)
    assert "synthetic_missing" in set(new["執行失敗原因"])
    assert any("未晉級" in item.value for item in at.warning)
    assert _files(rendered_definitions.parent) == before


def test_paired_eye_view_preserves_baselines_and_missing_predictions(
    rendered_definitions: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(rendered_definitions, monkeypatch)
    at.radio(key="results_definition").set_value("paired").run()
    assert not at.exception
    assert not at.error
    rows = _table(at, "執行失敗原因")
    assert set(rows["條件"]) == {"A_r100", "A_r050"}
    assert set(rows["參考定義"]) == {"v1", "v2"}
    assert len(rows) == 4
    baseline = rows.loc[rows["條件"] == "A_r100"]
    assert baseline["預測重心 x"].nunique() == 1
    assert baseline["預測重心 y"].nunique() == 1
    assert baseline["參考重心 x"].nunique() == 2
    missing = rows.loc[rows["條件"] == "A_r050"]
    assert missing["預測重心 x"].isna().all()
    assert missing["重心偏差 px"].isna().all()
    assert set(missing["執行失敗原因"]) == {"synthetic_missing"}

    summary = _table(at, "用途")
    assert list(summary["條件"][:2]) == ["A_r100", "A_r100"]
    assert "B_r100" not in set(summary["條件"])
    at.checkbox(key="results_include_manual").check().run()
    assert set(_table(at, "用途").loc[lambda frame: frame["條件"] == "B_r100", "用途"]) == {
        "人工介入診斷"
    }


def test_distance_download_identifies_each_definition_and_keeps_prediction_fixed(
    rendered_definitions: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: dict[str, tuple[str, list[dict[str, str]]]] = {}
    original_download = results_view.download

    def capture(rows: list[dict[str, Any]], label: str, filename: str, key: str) -> None:
        decoded = list(
            csv.DictReader(io.StringIO(results_view.csv_bytes(rows).decode("utf-8-sig")))
        )
        downloads[key] = filename, decoded
        original_download(rows, label, filename, key)

    monkeypatch.setattr(results_view, "download", capture)
    at = _app(rendered_definitions, monkeypatch)
    at.radio(key="results_definition").set_value("paired").run()
    at.selectbox(key="results_distance_image").select(2).run()
    at.selectbox(key="results_distance_condition").select("auto_box_k050_point").run()
    assert not at.exception
    assert not at.error
    filename, rows = downloads["results_distances_download"]
    assert "paired" in filename
    assert {row["reference_definition"] for row in rows} == {"v1", "v2"}
    assert {row["reference_protocol"] for row in rows} == {
        "eye-visible-aperture-v1",
        "eye-visible-eyelid-v2",
    }
    assert {row["condition_id"] for row in rows} == {
        "A_r100",
        "A_r050",
        "auto_box_k050_point",
    }
    assert {row["image_id"] for row in rows} == {"2"}
    assert {row["visibility_status"] for row in rows} == {"unverified"}
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in ("condition_id", "measurement_type", "annotation_id_a", "annotation_id_b")
        )
        grouped.setdefault(key, []).append(row)
    assert all(len(pair) == 2 for pair in grouped.values())
    assert all(
        pair[0]["predicted_distance_px"] == pair[1]["predicted_distance_px"]
        for pair in grouped.values()
    )
    assert any(
        pair[0]["reference_distance_px"] != pair[1]["reference_distance_px"]
        for pair in grouped.values()
    )


def test_original_v1_config_remains_available_without_v2_assets(
    rendered_definitions: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(rendered_definitions.parent / "results.toml", monkeypatch)
    assert not at.exception
    assert not at.error
    assert at.radio(key="results_definition").options == ["v1 · 可見眼裂"]
    assert _table(at, "用途").iloc[0]["IoU ↑"] == pytest.approx(1)


def test_failure_explanation_distinguishes_historical_v1_and_v2_findings(
    rendered_definitions: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Substitute only the case identity to exercise real explanatory UI on invented geometry.
    name = "狗 17207 · 左眼邊界"
    cases = dict(results_view.FAILURE_CASES)
    old_note = cases[name][3]
    cases[name] = (1, 11, "left", old_note)
    monkeypatch.setattr(results_view, "FAILURE_CASES", cases)
    new_note = results_view.V2_CASE_NOTES[name]

    def visible_text(at: AppTest) -> str:
        return "\n".join(
            str(element.value)
            for kind in ("markdown", "caption", "info", "warning", "success")
            for element in at.get(kind)
        )

    at = _app(rendered_definitions, monkeypatch)
    assert not at.exception
    assert not at.error
    assert old_note in visible_text(at)
    assert new_note not in visible_text(at)
    at.radio(key="results_definition").set_value("v2").run()
    assert not at.exception
    assert not at.error
    assert new_note in visible_text(at)
    at.radio(key="results_definition").set_value("paired").run()
    assert old_note in visible_text(at)
    assert new_note in visible_text(at)
