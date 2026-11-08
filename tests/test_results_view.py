"""Streamlit result browsing uses synthetic files and never loads COCO or models."""

import copy
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("pycocotools.mask")

from streamlit.testing.v1 import AppTest  # noqa: E402
from test_results_workspace import (  # noqa: E402
    _change_table,
    _csv_file,
    _save,
    _seal,
    result_package,  # noqa: F401 - imported pytest fixture
)

APP = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


def _second_image(row: dict) -> dict:
    duplicate = copy.deepcopy(row)
    duplicate["image_id"] = 2
    for key in ("annotation_id", "annotation_id_a", "annotation_id_b"):
        if key in duplicate:
            duplicate[key] = 10
    if "prediction_id" in duplicate:
        duplicate["prediction_id"] = "p2"
    for key in ("predicted_mask_path", "mask_path"):
        if duplicate.get(key):
            duplicate[key] = f"image2/{duplicate[key]}"
    return duplicate


def _mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (48, 32), 0)
    mask.putpixel((2, 4), 255)
    mask.save(path)


@pytest.fixture
def rendered_package(result_package: Path) -> Path:  # noqa: F811 - pytest fixture dependency
    """Extend data-layer fixtures with two images and complete display metadata."""
    for source in ("automatic", "ablation", "auto_box"):
        directory = result_package.parent / source
        values = json.loads((directory / "evaluation_values.json").read_text())
        eye_table = "eyes" if source == "automatic" else "eye_metrics.csv"
        distance_table = "measurements" if source == "automatic" else "measurement_comparison.csv"
        for eye in values[eye_table]:
            eye.update(
                animal_category="synthetic",
                reference_cx_px=eye["predicted_cx_px"],
                reference_cy_px=eye["predicted_cy_px"],
                centroid_error_px=0.0,
                dice=0.6,
                reference_area_px=1,
                predicted_area_px=1,
            )
        values[eye_table] += [_second_image(row) for row in values[eye_table]]
        for eye in values[eye_table]:
            _mask(directory / eye["predicted_mask_path"])
        for distance in values[distance_table]:
            distance.update(
                predicted_cx_a_px=2.5,
                predicted_cy_a_px=4.5,
                predicted_cx_b_px=7.5,
                predicted_cy_b_px=4.5,
                signed_error_px=-0.5,
                absolute_error_px=0.5,
            )
        values[distance_table] += [_second_image(row) for row in values[distance_table]]
        for condition in values.get("condition_summary.csv", []):
            condition.update(eyes_expected=4, binocular_expected=2)
        _save(directory / "evaluation_values.json", values)
        filename = "segmentation.json" if source == "automatic" else "predictions.json"
        raw_predictions = json.loads((directory / filename).read_text())
        predictions = raw_predictions["results"] if source == "automatic" else raw_predictions
        predictions += [_second_image(row) for row in predictions]
        for prediction in predictions:
            prediction["roi_xyxy"] = [0, 0, 16, 12]
            prediction["box_xyxy"] = [1, 2, 9, 8] if source == "auto_box" else None
            prediction["hrnet_point_xy"] = [3.0, 4.0]
        _save(directory / filename, raw_predictions)
        _seal(directory)
    reference = result_package.parent / "reference"
    for filename in ("eye_centers.csv", "measurements.csv"):
        with (reference / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows += [_second_image(row) for row in rows]
        _csv_file(reference / filename, rows)
        if filename == "eye_centers.csv":
            for row in rows:
                _mask(reference / row["mask_path"])
    for image_id in (1, 2):
        path = reference / f"frozen/images/{image_id:012d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 32), "gray").save(path)
    _seal(reference, reference=True)
    return result_package


def _app(config: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("AEM_WORKSPACE", "Results Explorer")
    monkeypatch.setenv("AEM_RESULTS_CONFIG", str(config))
    monkeypatch.setenv("AEM_CONFIG", str(config.parent / "does-not-exist-phase1.toml"))
    return AppTest.from_file(str(APP), default_timeout=30).run()


def _summary(at: AppTest):
    return next(table.value for table in at.dataframe if "用途" in table.value.columns)


def test_results_are_independent_read_only_and_keep_prompt_roles(
    rendered_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {p: p.read_bytes() for p in rendered_package.parent.rglob("*") if p.is_file()}
    at = _app(rendered_package, monkeypatch)
    assert not at.exception
    assert not at.error
    assert at.radio(key="workspace").value == "Results Explorer"
    assert _summary(at).iloc[0]["條件"] == "A_r100"
    assert "B_r100" not in set(_summary(at)["條件"])
    assert any("γ＝0.5" in message.value and "未晉級" in message.value for message in at.warning)

    at.checkbox(key="results_include_manual").check().run()
    manual = _summary(at).loc[lambda frame: frame["條件"] == "B_r100"].iloc[0]
    assert manual["用途"] == "人工介入診斷"
    at.selectbox(key="results_eye_condition").select("auto_box_k030_only").run()
    assert not at.exception
    assert not at.error
    prompts = [
        json.loads(element.value) for element in at.json if "prompt_point_xy" in element.value
    ]
    assert prompts[0]["prompt_point_xy"] == [3.0, 4.0]
    assert prompts[1]["prompt_point_xy"] is None
    assert prompts[1]["hrnet_geometry_point_xy"] == [3.0, 4.0]
    assert {p: p.read_bytes() for p in rendered_package.parent.rglob("*") if p.is_file()} == before


def test_switching_images_resets_inapplicable_animal_and_keeps_distance_baselines(
    rendered_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(rendered_package, monkeypatch)
    assert at.selectbox(key="results_eye_animal_1").value == 2
    at.selectbox(key="results_eye_image").select(2).run()
    assert not at.exception
    assert not at.error
    assert at.selectbox(key="results_eye_animal_2").value == 10
    at.selectbox(key="results_distance_image").select(2).run()
    at.selectbox(key="results_distance_condition").select("auto_box_k030_only").run()
    distance_table = next(
        table.value for table in at.dataframe if "預測距離 px" in table.value.columns
    )
    assert set(distance_table["條件"]) == {"A_r100", "A_r050", "auto_box_k030_only"}
    assert set(distance_table["動物 A"]) == {10}
    assert set(distance_table["量測資格／可見性"]) == {"unverified"}
    at.selectbox(key="results_eye_image").select(1).run()
    assert at.selectbox(key="results_eye_animal_1").value == 2
    assert not at.exception


def test_missing_config_and_missing_mask_are_explained_without_unhandled_exception(
    rendered_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = rendered_package.parent / "missing.toml"
    at = _app(missing, monkeypatch)
    assert not at.exception
    assert any("無法載入成果" in message.value for message in at.error)
    assert not missing.exists()

    (rendered_package.parent / "reference/frozen/left.png").unlink()
    at = _app(rendered_package, monkeypatch)
    assert not at.exception
    assert any("找不到" in message.value for message in at.error)
    assert _summary(at).iloc[0]["條件"] == "A_r100"


def test_undefined_failed_eye_remains_visible_and_read_only(
    rendered_package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_eye(rows: list[dict]) -> None:
        eye = next(
            row
            for row in rows
            if row["condition_id"] == "A_r050"
            and row["image_id"] == 1
            and row["eye_side"] == "left"
        )
        eye.update(
            iou=None,
            dice=None,
            predicted_mask_path=None,
            prediction_status="failed",
            predicted_cx_px=None,
            predicted_cy_px=None,
            centroid_error_px=None,
            failure_reason="SYNTHETIC SAVED FAILURE",
        )

    _change_table(rendered_package, "ablation", "eye_metrics.csv", failed_eye)
    before = {p: p.read_bytes() for p in rendered_package.parent.rglob("*") if p.is_file()}
    at = _app(rendered_package, monkeypatch)
    assert not at.exception
    assert not at.error
    assert any("未定義" in caption.value and "failed" in caption.value for caption in at.caption)
    eye_table = next(table.value for table in at.dataframe if "執行失敗原因" in table.value.columns)
    assert "SYNTHETIC SAVED FAILURE" in set(eye_table["執行失敗原因"])
    assert {p: p.read_bytes() for p in rendered_package.parent.rglob("*") if p.is_file()} == before
