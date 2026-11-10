"""Published evidence and offline navigation do not require research assets."""

import csv
import io
import json
import shutil
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from animal_eye_metrology.paths import PROJECT_ROOT  # noqa: E402
from animal_eye_metrology.ui.training_view import load_records  # noqa: E402


def test_published_training_records_detect_tampering(tmp_path: Path) -> None:
    shutil.copytree(PROJECT_ROOT / "examples/training", tmp_path / "examples/training")
    pilot, recovery = load_records(tmp_path)
    assert len(pilot) == 4 and len(recovery) == 3
    path = tmp_path / "examples/training/recovery_comparison.csv"
    path.write_text(path.read_text().replace("0.6209558270610754", "0.99"))
    with pytest.raises(ValueError, match="integrity"):
        load_records(tmp_path)


def training_app(root: Path) -> AppTest:
    """Render only published evidence, with no private outputs or model assets."""
    shutil.copytree(PROJECT_ROOT / "examples/training", root / "examples/training")
    (root / "docs").mkdir()
    shutil.copyfile(PROJECT_ROOT / "docs/training.md", root / "docs/training.md")
    script = (
        "from pathlib import Path\n"
        "from animal_eye_metrology.ui.training_view import training_workspace\n"
        f"training_workspace(Path({str(root)!r}))\n"
    )
    return AppTest.from_string(script).run()


def test_training_view_defaults_to_expanded_results_without_private_outputs(tmp_path: Path) -> None:
    app = training_app(tmp_path)
    assert not app.exception
    assert app.header[0].value == "訓練與實驗"
    assert app.radio[0].value == "最新大型訓練"
    assert [metric.value for metric in app.metric] == ["7,727 張", "950 張", "950 張"]
    comparison = app.dataframe[0].value
    assert comparison.columns.tolist() == ["權重", "Development AP", "Holdout AP"]
    assert comparison["權重"].tolist() == [
        "原始預訓練",
        "微調最佳 epoch 1",
        "微調 − 原始（AP 點）",
    ]
    assert comparison["Development AP"].tolist() == pytest.approx(
        [58.38158830451957, 53.67784187953976, -4.70374642497981]
    )
    assert comparison["Holdout AP"].tolist() == pytest.approx(
        [59.00766623220915, 54.67512349871615, -4.332542733493005]
    )
    # The visible decision must stand on its own, outside the full methodology document.
    decision = "\n".join(
        node.value
        for kind in ("info", "success", "markdown")
        for node in app.get(kind)
        if not node.value.startswith("# Training and Experiments")
    )
    assert "原始權重" in decision
    assert "holdout 推論前" in decision
    assert not (tmp_path / "outputs").exists()


def test_training_view_navigates_all_experiments_without_private_outputs(tmp_path: Path) -> None:
    app = training_app(tmp_path)
    expanded_comparison = app.dataframe[0].value.copy()
    app.radio[0].set_value("第二輪微調").run()
    assert len(app.dataframe[0].value) == 3
    assert [metric.value for metric in app.metric][:2] == ["256 張", "64 張"]
    app.selectbox[0].set_value("yolo11n_seg").run()
    assert not app.exception
    app.radio[0].set_value("第一輪微調").run()
    assert len(app.dataframe[0].value) == 4
    app.radio[0].set_value("學習率診斷").run()
    assert len(app.dataframe[0].value) == 3
    app.radio[0].set_value("最新大型訓練").run()
    assert app.dataframe[0].value.equals(expanded_comparison)
    assert not app.exception
    assert not (tmp_path / "outputs").exists()


def test_training_view_rejects_tampered_expanded_evidence(tmp_path: Path) -> None:
    app = training_app(tmp_path)
    evidence = tmp_path / "examples/training/expanded_yolo26s_job_status.json"
    record = json.loads(evidence.read_text())
    record["result"]["finetuned_holdout_mask_ap"] = 0.99
    evidence.write_text(json.dumps(record))
    app.run()
    assert not app.exception
    assert len(app.error) == 1
    assert "無法讀取已驗證的實驗紀錄" in app.error[0].value
    assert not app.dataframe
    assert not app.metric


def test_expanded_download_matches_displayed_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: dict[str, bytes | str] = {}
    original_download = streamlit.download_button

    def capture_download(label: str, data: bytes | str, **kwargs: object) -> bool:
        downloads[str(kwargs["file_name"])] = data
        return original_download(label, data, **kwargs)

    monkeypatch.setattr(streamlit, "download_button", capture_download)
    app = training_app(tmp_path)
    assert not app.exception
    exported = downloads["expanded_yolo26s_comparison.csv"]
    assert isinstance(exported, bytes)
    records = list(csv.DictReader(io.StringIO(exported.decode("utf-8-sig"))))
    displayed = app.dataframe[0].value.to_dict(orient="records")
    assert len(records) == len(displayed) == 3
    for record, row in zip(records, displayed, strict=True):
        assert record["權重"] == row["權重"]
        assert float(record["Development AP"]) == pytest.approx(row["Development AP"])
        assert float(record["Holdout AP"]) == pytest.approx(row["Holdout AP"])
