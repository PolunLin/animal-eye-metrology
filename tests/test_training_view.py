"""Published evidence and offline navigation do not require research assets."""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
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


def test_training_view_navigates_without_private_outputs(tmp_path: Path) -> None:
    shutil.copytree(PROJECT_ROOT / "examples/training", tmp_path / "examples/training")
    (tmp_path / "docs").mkdir()
    shutil.copyfile(PROJECT_ROOT / "docs/training.md", tmp_path / "docs/training.md")
    script = (
        "from pathlib import Path\n"
        "from animal_eye_metrology.ui.training_view import training_workspace\n"
        f"training_workspace(Path({str(tmp_path)!r}))\n"
    )
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.header[0].value == "訓練與實驗"
    assert len(app.dataframe[0].value) == 3
    app.selectbox[0].set_value("yolo11n_seg").run()
    assert not app.exception
    app.radio[0].set_value("第一輪微調").run()
    assert len(app.dataframe[0].value) == 4
    app.radio[0].set_value("學習率診斷").run()
    assert len(app.dataframe[0].value) == 3
    assert not app.exception
    assert not (tmp_path / "outputs").exists()
