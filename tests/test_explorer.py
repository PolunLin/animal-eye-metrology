"""Native Streamlit persistence and filtering checks, using synthetic data only."""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("pycocotools.mask")

from streamlit.testing.v1 import AppTest  # noqa: E402

from animal_eye_metrology.analysis import analyze  # noqa: E402
from animal_eye_metrology.cli import main  # noqa: E402
from animal_eye_metrology.coco import load_coco  # noqa: E402
from animal_eye_metrology.review_store import (  # noqa: E402
    initialize_reviews,
    load_reviews,
    review_key,
    save_review,
)


def test_explorer_save_switch_instance_and_reopen(
    local_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = local_config.parent / "annotations.json"
    before = source.read_bytes()
    assert main(["analyze", "--config", str(local_config)]) == 0
    image_dir = local_config.parent / "images"
    image_dir.mkdir()
    for image_id in (10, 20, 50):
        Image.new("RGB", (100, 80), "gray").save(image_dir / f"{image_id:012d}.png")
    monkeypatch.setenv("AEM_CONFIG", str(local_config))
    app = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    assert at.radio(key="workspace").value == "Image Selection"
    assert not any(widget.key == "annotation_id" for widget in at.selectbox)
    at.checkbox(key="auto_advance_selection").uncheck().run()
    at.text_area[0].input("SYNTHETIC IMAGE SELECTION")
    next(widget for widget in at.button if widget.label == "Selected · 保留").click().run()
    assert not at.exception
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert at.text_area[0].value == "SYNTHETIC IMAGE SELECTION"
    at.radio(key="workspace").set_value("Eye Review").run()
    assert not at.exception
    assert at.selectbox(key="image_id").options == ["Image 10"]
    assert at.selectbox(key="annotation_id").value == 1001
    choices = {
        "Left Eye Visible": "Yes",
        "Right Eye Visible": "Uncertain",
        "Head Pose": "Front",
        "Occlusion": "Partial",
        "Eligibility": "Uncertain",
    }
    for widget in at.selectbox:
        if widget.label in choices:
            widget.select(choices[widget.label])
    at.text_area[0].input("SYNTHETIC UI TEST ONLY")
    next(widget for widget in at.button if widget.label == "儲存審核").click().run()
    assert not at.exception
    assert at.success
    at.selectbox(key="annotation_id").select(1002).run()
    assert not at.exception
    assert (
        next(widget for widget in at.selectbox if widget.label == "Left Eye Visible").value is None
    )
    reopened = AppTest.from_file(str(app), default_timeout=60).run()
    assert not reopened.exception
    reopened.radio(key="workspace").set_value("Eye Review").run()
    assert not reopened.exception
    assert (
        next(widget for widget in reopened.selectbox if widget.label == "Left Eye Visible").value
        == "Yes"
    )
    assert reopened.text_area[0].value == "SYNTHETIC UI TEST ONLY"
    loaded = load_reviews(
        local_config.parent / "result/eye_visibility_review_working.csv",
        analyze(load_coco(source)).review,
    )
    assert loaded[0]["notes"] == "SYNTHETIC UI TEST ONLY"
    assert loaded[1]["left_eye_visibility"] == ""
    assert source.read_bytes() == before


def test_progressive_filtering_preserves_previous_reviews_on_reject_and_reselect(
    local_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = local_config.parent / "annotations.json"
    assert main(["analyze", "--config", str(local_config)]) == 0
    image_dir = local_config.parent / "images"
    image_dir.mkdir()
    for image_id in (10, 20, 50):
        Image.new("RGB", (100, 80), "gray").save(image_dir / f"{image_id:012d}.png")
    template = analyze(load_coco(source)).review
    working = local_config.parent / "result/eye_visibility_review_working.csv"
    old = initialize_reviews(working, template)
    save_review(
        working,
        template,
        review_key(old[0]),
        {
            "left_eye_visibility": "Yes",
            "right_eye_visibility": "Yes",
            "head_pose": "Front",
            "occlusion": "None",
            "measurement_eligibility": "Yes",
            "notes": "EXISTING HUMAN DATA",
        },
        expected=old[0],
    )
    before = working.read_bytes()
    monkeypatch.setenv("AEM_CONFIG", str(local_config))
    app = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    assert working.read_bytes() == before
    at.radio(key="workspace").set_value("Eye Review").run()
    assert not any(widget.key == "annotation_id" for widget in at.selectbox)
    assert any("尚無 Selected" in item.value for item in at.info)
    at.radio(key="workspace").set_value("Image Selection").run()
    next(widget for widget in at.button if widget.label == "Selected · 保留").click().run()
    assert not at.exception
    assert at.selectbox(key="image_id").value == 20  # Save first, then advance.
    at.selectbox(key="selection_filter").select("Selected").run()
    assert at.selectbox(key="image_id").value == 10
    next(widget for widget in at.button if widget.label == "Rejected · 排除").click().run()
    at.radio(key="workspace").set_value("Eye Review").run()
    assert not any(widget.key == "annotation_id" for widget in at.selectbox)
    assert working.read_bytes() == before
    at.radio(key="workspace").set_value("Image Selection").run()
    at.selectbox(key="selection_filter").select("全部").run()
    at.selectbox(key="image_id").select(10).run()
    next(widget for widget in at.button if widget.label == "Selected · 保留").click().run()
    at.radio(key="workspace").set_value("Eye Review").run()
    assert not at.exception
    assert at.selectbox(key="image_id").options == ["Image 10"]
    assert at.text_area[0].value == "EXISTING HUMAN DATA"
    assert any("待確認" in item.value for item in at.info)
    assert working.read_bytes() == before
