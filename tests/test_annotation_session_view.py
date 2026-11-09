"""Dedicated v2 UI isolates human revisions from old GT and model-result workspaces."""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pycocotools")
Image = pytest.importorskip("PIL.Image")

from streamlit.testing.v1 import AppTest  # noqa: E402
from test_annotation_session import (  # noqa: E402
    _files,
    annotation_session,  # noqa: F401 - shared synthetic fixture
)

from animal_eye_metrology.annotation.annotation_session import (  # noqa: E402
    AnnotationSession,
    prepare,
    read_session,
    status,
)
from animal_eye_metrology.annotation.eye_annotations import V2_PROTOCOL_VERSION  # noqa: E402
from animal_eye_metrology.core.verification import sha256_file  # noqa: E402
from animal_eye_metrology.data.coco import load_coco  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


@pytest.fixture
def two_image_session(annotation_session: AnnotationSession) -> AnnotationSession:  # noqa: F811
    session = annotation_session
    dataset = load_coco(session.source.parent / "annotations.json")
    image_path = session.source.parent / "images/000000000020.png"
    Image.new("RGB", (100, 80), "navy").save(image_path)
    with session.source_proposal.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for annotation_id in (2001, 2002):
        annotation = dataset.annotations[annotation_id]
        rows.append(
            rows[0]
            | {
                "image_id": "20",
                "annotation_id": str(annotation_id),
                "animal_category": dataset.categories[annotation.category_id].name,
                "file_name": image_path.name,
                "coco_bbox_xywh": json.dumps(list(annotation.bbox)),
                "image_sha256": sha256_file(image_path),
            }
        )
    with session.source_proposal.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    session.source.write_text(
        session.source.read_text()
        .replace("expected_images = 1", "expected_images = 2")
        .replace("expected_instances = 2", "expected_instances = 4")
        .replace("expected_eyes = 4", "expected_eyes = 8")
    )
    return read_session(session.source)


def _environment(session: AnnotationSession, monkeypatch: pytest.MonkeyPatch) -> Path:
    old = session.source.parent / "phase3/v1_trap"
    old.mkdir(parents=True)
    (old / "eye_mask_annotations.csv").write_text("SYNTHETIC OLD DATA MUST NOT BE READ OR WRITTEN")
    monkeypatch.setenv("AEM_ANNOTATION_SESSION", str(session.source))
    monkeypatch.setenv("AEM_EYE_OUTPUT", str(old))
    monkeypatch.setenv("AEM_EYE_PROPOSAL", str(old / "nonexistent-proposal.csv"))
    monkeypatch.setenv("AEM_CONFIG", str(old / "nonexistent-phase1.toml"))
    monkeypatch.setenv("AEM_WORKSPACE", "Results Explorer")
    monkeypatch.setenv("AEM_RESULTS_CONFIG", str(old / "nonexistent-results.toml"))
    return old


def test_v2_mode_isolated_empty_drafts_and_image_switching(
    two_image_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = two_image_session
    prepare(session)
    old = _environment(session, monkeypatch)
    old_before = _files(old)
    immutable_before = {
        key: value
        for key, value in _files(session.directory).items()
        if not key.startswith("ground_truth/")
    }
    # The UI must use prepared originals, even if the original file later becomes unavailable.
    (session.source.parent / "images/000000000010.png").write_bytes(b"changed original image")
    at = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    assert not at.exception
    assert not at.error
    assert {metric.label: metric.value for metric in at.metric} == {
        "照片": "2",
        "眼睛標註位置（左右各一）": "8",
        "已保存": "0",
        "Mask 已接受": "0",
    }
    assert any("排除眼周毛髮" in message.value for message in at.info)
    assert any(V2_PROTOCOL_VERSION in caption.value for caption in at.caption)
    assert not (session.ground_truth / "masks").exists()
    assert status(session)["saved_eyes"] == 0

    next(button for button in at.button if button.label == "儲存草稿").click().run()
    assert not at.exception
    assert not at.error
    report = status(session)
    assert report["saved_eyes"] == 1 and report["eligible_eyes"] == 0
    with (session.ground_truth / "eye_mask_annotations.csv").open(newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 1 and saved[0]["protocol_version"] == V2_PROTOCOL_VERSION
    assert saved[0]["mask_path"] == saved[0]["reviewer"] == ""
    assert saved[0]["mask_review_status"] == "Pending"

    at.selectbox(key="eye_gt_image").select(20).run()
    assert not at.exception
    assert not at.error
    assert at.radio(key="eye_gt_annotation_20").value == 2001
    assert any("已載入版本 0" in caption.value for caption in at.caption)
    at.selectbox(key="eye_gt_image").select(10).run()
    assert at.radio(key="eye_gt_annotation_10").value == 1001
    assert any("已載入版本 1" in caption.value for caption in at.caption)
    reopened = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    assert not reopened.exception
    assert any("已載入版本 1" in caption.value for caption in reopened.caption)
    assert _files(old) == old_before
    assert {
        key: value
        for key, value in _files(session.directory).items()
        if not key.startswith("ground_truth/")
    } == immutable_before
    assert not (session.ground_truth / "masks").exists()


def test_unprepared_or_corrupted_session_never_falls_back_to_v1(
    annotation_session: AnnotationSession,  # noqa: F811 - shared synthetic fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = annotation_session
    old = _environment(session, monkeypatch)
    old_before = _files(old)
    at = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    assert not at.exception
    assert any("無法開啟獨立重標工作區" in message.value for message in at.error)
    assert not session.directory.exists()
    assert not at.radio
    assert _files(old) == old_before

    prepare(session)
    (session.images / "000000000010.png").write_bytes(b"corrupt prepared image")
    at = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    assert not at.exception
    assert any("雜湊不符" in message.value for message in at.error)
    assert not at.radio
    assert _files(old) == old_before


def test_radio_labels_keep_raw_review_values_and_eye_identity(
    two_image_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = two_image_session
    prepare(session)
    _environment(session, monkeypatch)
    app = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    assert app.radio(key="eye_gt_annotation_10").options == [
        "動物 A · test_species_0",
        "動物 B · test_species_0",
    ]
    assert app.radio(key="eye_gt_side").options == ["牠的左眼", "牠的右眼"]
    visible = next(r for r in app.radio if r.label == "眼部可見性")
    assert visible.options == ["完整可見", "部分可見", "看不到", "無法判斷"]
    assert visible.value is None
    assert next(r for r in app.radio if r.label == "輪廓審核結果").value is None
    app.radio(key="eye_gt_annotation_10").set_value(1002).run()
    app.radio(key="eye_gt_side").set_value("right").run()
    assert any("動物 B" in m.value and "牠的右眼" in m.value for m in app.markdown)
    next(r for r in app.radio if r.label == "眼部可見性").set_value("Uncertain")
    next(b for b in app.button if b.label == "儲存草稿").click().run()
    assert not app.exception and not app.error
    with (session.ground_truth / "eye_mask_annotations.csv").open(newline="") as handle:
        row = list(csv.DictReader(handle))[0]
    assert (
        row["annotation_id"],
        row["eye_side"],
        row["visibility"],
        row["mask_review_status"],
    ) == ("1002", "right", "Uncertain", "Pending")


def test_review_dropdown_custom_values_round_trip(
    annotation_session: AnnotationSession,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = annotation_session
    prepare(session)
    _environment(session, monkeypatch)
    app = (
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)"
    )
    at = AppTest.from_string(app, default_timeout=30).run()
    assert not at.text_input
    next(w for w in at.selectbox if w.label == "標註／審核者").select("__custom__").run()
    next(w for w in at.text_input if w.label == "其他審核者姓名").input("Synthetic Reviewer").run()
    next(w for w in at.selectbox if w.label == "原因／備註").select("__custom__").run()
    next(w for w in at.text_input if w.label == "其他原因／備註").input(
        "Synthetic custom reason"
    ).run()
    next(w for w in at.button if w.label == "儲存草稿").click().run()
    assert not at.exception
    assert not at.error
    reopened = AppTest.from_string(app, default_timeout=30).run()
    reviewer = next(w for w in reopened.selectbox if w.label == "標註／審核者")
    assert reviewer.value == "Synthetic Reviewer"
    assert (
        next(w for w in reopened.text_input if w.label == "其他原因／備註").value
        == "Synthetic custom reason"
    )
    next(w for w in reopened.selectbox if w.label == "原因／備註").select("影像模糊").run()
    assert not reopened.text_input
    next(w for w in reopened.button if w.label == "儲存草稿").click().run()
    assert not reopened.exception
    with (session.ground_truth / "eye_mask_annotations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["exclusion_reason"] == "影像模糊"
    assert rows[-1]["reviewer"] == "Synthetic Reviewer"


def test_reviewer_remembered_across_photos_and_eyes(
    two_image_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(two_image_session)
    _environment(two_image_session, monkeypatch)
    at = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()

    def reviewer():
        return next(w for w in at.selectbox if w.label == "標註／審核者")

    reviewer().select("__custom__").run()
    next(w for w in at.text_input if w.label == "其他審核者姓名").input("Alice").run()
    at.radio(key="eye_gt_side").set_value("right").run()
    assert reviewer().value == "Alice"
    next(w for w in at.button if w.label == "儲存草稿").click().run()
    at.selectbox(key="eye_gt_image").select(20).run()
    assert reviewer().value == "Alice"
    reviewer().select("__custom__").run()
    next(w for w in at.text_input if w.label == "其他審核者姓名").input("Bob").run()
    at.selectbox(key="eye_gt_image").select(10).run()
    assert reviewer().value == "Alice"  # Existing saved attribution is preserved.
    assert at.session_state["eye_gt_last_reviewer"] == "Bob"
    at.radio(key="eye_gt_annotation_10").set_value(1002).run()
    assert reviewer().value == "Bob"
    assert not at.exception and not at.error
