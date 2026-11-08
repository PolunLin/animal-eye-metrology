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
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert not at.error
    assert at.radio(key="workspace").options == ["Eye Mask Annotation"]
    assert at.radio(key="workspace").value == "Eye Mask Annotation"
    assert {metric.label: metric.value for metric in at.metric} == {
        "試標圖片": "2",
        "眼睛槽位": "8",
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
    assert at.selectbox(key="eye_gt_annotation_20").value == 2001
    assert any("已載入版本 0" in caption.value for caption in at.caption)
    at.selectbox(key="eye_gt_image").select(10).run()
    assert at.selectbox(key="eye_gt_annotation_10").value == 1001
    assert any("已載入版本 1" in caption.value for caption in at.caption)
    reopened = AppTest.from_file(str(APP), default_timeout=30).run()
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
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("無法開啟獨立重標工作區" in message.value for message in at.error)
    assert not session.directory.exists()
    assert not at.radio
    assert _files(old) == old_before

    prepare(session)
    (session.images / "000000000010.png").write_bytes(b"corrupt prepared image")
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("雜湊不符" in message.value for message in at.error)
    assert not at.radio
    assert _files(old) == old_before
