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
from animal_eye_metrology.ui.annotation_view import photo_progress_labels  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


def test_photo_progress_uses_latest_revision_and_ignores_empty_drafts() -> None:
    identities = [(10, 1001, "left"), (10, 1001, "right"), (20, 2001, "left")]
    accepted = {
        "image_id": "10",
        "annotation_id": "1001",
        "eye_side": "left",
        "annotation_version": "1",
        "mask_path": "synthetic.png",
        "visibility": "Visible",
        "mask_review_status": "Accepted",
    }
    rows = [
        accepted | {"annotation_version": "2", "mask_review_status": "Pending"},
        accepted,  # Older acceptance must not replace the latest draft.
        accepted
        | {
            "eye_side": "right",
            "mask_path": "",
            "visibility": "",
            "mask_review_status": "Pending",
            "reviewer": "Synthetic Reviewer",
        },
    ]
    assert photo_progress_labels(identities, rows) == {
        10: "部分已標註 1/2 · 待審核 1",
        20: "未標註 0/1",
    }


@pytest.mark.parametrize(
    ("review", "visibility", "expected"),
    [
        ("Accepted", "Visible", "已標註 2/2 · 審核通過"),
        ("Pending", "Visible", "已標註 2/2 · 待審核 1"),
        ("Uncertain", "Uncertain", "已標註 2/2 · 待審核 1"),
        ("Rejected", "Visible", "已標註 2/2 · 需修正 1"),
        ("Rejected", "NotVisible", "已標註 2/2 · 已審閱 · 已排除 1"),
    ],
)
def test_photo_progress_separates_review_and_exclusion(
    review: str, visibility: str, expected: str
) -> None:
    left = {
        "image_id": "10",
        "annotation_id": "1001",
        "eye_side": "left",
        "annotation_version": "1",
        "mask_path": "synthetic.png",
        "visibility": "Visible",
        "mask_review_status": "Accepted",
    }
    right = left | {
        "eye_side": "right",
        "mask_review_status": review,
        "visibility": visibility,
        "mask_path": "" if visibility == "NotVisible" else "synthetic-right.png",
    }
    assert photo_progress_labels([(10, 1001, "left"), (10, 1001, "right")], [left, right]) == {
        10: expected
    }


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


def test_photo_dropdown_progress_updates_after_save_without_changing_selection(
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
    at.selectbox(key="eye_gt_image").select(20).run()
    next(b for b in at.button if b.label == "儲存草稿").click().run()
    assert at.selectbox(key="eye_gt_image").value == 20
    assert "未標註 0/4" in at.selectbox(key="eye_gt_image").options[1]
    next(r for r in at.radio if r.label == "眼部可見性").set_value("NotVisible")
    next(b for b in at.button if b.label == "儲存草稿").click().run()
    assert not at.exception and not at.error
    selector = at.selectbox(key="eye_gt_image")
    assert selector.value == 20
    assert "#10（未標註 0/4）" in selector.options[0]
    assert "#20（部分已標註 1/4；待審核）" in selector.options[1]
    assert any("此照片：部分已標註 1/4 · 待審核 1" in c.value for c in at.caption)
    at.selectbox(key="eye_gt_image").select(10).run()
    at.selectbox(key="eye_gt_image").select(20).run()
    assert "部分已標註 1/4" in at.selectbox(key="eye_gt_image").options[1]


def test_nonfirst_photo_stays_selected_when_switching_animals_and_eyes(
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
    at.selectbox(key="eye_gt_image").select(20).run()
    for annotation_id, side in ((2002, "right"), (2001, "left"), (2002, "left")):
        at.radio(key="eye_gt_annotation_20").set_value(annotation_id).run()
        assert at.selectbox(key="eye_gt_image").value == 20
        at.radio(key="eye_gt_side").set_value(side).run()
        assert not at.exception and not at.error
        assert at.selectbox(key="eye_gt_image").value == 20
        assert at.radio(key="eye_gt_annotation_20").value == annotation_id
        assert at.radio(key="eye_gt_side").value == side
    assert status(two_image_session)["saved_eyes"] == 0


@pytest.mark.parametrize("trigger", ["animal", "eye", "canvas_rerun"])
def test_stale_photo_progress_wire_label_does_not_reset_photo(
    two_image_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch, trigger: str
) -> None:
    """A browser may resend its prior display label after another save changes progress."""
    from animal_eye_metrology.annotation.eye_annotations import (
        empty_geometry,
        load_targets,
        save_annotation,
    )

    session = two_image_session
    prepare(session)
    _environment(session, monkeypatch)
    at = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    at.selectbox(key="eye_gt_image").select(20).run()
    stale_label = at.selectbox(key="eye_gt_image").options[1]

    dataset = load_coco(session.source.parent / "annotations.json")
    targets = load_targets(session.proposal, dataset)
    save_annotation(
        session.ground_truth,
        targets,
        (20, 2001, "left"),
        empty_geometry(100, 80, dataset.annotations[2001].bbox),
        {
            "visibility": "NotVisible",
            "side_confirmed": "Unknown",
            "measurement_quality": "Unknown",
            "mask_review_status": "Pending",
            "reviewer": "Synthetic concurrent reviewer",
            "exclusion_reason": "Synthetic external save",
        },
        expected_version=0,
    )
    at.radio(key="eye_gt_annotation_20").set_value(2002).run()
    assert at.selectbox(key="eye_gt_image").value == 20
    assert at.selectbox(key="eye_gt_image").options[1] != stale_label

    # AppTest normally rewrites wire values using the newest formatter, hiding the
    # real-browser case where a neighboring control resends the previous label.
    if trigger == "animal":
        at.radio(key="eye_gt_annotation_20").set_value(2001)
    elif trigger == "eye":
        at.radio(key="eye_gt_side").set_value("right")
    states = at._tree.get_widget_states()
    photo_widget_id = at.selectbox(key="eye_gt_image").id
    photo_state = next(state for state in states.widgets if state.id == photo_widget_id)
    photo_state.string_value = stale_label
    at._run(states)

    assert not at.exception and not at.error
    assert at.selectbox(key="eye_gt_image").value == 20
    assert at.radio(key="eye_gt_annotation_20").value == (2001 if trigger == "animal" else 2002)
    assert at.radio(key="eye_gt_side").value == ("right" if trigger == "eye" else "left")
    assert "部分已標註 1/4" in at.selectbox(key="eye_gt_image").options[1]
    assert status(session)["saved_eyes"] == 1


def test_photo_selection_survives_annotation_panel_widget_cleanup(
    two_image_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main app conditionally renders tabs, so hidden controls may lose widget state."""
    prepare(two_image_session)
    _environment(two_image_session, monkeypatch)
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "if st.checkbox('Show annotation panel', value=True, key='show_panel'):\n"
        "    annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    at.selectbox(key="eye_gt_image").select(20).run()
    at.checkbox(key="show_panel").uncheck().run()
    assert not at.selectbox
    at.checkbox(key="show_panel").check().run()
    assert not at.exception and not at.error
    assert at.selectbox(key="eye_gt_image").value == 20
    assert status(two_image_session)["saved_eyes"] == 0


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


def _seed_reviewed_eye(session: AnnotationSession, key: tuple[int, int, str]) -> dict:
    """Seed a small accepted synthetic mask so presets must preserve real content."""
    from animal_eye_metrology.annotation.eye_annotations import (
        empty_geometry,
        load_targets,
        save_annotation,
    )

    dataset = load_coco(session.source.parent / "annotations.json")
    geometry = empty_geometry(100, 80, dataset.annotations[key[1]].bbox)
    geometry["contours"] = [{"operation": "add", "points": [[3, 3], [8, 3], [8, 7], [3, 7]]}]
    save_annotation(
        session.ground_truth,
        load_targets(session.proposal, dataset),
        key,
        geometry,
        {
            "visibility": "Visible",
            "side_confirmed": "Yes",
            "measurement_quality": "Pass",
            "mask_review_status": "Accepted",
            "reviewer": "Synthetic saved reviewer",
            "exclusion_reason": "Synthetic original review",
        },
        expected_version=0,
    )
    return geometry


def _assert_blur_form(at: AppTest) -> None:
    assert not at.exception and not at.error
    for label, expected in (
        ("眼部可見性", "Uncertain"),
        ("左右眼判斷", "Unknown"),
        ("量測品質", "Fail"),
        ("輪廓審核結果", "Rejected"),
    ):
        assert next(w for w in at.radio if w.label == label).value == expected
    assert next(w for w in at.selectbox if w.label == "原因／備註").value == "影像模糊"
    assert not next(w for w in at.checkbox if w.label == "我已確認眼側及輪廓符合規範").value


def test_blur_current_eye_stages_only_and_preserves_geometry_reviewer_and_version(
    two_image_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = two_image_session
    prepare(session)
    geometry = _seed_reviewed_eye(session, (10, 1001, "left"))
    _environment(session, monkeypatch)
    at = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    next(w for w in at.selectbox if w.label == "標註／審核者").select("__custom__").run()
    next(w for w in at.text_input if w.label == "其他審核者姓名").input(
        "Synthetic unsaved reviewer"
    ).run()
    next(w for w in at.checkbox if w.label == "我已確認眼側及輪廓符合規範").check().run()
    before = _files(session.directory)
    prefix = next(iter(at.session_state["eye_gt_drafts"]))

    next(w for w in at.button if w.label == "此眼影像模糊：自動填寫").click().run()
    _assert_blur_form(at)
    assert _files(session.directory) == before
    draft = at.session_state["eye_gt_drafts"][prefix]
    assert draft["geometry"] == geometry and draft["version"] == 1
    assert draft["quick_fill_pending"] is True
    reviewer = next(w for w in at.selectbox if w.label == "標註／審核者")
    assert reviewer.value == "Synthetic unsaved reviewer" or (
        reviewer.value == "__custom__"
        and next(w for w in at.text_input if w.label == "其他審核者姓名").value
        == "Synthetic unsaved reviewer"
    )

    at.radio(key="eye_gt_side").set_value("right").run()
    assert next(w for w in at.radio if w.label == "眼部可見性").value is None
    at.radio(key="eye_gt_side").set_value("left").run()
    _assert_blur_form(at)
    next(w for w in at.button if w.label == "儲存審核結果").click().run()
    assert not at.exception and not at.error
    with (session.ground_truth / "eye_mask_annotations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[-1]["annotation_version"] == "2"
    assert rows[-1]["reviewer"] == "Synthetic unsaved reviewer"
    assert rows[-1]["mask_review_status"] == "Rejected"
    assert rows[-1]["visibility"] == "Uncertain"
    assert rows[-1]["measurement_quality"] == "Fail"
    assert rows[-1]["exclusion_reason"] == "影像模糊"
    assert rows[-1]["mask_path"]
    assert not at.session_state["eye_gt_drafts"][prefix]["quick_fill_pending"]


def test_blur_whole_photo_persists_across_eyes_without_touching_another_photo(
    two_image_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = two_image_session
    prepare(session)
    geometry = _seed_reviewed_eye(session, (20, 2001, "left"))
    _environment(session, monkeypatch)
    at = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    at.selectbox(key="eye_gt_image").select(20).run()
    before = _files(session.directory)
    next(w for w in at.button if w.label == "整張照片太模糊：全部眼睛").click().run()
    for annotation_id in (2001, 2002):
        at.radio(key="eye_gt_annotation_20").set_value(annotation_id).run()
        for side in ("left", "right"):
            at.radio(key="eye_gt_side").set_value(side).run()
            _assert_blur_form(at)
            assert at.selectbox(key="eye_gt_image").value == 20
    assert _files(session.directory) == before
    drafts = at.session_state["eye_gt_drafts"]
    assert sum(bool(d.get("quick_fill_pending")) for d in drafts.values()) == 4
    accepted_draft = next(d for d in drafts.values() if d["version"] == 1)
    assert accepted_draft["geometry"] == geometry
    assert accepted_draft["record"]["reviewer"] == "Synthetic saved reviewer"

    at.selectbox(key="eye_gt_image").select(10).run()
    at.radio(key="eye_gt_side").set_value("left").run()
    assert next(w for w in at.radio if w.label == "眼部可見性").value is None
    assert next(w for w in at.radio if w.label == "輪廓審核結果").value is None
    at.selectbox(key="eye_gt_image").select(20).run()
    at.radio(key="eye_gt_annotation_20").set_value(2001).run()
    _assert_blur_form(at)
    next(w for w in at.button if w.label == "重新載入此眼睛最新版本").click().run()
    assert next(w for w in at.radio if w.label == "眼部可見性").value == "Visible"
    assert (
        sum(bool(d.get("quick_fill_pending")) for d in at.session_state["eye_gt_drafts"].values())
        == 3
    )
    assert _files(session.directory) == before


def test_blur_preset_does_not_bypass_concurrent_revision_check(
    two_image_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from animal_eye_metrology.annotation.eye_annotations import load_targets, save_annotation

    session = two_image_session
    prepare(session)
    geometry = _seed_reviewed_eye(session, (10, 1001, "left"))
    _environment(session, monkeypatch)
    at = AppTest.from_string(
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        "from animal_eye_metrology.paths import PROJECT_ROOT\n"
        "annotation_workspace(PROJECT_ROOT)",
        default_timeout=30,
    ).run()
    dataset = load_coco(session.source.parent / "annotations.json")
    save_annotation(
        session.ground_truth,
        load_targets(session.proposal, dataset),
        (10, 1001, "left"),
        geometry,
        {
            "visibility": "Visible",
            "side_confirmed": "Yes",
            "measurement_quality": "Pass",
            "mask_review_status": "Accepted",
            "reviewer": "Synthetic concurrent reviewer",
            "exclusion_reason": "Synthetic newer review",
        },
        expected_version=1,
    )
    after_external_save = _files(session.directory)
    next(w for w in at.button if w.label == "此眼影像模糊：自動填寫").click().run()
    _assert_blur_form(at)
    next(w for w in at.button if w.label == "儲存審核結果").click().run()
    assert not at.exception
    assert any("另一個視窗更新" in message.value for message in at.error)
    assert _files(session.directory) == after_external_save
