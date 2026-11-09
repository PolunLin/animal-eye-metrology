"""Synthetic candidate batches preserve source identity and never create eye GT."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from animal_eye_metrology.annotation.annotation_session import read_session, status
from animal_eye_metrology.annotation.batches import create_batch
from animal_eye_metrology.core.errors import Phase1Error


def source(root: Path, raw: dict, split: str, zipped: bool = False) -> None:
    images = root / ("data/train2017" if split == "train2017" else "data/coco/val2017")
    images.mkdir(parents=True)
    for row in raw["images"]:
        Image.new("RGB", (100, 80), (row["id"], 50, 80)).save(images / row["file_name"])
    if zipped:
        with ZipFile(root / "annotations_trainval2017.zip", "w") as archive:
            archive.writestr(f"annotations/instances_{split}.json", json.dumps(raw))
    else:
        annotations = root / f"data/coco/annotations/instances_{split}.json"
        annotations.parent.mkdir(parents=True)
        annotations.write_text(json.dumps(raw))
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/eye_mask_protocol_v2.md").write_text("SYNTHETIC eye-visible-eyelid-v2")
    # Historical full-dataset digest must not override the explicit new subset digest.
    (root / "outputs/phase1").mkdir(parents=True, exist_ok=True)
    (root / "outputs/phase1/run_manifest.json").write_text(
        json.dumps({"dataset": {"annotations_sha256": "f" * 64}})
    )


@pytest.mark.parametrize("split,zipped", [("val2017", False), ("train2017", True)])
def test_batch_can_use_either_source_and_preserves_all_selected_instances(
    tmp_path: Path, synthetic_coco: dict, split: str, zipped: bool
) -> None:
    source(tmp_path, synthetic_coco, split, zipped)
    config = create_batch(tmp_path, split, 2, 42)
    record = json.loads((config.parent / "batch_manifest.json").read_text())
    assert record["dataset_role"] == "unassigned"
    assert record["eye_suitability"] == "not_reviewed"
    assert len(record["selected_image_ids"]) == 2
    session = read_session(config)
    assert status(session)["saved_eyes"] == 0
    assert not (session.ground_truth / "masks").exists()
    second = create_batch(tmp_path, split, 1, 42)
    second_record = json.loads((second.parent / "batch_manifest.json").read_text())
    assert not set(record["selected_image_ids"]) & set(second_record["selected_image_ids"])
    subset = json.loads((config.parent / "annotations.json").read_text())
    expected = [
        r for r in synthetic_coco["annotations"] if r["image_id"] in record["selected_image_ids"]
    ]
    assert subset["annotations"] == expected


def test_insufficient_candidates_do_not_publish_a_partial_batch(
    tmp_path: Path, synthetic_coco: dict
) -> None:
    source(tmp_path, synthetic_coco, "val2017")
    with pytest.raises(Phase1Error, match="不足"):
        create_batch(tmp_path, "val2017", 100)
    assert not (tmp_path / "outputs/phase3/eye_annotation_batches").exists()


def test_invalid_source_rejected_before_accessing_files(tmp_path: Path) -> None:
    with pytest.raises(Phase1Error):
        create_batch(tmp_path, "../../other", 5)


def test_ui_can_create_and_reopen_a_source_selected_batch(
    tmp_path: Path, synthetic_coco: dict
) -> None:
    import shutil

    from streamlit.testing.v1 import AppTest

    from animal_eye_metrology.paths import PROJECT_ROOT

    source(tmp_path, synthetic_coco, "val2017")
    source(tmp_path, synthetic_coco, "train2017", zipped=True)
    shutil.copytree(PROJECT_ROOT / "app/assets", tmp_path / "app/assets")
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from animal_eye_metrology.ui.annotation_workspace import annotation_workspace\n"
        f"annotation_workspace(Path({str(tmp_path)!r}))\n",
        default_timeout=30,
    ).run()
    assert app.radio[0].options == ["val2017", "train2017"]
    app.radio[0].set_value("train2017")
    app.number_input[0].set_value(1)
    next(b for b in app.button if b.label == "建立待標註批次").click().run()
    assert not app.exception and not app.error
    selected = Path(app.session_state["active_annotation_batch"])
    manifest = json.loads((selected.parent / "batch_manifest.json").read_text())
    assert manifest["source_split"] == "train2017"
    assert status(read_session(selected))["saved_eyes"] == 0
    assert any(m.label == "照片" and m.value == "1" for m in app.metric)
