"""Invented prompts only: original coordinates, persistence and stale-window protection."""

import math
from pathlib import Path

import pytest

from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.eye_prompts import (
    crop_for_prompt,
    latest_prompt,
    load_prompts,
    save_prompt,
    validate_prompt,
)


def target() -> dict[str, str]:
    return {
        "image_id": "10",
        "annotation_id": "1001",
        "image_width": "100",
        "image_height": "80",
        "image_sha256": "a" * 64,
    }


def test_manual_prompt_roundtrip_separate_sides_and_conflicting_writer(tmp_path: Path) -> None:
    targets = [target()]
    geometry = {"point": [20.5, 20.5], "box": [18, 18, 24, 23]}
    assert load_prompts(tmp_path, targets) == []
    first = save_prompt(tmp_path, targets, (10, 1001, "left"), geometry, "SYNTHETIC QA", 0)
    save_prompt(tmp_path, targets, (10, 1001, "right"), geometry, "SYNTHETIC QA", 0)
    with pytest.raises(Phase1Error, match="newer prompt"):
        save_prompt(tmp_path, targets, (10, 1001, "left"), geometry, "SYNTHETIC QA", 0)
    second = save_prompt(tmp_path, targets, (10, 1001, "left"), geometry, "SYNTHETIC QA", 1)
    rows = load_prompts(tmp_path, targets)
    assert len(rows) == 3 and first in rows
    assert latest_prompt(rows, (10, 1001, "left")) == second
    assert latest_prompt(rows, (10, 1001, "right"))["prompt_version"] == "1"
    assert second["prompt_source"] == "manual_original_image"
    assert not list(tmp_path.rglob("*.png"))


def test_prompt_rejects_invalid_source_bounds_and_reviewer(tmp_path: Path) -> None:
    saved = save_prompt(
        tmp_path,
        [target()],
        (10, 1001, "left"),
        {"point": [20, 20], "box": [18, 18, 24, 23]},
        "SYNTHETIC QA",
        0,
    )
    for updates in (
        {"point_x": math.nan},
        {"point_y": 90},
        {"box_x0": -1},
        {"point_x": 18},
        {"reviewer": ""},
        {"prompt_source": "GT_centroid"},
        {"image_sha256": "b" * 64},
    ):
        with pytest.raises(Phase1Error):
            validate_prompt(saved | updates, target())
    assert len(load_prompts(tmp_path, [target()])) == 1


def test_crop_uses_box_only_and_clips_at_image_edges() -> None:
    row = {"box_x0": 18, "box_y0": 18, "box_x1": 24, "box_y1": 23}
    assert crop_for_prompt(row, 100, 80) == (5, 4, 37, 37)
    assert crop_for_prompt(row | {"point_x": 999, "point_y": 999}, 100, 80) == (5, 4, 37, 37)
    edge = {"box_x0": 0, "box_y0": 0, "box_x1": 3, "box_y1": 3}
    assert crop_for_prompt(edge, 100, 80) == (0, 0, 18, 18)


def test_native_logits_restore_to_known_original_grid_and_detect_corruption(tmp_path: Path) -> None:
    import json

    import numpy as np
    from PIL import Image

    from animal_eye_metrology.eye_poc import verify_run
    from animal_eye_metrology.verification import sha256_file

    Image.new("RGB", (8, 6), "gray").save(tmp_path / "input.png")
    (tmp_path / "request.json").write_text(json.dumps({"roi_xyxy": [2, 1, 5, 3]}))
    model_dir = tmp_path / "SYNTHETIC_QA_ONLY"
    model_dir.mkdir()
    logits = np.array([[1, -1, 1], [0, 0.1, -0.1]])
    np.save(model_dir / "roi_logits.npy", logits, allow_pickle=False)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1, 2] = mask[1, 4] = mask[2, 3] = 255
    Image.fromarray(mask).save(model_dir / "mask.png")
    manifest = {
        "results": [{"status": "passed", "model": "SYNTHETIC_QA_ONLY"}],
        "artifacts_sha256": {
            str(p.relative_to(tmp_path)): sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()
        },
    }
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify_run(tmp_path)["model_artifacts_validated"] == 1
    mask[0, 0] = 255
    Image.fromarray(mask).save(model_dir / "mask.png")
    with pytest.raises(Phase1Error, match="digest differs"):
        verify_run(tmp_path)
    manifest["artifacts_sha256"]["SYNTHETIC_QA_ONLY/mask.png"] = sha256_file(model_dir / "mask.png")
    path.write_text(json.dumps(manifest))
    with pytest.raises(Phase1Error, match="original-grid"):
        verify_run(tmp_path)


def test_prompt_workspace_saves_and_reopens_without_eye_masks(
    local_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import csv
    import json

    from PIL import Image
    from streamlit.testing.v1 import AppTest

    from animal_eye_metrology.coco import load_coco
    from animal_eye_metrology.verification import sha256_file

    dataset = load_coco(local_config.parent / "annotations.json")
    image, ann = dataset.images[10], dataset.annotations[1001]
    image_dir = local_config.parent / "images"
    image_dir.mkdir()
    image_path = image_dir / image.file_name
    Image.new("RGB", (image.width, image.height), "gray").save(image_path)
    row = {
        "image_id": "10",
        "annotation_id": "1001",
        "file_name": image.file_name,
        "image_width": str(image.width),
        "image_height": str(image.height),
        "animal_category": dataset.categories[ann.category_id].name,
        "coco_bbox_xywh": json.dumps(list(ann.bbox)),
        "proposed_eye_sides": '["left", "right"]',
        "image_sha256": sha256_file(image_path),
    }
    proposal = local_config.parent / "prompt_targets.csv"
    with proposal.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    prompt_dir = local_config.parent / "independent_prompts"
    monkeypatch.setenv("AEM_CONFIG", str(local_config))
    monkeypatch.setenv("AEM_WORKSPACE", "Eye Segmentation Prompts")
    monkeypatch.setattr(
        "animal_eye_metrology.eye_poc.settings",
        lambda _: {
            "proposal_csv": proposal,
            "prompts_directory": prompt_dir,
        },
    )
    app = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception and not at.error
    assert not prompt_dir.exists()
    prefix = next(iter(at.session_state["eye_prompt_drafts"]))
    at.session_state[prefix + "_canvas"] = {
        "geometry": {
            "view": [0, 0, image.width, image.height],
            "point": [20.5, 20.5],
            "box": [18, 18, 24, 23],
        }
    }
    next(w for w in at.text_input if w.label == "提示 Reviewer").input("SYNTHETIC QA ONLY")
    next(w for w in at.checkbox if "我只依原圖" in w.label).check()
    next(w for w in at.button if w.label == "保存固定提示").click().run()
    assert not at.exception and not at.error and at.success
    assert len(load_prompts(prompt_dir, [row])) == 1
    reopened = AppTest.from_file(str(app), default_timeout=60).run()
    assert not reopened.exception and not reopened.error
    draft = next(iter(reopened.session_state["eye_prompt_drafts"].values()))
    assert draft["version"] == 1 and draft["geometry"]["point"] == [20.5, 20.5]
    assert not list(prompt_dir.rglob("*.png"))
