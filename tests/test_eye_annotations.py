"""Invented masks only: pixel-center geometry, independent eyes and immutable revisions."""

import copy
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
np = pytest.importorskip("numpy")

from animal_eye_metrology.annotation.eye_annotations import (  # noqa: E402
    empty_geometry,
    initialize_store,
    latest_annotation,
    load_annotations,
    load_geometry,
    rasterize,
    save_annotation,
)
from animal_eye_metrology.core.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.core.verification import sha256_file  # noqa: E402


def target() -> dict[str, str]:
    return {
        "image_id": "10",
        "annotation_id": "1001",
        "file_name": "synthetic.png",
        "image_width": "10",
        "image_height": "8",
        "image_sha256": "a" * 64,
    }


def geometry() -> dict:
    value = empty_geometry(10, 8, (0, 0, 10, 8))
    value["contours"] = [{"operation": "add", "points": [[1, 1], [6, 1], [6, 5], [1, 5]]}]
    return value


def review(**updates) -> dict[str, str]:
    return {
        "visibility": "Visible",
        "side_confirmed": "Yes",
        "mask_review_status": "Accepted",
        "measurement_quality": "Pass",
        "reviewer": "SYNTHETIC QA ONLY",
        "exclusion_reason": "",
        **updates,
    }


def test_pixel_center_rasterization_holes_and_invalid_original_coordinates() -> None:
    value = geometry()
    mask = np.array(rasterize(value, 10, 8))
    assert set(np.unique(mask)) == {0, 255}
    assert np.count_nonzero(mask) == 20
    assert mask[1:5, 1:6].all()
    value["contours"].append({"operation": "erase", "points": [[2, 2], [4, 2], [4, 4], [2, 4]]})
    assert np.count_nonzero(np.array(rasterize(value, 10, 8))) == 16
    value["contours"][0]["points"][0][0] = -1
    with pytest.raises(Phase1Error, match="原圖座標"):
        rasterize(value, 10, 8)
    value = geometry()
    value["contours"][0]["points"][0][0] = float("nan")
    with pytest.raises(Phase1Error):
        rasterize(value, 10, 8)


def test_revisions_separate_eyes_reopen_and_conflicting_writer(tmp_path: Path) -> None:
    targets = [target(), target() | {"annotation_id": "1002"}]
    initialize_store(tmp_path, targets, "dataset-sha")
    assert load_annotations(tmp_path, targets) == []
    left = save_annotation(
        tmp_path, targets, (10, 1001, "left"), geometry(), review(), expected_version=0
    )
    metadata = json.loads((tmp_path / "revisions/10_1001_left_v1.json").read_text())
    assert metadata["tool"]["canvas_sha256"] == sha256_file(
        Path(__file__).resolve().parents[1] / "app/assets/eye_canvas.js"
    )
    first_mask = (tmp_path / left["mask_path"]).read_bytes()
    right = save_annotation(
        tmp_path, targets, (10, 1001, "right"), geometry(), review(), expected_version=0
    )
    second = save_annotation(
        tmp_path, targets, (10, 1001, "left"), geometry(), review(), expected_version=1
    )
    assert second["annotation_version"] == "2"
    assert left["mask_path"] != second["mask_path"] != right["mask_path"]
    assert (tmp_path / left["mask_path"]).read_bytes() == first_mask
    reopened = load_annotations(tmp_path, targets)
    assert len(reopened) == 3
    assert latest_annotation(reopened, (10, 1001, "right")) == right
    assert latest_annotation(reopened, (10, 1001, "left")) == second
    assert load_geometry(tmp_path, left) == geometry()
    other = save_annotation(
        tmp_path, targets, (10, 1002, "left"), geometry(), review(), expected_version=0
    )
    assert latest_annotation(load_annotations(tmp_path, targets), (10, 1001, "left")) == second
    assert other["annotation_id"] == "1002"
    before = (tmp_path / "eye_mask_annotations.csv").read_bytes()
    with pytest.raises(Phase1Error, match="另一個視窗"):
        save_annotation(
            tmp_path, targets, (10, 1001, "left"), geometry(), review(), expected_version=1
        )
    assert (tmp_path / "eye_mask_annotations.csv").read_bytes() == before
    # A crash before CSV commit must not let the next save overwrite an orphan PNG.
    orphan = tmp_path / "masks/10/1001_left_v3.png"
    orphan.write_bytes(first_mask)
    next_row = save_annotation(
        tmp_path, targets, (10, 1001, "left"), geometry(), review(), expected_version=2
    )
    assert next_row["annotation_version"] == "4"
    assert orphan.read_bytes() == first_mask


def test_missing_and_uncertain_eyes_stay_explicit_without_fake_masks(tmp_path: Path) -> None:
    targets = [target()]
    initialize_store(tmp_path, targets, "dataset-sha")
    blank = empty_geometry(10, 8, (0, 0, 10, 8))
    draft = save_annotation(
        tmp_path,
        targets,
        (10, 1001, "left"),
        blank,
        review(
            visibility="",
            side_confirmed="Unknown",
            mask_review_status="Pending",
            measurement_quality="Unknown",
            reviewer="",
        ),
        expected_version=0,
    )
    assert draft["mask_path"] == "" and not list(tmp_path.rglob("*.png"))
    with pytest.raises(Phase1Error, match="非空輪廓"):
        save_annotation(tmp_path, targets, (10, 1001, "left"), blank, review(), expected_version=1)
    with pytest.raises(Phase1Error, match="量測 Pass"):
        save_annotation(
            tmp_path,
            targets,
            (10, 1001, "right"),
            geometry(),
            review(visibility="Partial"),
            expected_version=0,
        )
    invalid = copy.deepcopy(targets)
    invalid[0]["image_sha256"] = "b" * 64
    with pytest.raises(Phase1Error, match="來源／規範"):
        initialize_store(tmp_path, invalid, "dataset-sha")


def test_csv_failure_rolls_back_new_files_and_corruption_is_reported(
    tmp_path: Path, monkeypatch
) -> None:
    import animal_eye_metrology.annotation.eye_annotations as module

    targets = [target()]
    initialize_store(tmp_path, targets, "dataset-sha")
    initial = (tmp_path / "eye_mask_annotations.csv").read_bytes()
    with monkeypatch.context() as context:

        def fail(*args, **kwargs):
            raise OSError("SYNTHETIC DISK FAILURE")

        context.setattr(module, "_write_atomic", fail)
        with pytest.raises(OSError):
            save_annotation(
                tmp_path, targets, (10, 1001, "left"), geometry(), review(), expected_version=0
            )
    assert (tmp_path / "eye_mask_annotations.csv").read_bytes() == initial
    assert not list(tmp_path.rglob("*.png")) and not list((tmp_path / "revisions").glob("*.json"))
    row = save_annotation(
        tmp_path, targets, (10, 1001, "left"), geometry(), review(), expected_version=0
    )
    Image.new("L", (10, 8), 255).save(tmp_path / row["mask_path"])
    with pytest.raises(Phase1Error, match="損壞"):
        load_annotations(tmp_path, targets)


def test_only_v2_protocol_is_available(tmp_path: Path) -> None:
    from animal_eye_metrology.annotation.eye_annotations import SUPPORTED_PROTOCOLS

    assert SUPPORTED_PROTOCOLS == {"eye-visible-eyelid-v2"}
    with pytest.raises(Phase1Error, match="不支援"):
        initialize_store(
            tmp_path, [target()], "synthetic", protocol_version="eye-visible-aperture-v1"
        )
    assert not list(tmp_path.iterdir())
