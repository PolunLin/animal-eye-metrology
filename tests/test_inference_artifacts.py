"""Synthetic engineering checks; these fixtures are not research annotations."""

import csv
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.inference.inference_artifacts import (
    ANIMAL_COLORS,
    animal_segmentation_views,
    build_measurements,
    csv_bytes,
    render_overlays,
)
from animal_eye_metrology.measurement.metrology import mask_centroid


def eye(animal: str, side: str, center: tuple | None, **updates: object) -> dict:
    return {
        "prediction_id": animal,
        "eye_side": side,
        "condition_id": "A_r100",
        "category": "dog",
        "localization_status": "completed",
        "segmentation_status": "completed" if center is not None else "empty_mask",
        "failure_reason": "" if center is not None else "empty_mask",
        "centroid_x_px": center[0] if center is not None else None,
        "centroid_y_px": center[1] if center is not None else None,
        **updates,
    }


def test_missing_eye_does_not_remove_other_measurable_pairs_or_create_zero() -> None:
    rows = [
        eye("a", "left", None),
        eye("a", "right", (1.5, 2.5)),
        eye("b", "left", (4.5, 3.5)),
        eye("b", "right", (4.5, 6.5)),
    ]
    measurements = build_measurements(rows, ["A_r100"])
    assert len(measurements) == 3
    assert measurements[0]["distance_px"] is None
    assert measurements[0]["status"] == "unavailable"
    assert measurements[0]["reason"] == "a:left:empty_mask"
    assert measurements[1]["distance_px"] == 3
    assert measurements[2]["measurement_type"] == "right_to_right"
    assert measurements[2]["distance_px"] == 5
    assert {row["qualification"] for row in measurements} == {"unverified"}
    assert {row["units"] for row in measurements} == {"px"}


def test_unsupported_and_pose_failed_animals_retain_all_unavailable_measurements() -> None:
    rows = [
        eye("a", side, None, segmentation_status="skipped", failure_reason="unsupported_category")
        for side in ("left", "right")
    ] + [
        eye("b", side, None, segmentation_status="skipped", failure_reason="pose_inference_failed")
        for side in ("left", "right")
    ]
    measured = build_measurements(rows, ["A_r100"])
    assert len(measured) == 3
    assert {row["status"] for row in measured} == {"unavailable"}
    assert all(row["distance_px"] is None for row in measured)
    assert "unsupported_category" in measured[-1]["reason"]
    assert "pose_inference_failed" in measured[-1]["reason"]
    assert build_measurements([], ["A_r100"]) == []


def test_bird_domain_propagates_to_each_distance_without_changing_geometry() -> None:
    rows = [
        eye("a", "left", (1.5, 2.5)),
        eye("a", "right", (4.5, 6.5)),
        eye("b", "left", (8.5, 3.5)),
        eye("b", "right", (8.5, 6.5)),
    ]
    before = build_measurements(rows, ["A_r100"])
    for row in rows:
        is_bird = row["prediction_id"] == "a"
        row.update(
            category="bird" if is_bird else "dog",
            pose_domain="out_of_training_domain" if is_bird else "ap10k_mammal_mapping",
            experimental=is_bird,
        )
    after = build_measurements(rows, ["A_r100"])
    assert [row["distance_px"] for row in after] == [row["distance_px"] for row in before]
    assert [row["experimental"] for row in after] == [True, False, True]
    assert after[-1]["pose_domain_a"] == "out_of_training_domain"
    assert after[-1]["pose_domain_b"] == "ap10k_mammal_mapping"
    assert {row["qualification"] for row in after} == {"unverified"}
    assert {row["status"] for row in after} == {"candidate"}
    rows[0].update(centroid_x_px=None, centroid_y_px=None, segmentation_status="empty_mask")
    missing = build_measurements(rows, ["A_r100"])[0]
    assert missing["experimental"] and missing["distance_px"] is None
    assert missing["status"] == "unavailable"


def test_named_anatomical_sides_not_sorted_and_centroids_not_offset_twice() -> None:
    mask = np.zeros((7, 12), dtype=np.uint8)
    mask[2, 10] = 1
    center = mask_centroid(mask)
    rows = [eye("a", "left", center), eye("a", "right", (1.5, 2.5))]
    measured = build_measurements(rows, ["A_r100"])[0]
    assert measured["eye_side_a"] == "left"
    assert measured["cx_a_px"] == 10.5
    assert measured["cx_b_px"] == 1.5
    assert measured["distance_px"] == 9
    # A calculated zero is distinct from unavailable and is still only a candidate.
    rows[1] = eye("a", "right", center)
    measured = build_measurements(rows, ["A_r100"])[0]
    assert measured["status"] == "candidate"
    assert measured["distance_px"] == 0


@pytest.mark.parametrize(
    "problem", ["duplicate", "side", "missing", "partial", "nonfinite", "condition"]
)
def test_invalid_eye_records_are_rejected(problem: str) -> None:
    rows = [eye("a", "left", (1.5, 2.5)), eye("a", "right", (3.5, 4.5))]
    if problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "side":
        rows[0]["eye_side"] = "screen_left"
    elif problem == "missing":
        rows.pop()
    elif problem == "partial":
        rows[0]["centroid_y_px"] = None
    elif problem == "nonfinite":
        rows[0]["centroid_y_px"] = float("nan")
    else:
        rows[0]["condition_id"] = "A_r050"
    with pytest.raises(Phase1Error):
        build_measurements(rows, ["A_r100"])


def test_conditions_do_not_cross_mix_prediction_centers() -> None:
    rows = [eye("a", "left", (1.5, 2.5)), eye("a", "right", (4.5, 6.5))]
    rows += [
        eye("a", "left", (1.5, 2.5), condition_id="A_r050"),
        eye("a", "right", None, condition_id="A_r050"),
    ]
    measured = build_measurements(rows, ["A_r100", "A_r050"])
    assert [row["distance_px"] for row in measured] == [5, None]
    assert [row["condition_id"] for row in measured] == ["A_r100", "A_r050"]


def test_csv_retains_late_columns_unicode_null_and_nested_values() -> None:
    result = csv_bytes([{"reason": "缺眼", "distance": None}, {"reason": "", "roi": [1, 2, 3, 4]}])
    rows = list(csv.DictReader(io.StringIO(result.decode("utf-8-sig"))))
    assert rows[0] == {"reason": "缺眼", "distance": "", "roi": ""}
    assert rows[1]["roi"] == "[1, 2, 3, 4]"


def test_empty_csv_preserves_explicit_schema_without_fabricating_rows() -> None:
    result = csv_bytes([], ["prediction_id", "status", "distance_px"])
    decoded = result.decode("utf-8-sig")
    assert decoded == "prediction_id,status,distance_px\n"
    assert list(csv.DictReader(io.StringIO(decoded))) == []


def test_overlay_retains_original_grid_and_never_changes_saved_masks(tmp_path: Path) -> None:
    Image.new("RGB", (320, 240), "gray").save(tmp_path / "input.png")
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[80:100, 100:120] = 255
    Image.fromarray(mask).save(tmp_path / "eye.png")
    original_mask_bytes = (tmp_path / "eye.png").read_bytes()
    rows = [
        eye(
            "a",
            "left",
            (110, 90),
            mask_path="eye.png",
            roi_xyxy=[80, 70, 140, 130],
            point_x_px=110,
            point_y_px=90,
        ),
        eye("a", "right", None),
    ]
    result = {
        "input": {"image_path": "input.png"},
        "conditions": ["A_r100"],
        "animals": [
            {
                "prediction_id": "a",
                "category": "dog",
                "accepted": True,
                "supported": True,
                "bbox_xyxy": [50, 50, 220, 210],
            }
        ],
        "eyes": rows,
        "measurements": build_measurements(rows, ["A_r100"]),
    }
    artifacts = render_overlays(tmp_path, result)
    assert artifacts["overlay"] == artifacts["A_r100_overlay"]
    with Image.open(tmp_path / artifacts["overlay"]) as overlay:
        assert overlay.size == (320, 240)
        # Interior foreground has an eye tint; missing eyes introduce no origin marker.
        assert overlay.getpixel((105, 85)) != (128, 128, 128)
        assert overlay.getpixel((0, 0)) == (128, 128, 128)
    assert (tmp_path / "eye.png").read_bytes() == original_mask_bytes


def test_overlay_rejects_mask_outside_run_and_wrong_pixel_grid(tmp_path: Path) -> None:
    Image.new("RGB", (20, 20)).save(tmp_path / "input.png")
    Image.new("L", (10, 10)).save(tmp_path / "mask.png")
    result = {
        "input": {"image_path": "input.png"},
        "conditions": ["A_r100"],
        "animals": [],
        "eyes": [eye("a", "left", None, mask_path="mask.png")],
        "measurements": [],
    }
    with pytest.raises(Phase1Error, match="original-size"):
        render_overlays(tmp_path, result)
    result["eyes"][0]["mask_path"] = "../outside.png"
    with pytest.raises(Phase1Error, match="inside"):
        render_overlays(tmp_path, result)


def animal_view_fixture(tmp_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Irregular silhouettes with a hole and a separate component, not research GT."""
    Image.new("RGB", (320, 240), (100, 100, 100)).save(tmp_path / "input.png")
    first = np.zeros((240, 320), dtype=np.uint8)
    first[30:150, 30:100] = 255
    first[120:150, 100:140] = 255
    first[60:90, 50:80] = 0
    first[170:180, 30:40] = 255
    second = np.zeros_like(first)
    second[40:140, 210:280] = 255
    Image.fromarray(first).save(tmp_path / "first.png")
    Image.fromarray(second).save(tmp_path / "second.png")
    result = {
        "input": {"image_path": "input.png"},
        "animals": [
            {
                "prediction_id": "a",
                "category": "dog",
                "accepted": True,
                "bbox_xyxy": [20, 20, 150, 190],
                "mask_path": "first.png",
            },
            {
                "prediction_id": "b",
                "category": "cat",
                "accepted": True,
                "supported": False,
                "bbox_xyxy": [200, 30, 290, 150],
                "mask_path": "second.png",
            },
        ],
    }
    return result, first, second


def test_animal_views_show_true_silhouette_holes_components_and_independent_colors(
    tmp_path: Path,
) -> None:
    result, first, second = animal_view_fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    views = animal_segmentation_views(tmp_path, result)
    overlay, instances = (np.asarray(views[key]) for key in ("overlay", "instances"))
    assert views["overlay"].size == views["instances"].size == (320, 240)
    # The drawn contour follows the mask, not its larger rectangular detector box.
    assert tuple(overlay[30, 50]) == ANIMAL_COLORS[0]
    assert tuple(overlay[31, 50]) == ANIMAL_COLORS[0]
    assert tuple(overlay[32, 50]) != ANIMAL_COLORS[0]
    assert tuple(overlay[100, 130]) == (100, 100, 100)
    assert tuple(overlay[20, 80]) == (100, 100, 100)
    # Both the actual hole and the disconnected foreground component remain visible.
    assert tuple(overlay[70, 60]) == (100, 100, 100)
    assert tuple(overlay[60, 49]) == ANIMAL_COLORS[0]
    assert tuple(instances[175, 35]) == ANIMAL_COLORS[0]
    assert tuple(instances[90, 240]) == ANIMAL_COLORS[1]
    assert ANIMAL_COLORS[0] != ANIMAL_COLORS[1]
    assert np.array_equal(np.any(instances != 0, axis=2), (first != 0) | (second != 0))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_optional_boxes_are_separate_and_never_used_as_missing_mask_silhouettes(
    tmp_path: Path,
) -> None:
    result, _, _ = animal_view_fixture(tmp_path)
    result["animals"][0]["mask_path"] = None
    Image.new("L", (320, 240), 0).save(tmp_path / "second.png")
    plain = animal_segmentation_views(tmp_path, result)
    boxed = animal_segmentation_views(tmp_path, result, show_boxes=True)
    assert np.all(np.asarray(plain["overlay"]) == 100)
    assert not np.asarray(plain["instances"]).any()
    assert not np.asarray(boxed["instances"]).any()
    assert boxed["overlay"].getpixel((20, 100)) == (245, 245, 245)


def test_animal_view_ignores_low_score_masks_and_handles_no_animals(tmp_path: Path) -> None:
    result, _, _ = animal_view_fixture(tmp_path)
    for row in result["animals"]:
        row["accepted"] = False
        row["mask_path"] = "not-present.png"
    for animals in (result["animals"], []):
        views = animal_segmentation_views(tmp_path, result | {"animals": animals})
        assert np.all(np.asarray(views["overlay"]) == 100)
        assert not np.asarray(views["instances"]).any()


@pytest.mark.parametrize("bad_mask", ["../outside.png", "wrong_size.png"])
def test_animal_view_rejects_unsafe_paths_or_wrong_mask_dimensions(
    tmp_path: Path,
    bad_mask: str,
) -> None:
    result, _, _ = animal_view_fixture(tmp_path)
    Image.new("L", (20, 20)).save(tmp_path / "wrong_size.png")
    result["animals"][0]["mask_path"] = bad_mask
    with pytest.raises(Phase1Error):
        animal_segmentation_views(tmp_path, result)
