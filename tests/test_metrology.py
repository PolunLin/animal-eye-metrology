"""Invented geometry and temporary annotations only; not animal research results."""

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from animal_eye_metrology.core.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.measurement.metrology import (  # noqa: E402
    build_measurements,
    euclidean_distance,
    mask_centroid,
    mask_overlap,
)


def test_mask_overlap_known_sets_empty_masks_and_invalid_inputs() -> None:
    a = np.array([[1, 1, 0, 0]])
    b = np.array([[0, 1, 1, 0]])
    overlap = mask_overlap(a, b)
    assert overlap == {"intersection_px": 1, "union_px": 3, "iou": 1 / 3, "dice": 0.5}
    assert mask_overlap(b, a) == overlap
    assert mask_overlap(a, a)["iou"] == mask_overlap(a, a)["dice"] == 1
    assert mask_overlap(a, np.array([[0, 0, 1, 1]]))["iou"] == 0
    empty = np.zeros_like(a)
    assert mask_overlap(empty, a)["iou"] == mask_overlap(empty, a)["dice"] == 0
    assert mask_overlap(empty, empty)["iou"] is None
    assert mask_overlap(empty, empty)["dice"] is None
    for invalid in (np.array([[255]]), np.array([[np.nan]]), np.zeros((0, 1))):
        with pytest.raises(Phase1Error):
            mask_overlap(invalid, invalid)
    with pytest.raises(Phase1Error, match="same original-image"):
        mask_overlap(a, np.zeros((2, 2)))


def test_centroid_uses_original_pixel_centers_and_translates_correctly() -> None:
    mask = np.zeros((9, 12), dtype=np.uint8)
    mask[1:3, 2:6] = 1
    assert mask_centroid(mask) == (4.0, 2.0)
    shifted = np.zeros_like(mask)
    shifted[4:6, 4:8] = 1
    assert mask_centroid(shifted) == (6.0, 5.0)
    assert mask_centroid(np.array([[1]], dtype=bool)) == (0.5, 0.5)
    # A disconnected mask's centroid can lie outside foreground; do not snap it.
    assert mask_centroid(np.array([[1, 0, 1]])) == (1.5, 0.5)


def test_empty_masks_invalid_encodings_and_undefined_distances() -> None:
    assert mask_centroid(np.zeros((3, 5))) is None
    assert euclidean_distance(None, (2, 3)) is None
    assert euclidean_distance((1, 2), (1, 2)) == 0
    assert euclidean_distance((0, 0), (3, 4)) == 5
    assert euclidean_distance((3, 4), (0, 0)) == 5
    for invalid in (
        np.array([[255]]),
        np.array([[float("nan")]]),
        np.zeros((0, 2)),
        np.zeros((2, 2, 1)),
    ):
        with pytest.raises(Phase1Error):
            mask_centroid(invalid)
    with pytest.raises(Phase1Error):
        euclidean_distance((float("inf"), 0), (0, 0))


def invented_centers() -> list[dict]:
    rows = []
    for image_id in (10, 20, 30):
        for annotation_id, offset in ((1, 0), (2, 6)):
            for side, point in (("left", (0.5 + offset, 0.5)), ("right", (3.5 + offset, 4.5))):
                rows.append(
                    {
                        "image_id": image_id,
                        "annotation_id": annotation_id,
                        "eye_side": side,
                        "annotation_version": 7,
                        "centroid_x_px": point[0],
                        "centroid_y_px": point[1],
                    }
                )
    return rows


def test_pairs_use_same_image_same_animal_binocular_and_distinct_right_eyes() -> None:
    centers = invented_centers()
    result = build_measurements(centers[::-1])
    assert len(result) == 9
    binocular = [r for r in result if r["measurement_type"] == "binocular"]
    pairs = [r for r in result if r["measurement_type"] == "right_eye_pair"]
    assert len(binocular) == 6 and all(r["distance_px"] == 5 for r in binocular)
    assert len(pairs) == 3 and all(r["distance_px"] == 6 for r in pairs)
    assert all(
        r["annotation_id_a"] == r["annotation_id_b"]
        and (r["eye_side_a"], r["eye_side_b"]) == ("left", "right")
        for r in binocular
    )
    assert all(
        r["annotation_id_a"] < r["annotation_id_b"]
        and r["eye_side_a"] == r["eye_side_b"] == "right"
        for r in pairs
    )
    assert {r["image_id"] for r in pairs} == {10, 20, 30}
    with pytest.raises(Phase1Error, match="Duplicate"):
        build_measurements([*centers, centers[0]])
    with pytest.raises(Phase1Error, match="Both left/right"):
        build_measurements(centers[1:])
    centers[0] = centers[0] | {"centroid_x_px": None, "centroid_y_px": None}
    undefined = [r for r in build_measurements(centers) if r["distance_px"] is None]
    assert len(undefined) == 1 and undefined[0]["measurement_status"] == "Undefined"
