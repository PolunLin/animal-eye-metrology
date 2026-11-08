"""Synthetic rendering checks; these fixtures are not animal observations."""

from copy import deepcopy

import pytest

Image = pytest.importorskip("PIL.Image")

from animal_eye_metrology.core.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.ui.results_overlay import (  # noqa: E402
    BACKGROUND,
    BINOCULAR_LINE,
    POINT,
    PREDICTION,
    REFERENCE,
    RIGHT_PAIR_LINE,
    render_eye_comparison,
    render_measurement_overview,
)


def colors(image: Image.Image) -> set[tuple[int, int, int]]:
    return {color for _, color in image.getcolors(image.width * image.height)}


def test_pixel_edges_and_saved_centers_are_transformed_without_half_pixel_shift() -> None:
    image = Image.new("RGB", (8, 8), "black")
    reference = Image.new("L", image.size)
    reference.putpixel((3, 4), 255)
    metrics = {"reference_cx_px": 3.5, "reference_cy_px": 4.5}
    overlay = render_eye_comparison(image, None, reference, metrics, {}, (2, 2, 6, 6), size=400)
    # Pixel (3,4) spans x=[100,200], y=[200,300] in the common viewport.
    assert overlay.getpixel((100, 250)) == REFERENCE
    assert overlay.getpixel((150, 200)) == REFERENCE
    assert overlay.getpixel((150, 246)) == REFERENCE  # Centroid square top edge.
    assert overlay.getpixel((146, 250)) == REFERENCE
    assert overlay.getpixel((150, 250)) == (0, 0, 0)
    assert overlay.getpixel((96, 200)) == (0, 0, 0)  # No center at integer (3,4).


def test_box_only_never_draws_geometry_center_as_sam_prompt_and_inputs_stay_unchanged() -> None:
    image = Image.new("RGB", (10, 10), (20, 30, 40))
    reference = Image.new("L", image.size)
    predicted = Image.new("L", image.size)
    predicted.putpixel((4, 4), 255)
    metrics = {"predicted_cx_px": 4.5, "predicted_cy_px": 4.5}
    record = {"point_xy": None, "hrnet_point_xy": [5.5, 5.5], "box_xyxy": [3, 3, 7, 7]}
    record_before, metrics_before = deepcopy(record), deepcopy(metrics)
    bytes_before = [value.tobytes() for value in (image, reference, predicted)]
    overlay = render_eye_comparison(
        image, predicted, reference, metrics, record, (0, 0, 10, 10), 200
    )
    assert POINT not in colors(overlay)
    assert overlay.getpixel((90, 85)) == PREDICTION  # Circle around saved pixel-center (4.5,4.5).
    with_point = render_eye_comparison(
        image, predicted, reference, metrics, record | {"point_xy": [5.5, 5.5]}, (0, 0, 10, 10), 200
    )
    assert with_point.getpixel((110, 110)) == POINT
    assert [value.tobytes() for value in (image, reference, predicted)] == bytes_before
    assert record == record_before and metrics == metrics_before


def test_crop_does_not_invent_reference_edges_and_clips_overlays_before_letterboxing() -> None:
    image = Image.new("RGB", (10, 10), "black")
    full = Image.new("L", image.size, 255)
    overlay = render_eye_comparison(
        image,
        None,
        full,
        {},
        {"point_xy": [5, 1]},
        (2, 2, 8, 5),
        size=240,
    )
    assert REFERENCE not in colors(overlay)  # View is inside a full mask.
    assert POINT not in colors(overlay)  # Point outside view must not land in padding.
    assert overlay.getpixel((120, 20)) == BACKGROUND
    assert overlay.getpixel((120, 120)) == (0, 0, 0)


@pytest.mark.parametrize("which", ["prediction", "reference"])
def test_misaligned_masks_are_rejected(which: str) -> None:
    image = Image.new("RGB", (10, 10))
    correct, wrong = Image.new("L", (10, 10)), Image.new("L", (10, 9))
    predicted = wrong if which == "prediction" else correct
    reference = wrong if which == "reference" else correct
    with pytest.raises(Phase1Error, match="dimensions differ"):
        render_eye_comparison(image, predicted, reference, {}, {}, (0, 0, 10, 10))


@pytest.mark.parametrize("view", [(0, 0, 0, 3), (-1, 0, 3, 3), (0, 0, 11, 10)])
def test_invalid_viewport_is_not_silently_shifted(view: tuple[int, int, int, int]) -> None:
    image, mask = Image.new("RGB", (10, 10)), Image.new("L", (10, 10))
    with pytest.raises(Phase1Error, match="viewport"):
        render_eye_comparison(image, None, mask, {}, {}, view)


def test_undefined_centroids_are_omitted_and_nonfinite_coordinates_rejected() -> None:
    image, mask = Image.new("RGB", (10, 10)), Image.new("L", (10, 10))
    result = render_eye_comparison(
        image, None, mask, {"predicted_cx_px": None, "predicted_cy_px": None}, {}, (0, 0, 10, 10)
    )
    assert PREDICTION not in colors(result)
    with pytest.raises(Phase1Error, match="Non-finite"):
        render_eye_comparison(
            image, None, mask, {}, {"point_xy": [float("nan"), 1]}, (0, 0, 10, 10)
        )


def test_overview_uses_saved_endpoints_in_either_direction_and_does_not_recompute_distance() -> (
    None
):
    image = Image.new("RGB", (100, 80), "black")
    rows = [
        {
            "measurement_type": "binocular",
            # This deliberately differs from endpoint length: a renderer must not evaluate it.
            "predicted_distance_px": 999,
            "predicted_cx_a_px": 80.5,
            "predicted_cy_a_px": 20.5,
            "predicted_cx_b_px": 10.5,
            "predicted_cy_b_px": 20.5,
        },
        {
            "measurement_type": "right_eye_pair",
            "predicted_distance_px": 5,
            "predicted_cx_a_px": 50.5,
            "predicted_cy_a_px": 60.5,
            "predicted_cx_b_px": 50.5,
            "predicted_cy_b_px": 30.5,
        },
    ]
    before = deepcopy(rows)
    overlay = render_measurement_overview(image, [], [], rows)
    assert overlay.size == image.size
    assert overlay.getpixel((40, 20)) == BINOCULAR_LINE
    assert overlay.getpixel((50, 40)) == RIGHT_PAIR_LINE
    assert rows == before
    assert image.getpixel((40, 20)) == (0, 0, 0)


def test_overview_retains_missing_outputs_and_rejects_misaligned_eye_mask_pairs() -> None:
    image = Image.new("RGB", (100, 80), "black")
    eyes = [{"annotation_id": 123, "eye_side": "right", "predicted_cx_px": None}]
    missing_distance = [{"measurement_type": "binocular", "predicted_distance_px": None}]
    overlay = render_measurement_overview(image, [None], eyes, missing_distance)
    assert overlay.tobytes() == image.tobytes()
    with pytest.raises(Phase1Error, match="corresponding mask"):
        render_measurement_overview(image, [], eyes, missing_distance)
    with pytest.raises(Phase1Error, match="dimensions differ"):
        render_measurement_overview(image, [Image.new("L", (100, 79))], eyes, missing_distance)
    with pytest.raises(Phase1Error, match="coordinates"):
        render_measurement_overview(
            image, [None], eyes, [{"measurement_type": "binocular", "predicted_distance_px": 5}]
        )
