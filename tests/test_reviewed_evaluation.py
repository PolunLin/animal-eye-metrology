"""Synthetic matching/scoring fixtures only; no animal data or model downloads."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL.Image")

from animal_eye_metrology.core.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.measurement.reviewed_evaluation import (  # noqa: E402
    match_mask_arrays,
    score_distance,
    score_eye,
    summarize,
)


def animal(identity: str, mask: np.ndarray, category: str = "cat", *, target=False) -> dict:
    return {
        "annotation_id" if target else "prediction_id": identity,
        "category": category,
        "bbox_xyxy": [0, 0, mask.shape[1], mask.shape[0]],
        "mask": mask,
    }


def test_matching_keeps_category_and_all_competing_instances() -> None:
    mask = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    prediction = animal("p1", mask)
    target = animal("10", mask, target=True)
    matches, audit = match_mask_arrays([prediction], [target])
    assert matches["10"]["prediction_id"] == "p1"
    assert audit[0]["mask_iou"] == audit[0]["bbox_iou"] == 1
    assert match_mask_arrays([prediction], [animal("10", mask, "dog", target=True)])[0] == {}
    # An excluded-eye animal must still compete, preventing ambiguous identity reuse.
    assert match_mask_arrays([prediction], [target, animal("11", mask, target=True)])[0] == {}
    assert match_mask_arrays([prediction, animal("p2", mask)], [target])[0] == {}


def test_matching_requires_both_mask_and_box_thresholds_and_margin() -> None:
    target = animal("10", np.array([[1, 1, 1, 1, 0, 0]]), target=True)
    prediction = animal("p1", np.array([[1, 1, 0, 0, 0, 0]]))
    assert match_mask_arrays([prediction], [target])[0]["10"]["mask_iou"] == 0.5
    prediction["bbox_xyxy"] = [0, 0, 1, 1]
    assert match_mask_arrays([prediction], [target])[0] == {}
    prediction["bbox_xyxy"] = target["bbox_xyxy"]
    prediction["mask"] = np.array([[1, 0, 0, 0, 0, 0]])
    assert match_mask_arrays([prediction], [target])[0] == {}
    matches, audit = match_mask_arrays([], [target])
    assert matches == {} and audit[0]["reason"] == "no_accepted_animal_masks"


def test_matching_rejects_duplicate_identities_and_grid_mismatch() -> None:
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(Phase1Error, match="Duplicate"):
        match_mask_arrays([], [animal("1", mask, target=True)] * 2)
    with pytest.raises(Phase1Error, match="same original-image"):
        match_mask_arrays([animal("p1", mask)], [animal("1", np.ones((1, 2)), target=True)])


def test_scores_preserve_pixel_centers_and_explicit_failure_denominators() -> None:
    reference = np.zeros((8, 8), dtype=bool)
    reference[1, 1] = True
    shifted = np.zeros_like(reference)
    shifted[5, 4] = True
    exact = score_eye(reference, reference)
    different = score_eye(reference, shifted)
    missing = score_eye(reference, None, "unmatched_animal")
    empty = score_eye(reference, np.zeros_like(reference))
    assert exact["reference_cx_px"] == exact["predicted_cy_px"] == 1.5
    assert different["centroid_error_px"] == 5
    assert missing["centroid_error_px"] is empty["centroid_error_px"] is None
    assert missing["failure_reason"] == "unmatched_animal"
    summary = summarize([exact, different, missing, empty], [])
    assert summary["mean_iou_all_targets_failure_as_zero"] == 0.25
    assert summary["mean_dice_all_targets_failure_as_zero"] == 0.25
    assert summary["mean_centroid_error_px"] == 2.5
    assert summary["eyes_expected"] == 4 and summary["centroid_defined"] == 2
    assert summary["inference_failed"] == summary["empty_masks"] == 1


def test_pair_errors_exclude_undefined_values_but_preserve_counts() -> None:
    exact = score_distance((0.5, 0.5), (3.5, 4.5), (0.5, 0.5), (3.5, 4.5))
    larger = score_distance((0.5, 0.5), (3.5, 4.5), (0.5, 0.5), (6.5, 8.5))
    missing = score_distance((0.5, 0.5), (3.5, 4.5), None, (3.5, 4.5), "pose_failed")
    assert larger["reference_distance_px"] == 5 and larger["predicted_distance_px"] == 10
    assert missing["predicted_distance_px"] is missing["absolute_error_px"] is None
    assert missing["failure_reason"] == "pose_failed"
    rows = [{"measurement_type": "binocular", **r} for r in (exact, larger, missing)]
    summary = summarize([], rows)
    assert summary["binocular_expected"] == 3 and summary["binocular_defined"] == 2
    assert summary["binocular_failed"] == 1 and summary["binocular_mae_px"] == 2.5
    assert summary["binocular_rmse_px"] == pytest.approx((25 / 2) ** 0.5)
    assert summary["right_eye_pair_mae_px"] is None


def test_empty_reference_and_invalid_prediction_are_rejected() -> None:
    with pytest.raises(Phase1Error, match="nonempty"):
        score_eye(np.zeros((2, 2)), None)
    with pytest.raises(Phase1Error, match="binary"):
        score_eye(np.ones((2, 2)), np.full((2, 2), 255))
    with pytest.raises(Phase1Error, match="eligible endpoints"):
        score_distance(None, (0, 0), (1, 1), (2, 2))


def test_signed_distance_and_coincident_points_remain_defined() -> None:
    result = score_distance((0, 0), (3, 4), (1, 1), (1, 1))
    assert result["predicted_distance_px"] == 0
    assert result["signed_error_px"] == -5 and result["absolute_error_px"] == 5
    assert result["measurement_status"] == "defined"
