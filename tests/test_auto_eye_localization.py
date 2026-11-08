"""Synthetic QA only; these arrays are not research annotations or model results."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from animal_eye_metrology import auto_eye_localization as auto
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.metrology import build_measurements


def test_config_preserves_virtual_environment_executable_symlink() -> None:
    cfg = auto.config(Path("configs/auto_eye_localization.toml"))
    assert cfg["pose_python"].parts[-3:] == (".venv-eye-pose", "bin", "python")


def pose(points: list[list[float]]) -> dict:
    return {
        "status": "passed",
        "keypoints_xy": points,
        "scores": [0.8, 0.9],
        "geometry": {"input_center": [50, 40], "input_scale": [125, 125]},
    }


def test_anatomical_sides_are_not_sorted_by_screen_x_and_edge_crop_is_clipped() -> None:
    crops, error = auto.eye_crops(pose([[22.5, 10.5], [2.5, 10.5]]), 100, 80)
    assert error == ""
    assert crops == [[2, 0, 43, 31], [0, 0, 23, 31]]
    # Increasing the predicted inter-eye distance increases context, without GT input.
    wider, _ = auto.eye_crops(pose([[42.5, 10.5], [2.5, 10.5]]), 100, 80)
    assert wider == [[2, 0, 83, 51], [0, 0, 43, 51]]


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"status": "failed"}, "pose_inference_failed"),
        ({"keypoints_xy": [[math.nan, 2], [3, 4]]}, "nonfinite_or_invalid_eye_output"),
        ({"scores": [0, 0.8]}, "nonpositive_heatmap_score"),
        ({"keypoints_xy": [[100, 10], [10, 10]]}, "eye_outside_image"),
        ({"keypoints_xy": [[10, 10], [10, 10]]}, "coincident_eye_points"),
        (
            {"geometry": {"input_center": [90, 70], "input_scale": [5, 5]}},
            "eye_outside_native_animal_crop",
        ),
    ],
)
def test_invalid_eye_pair_has_no_manual_fallback(updates: dict, reason: str) -> None:
    assert auto.eye_crops(pose([[15, 10], [10, 10]]) | updates, 100, 80) == (None, reason)


def test_box_conversion_and_point_membership_follow_image_edge_coordinates() -> None:
    assert auto.xywh_to_xyxy([2.5, 3.5, 10, 20]) == [2.5, 3.5, 12.5, 23.5]
    for box in ([0, 0, 0, 2], [0, 0, -2, 3], [0, 0, math.nan, 2]):
        with pytest.raises(Phase1Error):
            auto.xywh_to_xyxy(box)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 3] = 1
    assert auto.point_in_mask([3.0, 2.0], mask) is True
    assert auto.point_in_mask([3.999, 2.999], mask) is True
    assert auto.point_in_mask([4.0, 2.0], mask) is False
    assert auto.point_in_mask([5.0, 2.0], mask) is False
    assert auto.point_in_mask(None, mask) is None


def test_input_freeze_requires_no_reference_or_manual_prompt_files(tmp_path: Path) -> None:
    images = tmp_path / "source"
    images.mkdir()
    Image.new("RGB", (12, 10), "gray").save(images / "000000000001.jpg")
    source = tmp_path / "animals.json"
    source.write_text(
        json.dumps(
            [
                {"image_id": 1, "prediction_id": "p1", "bbox": [1, 2, 7, 6], "score": 0.7},
                {"image_id": 1, "prediction_id": "p2", "bbox": [2, 2, 2, 2], "score": 0.2},
            ]
        )
    )
    auto.prepare(
        {
            "output_directory": tmp_path,
            "images_directory": images,
            "predictions": source,
            "image_ids": [1],
            "expected_predictions": 1,
            "animal_score_threshold": 0.5,
            "primary_reference": tmp_path / "DOES_NOT_EXIST",
            "animal_matches": tmp_path / "DOES_NOT_EXIST.csv",
        }
    )
    cases = auto.read_json(tmp_path / "inputs/pose_cases.json")
    assert len(cases) == 1 and cases[0]["bbox_xyxy"] == [1, 2, 8, 8]
    assert set(cases[0]) == {
        "image_id",
        "prediction_id",
        "image",
        "image_sha256",
        "bbox_xyxy",
    }
    auto.check_inputs(tmp_path)
    (tmp_path / "inputs/images/000000000001.jpg").write_bytes(b"CORRUPTED QA")
    with pytest.raises(Phase1Error, match="Frozen input changed"):
        auto.check_inputs(tmp_path)


def test_evaluation_keeps_missing_and_empty_targets_and_never_repairs_sides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "evaluation_inputs/reference"
    reference.mkdir(parents=True)
    (tmp_path / "pose").mkdir()
    centers, outcomes = [], []
    for aid in (1, 2):
        for side, x in (("left", 1), ("right", 5)):
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:4, x : x + 2] = 255
            relative = f"{aid}_{side}.png"
            Image.fromarray(mask).save(reference / relative)
            centers.append(
                {
                    "image_id": 10,
                    "annotation_id": aid,
                    "eye_side": side,
                    "animal_category": "SYNTHETIC",
                    "annotation_version": 1,
                    "centroid_x_px": x + 1.0,
                    "centroid_y_px": 3.0,
                    "mask_path": relative,
                }
            )
            # Both sides of animal 1 deliberately point to the opposite eye.
            opposite_x = 5 if side == "left" else 1
            predicted = np.zeros((8, 8), dtype=np.uint8)
            if aid == 1:
                predicted[2:4, opposite_x : opposite_x + 2] = 255
            mask_path = f"pred_{aid}_{side}.png"
            Image.fromarray(predicted).save(tmp_path / mask_path)
            outcomes.append(
                {
                    "image_id": 10,
                    "prediction_id": f"p{aid}",
                    "eye_side": side,
                    "status": "failed" if (aid, side) == (2, "right") else "passed",
                    "point_xy": [opposite_x + 0.5, 2.5] if aid == 1 else None,
                    "point_score": 0.8,
                    "roi_xyxy": [4, 1, 8, 5] if aid == 1 else None,
                    "mask_path": mask_path,
                    "failure_reason": "synthetic_failure" if aid == 2 else "",
                }
            )
    monkeypatch.setattr(auto, "verify", lambda _: {})
    monkeypatch.setattr(auto, "derive", lambda _: (centers, build_measurements(centers)))
    with (tmp_path / "evaluation_inputs/animal_matches.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "annotation_id",
                "prediction_id",
                "match_status",
                "confidence_threshold",
                "iou_threshold",
            ],
        )
        writer.writeheader()
        for aid in (1, 2):
            writer.writerow(
                dict(
                    image_id=10,
                    annotation_id=aid,
                    prediction_id=f"p{aid}",
                    match_status="TP",
                    confidence_threshold=0.5,
                    iou_threshold=0.5,
                )
            )
    auto.save_json(
        tmp_path / "pose/pose_results.json",
        {"results": [{"prediction_id": f"p{aid}", "status": "passed"} for aid in (1, 2, 3)]},
    )
    auto.save_json(tmp_path / "segmentation.json", {"results": outcomes})
    rows, measurements, summary = auto.evaluate_saved(tmp_path)
    assert len(rows) == 4 and summary["reference_eyes"] == 4
    assert summary["points_in_opposite_mask"] == 2
    assert summary["points_in_target_mask"] == 0
    assert summary["mean_iou_all_targets_failure_as_zero"] == 0
    assert rows[0]["reference_pixels_outside_roi"] == 4
    assert summary["empty_masks"] == 1 and summary["eye_inference_failed"] == 1
    assert summary["centroid_error_px_defined"] == 2
    assert summary["binocular"]["defined"] == 1
    assert summary["right_eye_pair"]["defined"] == 0
    assert summary["unmatched_prediction_ids"] == ["p3"]
    assert sum(r["predicted_distance_px"] is None for r in measurements) == 2


def test_saved_logit_verification_detects_restore_and_candidate_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = np.array([[[1, 0], [-1, 3]], [[-1, -1], [1, 0]], [[1, 1], [1, 1]]])
    np.save(tmp_path / "logits.npy", logits)
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:4, 4:6] = [[255, 0], [0, 255]]
    Image.fromarray(mask).save(tmp_path / "mask.png")
    result = {
        "status": "passed",
        "logits_path": "logits.npy",
        "mask_path": "mask.png",
        "roi_xyxy": [4, 2, 6, 4],
        "candidate_scores": [0.9, 0.8, 0.7],
        "selected_candidate": 0,
    }
    auto.save_json(tmp_path / "segmentation.json", {"results": [result]})
    auto.save_json(tmp_path / "run_manifest.json", {"files_sha256": {}})
    auto.save_json(tmp_path / "summary.json", {})
    auto.save_json(tmp_path / "evaluation_values.json", {"eyes": [], "measurements": []})
    monkeypatch.setattr(auto, "evaluate_saved", lambda _: ([], [], {}))
    assert auto.verify_run(tmp_path)["status"] == "passed"
    # Same logits placed one pixel away must fail original-grid reconstruction.
    Image.fromarray(np.roll(mask, 1, axis=1)).save(tmp_path / "mask.png")
    with pytest.raises(Phase1Error, match="full-image mask"):
        auto.verify_run(tmp_path)
    Image.fromarray(mask).save(tmp_path / "mask.png")
    result["selected_candidate"] = 2
    (tmp_path / "segmentation.json").write_text(json.dumps({"results": [result]}))
    with pytest.raises(Phase1Error, match="Candidate selection changed"):
        auto.verify_run(tmp_path)
