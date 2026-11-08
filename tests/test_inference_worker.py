"""Synthetic arrays test failure semantics; they are not research model results."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.inference import inference_worker
from animal_eye_metrology.inference.inference_capabilities import COCO_ANIMAL_CATEGORIES
from animal_eye_metrology.inference.inference_worker import detection_records, segment_case


class Predictor:
    def __init__(self, *, empty: bool = False, invalid: bool = False) -> None:
        self.empty = empty
        self.invalid = invalid
        self.arguments: dict[str, Any] = {}
        self.image: np.ndarray | None = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

    def predict(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.arguments = kwargs
        assert self.image is not None
        logits = np.full((3, *self.image.shape[:2]), -1.0, dtype=np.float32)
        if not self.empty:
            logits[0, :, :] = 1.0
            logits[1, 1:3, 2:4] = 1.0
        if self.invalid:
            logits[0, 0, 0] = np.nan
        return logits, np.array([0.1, 0.9, 0.2]), np.zeros((3, 256, 256))


@pytest.fixture
def case() -> dict[str, Any]:
    return {
        "prediction_id": "animal_0001",
        "eye_side": "left",
        "condition_id": "A_r100",
        "point_xy": [5.123456789, 5.5],
        "roi_xyxy": [3, 4, 9, 10],
    }


def test_sam_keeps_point_only_argmax_selection_original_coordinates_and_empty_pixels(
    tmp_path: Path, case: dict[str, Any]
) -> None:
    predictor = Predictor()
    image = np.arange(16 * 18 * 3, dtype=np.uint8).reshape(16, 18, 3)
    result = segment_case(predictor, image, case, tmp_path)
    assert result["status"] == "completed"
    assert result["selected_candidate"] == 1  # Highest quality, not largest area.
    assert result["mask_area_px"] == 4
    assert result["eye_side"] == "left"
    np.testing.assert_array_equal(predictor.image, image[4:10, 3:9])
    np.testing.assert_array_equal(
        predictor.arguments["point_coords"],
        np.asarray([case["point_xy"]], dtype=np.float32) - [3, 4],
    )
    assert predictor.arguments["box"] is None
    assert predictor.arguments["multimask_output"] is True
    assert predictor.arguments["return_logits"] is True
    mask = np.asarray(Image.open(tmp_path / result["mask_path"]))
    expected = np.zeros((16, 18), dtype=np.uint8)
    expected[5:7, 5:7] = 255
    np.testing.assert_array_equal(mask, expected)


def test_empty_sam_mask_is_saved_but_not_reported_as_completed(
    tmp_path: Path, case: dict[str, Any]
) -> None:
    result = segment_case(Predictor(empty=True), np.zeros((16, 18, 3)), case, tmp_path)
    assert result["status"] == "empty_mask"
    assert result["failure_reason"] == "empty_eye_mask"
    assert result["mask_area_px"] == 0
    assert (tmp_path / result["mask_path"]).is_file()


@pytest.mark.parametrize(
    "updates",
    [
        {"prediction_id": "../../escape"},
        {"eye_side": "screen_left"},
        {"condition_id": "gamma_0.5"},
        {"roi_xyxy": [3, 4, 99, 10]},
        {"roi_xyxy": [3.0, 4, 9, 10]},
        {"point_xy": [float("nan"), 5]},
        {"point_xy": [9, 5]},
    ],
)
def test_invalid_eye_request_is_explicit_failure_without_prompt_repair(
    tmp_path: Path, case: dict[str, Any], updates: dict[str, Any]
) -> None:
    predictor = Predictor()
    result = segment_case(predictor, np.zeros((16, 18, 3)), case | updates, tmp_path)
    assert result["status"] == "failed"
    assert result["failure_reason"]
    assert result["mask_path"] is None
    assert predictor.image is None


def test_nonfinite_sam_output_is_failed_not_empty(tmp_path: Path, case: dict[str, Any]) -> None:
    result = segment_case(Predictor(invalid=True), np.zeros((16, 18, 3)), case, tmp_path)
    assert result["status"] == "failed"
    assert "invalid logits" in result["failure_reason"]
    assert result["mask_path"] is None


def test_each_eye_failure_does_not_prevent_other_eye(tmp_path: Path, case: dict[str, Any]) -> None:
    first = segment_case(Predictor(invalid=True), np.zeros((16, 18, 3)), case, tmp_path)
    second = segment_case(
        Predictor(), np.zeros((16, 18, 3)), case | {"eye_side": "right"}, tmp_path
    )
    assert first["status"] == "failed" and second["status"] == "completed"


def test_detector_enables_cat_and_retains_below_threshold_animals(tmp_path: Path) -> None:
    names = {0: "person", 1: "dog", 2: "cat", 3: "bear"}
    boxes = np.array(
        [[0, 0, 4, 4, 0.8, 0], [1, 1, 5, 5, 0.7, 1], [2, 2, 6, 6, 0.6, 2], [3, 3, 7, 7, 0.49, 3]]
    )
    masks = np.ones((4, 8, 10), dtype=np.float32)
    rows, nonanimals = detection_records(boxes, masks, names, 10, 8, tmp_path, 0.5)
    assert nonanimals == 1 and len(rows) == 3
    assert [row["prediction_id"] for row in rows] == ["animal_0002", "animal_0003", "animal_0004"]
    dog, cat, bear = rows
    assert dog["accepted"] and dog["supported"] and dog["status"] == "completed"
    assert cat["accepted"] and cat["supported"] and cat["status"] == "completed"
    assert cat["pose_domain"] == "ap10k_mammal_mapping" and not cat["experimental"]
    assert not bear["accepted"] and bear["supported"] and bear["status"] == "below_threshold"
    assert dog["mask_area_px"] == 80 and cat["mask_area_px"] == 80
    assert bear["mask_path"] is None and bear["mask_status"] == "not_run"


def test_every_coco_animal_is_enabled_but_bird_keeps_domain_warning(tmp_path: Path) -> None:
    names = dict(enumerate(COCO_ANIMAL_CATEGORIES))
    boxes = np.asarray([[1, 1, 5, 5, 0.8, index] for index in names])
    rows, nonanimals = detection_records(
        boxes, np.ones((10, 8, 10), dtype=np.uint8), names, 10, 8, tmp_path, 0.5
    )
    assert nonanimals == 0 and len(rows) == 10
    assert all(row["accepted"] and row["supported"] for row in rows)
    assert all(row["status"] == "completed" and row["mask_status"] == "completed" for row in rows)
    experimental = [row for row in rows if row["experimental"]]
    assert [row["category"] for row in experimental] == ["bird"]
    assert experimental[0]["pose_domain"] == "out_of_training_domain"
    assert experimental[0]["capability_reason"] == "bird_not_in_ap10k_mammal_training_domain"
    assert all(row["failure_reason"] == "" for row in rows)


@pytest.mark.parametrize("masks", [None, np.zeros((1, 2, 2)), np.full((1, 8, 10), np.nan)])
def test_invalid_instance_mask_keeps_usable_bbox_with_separate_failure(
    tmp_path: Path, masks: np.ndarray | None
) -> None:
    rows, _ = detection_records(
        np.array([[0, 0, 4, 4, 0.8, 0]]), masks, {0: "dog"}, 10, 8, tmp_path, 0.5
    )
    assert rows[0]["accepted"] is True
    assert rows[0]["status"] == "completed"
    assert rows[0]["mask_status"] == "failed"
    assert rows[0]["mask_path"] is None


def test_empty_instance_mask_is_distinct_from_detection_failure(tmp_path: Path) -> None:
    rows, _ = detection_records(
        np.array([[0, 0, 4, 4, 0.8, 0]]), np.zeros((1, 8, 10)), {0: "dog"}, 10, 8, tmp_path, 0.5
    )
    assert rows[0]["accepted"] and rows[0]["mask_status"] == "empty_mask"
    assert rows[0]["mask_area_px"] == 0


def test_no_detections_is_completed_empty_result(tmp_path: Path) -> None:
    assert detection_records(np.zeros((0, 6)), None, {0: "dog"}, 10, 8, tmp_path, 0.5) == ([], 0)


@pytest.mark.parametrize(
    "box", [[0, 0, 2, 2, float("nan"), 0], [0, 0, 2, 2, 0.8, 0.2], [0, 0, 2, 2, 1.2, 0]]
)
def test_invalid_detection_arrays_are_not_reported_as_no_animals(
    tmp_path: Path, box: list[float]
) -> None:
    with pytest.raises(Phase1Error):
        detection_records(np.array([box]), None, {0: "dog"}, 10, 8, tmp_path, 0.5)


def test_invalid_box_is_not_accepted_for_pose(tmp_path: Path) -> None:
    rows, _ = detection_records(
        np.array([[-1, 0, 4, 4, 0.8, 0]]), np.zeros((1, 8, 10)), {0: "dog"}, 10, 8, tmp_path, 0.5
    )
    assert not rows[0]["accepted"]
    assert rows[0]["status"] == "failed"
    assert rows[0]["failure_reason"] == "invalid_animal_box"


def test_worker_load_failure_returns_nonzero_and_structured_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"output": str(tmp_path)}))

    def unavailable(_: dict[str, Any]) -> None:
        raise RuntimeError("synthetic model unavailable")

    monkeypatch.setattr(inference_worker, "detection", unavailable)
    monkeypatch.setattr(sys, "argv", ["worker", "--stage", "detection", "--request", str(request)])
    assert inference_worker.main() == 1
    error = json.loads((tmp_path / "detection_error.json").read_text())
    assert error["status"] == "failed"
    assert "synthetic model unavailable" in error["failure_reason"]
    assert not (tmp_path / "detection.json").exists()
