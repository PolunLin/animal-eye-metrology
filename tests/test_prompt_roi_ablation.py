"""Synthetic QA for the approved factorial geometry and independent SAM requests."""

import copy
import json
import sys
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.research import prompt_roi_ablation as ablation


def case() -> dict:
    return {
        "case_id": "1_101_left",
        "image_id": 1,
        "annotation_id": 101,
        "prediction_id": "SYNTHETIC_QA_1",
        "eye_side": "left",
        "image_width": 100,
        "image_height": 80,
        "image_path": "inputs/images/000000000001.jpg",
        "image_sha256": "a" * 64,
        "hrnet_point_xy": [40.25, 30.75],
        "hrnet_score": 0.8,
        "hrnet_eye_distance_px": 10.0,
        "manual_point_xy": [43.25, 32.75],
        "manual_box_xyxy": [41.0, 31.0, 45.0, 35.0],
    }


def condition(group: str, factor: float = 0.5) -> dict:
    return next(
        c for c in ablation.conditions() if c["group"] == group and c["roi_factor"] == factor
    )


def test_small_roi_scales_the_minimum_and_preserves_fractional_pixel_coverage() -> None:
    # The baseline length is 32 despite d=10. The half-size crop must be 16, not 32.
    assert ablation.roi_for_case(case(), 0.5) == ([32, 22, 49, 39], False)
    assert ablation.roi_for_case(case(), 1.0) == ([24, 14, 57, 47], False)
    # Once 2*d dominates, the same factor acts on that complete length.
    assert ablation.roi_for_case(case() | {"hrnet_eye_distance_px": 30.0}, 0.5) == (
        [25, 15, 56, 46],
        False,
    )


def test_image_clipping_does_not_recenter_the_crop() -> None:
    source = case() | {"hrnet_point_xy": [3.25, 5.75]}
    assert ablation.roi_for_case(source, 1.5) == ([0, 0, 28, 30], True)
    with pytest.raises(Phase1Error, match="Unapproved ROI factor"):
        ablation.roi_for_case(source, 0.75)
    with pytest.raises(Phase1Error, match="Invalid deterministic crop"):
        ablation.roi_for_case(source | {"hrnet_point_xy": [-100.0, -100.0]}, 1.0)


def test_point_replacement_keeps_roi_and_conflicting_prompt_geometry_unchanged() -> None:
    source = case()
    original = copy.deepcopy(source)
    auto = ablation.request_for_case(source, condition("A"))
    manual = ablation.request_for_case(source, condition("B"))
    conflict = ablation.request_for_case(source, condition("C"))
    both_manual = ablation.request_for_case(source, condition("D"))
    assert auto["roi_xyxy"] == manual["roi_xyxy"] == conflict["roi_xyxy"] == both_manual["roi_xyxy"]
    assert conflict["point_xy"] == [40.25, 30.75]
    assert conflict["box_xyxy"] == [41.0, 31.0, 45.0, 35.0]
    assert conflict["point_outside_prompt_box"] is True
    assert both_manual["point_outside_prompt_box"] is False
    assert manual["point_score"] is None
    assert auto["human_assisted"] is False and manual["human_assisted"] is True
    assert source == original


def test_global_float32_rounding_precedes_roi_translation_without_half_pixel_shift() -> None:
    source = case() | {
        "manual_point_xy": [43.123456789, 32.987654321],
        "manual_box_xyxy": [41.123456789, 31.23456789, 45.3456789, 35.456789],
    }
    request = ablation.request_for_case(source, condition("D"))
    local_point = np.asarray(request["point_coords_in_roi"])[0]
    restored_point = local_point + [32, 22]
    assert np.array_equal(restored_point, np.asarray(source["manual_point_xy"], dtype=np.float32))
    restored_box = np.asarray(request["box_coords_in_roi"]) + [32, 22, 32, 22]
    assert np.array_equal(restored_box, np.asarray(source["manual_box_xyxy"], dtype=np.float32))
    assert local_point[0] != source["manual_point_xy"][0] - 32


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"manual_point_xy": [90.0, 70.0]}, "frozen point"),
        ({"manual_point_xy": [float("nan"), 30.0]}, "frozen point"),
        ({"manual_box_xyxy": [0.0, 0.0, 100.0, 80.0]}, "frozen box"),
        ({"manual_box_xyxy": [45.0, 31.0, 41.0, 35.0]}, "frozen box"),
    ],
)
def test_invalid_manual_geometry_is_rejected_without_repair(updates: dict, message: str) -> None:
    source = case() | updates
    with pytest.raises(Phase1Error, match=message):
        ablation.request_for_case(source, condition("D"))


def test_factorial_contract_has_all_twelve_distinct_conditions_and_one_reused_cell() -> None:
    rows = ablation.conditions()
    actual = {(r["point_source"], r["box_source"], r["roi_factor"]) for r in rows}
    assert actual == set(product(("hrnet", "manual"), ("none", "manual"), (0.5, 1.0, 1.5)))
    assert len(rows) == len({r["condition_id"] for r in rows}) == 12
    requests = [ablation.request_for_case(case(), row) for row in rows]
    assert [r["condition_id"] for r in requests if r["reused_baseline"]] == ["A_r100"]


def test_selected_mask_uses_quality_score_then_zero_threshold_on_original_grid() -> None:
    logits = np.array(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[-1.0, 0.0, 0.1], [2.0, -2.0, 0.0]],
            [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
        ]
    )
    actual, selected = ablation.selected_mask(logits, [0.1, 0.9, 0.8], (7, 9), [3, 2, 6, 4])
    expected = np.zeros((7, 9), dtype=np.uint8)
    expected[2, 5] = 255
    expected[3, 3] = 255
    assert selected == 1
    assert np.array_equal(actual, expected)
    empty, selected = ablation.selected_mask(logits, [0.1, 0.2, 0.9], (7, 9), [3, 2, 6, 4])
    assert selected == 2 and not empty.any()
    # A quality-score tie preserves deterministic first-index selection, not an area heuristic.
    _, selected = ablation.selected_mask(logits, [0.9, 0.9, 0.8], (7, 9), [3, 2, 6, 4])
    assert selected == 0


@pytest.mark.parametrize(
    "fault", ("missing_candidate", "wrong_crop_size", "nan_logits", "nan_score")
)
def test_invalid_native_predictions_cannot_produce_a_measurement_mask(fault: str) -> None:
    logits = np.zeros((3, 2, 3))
    scores = [0.2, 0.8, 0.7]
    if fault == "missing_candidate":
        scores = [0.2, 0.8]
    elif fault == "wrong_crop_size":
        logits = np.zeros((3, 2, 4))
    elif fault == "nan_logits":
        logits[0, 0, 0] = np.nan
    else:
        scores[1] = np.nan
    with pytest.raises(Phase1Error, match="Invalid native logits"):
        ablation.selected_mask(logits, scores, (7, 9), [3, 2, 6, 4])


class FakePredictor:
    def __init__(self) -> None:
        self.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="float32")]))
        self.images: list[np.ndarray] = []
        self.calls: list[dict] = []

    def set_image(self, image: np.ndarray) -> None:
        self.images.append(image.copy())
        self.shape = image.shape[:2]

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls.append(kwargs)
        logits = np.zeros((3, *self.shape), dtype=np.float32)
        logits[1, 0, 0] = 1.0
        return logits, np.array([0.1, 0.9, 0.2]), np.zeros((3, 2, 2))


def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakePredictor:
    inputs = tmp_path / "inputs"
    (inputs / "images").mkdir(parents=True)
    checkpoint = tmp_path / "SYNTHETIC_CHECKPOINT"
    checkpoint.write_bytes(b"not a real model")
    source = {"commit": "SYNTHETIC_QA_ONLY"}
    (inputs / "baseline_segmentation.json").write_text(
        json.dumps(
            {
                "sam_source": source,
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    )
    (inputs / "manifest.json").write_text(json.dumps({"files_sha256": {}}))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="float32",
            set_num_threads=lambda _: None,
            set_num_interop_threads=lambda _: None,
            manual_seed=lambda _: None,
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setattr(
        ablation,
        "settings",
        lambda _: {
            "models": {"sam2_1_tiny": {"checkpoint": checkpoint.name}},
            "weights_directory": tmp_path,
        },
    )
    monkeypatch.setattr(ablation, "validate_source", lambda *_: source)
    predictor = FakePredictor()
    monkeypatch.setattr(ablation, "load_predictor", lambda *_: predictor)
    return predictor


def test_embedding_reuse_keeps_prompt_calls_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictor = fake_runtime(tmp_path, monkeypatch)
    source = case()
    image_path = tmp_path / source["image_path"]
    Image.new("RGB", (100, 80), "gray").save(image_path)
    source["image_sha256"] = sha256_file(image_path)
    requests = [ablation.request_for_case(source, condition(g)) for g in ("B", "C")]
    (tmp_path / "inputs/requests.json").write_text(json.dumps(requests))
    ablation.run_predictions(
        {"output_directory": tmp_path, "sam_runtime": tmp_path / "unused.toml"}
    )
    assert len(predictor.images) == 1 and len(predictor.calls) == 2
    for request, call in zip(requests, predictor.calls, strict=True):
        assert call["mask_input"] is None and call["multimask_output"] is True
        assert call["return_logits"] is True
        assert np.array_equal(call["point_coords"], request["point_coords_in_roi"])
    assert predictor.calls[0]["box"] is None
    assert np.array_equal(predictor.calls[1]["box"], requests[1]["box_coords_in_roi"])
    outcomes = json.loads((tmp_path / "predictions.json").read_text())
    assert all(r["status"] == "passed" and r["selected_candidate"] == 1 for r in outcomes)


def test_failed_new_image_never_reuses_pixels_from_previous_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = fake_runtime(tmp_path, monkeypatch)
    source = case()
    Image.new("RGB", (100, 80), "gray").save(tmp_path / source["image_path"])
    source["image_sha256"] = sha256_file(tmp_path / source["image_path"])
    broken = source | {
        "case_id": "2_202_left",
        "image_id": 2,
        "annotation_id": 202,
        "image_path": "inputs/images/000000000002.jpg",
        "image_sha256": "0" * 64,
    }
    Image.new("RGB", (100, 80), "red").save(tmp_path / broken["image_path"])
    requests = [ablation.request_for_case(source, condition("B"))]
    requests += [ablation.request_for_case(broken, condition(g)) for g in ("B", "C")]
    (tmp_path / "inputs/requests.json").write_text(json.dumps(requests))
    ablation.run_predictions(
        {"output_directory": tmp_path, "sam_runtime": tmp_path / "unused.toml"}
    )
    outcomes = json.loads((tmp_path / "predictions.json").read_text())
    assert [r["status"] for r in outcomes] == ["passed", "failed", "failed"]
    assert len(predictor.calls) == 1
