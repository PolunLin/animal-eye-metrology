"""Synthetic QA for frozen geometry and isolated automatic SAM box requests."""

import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from animal_eye_metrology import auto_eye_box as experiment
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.verification import sha256_file


def case() -> dict:
    return {
        "case_id": "synthetic_prediction_1_left",
        "image_id": 1,
        "prediction_id": "synthetic_prediction_1",
        "eye_side": "left",
        "image_width": 100,
        "image_height": 80,
        "image_path": "inputs/images/000000000001.jpg",
        "image_sha256": "a" * 64,
        "hrnet_point_xy": [40.25, 30.75],
        "hrnet_score": 0.8,
        "hrnet_eye_distance_px": 30.0,
        # Intentionally independent of a newly calculated ROI: this is the frozen input.
        "roi_xyxy": [21, 11, 58, 48],
    }


def condition(group: str = "point", gamma: float = 0.5) -> dict:
    return next(
        row for row in experiment.conditions() if row["group"] == group and row["gamma"] == gamma
    )


def test_six_conditions_prespecify_only_the_midscale_point_method() -> None:
    rows = experiment.conditions()
    assert len(rows) == len({row["condition_id"] for row in rows}) == 6
    assert {(row["gamma"], row["group"]) for row in rows} == {
        (gamma, group) for gamma in (0.3, 0.5, 0.7) for group in ("point", "only")
    }
    assert [row["condition_id"] for row in rows if row["is_primary"]] == ["auto_box_k050_point"]
    assert all(row["roi_factor"] == 1.0 and row["box_source"] == "geometry" for row in rows)


def test_box_only_removes_the_positive_prompt_without_relocating_geometry() -> None:
    source = case()
    original = copy.deepcopy(source)
    with_point = experiment.request_for_case(source, condition("point"))
    only = experiment.request_for_case(source, condition("only"))
    assert source == original
    assert with_point["roi_xyxy"] == only["roi_xyxy"] == source["roi_xyxy"]
    assert with_point["box_xyxy"] == only["box_xyxy"] == [32.75, 23.25, 47.75, 38.25]
    assert with_point["hrnet_point_xy"] == only["hrnet_point_xy"] == source["hrnet_point_xy"]
    assert with_point["point_coords_in_roi"] == [[19.25, 19.75]]
    assert only["point_coords_in_roi"] is None and only["point_xy"] is None
    assert only["point_source"] == "none"
    assert with_point["geometry_status"] == only["geometry_status"] == "passed"


def test_minimum_box_and_boundary_clip_do_not_recenter_eye_or_roi() -> None:
    source = case() | {
        "hrnet_point_xy": [1.25, 2.75],
        "hrnet_eye_distance_px": 10.0,
        "roi_xyxy": [0, 0, 33, 35],
    }
    request = experiment.request_for_case(source, condition(gamma=0.3))
    assert request["box_xyxy"] == [0.0, 0.0, 5.25, 6.75]
    assert request["hrnet_point_xy"] == [1.25, 2.75]
    assert request["roi_xyxy"] == [0, 0, 33, 35]
    assert request["point_coords_in_roi"] == [[1.25, 2.75]]


def test_float32_global_coordinates_precede_translation_without_half_pixel_shift() -> None:
    source = case() | {
        "hrnet_point_xy": [40.123456789, 30.987654321],
        "hrnet_eye_distance_px": 30.123456789,
    }
    request = experiment.request_for_case(source, condition())
    restored_point = np.asarray(request["point_coords_in_roi"])[0] + source["roi_xyxy"][:2]
    assert np.array_equal(restored_point, np.asarray(source["hrnet_point_xy"], dtype=np.float32))
    length = 0.5 * source["hrnet_eye_distance_px"]
    x, y = source["hrnet_point_xy"]
    expected = np.asarray(
        [x - length / 2, y - length / 2, x + length / 2, y + length / 2], dtype=np.float32
    )
    restored_box = np.asarray(request["box_coords_in_roi"]) + source["roi_xyxy"][:2] * 2
    assert np.array_equal(restored_box, expected)
    assert np.array_equal(request["box_xyxy"], expected)


@pytest.mark.parametrize("distance", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_eye_distance_is_retained_as_failed_geometry(distance: float) -> None:
    request = experiment.request_for_case(case() | {"hrnet_eye_distance_px": distance}, condition())
    assert request["geometry_status"] == "failed"
    assert request["geometry_failure_reason"]
    assert request["box_xyxy"] is None and request["box_coords_in_roi"] is None
    # Failure diagnostics must remain valid strict JSON, including nonfinite source inputs.
    json.dumps(request, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "sanitized"),
    [
        ("hrnet_point_xy", [float("nan"), 30.0], [None, 30.0]),
        ("hrnet_point_xy", [40.0, float("inf")], [40.0, None]),
        ("hrnet_score", float("nan"), None),
        ("hrnet_score", float("inf"), None),
    ],
)
def test_nonfinite_point_or_score_retains_strict_json_failure_record(
    field: str, value: object, sanitized: object
) -> None:
    request = experiment.request_for_case(case() | {field: value}, condition())
    assert request["geometry_status"] == "failed" and request["geometry_failure_reason"]
    assert request[field] == sanitized
    assert request["box_xyxy"] is None and request["point_coords_in_roi"] is None
    json.dumps(request, allow_nan=False)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"manual_point_xy": [40.0, 30.0]},
        {"manual_box_xyxy": [32.0, 22.0, 48.0, 39.0]},
        {"annotation_id": 123},
    ],
)
def test_reference_or_manual_schema_is_rejected_instead_of_silently_dropped(
    forbidden: dict,
) -> None:
    with pytest.raises(Phase1Error):
        experiment.request_for_case(case() | forbidden, condition())


def test_unapproved_condition_cannot_change_gamma_or_roi_policy() -> None:
    for changed in (condition() | {"gamma": 0.9}, condition() | {"roi_factor": 0.5}):
        with pytest.raises(Phase1Error):
            experiment.request_for_case(case(), changed)


class FakePredictor:
    def __init__(self, fail_first_decode: bool = False) -> None:
        self.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="float32")]))
        self.images: list[np.ndarray] = []
        self.calls: list[dict] = []
        self.fail_first_decode = fail_first_decode

    def set_image(self, image: np.ndarray) -> None:
        self.images.append(image.copy())
        self.shape = image.shape[:2]

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls.append(kwargs)
        if self.fail_first_decode and len(self.calls) == 1:
            raise RuntimeError("Synthetic decoder failure")
        logits = np.zeros((3, *self.shape), dtype=np.float32)
        # The highest-quality mask deliberately fills the ROI, including pixels outside the box.
        logits[1] = 1.0
        logits[2, 0, 0] = 2.0
        return logits, np.array([0.1, 0.9, 0.2]), np.zeros((3, 2, 2))


def fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requests: list[dict],
    fail_first_decode: bool = False,
) -> FakePredictor:
    inputs = tmp_path / "inputs"
    (inputs / "images").mkdir(parents=True)
    checkpoint = tmp_path / "SYNTHETIC_CHECKPOINT"
    checkpoint.write_bytes(b"not a real model")
    source = {"commit": "SYNTHETIC_QA_ONLY"}
    (inputs / "baseline_segmentation.json").write_text(
        json.dumps({"sam_source": source, "checkpoint_sha256": sha256_file(checkpoint)})
    )
    monkeypatch.setattr(experiment, "check_inputs", lambda _: requests)
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
        experiment,
        "settings",
        lambda _: {
            "models": {"sam2_1_tiny": {"checkpoint": checkpoint.name}},
            "weights_directory": tmp_path,
        },
    )
    monkeypatch.setattr(experiment, "validate_source", lambda *_: source)
    predictor = FakePredictor(fail_first_decode)
    monkeypatch.setattr(experiment, "load_predictor", lambda *_: predictor)
    return predictor


def save_image(tmp_path: Path, source: dict, color: str = "gray") -> dict:
    path = tmp_path / source["image_path"]
    Image.new("RGB", (source["image_width"], source["image_height"]), color).save(path)
    return source | {"image_sha256": sha256_file(path)}


def run_fake(tmp_path: Path) -> list[dict]:
    experiment.run_predictions(
        {"output_directory": tmp_path, "sam_runtime": tmp_path / "unused.toml"}
    )
    return json.loads((tmp_path / "predictions.json").read_text())


def test_sam_calls_are_independent_and_masks_are_not_clipped_to_prompt_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict] = []
    predictor = fake_runtime(tmp_path, monkeypatch, requests)
    source = save_image(tmp_path, case())
    requests.extend(experiment.request_for_case(source, condition(g)) for g in ("point", "only"))
    outcomes = run_fake(tmp_path)
    assert len(predictor.images) == 1 and len(predictor.calls) == 2
    first, only = predictor.calls
    assert np.array_equal(first["point_coords"], [[19.25, 19.75]])
    assert np.array_equal(first["point_labels"], [1])
    assert only["point_coords"] is None and only["point_labels"] is None
    for request, call, result in zip(requests, predictor.calls, outcomes, strict=True):
        assert call["mask_input"] is None and call["multimask_output"] is True
        assert call["return_logits"] is True
        assert np.array_equal(call["box"], request["box_coords_in_roi"])
        assert result["status"] == "passed" and result["selected_candidate"] == 1
        mask = np.asarray(Image.open(tmp_path / result["mask_path"]))
        x0, y0, x1, y1 = source["roi_xyxy"]
        assert np.count_nonzero(mask) == (x1 - x0) * (y1 - y0)
        assert mask[y0, x0] == 255  # This point is outside the narrower prompt box.
    manifest = json.loads((tmp_path / "predictions_manifest.json").read_text())
    assert manifest["predictions_sha256"] == sha256_file(tmp_path / "predictions.json")


def test_failed_new_image_cannot_reuse_previous_image_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict] = []
    predictor = fake_runtime(tmp_path, monkeypatch, requests)
    source = save_image(tmp_path, case())
    broken = save_image(
        tmp_path,
        case()
        | {
            "case_id": "synthetic_prediction_2_left",
            "prediction_id": "synthetic_prediction_2",
            "image_id": 2,
            "image_path": "inputs/images/000000000002.jpg",
        },
        "red",
    ) | {"image_sha256": "0" * 64}
    requests.append(experiment.request_for_case(source, condition()))
    requests.extend(experiment.request_for_case(broken, condition(g)) for g in ("point", "only"))
    outcomes = run_fake(tmp_path)
    assert [r["status"] for r in outcomes] == ["passed", "failed", "failed"]
    assert len(predictor.calls) == len(predictor.images) == 1
    assert all("Frozen image changed" in r["failure_reason"] for r in outcomes[1:])


def test_geometry_failure_retains_row_without_calling_sam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = [
        experiment.request_for_case(case() | {"hrnet_eye_distance_px": float("nan")}, condition())
    ]
    predictor = fake_runtime(tmp_path, monkeypatch, requests)
    outcomes = run_fake(tmp_path)
    assert len(outcomes) == 1 and outcomes[0]["status"] == "failed"
    assert outcomes[0]["hrnet_eye_distance_px"] is None
    assert "Invalid frozen eye separation" in outcomes[0]["failure_reason"]
    assert not predictor.images and not predictor.calls


def test_decoder_failure_invalidates_cached_embedding_before_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict] = []
    predictor = fake_runtime(tmp_path, monkeypatch, requests, fail_first_decode=True)
    source = save_image(tmp_path, case())
    requests.extend(experiment.request_for_case(source, condition(g)) for g in ("point", "only"))
    outcomes = run_fake(tmp_path)
    assert [r["status"] for r in outcomes] == ["failed", "passed"]
    assert len(predictor.images) == len(predictor.calls) == 2
    assert predictor.calls[1]["point_coords"] is None
    assert predictor.calls[1]["mask_input"] is None
