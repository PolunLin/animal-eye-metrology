"""Portable evaluation integration tests use invented masks, never model inference."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL.Image")
pytest.importorskip("pycocotools.mask")

from animal_eye_metrology.core.verification import sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reviewed_evaluation_cli", ROOT / "scripts/evaluate_reviewed.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def invented_records() -> tuple[dict, list[dict], dict]:
    image = {
        "image_id": 7,
        "file_name": "invented.jpg",
        "image_sha256": "a" * 64,
        "height": 6,
        "width": 8,
        "batch": "synthetic",
        "targets": [],
    }
    cases, animals, eyes = [], [], []
    for annotation_id, prediction_id, x in ((10, "p1", 0), (20, "p2", 4)):
        animal = np.zeros((6, 8), dtype=np.uint8)
        animal[:, x : x + 4] = 1
        image["targets"].append(
            {
                "annotation_id": str(annotation_id),
                "animal_category": "cat",
                "coco_bbox_xywh": json.dumps([x, 0, 4, 6]),
                "segmentation": evaluation.encode(animal),
            }
        )
        animals.append(
            {
                "prediction_id": prediction_id,
                "category": "cat",
                "accepted": True,
                "mask_status": "completed",
                "bbox_xyxy": [x, 0, x + 4, 6],
                "mask_rle": evaluation.encode(animal),
            }
        )
        for side, y in (("left", 1), ("right", 3)):
            eye = np.zeros((6, 8), dtype=np.uint8)
            eye[y, x + 1] = 1
            cases.append(
                {
                    "image_id": 7,
                    "annotation_id": annotation_id,
                    "eye_side": side,
                    "annotation_version": 1,
                    "eligible": True,
                    "human_reviewed": True,
                    "reference_mask_rle": evaluation.encode(eye),
                }
            )
            eyes.append(
                {
                    "prediction_id": prediction_id,
                    "eye_side": side,
                    "condition_id": "A_r100",
                    "segmentation_status": "completed",
                    "failure_reason": "",
                    "mask_rle": evaluation.encode(eye),
                }
            )
    prediction = {
        "image_id": 7,
        "image_sha256": image["image_sha256"],
        "baseline": "A_r100",
        "run_id": "synthetic",
        "status": "completed",
        "animals": animals,
        "eyes": eyes,
    }
    return image, cases, prediction


def test_lossless_rle_replay_and_full_expected_pair_scope(tmp_path: Path) -> None:
    image, cases, prediction = invented_records()
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    assert np.array_equal(evaluation.decode(evaluation.encode(mask), 2, 3), mask)
    reference = {"images": [image], "cases": cases, "summary": {"eligible_eyes": 4}}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps({"reference_sha256": sha256_file(reference_path), "images": [prediction]})
    )
    out = tmp_path / "scores"
    evaluation.score(
        argparse.Namespace(reference=reference_path, predictions=prediction_path, output=out)
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["eyes_expected"] == summary["centroid_defined"] == 4
    assert summary["mean_iou_all_targets_failure_as_zero"] == 1
    assert summary["binocular_expected"] == summary["binocular_defined"] == 2
    assert summary["right_eye_pair_expected"] == summary["right_eye_pair_defined"] == 1
    assert summary["binocular_mae_px"] == summary["right_eye_pair_mae_px"] == 0
    assert (out / "reviews.csv").is_file() and (out / "matches.csv").is_file()


def test_unqualified_animals_still_compete_for_identity() -> None:
    image, cases, prediction = invented_records()
    # Only animal 10 has eligible eyes, but animal 20 has an identical silhouette.
    image["targets"][1]["segmentation"] = image["targets"][0]["segmentation"]
    image["targets"][1]["coco_bbox_xywh"] = image["targets"][0]["coco_bbox_xywh"]
    for case in cases[2:]:
        case["eligible"] = False
    prediction["animals"] = prediction["animals"][:1]
    eyes, pairs, audit = evaluation.image_scores(image, cases, prediction)
    assert len(audit) == 2 and not any(row["matched"] for row in audit)
    assert len(eyes) == 2 and all(row["prediction_status"] == "inference_failed" for row in eyes)
    assert len(pairs) == 1 and pairs[0]["predicted_distance_px"] is None


def test_missing_right_eye_preserves_both_distance_denominators() -> None:
    image, cases, prediction = invented_records()
    eye = prediction["eyes"][1]
    eye.update(mask_rle=None, segmentation_status="failed", failure_reason="pose_failed")
    eyes, pairs, _ = evaluation.image_scores(image, cases, prediction)
    summary = evaluation.summarize(eyes, pairs)
    assert summary["eyes_expected"] == 4 and summary["centroid_defined"] == 3
    assert summary["mean_iou_all_targets_failure_as_zero"] == 0.75
    assert summary["binocular_expected"] == 2 and summary["binocular_defined"] == 1
    assert summary["right_eye_pair_expected"] == 1 and summary["right_eye_pair_defined"] == 0
    unavailable = [p for p in pairs if p["predicted_distance_px"] is None]
    assert len(unavailable) == 2
    assert all(p["absolute_error_px"] is None and p["failure_reason"] for p in unavailable)


def test_eye_channels_are_not_swapped_to_improve_overlap() -> None:
    image, cases, prediction = invented_records()
    left, right = prediction["eyes"][:2]
    left["mask_rle"], right["mask_rle"] = right["mask_rle"], left["mask_rle"]
    eyes, _, _ = evaluation.image_scores(image, cases, prediction)
    swapped = [row for row in eyes if row["annotation_id"] == 10]
    assert len(swapped) == 2 and all(row["iou"] == 0 for row in swapped)
    assert all(row["centroid_error_px"] == 2 for row in swapped)


def test_duplicate_prediction_eyes_are_rejected_within_the_same_condition() -> None:
    image, cases, prediction = invented_records()
    duplicate = copy.deepcopy(prediction["eyes"][0])
    prediction["eyes"].append(duplicate)
    with pytest.raises(ValueError, match="Duplicate prediction eye"):
        evaluation.image_scores(image, cases, prediction)
    # A different condition does not replace the A_r100 eye or create ambiguity.
    duplicate["condition_id"] = "A_r050"
    eyes, _, _ = evaluation.image_scores(image, cases, prediction)
    assert len(eyes) == 4 and all(row["iou"] == 1 for row in eyes)


@pytest.mark.parametrize("duplicate_kind", ["image", "eligible_eye", "excluded_eye"])
def test_duplicate_reference_identities_are_rejected(tmp_path: Path, duplicate_kind: str) -> None:
    image, cases, prediction = invented_records()
    reference = {"images": [image], "cases": cases, "summary": {}}
    if duplicate_kind == "image":
        reference["images"].append(copy.deepcopy(image))
    else:
        duplicate = copy.deepcopy(cases[0])
        if duplicate_kind == "excluded_eye":
            duplicate["eligible"] = False
        reference["cases"].append(duplicate)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps({"reference_sha256": sha256_file(reference_path), "images": [prediction]})
    )
    out = tmp_path / "scores"
    with pytest.raises(ValueError, match="Duplicate reference"):
        evaluation.score(
            argparse.Namespace(reference=reference_path, predictions=prediction_path, output=out)
        )
    assert not out.exists()


def test_image_scoring_rejects_duplicate_and_foreign_cases() -> None:
    image, cases, prediction = invented_records()
    with pytest.raises(ValueError, match="Duplicate reference eye"):
        evaluation.image_scores(image, cases + [copy.deepcopy(cases[0])], prediction)
    foreign = copy.deepcopy(cases)
    foreign[0]["image_id"] = 99
    with pytest.raises(ValueError, match="image/animal scope"):
        evaluation.image_scores(image, foreign, prediction)


@pytest.mark.parametrize(
    "change", ["snapshot_hash", "image_hash", "image_id", "duplicate", "baseline"]
)
def test_replay_rejects_identity_or_snapshot_mismatch(tmp_path: Path, change: str) -> None:
    image, cases, prediction = invented_records()
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps({"images": [image], "cases": cases, "summary": {}}))
    predictions = {"reference_sha256": sha256_file(reference_path), "images": [prediction]}
    if change == "snapshot_hash":
        predictions["reference_sha256"] = "b" * 64
    elif change == "image_hash":
        prediction["image_sha256"] = "b" * 64
    elif change == "image_id":
        prediction["image_id"] = 999
    elif change == "duplicate":
        predictions["images"].append(copy.deepcopy(prediction))
    else:
        prediction["baseline"] = "A_r050"
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(predictions))
    out = tmp_path / "scores"
    with pytest.raises(ValueError, match="reference snapshot|identities differ|baseline mismatch"):
        evaluation.score(
            argparse.Namespace(reference=reference_path, predictions=prediction_path, output=out)
        )
    assert not out.exists()


def test_inference_rejects_changed_reference_source_before_running_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "annotations.csv"
    source.write_text("original")
    reference = {"source_artifacts_sha256": {source.name: sha256_file(source)}}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    source.write_text("changed")
    monkeypatch.setattr(evaluation, "load_config", lambda path: {})
    monkeypatch.setattr(evaluation, "preflight", lambda config: [{"status": "ready"}])
    monkeypatch.setattr(evaluation, "protected_hashes", lambda root, config: {})
    monkeypatch.setattr(
        evaluation,
        "run_inference",
        lambda *args: pytest.fail("A changed reference must be rejected before model execution."),
    )
    out = tmp_path / "inference"
    with pytest.raises(ValueError, match="Reference source changed"):
        evaluation.infer(
            argparse.Namespace(
                root=tmp_path,
                reference=reference_path,
                config=tmp_path / "config.toml",
                output=out,
            )
        )
    assert not out.exists()


def test_published_portable_results_reproduce_every_summary_value(tmp_path: Path) -> None:
    public = ROOT / "examples/reviewed_validation"
    reference = public / "reference.json"
    predictions = public / "predictions.json"
    expected = json.loads((public / "summary.json").read_text())
    output = tmp_path / "replayed"
    evaluation.score(
        argparse.Namespace(reference=reference, predictions=predictions, output=output)
    )
    assert json.loads((output / "summary.json").read_text()) == expected
