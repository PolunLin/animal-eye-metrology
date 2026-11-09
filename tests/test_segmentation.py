"""Invented masks only: category identity, geometry, export, and official matching."""

import copy
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pycocotools")

from pycocotools import mask as mask_utils  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

from animal_eye_metrology.core.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.data.coco import Category, CocoDataset, ImageRecord  # noqa: E402
from animal_eye_metrology.models.segmentation import (  # noqa: E402
    animal_class_mapping,
    clip_box_roundoff,
    coco_predictions,
    evaluate_predictions,
    mask_overlap,
    normalize_predictions,
)


def rle(mask):
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(mask.shape), "counts": encoded["counts"].decode("ascii")}


def test_original_geometry_category_mapping_and_coco_roundtrip() -> None:
    dataset = CocoDataset(
        {100: Category(100, "invented_a", "animal"), 107: Category(107, "invented_b", "animal")},
        {},
        {},
        "",
        {},
    )
    mapping = animal_class_mapping({0: "other", 1: "invented_b", 3: "invented_a"}, dataset)
    assert mapping == {3: 100, 1: 107}
    with pytest.raises(Phase1Error, match="exactly one"):
        animal_class_mapping({0: "invented_a"}, dataset)
    image = ImageRecord(10, "invented.png", 9, 5)
    masks = np.zeros((2, 5, 9), dtype=np.uint8)
    masks[0, 1:4, 2:7] = 1
    boxes = np.array([[2, 1, 7, 4, 0.9, 3], [1, 1, 3, 3, 0.8, 0]])
    normalized = normalize_predictions(image, boxes, masks, mapping)
    assert len(normalized) == 1  # Non-animal model class is excluded.
    assert normalized[0]["bbox"] == [2, 1, 5, 3]
    exported = json.loads(json.dumps(coco_predictions(normalized)))
    assert "bbox" not in exported[0]  # loadRes must compute area from segmentation.
    assert exported[0]["category_id"] == 100
    assert exported[0]["image_id"] == 10
    np.testing.assert_array_equal(mask_utils.decode(exported[0]["segmentation"]), masks[0])
    assert normalize_predictions(image, np.empty((0, 6)), None, mapping) == []
    with pytest.raises(Phase1Error, match="original HxW"):
        normalize_predictions(image, boxes, np.zeros((2, 9, 5)), mapping)
    boxes[0, 2] = 100  # Display/letterbox coordinates must never be silently accepted.
    with pytest.raises(Phase1Error, match="original image coordinate"):
        normalize_predictions(image, boxes, masks, mapping)


def test_mask_iou_and_dice_known_geometry_and_undefined_empty() -> None:
    a = np.array([[1, 1, 1], [1, 0, 0]], dtype=np.uint8)
    b = np.array([[0, 1, 1], [0, 1, 1]], dtype=np.uint8)
    assert mask_overlap(a, b) == pytest.approx((2 / 6, 4 / 8))
    assert mask_overlap(a, a) == (1.0, 1.0)
    assert mask_overlap(a * 0, b * 0) == (None, None)
    with pytest.raises(Phase1Error, match="same|share"):
        mask_overlap(a, b.T)


@pytest.fixture
def invented_evaluation():
    a = np.zeros((6, 10), dtype=np.uint8)
    b, crowd = a.copy(), a.copy()
    a[1:3, 1:3] = 1
    b[1:3, 4:6] = 1
    crowd[4:6, 7:10] = 1
    categories = [{"id": 100, "name": "invented_a"}, {"id": 107, "name": "invented_b"}]
    anns = [
        {
            "id": key,
            "image_id": 10,
            "category_id": 100,
            "segmentation": rle(mask),
            "bbox": list(mask_utils.toBbox(rle(mask))),
            "area": int(mask.sum()),
            "iscrowd": iscrowd,
        }
        for key, mask, iscrowd in ((9001, a, 0), (9002, b, 0), (9003, crowd, 1))
    ]
    truth = COCO()
    truth.dataset = {
        "info": {"description": "SYNTHETIC, NOT RESEARCH RESULTS"},
        "images": [{"id": key, "height": 6, "width": 10} for key in (10, 20)],
        "categories": categories,
        "annotations": anns,
    }
    truth.createIndex()

    def predicted(key, mask, score, category=100, image_id=10):
        return {
            "prediction_id": key,
            "image_id": image_id,
            "category_id": category,
            "segmentation": rle(mask),
            "bbox": list(mask_utils.toBbox(rle(mask))),
            "score": score,
        }

    predictions = [
        predicted("a", a, 0.95),
        predicted("duplicate", a, 0.9),
        predicted("wrong_class", b, 0.9, 107),
        predicted("crowd1", crowd, 0.8),
        predicted("crowd2", crowd, 0.7),
        predicted("low_score", b, 0.1),
        predicted("negative_image", a, 0.9, image_id=20),
    ]
    return truth, predictions


def test_official_matching_duplicates_wrong_classes_crowd_negatives_and_score_sweep(
    invented_evaluation,
) -> None:
    truth, predictions = invented_evaluation
    before = copy.deepcopy(truth.dataset)
    metrics, rows = evaluate_predictions(truth, predictions, [10, 20], [100, 107])
    point = metrics["diagnostic"]
    assert (point["tp"], point["fp"], point["fn"]) == (1, 3, 1)
    assert point["precision"] == 0.25
    assert point["recall"] == 0.5
    assert point["matched_mean_iou"] == point["matched_mean_dice"] == 1
    assert point["ignored_predictions"] == 2  # Crowd allows multiple ignored overlaps.
    assert point["ignored_ground_truth"] == 1
    assert point["below_confidence"] == 1
    assert metrics["coco_segm"]["mask_ar100"] == 1  # AP path retains low-score true positive.
    assert metrics["per_category"]["107"]["mask_ap"] is None  # No GT in this category.
    assert [r["annotation_id"] for r in rows if r["match_status"] == "TP"] == [9001]
    assert next(r for r in rows if r["prediction_id"] == "duplicate")["match_status"] == "FP"
    assert next(r for r in rows if r["match_status"] == "FN")["annotation_id"] == 9002
    assert truth.dataset == before


def test_empty_predictions_keep_false_negatives_and_undefined_precision(
    invented_evaluation,
) -> None:
    truth, _ = invented_evaluation
    metrics, rows = evaluate_predictions(truth, [], [10, 20], [100, 107])
    assert metrics["coco_segm"]["mask_ap"] == 0
    assert metrics["diagnostic"]["tp"] == metrics["diagnostic"]["fp"] == 0
    assert metrics["diagnostic"]["fn"] == 2
    assert metrics["diagnostic"]["precision"] is None
    assert metrics["diagnostic"]["recall"] == 0
    assert metrics["diagnostic"]["matched_mean_dice"] is None
    assert sum(row["match_status"] == "FN" for row in rows) == 2
    metrics, _ = evaluate_predictions(truth, [], [20], [107])
    assert metrics["coco_segm"]["mask_ap"] is None
    assert metrics["diagnostic"]["recall"] is None


def test_bbox_recovery_does_not_hide_mask_quality_failure(invented_evaluation) -> None:
    truth, predictions = invented_evaluation
    prediction = copy.deepcopy(predictions[0])
    prediction["segmentation"] = rle(np.zeros((6, 10), dtype=np.uint8))
    metrics, rows = evaluate_predictions(truth, [prediction], [10, 20], [100, 107])
    assert metrics["diagnostic"]["tp"] == 0
    assert metrics["detection_diagnostic"]["tp"] == 1
    assert metrics["detection_diagnostic"]["mask_misses_with_bbox_match"] == 1
    a = next(row for row in rows if row["annotation_id"] == 9001)
    b = next(row for row in rows if row["annotation_id"] == 9002)
    assert a["failure_type"] == "bbox_matched_but_mask_unmatched"
    assert b["failure_type"] == "not_recovered_by_bbox_or_mask"


def test_box_roundoff_allows_float_resize_error_but_rejects_wrong_coordinates() -> None:
    import numpy as np

    boxes = np.array([[1, 2, 640, 426.0000305175781, 0.9, 20]])
    clipped, count = clip_box_roundoff(boxes, 640, 426)
    assert count == 1
    assert clipped[0, 3] == 426
    assert boxes[0, 3] > 426  # Never modify the model array in place.
    np.testing.assert_array_equal(clipped[0, [0, 1, 2, 4, 5]], boxes[0, [0, 1, 2, 4, 5]])
    boxes[0, 3] = 426.1
    with pytest.raises(Phase1Error, match="roundoff"):
        clip_box_roundoff(boxes, 640, 426)
