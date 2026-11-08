"""Original-coordinate prediction export and official COCO segmentation evaluation."""

import contextlib
import copy
import io
from collections import Counter
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from animal_eye_metrology.coco import CocoDataset, ImageRecord
from animal_eye_metrology.errors import Phase1Error

MATCH_FIELDS = (
    "image_id",
    "prediction_id",
    "annotation_id",
    "category_id",
    "animal_category",
    "match_status",
    "score",
    "mask_iou",
    "dice",
    "confidence_threshold",
    "iou_threshold",
    "failure_type",
)


def animal_class_mapping(names: dict[int, str], dataset: CocoDataset) -> dict[int, int]:
    """Map names, never a numeric offset, between model indices and sparse COCO IDs."""
    mapping = {}
    for category in dataset.animal_categories.values():
        matches = [int(key) for key, name in names.items() if name == category.name]
        if len(matches) != 1:
            raise Phase1Error(f"Model must contain exactly one class named {category.name}.")
        mapping[matches[0]] = category.id
    return mapping


def normalize_predictions(
    image: ImageRecord,
    boxes: np.ndarray,
    masks: np.ndarray | None,
    class_mapping: dict[int, int],
) -> list[dict[str, Any]]:
    """Require native retina masks and xyxy boxes already restored to the original image."""
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise Phase1Error("Expected finite Nx6 original-coordinate xyxy/score/class predictions.")
    if not len(boxes):
        return []
    if masks is None or masks.shape != (len(boxes), image.height, image.width):
        raise Phase1Error("retina_masks must have original HxW dimensions and align with boxes.")
    if not np.isfinite(masks).all() or np.any((masks < 0) | (masks > 1)):
        raise Phase1Error("Invalid native mask values.")
    results = []
    for index, (x1, y1, x2, y2, score, class_id) in enumerate(boxes):
        if class_id != int(class_id) or not 0 <= score <= 1:
            raise Phase1Error("Invalid class index or confidence.")
        if int(class_id) not in class_mapping:
            continue
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise Phase1Error("Prediction box is not in the original image coordinate system.")
        binary = np.asarray(masks[index] >= 0.5, dtype=np.uint8)
        encoded = mask_utils.encode(np.asfortranarray(binary))
        results.append(
            {
                "image_id": image.id,
                "prediction_id": f"image_{image.id}_prediction_{index + 1:03d}",
                "category_id": class_mapping[int(class_id)],
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "segmentation": {
                    "size": [image.height, image.width],
                    "counts": encoded["counts"].decode("ascii"),
                },
            }
        )
    return results


def coco_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Omit bbox so COCO.loadRes derives area from the mask, rather than its box."""
    fields = ("image_id", "category_id", "segmentation", "score", "prediction_id")
    return [{key: prediction[key] for key in fields} for prediction in predictions]


def mask_overlap(prediction: np.ndarray, truth: np.ndarray) -> tuple[float | None, float | None]:
    if prediction.shape != truth.shape:
        raise Phase1Error("Masks must share the original image shape.")
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    intersection = int(np.logical_and(prediction, truth).sum())
    union = int(np.logical_or(prediction, truth).sum())
    total = int(prediction.sum()) + int(truth.sum())
    return (intersection / union if union else None, 2 * intersection / total if total else None)


def _evaluate(
    truth: COCO,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    category_ids: list[int],
    *,
    diagnostic_iou: float | None = None,
    iou_type: str = "segm",
) -> COCOeval:
    # COCO.loadRes indexes anns[0]; construct an indexed empty result explicitly.
    with contextlib.redirect_stdout(io.StringIO()):
        if predictions:
            exported = copy.deepcopy(coco_predictions(predictions))
            if iou_type == "bbox":
                for row, prediction in zip(exported, predictions, strict=True):
                    row["bbox"] = prediction["bbox"]
            detected = truth.loadRes(exported)
        else:
            detected = COCO()
            detected.dataset = {
                "images": [truth.imgs[key] for key in image_ids],
                "categories": [truth.cats[key] for key in category_ids],
                "annotations": [],
            }
            detected.createIndex()
        evaluator = COCOeval(truth, detected, iou_type)
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = category_ids
        evaluator.params.maxDets = [1, 10, 100]
        if diagnostic_iou is not None:
            evaluator.params.iouThrs = np.array([diagnostic_iou])
            evaluator.params.areaRng = [[0, 1e10]]
            evaluator.params.areaRngLbl = ["all"]
        evaluator.evaluate()
        evaluator.accumulate()
        if diagnostic_iou is None:
            evaluator.summarize()
    return evaluator


def evaluate_predictions(
    truth: COCO,
    predictions: list[dict[str, Any]],
    image_ids: list[int],
    category_ids: list[int],
    *,
    confidence: float = 0.5,
    iou: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """COCO AP/AR and a separate fixed-operating-point report from COCOeval matches."""
    # COCOeval annotates GT records with ignore flags and canonical RLEs in memory.
    truth = copy.deepcopy(truth)
    if len({p["prediction_id"] for p in predictions}) != len(predictions):
        raise Phase1Error("Duplicate prediction IDs.")
    if any(
        p["image_id"] not in image_ids or p["category_id"] not in category_ids for p in predictions
    ):
        raise Phase1Error("Prediction is outside the evaluation image/category population.")
    full = _evaluate(truth, predictions, image_ids, category_ids)
    names = (
        "mask_ap",
        "mask_ap50",
        "mask_ap75",
        "mask_ap_small",
        "mask_ap_medium",
        "mask_ap_large",
        "mask_ar1",
        "mask_ar10",
        "mask_ar100",
        "mask_ar_small",
        "mask_ar_medium",
        "mask_ar_large",
    )
    metrics: dict[str, Any] = {
        "scope": "manually_selected_POC_not_unbiased_generalization",
        "score_units": "0_to_1; null means undefined",
        "image_ids": image_ids,
        "category_ids": category_ids,
        "coco_segm": {
            name: float(value) if value >= 0 else None
            for name, value in zip(names, full.stats, strict=True)
        },
        "per_category": {},
    }
    for index, category in enumerate(full.params.catIds):
        values = full.eval["precision"][:, :, index, 0, -1]
        valid = values[values >= 0]
        metrics["per_category"][str(category)] = {
            "category": truth.cats[category]["name"],
            "mask_ap": float(valid.mean()) if valid.size else None,
        }
    visible = [p for p in predictions if p["score"] >= confidence]
    diagnostic = _evaluate(truth, visible, image_ids, category_ids, diagnostic_iou=iou)
    box_diagnostic = _evaluate(
        truth, visible, image_ids, category_ids, diagnostic_iou=iou, iou_type="bbox"
    )
    box_counts: Counter[str] = Counter()
    box_matched_gt = set()
    for evaluated in box_diagnostic.evalImgs:
        if evaluated is None:
            continue
        for index, gt_id in enumerate(evaluated["dtMatches"][0]):
            if not evaluated["dtIgnore"][0, index]:
                box_counts["tp" if gt_id else "fp"] += 1
                if gt_id:
                    box_matched_gt.add(int(gt_id))
        box_counts["fn"] += sum(
            not ignored and not matched
            for ignored, matched in zip(
                evaluated["gtIgnore"], evaluated["gtMatches"][0], strict=True
            )
        )
    rows = []
    for evaluated in diagnostic.evalImgs:
        if evaluated is None:
            continue
        image_id, category_id = evaluated["image_id"], evaluated["category_id"]
        base = {
            "image_id": image_id,
            "category_id": category_id,
            "animal_category": truth.cats[category_id]["name"],
            "confidence_threshold": confidence,
            "iou_threshold": iou,
        }
        for index, detected_id in enumerate(evaluated["dtIds"]):
            detected = diagnostic.cocoDt.anns[detected_id]
            gt_id = int(evaluated["dtMatches"][0, index])
            ignored = bool(evaluated["dtIgnore"][0, index])
            overlap = (None, None)
            if gt_id and not ignored:
                overlap = mask_overlap(
                    diagnostic.cocoDt.annToMask(detected), truth.annToMask(truth.anns[gt_id])
                )
            rows.append(
                base
                | {
                    "prediction_id": detected["prediction_id"],
                    "annotation_id": gt_id or "",
                    "match_status": "ignored_prediction" if ignored else "TP" if gt_id else "FP",
                    "score": detected["score"],
                    "mask_iou": overlap[0],
                    "dice": overlap[1],
                }
            )
        for index, gt_id in enumerate(evaluated["gtIds"]):
            ignored = bool(evaluated["gtIgnore"][index])
            if ignored or not evaluated["gtMatches"][0, index]:
                rows.append(
                    base
                    | {
                        "prediction_id": "",
                        "annotation_id": gt_id,
                        "match_status": "ignored_ground_truth" if ignored else "FN",
                        "score": None,
                        "mask_iou": None,
                        "dice": None,
                    }
                )
    for prediction in predictions:
        if prediction["score"] < confidence:
            rows.append(
                {
                    "image_id": prediction["image_id"],
                    "prediction_id": prediction["prediction_id"],
                    "annotation_id": "",
                    "category_id": prediction["category_id"],
                    "animal_category": truth.cats[prediction["category_id"]]["name"],
                    "match_status": "below_confidence",
                    "score": prediction["score"],
                    "mask_iou": None,
                    "dice": None,
                    "confidence_threshold": confidence,
                    "iou_threshold": iou,
                }
            )
    rows.sort(
        key=lambda row: (row["image_id"], str(row["prediction_id"]), str(row["annotation_id"]))
    )
    for row in rows:
        if row["match_status"] == "FN":
            row["failure_type"] = (
                "bbox_matched_but_mask_unmatched"
                if row["annotation_id"] in box_matched_gt
                else "not_recovered_by_bbox_or_mask"
            )
        else:
            row["failure_type"] = ""

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(row["match_status"] for row in subset)
        tp, fp, fn = (counts[key] for key in ("TP", "FP", "FN"))
        matched = [row for row in subset if row["match_status"] == "TP"]
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "matched_mean_iou": float(np.mean([row["mask_iou"] for row in matched]))
            if matched
            else None,
            "matched_mean_dice": float(np.mean([row["dice"] for row in matched]))
            if matched
            else None,
            "ignored_predictions": counts["ignored_prediction"],
            "ignored_ground_truth": counts["ignored_ground_truth"],
            "below_confidence": counts["below_confidence"],
        }

    metrics["diagnostic"] = {
        "matching": "COCOeval greedy at fixed operating point",
        "confidence": confidence,
        "mask_iou": iou,
        **summarize(rows),
    }
    metrics["detection_diagnostic"] = {
        "matching": "COCOeval bbox at same confidence and IoU operating point",
        "confidence": confidence,
        "bbox_iou": iou,
        **{key: int(box_counts[key]) for key in ("tp", "fp", "fn")},
        "mask_misses_with_bbox_match": sum(
            row["failure_type"] == "bbox_matched_but_mask_unmatched" for row in rows
        ),
        "interpretation": "bbox recovery vs mask recovery; not proof of a unique failure cause",
    }
    metrics["per_image"] = {
        str(key): summarize([row for row in rows if row["image_id"] == key]) for key in image_ids
    }
    return metrics, rows
