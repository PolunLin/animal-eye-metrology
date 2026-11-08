"""Deterministic descriptive statistics; no eye or measurement eligibility inference."""

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from animal_eye_metrology.coco import Annotation, CocoDataset

GROUPS = ("multi_any", "multi_same_category", "multi_different_categories")
CATEGORY_FIELDS = (
    "category_id",
    "category_name",
    "total_annotations",
    "crowd_annotations",
    "eligible_instances",
    "images_with_annotations",
    "images_with_eligible_instances",
    "images_with_at_least_two_eligible_instances",
)
INSTANCE_FIELDS = (
    "image_id",
    "annotation_id",
    "category_id",
    "category_name",
    "iscrowd",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "bbox_area_px2",
    "segmentation_area_px2",
    "bbox_area_fraction",
    "segmentation_area_fraction",
)
AREA_FIELDS = (
    "category_id",
    "category_name",
    "metric",
    "count",
    "min",
    "p25",
    "median",
    "p75",
    "max",
    "mean",
)
CANDIDATE_FIELDS = (
    "image_id",
    "file_name",
    "width",
    "height",
    "animal_count",
    "category_ids",
    "category_names",
    "category_counts",
    "repeated_category_ids",
    "annotation_ids",
)
REVIEW_MANUAL_FIELDS = (
    "left_eye_visibility",
    "right_eye_visibility",
    "head_pose",
    "occlusion",
    "eye_resolution",
    "measurement_eligibility",
    "exclusion_reason",
    "reviewer",
    "review_protocol_version",
    "notes",
)
REVIEW_FIELDS = (
    "image_id",
    "file_name",
    "annotation_id",
    "category_id",
    "animal_category",
    "animal_count",
    "category_animal_count",
    *REVIEW_MANUAL_FIELDS,
)


@dataclass(frozen=True)
class AnalysisResult:
    categories: list[dict[str, Any]]
    instances: list[dict[str, Any]]
    area_summary: list[dict[str, Any]]
    candidates: dict[str, list[dict[str, Any]]]
    review: list[dict[str, Any]]
    eligible_by_image: dict[int, list[Annotation]]

    def summary(self) -> dict[str, Any]:
        return {
            "animal_categories": len(self.categories),
            "animal_annotations": len(self.instances),
            "crowd_animal_annotations": sum(row["crowd_annotations"] for row in self.categories),
            "eligible_animal_instances": sum(row["eligible_instances"] for row in self.categories),
            "candidate_images": {key: len(rows) for key, rows in self.candidates.items()},
            "manual_review_rows": len(self.review),
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return dict.fromkeys(("min", "p25", "median", "p75", "max", "mean")) | {"count": 0}

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def analyze(dataset: CocoDataset) -> AnalysisResult:
    animals = dataset.animal_categories
    by_category: dict[int, list[Annotation]] = defaultdict(list)
    eligible_by_image: dict[int, list[Annotation]] = defaultdict(list)
    instances = []
    for ann in sorted(dataset.annotations.values(), key=lambda item: (item.image_id, item.id)):
        if ann.category_id not in animals:
            continue
        by_category[ann.category_id].append(ann)
        if ann.iscrowd == 0:
            eligible_by_image[ann.image_id].append(ann)
        image = dataset.images[ann.image_id]
        image_area = image.width * image.height
        x, y, width, height = ann.bbox
        instances.append(
            {
                "image_id": ann.image_id,
                "annotation_id": ann.id,
                "category_id": ann.category_id,
                "category_name": animals[ann.category_id].name,
                "iscrowd": ann.iscrowd,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
                "bbox_area_px2": width * height,
                "segmentation_area_px2": ann.area,
                "bbox_area_fraction": width * height / image_area,
                "segmentation_area_fraction": ann.area / image_area,
            }
        )
    categories, area_summary = [], []
    eligible_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in instances:
        if row["iscrowd"] == 0:
            eligible_rows[row["category_id"]].append(row)
    for key, category in animals.items():
        all_annotations = by_category[key]
        eligible = [ann for ann in all_annotations if ann.iscrowd == 0]
        per_image = Counter(ann.image_id for ann in eligible)
        categories.append(
            {
                "category_id": key,
                "category_name": category.name,
                "total_annotations": len(all_annotations),
                "crowd_annotations": len(all_annotations) - len(eligible),
                "eligible_instances": len(eligible),
                "images_with_annotations": len({ann.image_id for ann in all_annotations}),
                "images_with_eligible_instances": len(per_image),
                "images_with_at_least_two_eligible_instances": sum(
                    n >= 2 for n in per_image.values()
                ),
            }
        )
        for metric in (
            "bbox_area_px2",
            "segmentation_area_px2",
            "bbox_area_fraction",
            "segmentation_area_fraction",
        ):
            area_summary.append(
                {
                    "category_id": key,
                    "category_name": category.name,
                    "metric": metric,
                    **_distribution([row[metric] for row in eligible_rows[key]]),
                }
            )
    candidates: dict[str, list[dict[str, Any]]] = {key: [] for key in GROUPS}
    review = []
    for image_id, annotations in sorted(eligible_by_image.items()):
        if len(annotations) < 2:
            continue
        image = dataset.images[image_id]
        counts = Counter(ann.category_id for ann in annotations)
        repeated = sorted(key for key, count in counts.items() if count >= 2)
        row = {
            "image_id": image_id,
            "file_name": image.file_name,
            "width": image.width,
            "height": image.height,
            "animal_count": len(annotations),
            "category_ids": _json(sorted(counts)),
            "category_names": _json([animals[key].name for key in sorted(counts)]),
            "category_counts": _json({str(key): counts[key] for key in sorted(counts)}),
            "repeated_category_ids": _json(repeated),
            "annotation_ids": _json([ann.id for ann in annotations]),
        }
        candidates["multi_any"].append(row)
        if repeated:
            candidates["multi_same_category"].append(row)
        if len(counts) >= 2:
            candidates["multi_different_categories"].append(row)
        for ann in annotations:
            review.append(
                {
                    "image_id": image_id,
                    "file_name": image.file_name,
                    "annotation_id": ann.id,
                    "category_id": ann.category_id,
                    "animal_category": animals[ann.category_id].name,
                    "animal_count": len(annotations),
                    "category_animal_count": counts[ann.category_id],
                    **dict.fromkeys(REVIEW_MANUAL_FIELDS, ""),
                }
            )
    return AnalysisResult(
        categories, instances, area_summary, candidates, review, dict(eligible_by_image)
    )
