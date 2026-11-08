import copy
import json
from pathlib import Path

import pytest

from animal_eye_metrology.analysis import REVIEW_MANUAL_FIELDS, analyze
from animal_eye_metrology.coco import load_coco
from animal_eye_metrology.review_sample import sample_images


def test_counts_and_overlapping_candidate_groups(local_config: Path) -> None:
    dataset = load_coco(local_config.parent / "annotations.json")
    result = analyze(dataset)
    assert result.summary() == {
        "animal_categories": 10,
        "animal_annotations": 12,
        "crowd_animal_annotations": 3,
        "eligible_animal_instances": 9,
        "candidate_images": {
            "multi_any": 3,
            "multi_same_category": 2,
            "multi_different_categories": 2,
        },
        "manual_review_rows": 7,
    }
    assert [row["image_id"] for row in result.candidates["multi_any"]] == [10, 20, 50]
    assert [row["image_id"] for row in result.candidates["multi_same_category"]] == [10, 50]
    assert [row["image_id"] for row in result.candidates["multi_different_categories"]] == [20, 50]
    assert result.categories[0] == {
        "category_id": 100,
        "category_name": "test_species_0",
        "total_annotations": 6,
        "crowd_annotations": 0,
        "eligible_instances": 6,
        "images_with_annotations": 4,
        "images_with_eligible_instances": 4,
        "images_with_at_least_two_eligible_instances": 2,
    }
    assert result.categories[1]["eligible_instances"] == 3
    assert result.categories[1]["images_with_at_least_two_eligible_instances"] == 0
    assert result.categories[2]["total_annotations"] == 2
    assert result.categories[2]["images_with_eligible_instances"] == 0
    assert result.categories[3]["total_annotations"] == 0
    assert json.loads(result.candidates["multi_same_category"][-1]["repeated_category_ids"]) == [
        100
    ]
    assert json.loads(result.candidates["multi_any"][-1]["annotation_ids"]) == [5001, 5002, 5003]


def test_bbox_area_is_distinct_from_official_segmentation_area(local_config: Path) -> None:
    result = analyze(load_coco(local_config.parent / "annotations.json"))
    assert result.instances[0]["bbox_area_px2"] == 200
    assert result.instances[0]["segmentation_area_px2"] == 90
    assert result.instances[0]["bbox_area_fraction"] == pytest.approx(0.025)
    assert result.instances[0]["segmentation_area_fraction"] == pytest.approx(0.01125)
    bbox_summary = result.area_summary[0]
    # Species 0 has five 200-pixel boxes and one 400-pixel box, including single-animal images.
    assert bbox_summary["count"] == 6
    assert bbox_summary["min"] == bbox_summary["median"] == bbox_summary["p75"] == 200
    assert bbox_summary["max"] == 400
    assert bbox_summary["mean"] == pytest.approx(1400 / 6)
    empty = next(row for row in result.area_summary if row["category_id"] == 114)
    assert empty["count"] == 0
    assert all(empty[name] is None for name in ("min", "p25", "median", "p75", "max", "mean"))


def test_linear_quantiles_and_singleton_groups(local_config: Path) -> None:
    path = local_config.parent / "annotations.json"
    data = json.loads(path.read_text())
    data["annotations"] = data["annotations"][:2]
    path.write_text(json.dumps(data))
    result = analyze(load_coco(path))
    summary = result.area_summary[0]
    assert (summary["p25"], summary["median"], summary["p75"]) == (250, 300, 350)
    data["annotations"] = data["annotations"][:1]
    path.write_text(json.dumps(data))
    singleton = analyze(load_coco(path)).area_summary[0]
    assert all(singleton[name] == 200 for name in ("min", "p25", "median", "p75", "max", "mean"))


def test_manual_fields_are_blank_and_rows_are_per_animal(local_config: Path) -> None:
    result = analyze(load_coco(local_config.parent / "annotations.json"))
    assert [row["annotation_id"] for row in result.review] == [
        1001,
        1002,
        2001,
        2002,
        5001,
        5002,
        5003,
    ]
    assert all(row[field] == "" for row in result.review for field in REVIEW_MANUAL_FIELDS)
    assert result.review[-1]["animal_count"] == 3
    assert result.review[-1]["category_animal_count"] == 1


def test_ordering_is_stable_and_source_annotations_are_unchanged(local_config: Path) -> None:
    path = local_config.parent / "annotations.json"
    dataset = load_coco(path)
    before = copy.deepcopy(dataset)
    expected = analyze(dataset)
    assert dataset == before
    data = json.loads(path.read_text())
    for field in ("categories", "images", "annotations"):
        data[field].reverse()
    path.write_text(json.dumps(data))
    assert analyze(load_coco(path)) == expected


def test_empty_annotations_yield_headers_and_no_candidates(local_config: Path) -> None:
    path = local_config.parent / "annotations.json"
    data = json.loads(path.read_text())
    data["annotations"] = []
    path.write_text(json.dumps(data))
    result = analyze(load_coco(path))
    assert len(result.categories) == 10
    assert not result.review
    assert all(not rows for rows in result.candidates.values())


def test_review_sampling_is_repeatable_without_replacement_and_order_bias() -> None:
    candidates = [{"image_id": key, "category_counts": '{"24":2}'} for key in range(40)]
    populations, draws = sample_images(candidates, {"zebra": 24})
    assert sample_images(list(reversed(candidates)), {"zebra": 24}) == (populations, draws)
    assert populations["zebra"] == list(range(40))
    assert len(draws["zebra"]) == len(set(draws["zebra"])) == 15
    assert set(draws["zebra"]) <= set(populations["zebra"])
    assert draws != sample_images(candidates, {"zebra": 24}, seed=43)[1]


def test_review_sampling_requires_two_target_instances_and_uses_small_population() -> None:
    candidates = [
        {"image_id": 1, "category_counts": '{"24":2,"25":1}'},
        {"image_id": 2, "category_counts": '{"24":1,"25":2}'},
        {"image_id": 3, "category_counts": '{"24":3,"25":2}'},
    ]
    populations, draws = sample_images(candidates, {"zebra": 24, "giraffe": 25, "dog": 18})
    assert populations == {"zebra": [1, 3], "giraffe": [2, 3], "dog": []}
    assert {key: sorted(value) for key, value in draws.items()} == populations
