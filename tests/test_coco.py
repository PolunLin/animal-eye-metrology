import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from animal_eye_metrology.coco import load_coco
from animal_eye_metrology.errors import Phase1Error


def test_metadata_categories_and_input_digest(local_config: Path) -> None:
    path = local_config.parent / "annotations.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    data = load_coco(path, expected_sha256=digest)
    assert len(data.animal_categories) == 10
    assert list(data.animal_categories) == [100 + 7 * index for index in range(10)]
    assert data.source_sha256 == digest
    with pytest.raises(Phase1Error, match="SHA-256"):
        load_coco(path, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image_id", 987654, "unknown image"),
        ("category_id", 987654, "unknown image or category"),
        ("id", True, "integer"),
        ("iscrowd", 2, "0 or 1"),
        ("iscrowd", True, "integer"),
        ("bbox", [0, 0, -1, 1], "nonnegative"),
        ("bbox", [0, 0, 1], "bbox"),
        ("bbox", [0, 0, float("inf"), 1], "finite"),
        ("area", -1, "nonnegative"),
        ("area", float("nan"), "finite"),
        ("segmentation", [], "polygons or RLE"),
        ("segmentation", [[0, 0, 1, 1]], "coordinate pairs"),
        ("segmentation", {"size": [1, 1], "counts": [1]}, "RLE size"),
        ("segmentation", {"size": [80, 100], "counts": [3]}, "cover the image"),
        ("segmentation", {"size": [80, 100], "counts": "!"}, "RLE characters"),
        ("segmentation", {"size": [80, 100], "counts": "0"}, "cover the image"),
        ("segmentation", {"size": [80, 100], "counts": "o"}, "truncated"),
        ("segmentation", {"size": [80, 100], "counts": "O"}, "out of range"),
    ],
)
def test_invalid_annotations_fail_explicitly(
    local_config: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    path = local_config.parent / "annotations.json"
    data = json.loads(path.read_text())
    data["annotations"][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(Phase1Error, match=message):
        load_coco(path)


@pytest.mark.parametrize("section", ["categories", "images", "annotations"])
def test_duplicate_ids_fail(local_config: Path, section: str) -> None:
    path = local_config.parent / "annotations.json"
    data = json.loads(path.read_text())
    data[section].append(data[section][0])
    path.write_text(json.dumps(data))
    with pytest.raises(Phase1Error, match="Duplicate"):
        load_coco(path)


def test_wrong_animal_metadata_count_fails(local_config: Path) -> None:
    path = local_config.parent / "annotations.json"
    data = json.loads(path.read_text())
    data["categories"][0]["supercategory"] = "other"
    path.write_text(json.dumps(data))
    with pytest.raises(Phase1Error, match="found 9"):
        load_coco(path)


@pytest.mark.parametrize("name", ["../outside.png", "/outside.png", "folder\\outside.png"])
def test_image_path_traversal_is_rejected(local_config: Path, name: str) -> None:
    path = local_config.parent / "annotations.json"
    data = json.loads(path.read_text())
    data["images"][0]["file_name"] = name
    path.write_text(json.dumps(data))
    with pytest.raises(Phase1Error, match="Unsafe"):
        load_coco(path)


def test_json_parse_error_is_actionable(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{")
    with pytest.raises(Phase1Error, match="Invalid annotation JSON"):
        load_coco(path)


@pytest.mark.parametrize(
    "segmentation",
    [
        [[1, 2, 11, 2, 11, 11, 1, 11], [30, 20, 34, 20, 34, 24, 30, 24]],
        {"size": [80, 100], "counts": [0, 8000]},
        # Fixed test encodings, with no dependency on a mask renderer.
        {"size": [80, 100], "counts": "Pj7"},  # Runs: [8000], empty foreground.
        {"size": [80, 100], "counts": "0Pj7"},  # Runs: [0, 8000], full foreground.
        {"size": [80, 100], "counts": "234O1^i7"},  # [2, 3, 4, 2, 5, 7984], negative delta.
        {"size": [80, 100], "counts": "Ub15[20kR6"},  # [1605, 5, 75, 5, 6310], separate runs.
    ],
)
def test_valid_segmentation_encodings_are_preserved(local_config: Path, segmentation: Any) -> None:
    path = local_config.parent / "annotations.json"
    data = json.loads(path.read_text())
    data["annotations"][0]["segmentation"] = segmentation
    path.write_text(json.dumps(data))
    dataset = load_coco(path)
    assert dataset.annotations[1001].segmentation == segmentation
    assert dataset.annotations[1001].image_id == 10
