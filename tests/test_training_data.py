"""Native-mask and split protection checks using synthetic data, never research GT."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pycocotools")

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.data.training_data import (
    TRAIN_MEMBER,
    FrozenCocoDataset,
    group_images,
    prepare_training_data,
)


def make_source(root: Path, count: int = 10) -> dict:
    images = root / "data/train2017"
    images.mkdir(parents=True)
    payload = {
        "info": {"description": "Synthetic engineering fixture, not research ground truth"},
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 16, "name": "bird", "supercategory": "animal"},
            {"id": 18, "name": "dog", "supercategory": "animal"},
        ],
    }
    for key in range(1, count + 1):
        pixels = np.random.default_rng(key).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        name = f"{key:012d}.jpg"
        Image.fromarray(pixels).save(images / name)
        payload["images"].append({"id": key, "file_name": name, "width": 32, "height": 24})
        for offset, category, box, polygons, area in (
            (1, 1, [10, 12, 5, 6], [[10, 12, 15, 12, 15, 18, 10, 18]], 30),
            (2, 16, [20, 10, 5, 5], [[20, 10, 25, 10, 25, 15, 20, 15]], 25),
            (
                3,
                18,
                [1, 1, 13, 4],
                [[1, 1, 5, 1, 5, 5, 1, 5], [10, 1, 14, 1, 14, 5, 10, 5]],
                32,
            ),
        ):
            payload["annotations"].append(
                {
                    "id": key * 10 + offset,
                    "image_id": key,
                    "category_id": category,
                    "bbox": box,
                    "segmentation": polygons,
                    "area": area,
                    "iscrowd": 0,
                }
            )
    write_annotations(root, payload)
    return payload


def write_annotations(root: Path, payload: dict) -> None:
    with ZipFile(root / "annotations_trainval2017.zip", "w") as archive:
        archive.writestr(TRAIN_MEMBER, json.dumps(payload))


def prepare(root: Path, directory: str = "frozen", count: int = 10) -> Path:
    return prepare_training_data(
        repository_root=root,
        output_directory=root / directory,
        fitting_size=2,
        development_size=1,
        holdout_size=1,
        candidate_limit=count,
    )


def test_original_ids_all_categories_and_disconnected_masks_survive(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    original = make_source(tmp_path)
    manifest_path = prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    dataset = FrozenCocoDataset(manifest_path.parent / "fitting.json")
    image, target = dataset[0]
    key = target["image_id"].item()
    expected_annotations = [row for row in original["annotations"] if row["image_id"] == key]
    assert dataset.coco.imgToAnns[key] == expected_annotations
    assert image.shape == (3, 24, 32)
    assert image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1
    assert target["labels"].tolist() == [1, 16, 18]
    assert target["annotation_id"].tolist() == [key * 10 + offset for offset in (1, 2, 3)]
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64
    assert target["masks"].dtype == torch.uint8
    dog_mask = target["masks"][2]
    assert dog_mask.sum().item() == 32
    assert dog_mask[2, 2] == dog_mask[2, 11] == 1
    assert dog_mask[2, 7] == 0  # No invented connecting polygon between components.
    assert manifest["status"] == "frozen"
    assert manifest["source_unchanged"] is True


def test_crowd_in_non_animal_category_excludes_whole_image(tmp_path: Path) -> None:
    original = make_source(tmp_path)
    original["annotations"][0]["iscrowd"] = 1
    write_annotations(tmp_path, original)
    manifest = json.loads(prepare(tmp_path).read_text())
    selected = set().union(*map(set, manifest["split_image_ids"].values()))
    assert 1 not in selected
    assert manifest["exclusion_counts"] == {"crowd_any_category": 1}


def test_invalid_boxes_and_bad_or_missing_images_are_not_silently_used(tmp_path: Path) -> None:
    original = make_source(tmp_path)
    original["annotations"][0]["bbox"] = [1, 1, 0, 3]
    write_annotations(tmp_path, original)
    (tmp_path / "data/train2017/000000000002.jpg").write_bytes(b"not an image")
    (tmp_path / "data/train2017/000000000003.jpg").unlink()
    manifest = json.loads(prepare(tmp_path).read_text())
    assert manifest["exclusion_counts"] == {
        "invalid_or_out_of_bounds_bbox": 1,
        "image_decode_failed": 1,
        "missing_image": 1,
    }
    assert not {1, 2, 3} & set().union(*map(set, manifest["split_image_ids"].values()))


def test_grouped_duplicates_never_cross_splits_and_selection_is_reproducible(
    tmp_path: Path,
) -> None:
    make_source(tmp_path)
    source = tmp_path / "data/train2017/000000000001.jpg"
    (tmp_path / "data/train2017/000000000010.jpg").write_bytes(source.read_bytes())
    first_path = prepare(tmp_path, "one")
    first = json.loads(first_path.read_text())
    second = json.loads(prepare(tmp_path, "two").read_text())
    split_sets = [
        set(first["split_image_ids"][split]) for split in ("holdout", "development", "fitting")
    ]
    assert len(set.union(*split_sets)) == sum(map(len, split_sets)) == 4
    assert 10 not in set.union(*split_sets)
    assert first["split_image_ids"] == second["split_image_ids"]
    groups = json.loads((first_path.parent / "groups.json").read_text())
    assert any({row["id"] for row in group["members"]} == {1, 10} for group in groups)


def test_dhash_neighbors_form_transitive_group() -> None:
    rows = [
        {"id": key, "sha256": f"file{key}", "rgb_sha256": f"rgb{key}", "dhash": digest}
        for key, digest in ((1, "0000"), (2, "0003"), (3, "000f"), (4, "ffff"))
    ]
    assert [[row["id"] for row in group] for group in group_images(rows)] == [[1, 2, 3], [4]]


def test_frozen_dataset_rejects_changed_source(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    make_source(tmp_path)
    manifest = prepare(tmp_path)
    dataset = FrozenCocoDataset(manifest.parent / "fitting.json")
    key = dataset.image_ids[0]
    (tmp_path / "data/train2017" / f"{key:012d}.jpg").write_bytes(b"changed")
    with pytest.raises(Phase1Error, match="digest changed"):
        dataset[0]


def test_training_preparation_rejects_existing_output(tmp_path: Path) -> None:
    make_source(tmp_path)
    prepare(tmp_path)
    with pytest.raises(Phase1Error, match="overwrite"):
        prepare(tmp_path)


def test_insufficient_supply_is_reported_without_relaxing_rules(tmp_path: Path) -> None:
    make_source(tmp_path, count=3)
    with pytest.raises(Phase1Error, match="Need 4 groups"):
        prepare(tmp_path, count=4)
    manifest = json.loads((tmp_path / "frozen/manifest.json").read_text())
    assert manifest["status"] == "insufficient_supply"
    assert not (tmp_path / "frozen/fitting.json").exists()


def test_val_images_cannot_be_substituted_for_train(tmp_path: Path) -> None:
    make_source(tmp_path)
    with pytest.raises(Phase1Error, match="train2017"):
        prepare_training_data(
            repository_root=tmp_path,
            output_directory=tmp_path / "output",
            image_root=tmp_path / "data/val2017",
        )


def test_frozen_subset_detects_changed_class_labels(tmp_path: Path) -> None:
    make_source(tmp_path)
    manifest = prepare(tmp_path)
    subset = manifest.parent / "fitting.json"
    payload = json.loads(subset.read_text())
    payload["annotations"][0]["category_id"] = 18
    subset.write_text(json.dumps(payload))
    with pytest.raises(Phase1Error, match="subset digest"):
        FrozenCocoDataset(subset)
