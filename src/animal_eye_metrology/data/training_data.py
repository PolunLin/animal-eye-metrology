"""Freeze local COCO training subsets while retaining native instance masks and IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file

if TYPE_CHECKING:
    from torch import Tensor

LOGGER = logging.getLogger(__name__)
TRAIN_MEMBER = "annotations/instances_train2017.json"
SPLIT_NAMES = ("holdout", "development", "fitting")


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _coco(payload: dict[str, Any]) -> COCO:
    with redirect_stdout(io.StringIO()):
        result = COCO()
        result.dataset = payload
        result.createIndex()
    return result


def difference_hash(image: Image.Image) -> int:
    """Return a 64-bit horizontal dHash; proximity is not scene independence proof."""
    gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    result = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).flat:
        result = (result << 1) | int(bit)
    return result


def group_images(
    rows: list[dict[str, Any]], *, maximum_hamming_distance: int = 2
) -> list[list[dict[str, Any]]]:
    """Use transitive groups of exact bytes/RGB or dHash neighbors, never across splits."""
    if type(maximum_hamming_distance) is not int or not 0 <= maximum_hamming_distance <= 64:
        raise Phase1Error("dHash distance must be between 0 and 64.")
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, left in enumerate(rows):
        for j in range(i):
            right = rows[j]
            if (
                left["sha256"] == right["sha256"]
                or left["rgb_sha256"] == right["rgb_sha256"]
                or (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
                <= maximum_hamming_distance
            ):
                parent[find(i)] = find(j)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    return sorted(
        (sorted(group, key=lambda row: row["id"]) for group in grouped.values()),
        key=lambda group: group[0]["id"],
    )


def _validate_annotations(
    image: dict[str, Any], annotations: list[dict[str, Any]], category_ids: set[int]
) -> None:
    width, height = image["width"], image["height"]
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise Phase1Error("invalid_dimensions")
    for row in annotations:
        if row.get("iscrowd", 0) != 0:
            raise Phase1Error("crowd_any_category")
        if row["category_id"] not in category_ids:
            raise Phase1Error("unknown_category")
        box = row.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise Phase1Error("invalid_bbox")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in box):
            raise Phase1Error("invalid_bbox")
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise Phase1Error("invalid_or_out_of_bounds_bbox")
        area = row.get("area")
        if type(area) not in (int, float) or not math.isfinite(area) or area <= 0:
            raise Phase1Error("invalid_area")


def _image_record(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise Phase1Error("missing_image")
    contents = path.read_bytes()
    if not contents:
        raise Phase1Error("empty_image")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
        with Image.open(io.BytesIO(contents)) as image:
            image.load()
            rgb = image.convert("RGB")
    except (OSError, ValueError, SyntaxError) as error:
        raise Phase1Error("image_decode_failed") from error
    if rgb.size != (metadata["width"], metadata["height"]):
        raise Phase1Error("image_dimensions_differ")
    return {
        "id": metadata["id"],
        "file_name": metadata["file_name"],
        "sha256": _digest(contents),
        "rgb_sha256": _digest(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()),
        "dhash": f"{difference_hash(rgb):016x}",
        "size_bytes": len(contents),
    }


def _read_source(archive_path: Path) -> tuple[dict[str, Any], str]:
    with ZipFile(archive_path) as archive:
        payload = archive.read(TRAIN_MEMBER)
    digest = _digest(payload)
    source = json.loads(payload)
    if not isinstance(source, dict):
        raise Phase1Error("COCO training annotations must be an object.")
    for field in ("images", "annotations", "categories"):
        if not isinstance(source.get(field), list):
            raise Phase1Error(f"COCO training annotations lack {field}.")
        identifiers = [row["id"] for row in source[field]]
        if len(set(identifiers)) != len(identifiers):
            raise Phase1Error(f"Duplicate COCO {field} IDs.")
    return source, digest


def prepare_training_data(
    *,
    repository_root: Path,
    output_directory: Path,
    fitting_size: int = 256,
    development_size: int = 64,
    holdout_size: int = 64,
    seed: int = 42,
    split_seed: int = 43,
    candidate_limit: int = 768,
    maximum_hamming_distance: int = 2,
    annotation_archive: Path | None = None,
    image_root: Path | None = None,
) -> Path:
    """Freeze all-category native-mask COCO subsets from train2017 without model calls.

    Sampling uses two independent RNGs: seed chooses the candidate audit window;
    split_seed shuffles valid group representatives, assigning holdout then development
    then fitting. No labels, image files, or pre-existing output files are modified.
    """
    sizes = (holdout_size, development_size, fitting_size)
    if any(type(value) is not int or value <= 0 for value in sizes):
        raise Phase1Error("All three split sizes must be positive integers.")
    if type(candidate_limit) is not int or candidate_limit < sum(sizes):
        raise Phase1Error("Candidate limit must cover the requested split sizes.")
    if type(maximum_hamming_distance) is not int or not 0 <= maximum_hamming_distance <= 64:
        raise Phase1Error("dHash distance must be between 0 and 64.")
    root = Path(repository_root).resolve()
    output = Path(output_directory).resolve()
    archive_path = (
        Path(annotation_archive).resolve()
        if annotation_archive is not None
        else root / "annotations_trainval2017.zip"
    )
    images_path = Path(image_root).resolve() if image_root is not None else root / "data/train2017"
    if images_path.name != "train2017":
        raise Phase1Error("Only the train2017 image directory is accepted for this pilot.")
    if output.exists():
        raise Phase1Error(f"Refusing to overwrite existing training data: {output}")
    if output == images_path or output.is_relative_to(images_path):
        raise Phase1Error("Training output must not be inside the image source.")
    archive_digest = sha256_file(archive_path)
    source, member_digest = _read_source(archive_path)
    images = {row["id"]: row for row in source["images"]}
    categories = {row["id"] for row in source["categories"]}
    animal_ids = {row["id"] for row in source["categories"] if row.get("supercategory") == "animal"}
    animal_counts = Counter(
        row["image_id"]
        for row in source["annotations"]
        if row["category_id"] in animal_ids and not row.get("iscrowd", 0)
    )
    candidates = sorted(key for key in images if animal_counts[key] >= 2)
    random.Random(seed).shuffle(candidates)
    window = candidates[:candidate_limit]
    window_set = set(window)
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source["annotations"]:
        if row["image_id"] in window_set:
            annotations[row["image_id"]].append(row)
    for rows in annotations.values():
        rows.sort(key=lambda row: row["id"])
    # Only the bounded audit window is indexed or mask-decoded by pycocotools.
    truth = _coco(
        {
            "images": [images[key] for key in window],
            "annotations": [row for key in window for row in annotations[key]],
            "categories": source["categories"],
        }
    )
    valid, excluded = [], []
    mask_digests: dict[str, str] = {}
    for order, key in enumerate(window):
        metadata = images[key]
        try:
            name = metadata["file_name"]
            if Path(name).is_absolute() or len(Path(name).parts) != 1 or "\\" in name:
                raise Phase1Error("unsafe_image_filename")
            _validate_annotations(metadata, annotations[key], categories)
            record = _image_record(images_path / name, metadata)
            local_mask_digests = {}
            for annotation in annotations[key]:
                try:
                    mask = truth.annToMask(annotation)
                except (ValueError, TypeError, KeyError, OverflowError) as error:
                    raise Phase1Error("mask_decode_failed") from error
                if mask.shape != (metadata["height"], metadata["width"]) or not mask.any():
                    raise Phase1Error("empty_or_wrong_size_mask")
                if not np.isin(mask, (0, 1)).all():
                    raise Phase1Error("non_binary_mask")
                local_mask_digests[str(annotation["id"])] = _digest(
                    np.asarray(mask, dtype=np.uint8).tobytes()
                )
            mask_digests.update(local_mask_digests)
            valid.append(record | {"candidate_order": order})
        except Phase1Error as error:
            excluded.append({"image_id": key, "candidate_order": order, "reason": str(error)})
        if (order + 1) % 64 == 0:
            LOGGER.info("training_data_candidates_checked=%d valid=%d", order + 1, len(valid))
    groups = group_images(valid, maximum_hamming_distance=maximum_hamming_distance)
    representatives = [group[0] for group in groups]
    random.Random(split_seed).shuffle(representatives)
    group_by_id = {row["id"]: group for group in groups for row in group}
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "exclusions.json", excluded)
    _json(
        output / "groups.json",
        [{"group_id": f"train_{group[0]['id']:012d}", "members": group} for group in groups],
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "insufficient_supply",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_split": "train2017",
        "repository_root": str(root),
        "image_root": str(images_path),
        "annotation_archive": str(archive_path),
        "annotation_archive_sha256": archive_digest,
        "annotation_member": TRAIN_MEMBER,
        "annotation_member_sha256": member_digest,
        "preparation_source_sha256": sha256_file(Path(__file__)),
        "execution_argv": [sys.executable, *sys.argv],
        "seed": seed,
        "split_seed": split_seed,
        "candidate_limit": candidate_limit,
        "candidate_image_ids": window,
        "selection_rule": "at least two non-crowd animals; retain every category annotation",
        "exclusion_rule": "any-category crowd, invalid boxes/masks, absent or invalid images",
        "maximum_dhash_hamming_distance": maximum_hamming_distance,
        "grouping_rule": "transitive exact file/RGB matches or 64-bit dHash distance <= threshold",
        "group_representative_rule": "lowest original image_id",
        "split_assignment_order": list(SPLIT_NAMES),
        "counts": {
            "source_images": len(images),
            "source_annotations": len(source["annotations"]),
            "multi_animal_candidates": len(candidates),
            "checked_candidates": len(window),
            "valid_images": len(valid),
            "excluded_images": len(excluded),
            "automatic_groups": len(groups),
            "duplicate_members_not_selected": len(valid) - len(groups),
        },
        "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
        "requested_sizes": dict(zip(SPLIT_NAMES, sizes, strict=True)),
        "versions": {name: version(name) for name in ("numpy", "Pillow", "pycocotools")},
        "limitations": [
            "Automatic dHash grouping is not human scene-independence review.",
            "COCO training-source holdout may overlap upstream model pretraining.",
            "Crowd exclusion and multi-animal sampling restrict the evaluation population.",
            "No eye ground truth or eye-distance accuracy is established.",
        ],
    }
    if len(groups) < sum(sizes):
        _json(output / "manifest.json", manifest)
        raise Phase1Error(
            f"Need {sum(sizes)} groups but found {len(groups)}; supply report: {output}"
        )
    selected_rows = []
    offset = 0
    split_ids = {}
    for split, size in zip(SPLIT_NAMES, sizes, strict=True):
        chosen = representatives[offset : offset + size]
        offset += size
        identifiers = {row["id"] for row in chosen}
        split_ids[split] = sorted(identifiers)
        chosen_annotations = [row for key in sorted(identifiers) for row in annotations[key]]
        selected_images = []
        for record in sorted(chosen, key=lambda row: row["id"]):
            metadata = images[record["id"]]
            group = group_by_id[record["id"]]
            group_id = f"train_{group[0]['id']:012d}"
            selected_images.append(metadata | {"aem_sha256": record["sha256"]})
            selected_rows.append(
                record
                | {
                    "split": split,
                    "group_id": group_id,
                    "group_members": json.dumps([row["id"] for row in group]),
                    "annotation_ids": json.dumps([row["id"] for row in annotations[record["id"]]]),
                    "non_crowd_animals": animal_counts[record["id"]],
                }
            )
        subset = {
            "info": source.get("info", {})
            | {
                "aem_training_data": {
                    "source_split": "train2017",
                    "experiment_split": split,
                    "image_root_relative_to_subset": os.path.relpath(images_path, output),
                    "mask_sha256_by_annotation": {
                        str(row["id"]): mask_digests[str(row["id"])] for row in chosen_annotations
                    },
                }
            },
            "licenses": source.get("licenses", []),
            "categories": source["categories"],
            "images": selected_images,
            "annotations": chosen_annotations,
        }
        _json(output / f"{split}.json", subset)
    with (output / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    # Recheck chosen byte sources, rather than trusting an earlier file inventory.
    for row in selected_rows:
        if sha256_file(images_path / row["file_name"]) != row["sha256"]:
            raise Phase1Error("A selected image changed during preparation.")
    if sha256_file(archive_path) != archive_digest:
        raise Phase1Error("The annotation archive changed during preparation.")
    manifest.update(
        {
            "status": "frozen",
            "source_unchanged": True,
            "split_image_ids": split_ids,
            "artifact_sha256": {
                path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()
            },
        }
    )
    _json(output / "manifest.json", manifest)
    return output / "manifest.json"


class FrozenCocoDataset:
    """Read original-size RGB tensors and exact native COCO masks from a frozen split."""

    def __init__(
        self, subset_json: Path, *, image_root: Path | None = None, verify_hashes: bool = True
    ) -> None:
        self.subset_json = Path(subset_json).resolve()
        if verify_hashes:
            manifest = json.loads((self.subset_json.parent / "manifest.json").read_text())
            expected = manifest.get("artifact_sha256", {}).get(self.subset_json.name)
            if manifest.get("status") != "frozen" or sha256_file(self.subset_json) != expected:
                raise Phase1Error("Frozen subset digest or manifest status changed.")
        payload = json.loads(self.subset_json.read_text())
        metadata = payload.get("info", {}).get("aem_training_data", {})
        if metadata.get("source_split") != "train2017":
            raise Phase1Error("Frozen training datasets must come from train2017.")
        self.image_root = (
            Path(image_root).resolve()
            if image_root is not None
            else (self.subset_json.parent / metadata["image_root_relative_to_subset"]).resolve()
        )
        self.verify_hashes = verify_hashes
        self.mask_digests = metadata["mask_sha256_by_annotation"]
        self.coco = _coco(payload)
        self.image_ids = sorted(self.coco.imgs)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        import torch

        identifier = self.image_ids[index]
        metadata = self.coco.imgs[identifier]
        name = metadata["file_name"]
        if Path(name).is_absolute() or len(Path(name).parts) != 1 or "\\" in name:
            raise Phase1Error("Unsafe image filename in frozen subset.")
        payload = (self.image_root / name).read_bytes()
        if self.verify_hashes and _digest(payload) != metadata["aem_sha256"]:
            raise Phase1Error(f"Frozen image digest changed: {identifier}")
        with Image.open(io.BytesIO(payload)) as source:
            rgb = np.array(source.convert("RGB"), copy=True)
        height, width = rgb.shape[:2]
        if (height, width) != (metadata["height"], metadata["width"]):
            raise Phase1Error(f"Frozen image dimensions changed: {identifier}")
        annotations = sorted(self.coco.imgToAnns[identifier], key=lambda row: row["id"])
        masks = []
        for row in annotations:
            mask = np.asarray(self.coco.annToMask(row), dtype=np.uint8)
            if self.verify_hashes and _digest(mask.tobytes()) != self.mask_digests[str(row["id"])]:
                raise Phase1Error(f"Frozen annotation mask changed: {row['id']}")
            masks.append(mask)
        boxes = [
            [x, y, x + w, y + h]
            for x, y, w, h in (annotation["bbox"] for annotation in annotations)
        ]
        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([row["category_id"] for row in annotations], dtype=torch.int64),
            "masks": torch.from_numpy(
                np.stack(masks) if masks else np.empty((0, height, width), dtype=np.uint8)
            ),
            "image_id": torch.tensor(identifier, dtype=torch.int64),
            "area": torch.tensor([row["area"] for row in annotations], dtype=torch.float32),
            "iscrowd": torch.tensor(
                [row.get("iscrowd", 0) for row in annotations], dtype=torch.int64
            ),
            "annotation_id": torch.tensor([row["id"] for row in annotations], dtype=torch.int64),
        }
        return torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32).div(255), targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=768)
    parser.add_argument("--fitting-size", type=int, default=256)
    parser.add_argument("--development-size", type=int, default=64)
    parser.add_argument("--holdout-size", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = prepare_training_data(
        repository_root=args.repo_root,
        output_directory=args.output,
        candidate_limit=args.candidate_limit,
        fitting_size=args.fitting_size,
        development_size=args.development_size,
        holdout_size=args.holdout_size,
    )
    LOGGER.info("training_data_manifest=%s", manifest)


if __name__ == "__main__":
    main()
