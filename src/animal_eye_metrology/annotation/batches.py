"""Create unreviewed v2 annotation batches from user-selected local COCO sources."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zipfile import ZipFile

from PIL import Image

from animal_eye_metrology.annotation.annotation_session import prepare, read_session
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file


def create_batch(root: Path, split: str, count: int, seed: int = 42) -> Path:
    """Freeze source identities, not eye GT or train/test assignments; never download."""
    root = root.resolve()
    if split not in {"train2017", "val2017"} or type(count) is not int or not 1 <= count <= 100:
        raise Phase1Error("來源須為 train2017／val2017，批次大小須為 1–100 張。")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise Phase1Error("Seed 必須為 0–4294967295 的整數。")
    source_file = root / f"data/coco/annotations/instances_{split}.json"
    image_root = root / ("data/train2017" if split == "train2017" else "data/coco/val2017")
    archive = root / "annotations_trainval2017.zip"
    member = f"annotations/instances_{split}.json"
    if source_file.is_file():
        payload = source_file.read_bytes()
        source = {
            "path": str(source_file.relative_to(root)),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    else:
        with ZipFile(archive) as zipped:
            payload = zipped.read(member)
        source = {
            "archive": archive.name,
            "archive_sha256": sha256_file(archive),
            "member": member,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    raw = json.loads(payload)
    categories = {r["id"]: r for r in raw["categories"]}
    animals = {key for key, r in categories.items() if r.get("supercategory") == "animal"}
    instances = defaultdict(list)
    for ann in raw["annotations"]:
        if ann["category_id"] in animals and not ann.get("iscrowd", 0):
            instances[ann["image_id"]].append(ann)
    images = {row["id"]: row for row in raw["images"]}
    candidates = sorted(key for key, anns in instances.items() if len(anns) >= 2)
    random.Random(seed).shuffle(candidates)
    # Exact identities already in prepared batches are excluded; no claim of scene independence.
    used_ids, used_hashes = set(), set()
    sessions = list((root / "outputs/phase3/eye_annotation_batches").glob("*/batch_manifest.json"))
    for path in sessions:
        prior = json.loads(path.read_text())
        if prior["source_split"] == split:
            used_ids.update(prior["selected_image_ids"])
        used_hashes.update(prior["image_sha256"].values())
    if split == "val2017":
        used_ids.update((125405, 20247, 333069))
    selected, rejected, hashes = [], [], {}
    for image_id in candidates:
        if image_id in used_ids:
            rejected.append({"image_id": image_id, "reason": "previously_selected_image"})
            continue
        row = images[image_id]
        path = image_root / row["file_name"]
        if Path(row["file_name"]).name != row["file_name"]:
            raise Phase1Error("來源圖片檔名超出指定資料夾。")
        try:
            digest = sha256_file(path)
            with Image.open(path) as im:
                im.load()
                if im.size != (row["width"], row["height"]):
                    raise ValueError("Image dimensions do not match source metadata.")
            if digest in used_hashes:
                rejected.append({"image_id": image_id, "reason": "duplicate_image_bytes"})
                continue
        except (OSError, ValueError) as error:
            rejected.append(
                {"image_id": image_id, "reason": "missing_or_invalid_image", "detail": str(error)}
            )
            continue
        selected.append(image_id)
        hashes[str(image_id)] = digest
        used_hashes.add(digest)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise Phase1Error(
            f"可讀取且未重複的候選只有 {len(selected)} 張，不足要求的 {count} 張；未建立批次。"
        )
    base = root / "outputs/phase3/eye_annotation_batches"
    name = f"{split}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
    directory = base / name
    directory.mkdir(parents=True, exist_ok=False)
    # Keep selected images' original annotations (including other classes) without relabeling them.
    subset = {
        "info": raw.get("info", {}),
        "licenses": raw.get("licenses", []),
        "categories": raw["categories"],
        "images": [images[key] for key in selected],
        "annotations": [r for r in raw["annotations"] if r["image_id"] in set(selected)],
    }
    (directory / "annotations.json").write_text(json.dumps(subset) + "\n")
    rows = []
    for image_id in selected:
        image = images[image_id]
        for ann in sorted(instances[image_id], key=lambda r: r["id"]):
            rows.append(
                {
                    "image_id": image_id,
                    "annotation_id": ann["id"],
                    "animal_category": categories[ann["category_id"]]["name"],
                    "file_name": image["file_name"],
                    "image_width": image["width"],
                    "image_height": image["height"],
                    "coco_bbox_xywh": json.dumps(ann["bbox"]),
                    "proposed_eye_sides": '["left","right"]',
                    "image_sha256": hashes[str(image_id)],
                }
            )
    with (directory / "proposal.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def relative(path: Path) -> str:
        return json.dumps(os.path.relpath(path, directory))

    (directory / "dataset.toml").write_text(
        f'[dataset]\nname = "COCO {split} annotation candidates"\n'
        'annotations = "annotations.json"\n'
        f"images = {relative(image_root)}\nexpected_animal_categories = {len(animals)}\n"
        f'expected_annotations_sha256 = "{sha256_file(directory / "annotations.json")}"\n'
        f"[output]\ndirectory = {relative(root / 'outputs/phase1')}\n"
    )
    config = directory / "session.toml"
    config.write_text(
        'phase1_config = "dataset.toml"\nproposal_csv = "proposal.csv"\ndirectory = "workspace"\n'
        f"protocol_document = {relative(root / 'docs/eye_mask_protocol_v2.md')}\n"
        'protocol_version = "eye-visible-eyelid-v2"\n'
        f"expected_images = {count}\nexpected_instances = {len(rows)}\n"
        f"expected_eyes = {len(rows) * 2}\n"
    )
    manifest = {
        "source_split": split,
        "source": source,
        "seed": seed,
        "requested_images": count,
        "selected_image_ids": selected,
        "image_sha256": hashes,
        "instances": len(rows),
        "eye_slots": len(rows) * 2,
        "eye_suitability": "not_reviewed",
        "dataset_role": "unassigned",
        "selection_rule": (
            "At least two non-crowd animals; sorted image IDs shuffled with seed; "
            "decode and size verified; exact-byte duplicates excluded. "
            "No eye-quality or model-based filtering."
        ),
        "near_duplicates_checked": False,
        "rejected": rejected,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    # Publish discovery marker only after the session is fully prepared.
    prepare(read_session(config))
    (directory / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return config
