"""Retain frozen fitting/development data and reserve previously unused image groups."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from pycocotools.coco import COCO

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file


def select_unused_groups(
    groups: list[dict[str, Any]], used_ids: set[int], size: int = 64, seed: int = 44
) -> list[dict[str, Any]]:
    eligible = [
        group
        for group in groups
        if not used_ids.intersection(row["id"] for row in group["members"])
    ]
    eligible.sort(key=lambda group: group["group_id"])
    if size < 1 or size > len(eligible):
        raise Phase1Error("Insufficient previously unused groups for fresh evaluation.")
    return random.Random(seed).sample(eligible, size)


def prepare_recovery_data(source: Path, output: Path, size: int = 64, seed: int = 44) -> Path:
    source, output = source.resolve(), output.resolve()
    original = json.loads((source / "manifest.json").read_text())
    if original["status"] != "frozen" or output.exists():
        raise Phase1Error("Source must be frozen and recovery data must not exist.")
    for name, digest in original["artifact_sha256"].items():
        if sha256_file(source / name) != digest:
            raise Phase1Error(f"Original frozen artifact changed: {name}")
    archive = Path(original["annotation_archive"])
    if sha256_file(archive) != original["annotation_archive_sha256"]:
        raise Phase1Error("COCO source archive changed.")
    groups = json.loads((source / "groups.json").read_text())
    used = {key for keys in original["split_image_ids"].values() for key in keys}
    selected = select_unused_groups(groups, used, size, seed)
    records = [min(group["members"], key=lambda row: row["id"]) for group in selected]
    image_ids = {row["id"] for row in records}
    with ZipFile(archive) as zipped:
        payload = zipped.read(original["annotation_member"])
    if hashlib.sha256(payload).hexdigest() != original["annotation_member_sha256"]:
        raise Phase1Error("COCO source annotation member changed.")
    raw = json.loads(payload)
    images = {row["id"]: row for row in raw["images"] if row["id"] in image_ids}
    annotations = [row for row in raw["annotations"] if row["image_id"] in image_ids]
    if any(row.get("iscrowd", 0) for row in annotations):
        raise Phase1Error("Unexpected crowd annotation in an originally accepted group.")
    truth = COCO()
    truth.dataset = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": raw["categories"],
    }
    truth.createIndex()
    masks = {}
    for row in annotations:
        mask = np.asarray(truth.annToMask(row), dtype=np.uint8)
        if not mask.any():
            raise Phase1Error("Empty original mask in fresh evaluation data.")
        masks[str(row["id"])] = hashlib.sha256(mask.tobytes()).hexdigest()
    image_root = Path(original["image_root"])
    for record in records:
        if sha256_file(image_root / record["file_name"]) != record["sha256"]:
            raise Phase1Error("Previously audited image changed.")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("fitting.json", "development.json"):
        content = json.loads((source / name).read_text())
        relative = content["info"]["aem_training_data"]["image_root_relative_to_subset"]
        if (output / relative).resolve() != image_root:
            raise Phase1Error("Recovery directory depth must preserve frozen image paths.")
        shutil.copyfile(source / name, output / name)
    content = {
        "info": copy.deepcopy(raw.get("info", {})),
        "licenses": raw.get("licenses", []),
        "categories": raw["categories"],
        "images": [
            images[row["id"]] | {"aem_sha256": row["sha256"]}
            for row in sorted(records, key=lambda row: row["id"])
        ],
        "annotations": sorted(annotations, key=lambda row: row["id"]),
    }
    content["info"]["aem_training_data"] = {
        "source_split": "train2017",
        "experiment_split": "fresh_holdout",
        "image_root_relative_to_subset": os.path.relpath(image_root, output),
        "mask_sha256_by_annotation": masks,
    }
    (output / "holdout.json").write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    selected_rows = []
    for group, record in zip(selected, records, strict=True):
        selected_rows.append(
            {"image_id": record["id"], "group_id": group["group_id"], "sha256": record["sha256"]}
        )
    with (output / "fresh_groups.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    manifest = {
        "status": "frozen",
        "source_split": "train2017",
        "previous_manifest_sha256": sha256_file(source / "manifest.json"),
        "annotation_archive_sha256": original["annotation_archive_sha256"],
        "annotation_member_sha256": original["annotation_member_sha256"],
        "fresh_holdout_seed": seed,
        "selection": "random.Random(seed).sample(groups sorted by group_id, size); "
        "exclude every previously used group",
        "split_image_ids": {
            "fitting": original["split_image_ids"]["fitting"],
            "development": original["split_image_ids"]["development"],
            "holdout": sorted(image_ids),
        },
        "previously_used_image_ids_excluded_from_fresh_holdout": sorted(used),
        "fresh_group_ids": sorted(group["group_id"] for group in selected),
        "fitting_and_development_byte_identical": all(
            sha256_file(output / name) == sha256_file(source / name)
            for name in ("fitting.json", "development.json")
        ),
        "used_for_parameter_selection": False,
        "limitations": [
            "COCO train-source data may overlap upstream pretraining.",
            "Automatic grouping does not establish human-confirmed scene independence.",
        ],
        "artifact_sha256": {
            path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_recovery_data(args.source, args.output))


if __name__ == "__main__":
    main()
