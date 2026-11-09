"""Freeze the owner's saved review state without modifying annotation workspaces."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animal_eye_metrology.annotation.annotation_session import (
    _check_scope,
    _dataset,
    read_session,
    verify_session,
)
from animal_eye_metrology.annotation.eye_annotations import (
    V2_PROTOCOL_VERSION,
    load_annotations,
    load_targets,
)
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise Phase1Error("Snapshot sources must be inside the project root.")
    return resolved.relative_to(root).as_posix()


def _identity(row: dict[str, Any]) -> tuple[int, int, str]:
    return int(row["image_id"]), int(row["annotation_id"]), row["eye_side"]


def _human_saved(
    row: dict[str, Any] | None,
    batch: str,
    imports: dict[tuple[str, int, int, str], int],
    reviewer: str,
) -> bool:
    """A prefilled reviewer name alone is not evidence of a human review."""
    return bool(
        row
        and row["reviewer"] == reviewer
        and int(row["annotation_version"]) > imports.get((batch, *_identity(row)), 0)
    )


def _encode_mask(path: Path, width: int, height: int) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils

    with Image.open(path) as opened:
        values = np.asarray(opened)
    if values.shape != (height, width) or not np.isin(values, (0, 255)).all():
        raise Phase1Error("Reviewed mask must be binary at original image dimensions.")
    binary = values != 0
    if not binary.any():
        raise Phase1Error("An eligible reviewed mask must not be empty.")
    encoded = mask_utils.encode(np.asfortranarray(binary, dtype=np.uint8))
    record = {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}
    if not np.array_equal(mask_utils.decode(record).astype(bool), binary):
        raise Phase1Error("Lossless reviewed mask encoding failed.")
    return record


def _summary(images: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(
        photos=len(images),
        eye_slots=len(cases),
        saved_eyes=sum(c["saved"] for c in cases),
        human_saved_eyes=sum(c["human_saved"] for c in cases),
        reviewed_eyes=sum(c["human_reviewed"] for c in cases),
        accepted_eyes=sum(
            c["human_reviewed"] and c["mask_review_status"] == "Accepted" for c in cases
        ),
        rejected_eyes=sum(
            c["human_reviewed"] and c["mask_review_status"] == "Rejected" for c in cases
        ),
        eligible_eyes=sum(c["eligible"] for c in cases),
        unannotated_eyes=sum(not c["saved"] for c in cases),
        pending_eyes=sum(c["saved"] and not c["human_reviewed"] for c in cases),
        fully_reviewed_photos=0,
        photos_with_eligible_eyes=0,
        binocular_pairs=0,
        right_right_pairs=0,
    )
    for image in images:
        slots = [c for c in cases if c["image_id"] == image["image_id"]]
        counts["fully_reviewed_photos"] += all(c["human_reviewed"] for c in slots)
        eligible = {(c["annotation_id"], c["eye_side"]) for c in slots if c["eligible"]}
        counts["photos_with_eligible_eyes"] += bool(eligible)
        counts["binocular_pairs"] += sum(
            (int(t["annotation_id"]), "left") in eligible
            and (int(t["annotation_id"]), "right") in eligible
            for t in image["targets"]
        )
        rights = sum(side == "right" for _, side in eligible)
        counts["right_right_pairs"] += rights * (rights - 1) // 2
    return {
        **counts,
        "eligible_eyes_by_category": dict(
            Counter(c["animal_category"] for c in cases if c["eligible"])
        ),
        "exclusions_by_reason": dict(
            Counter(
                c["exclusion_reason"] or c["eligibility_reason"] for c in cases if not c["eligible"]
            )
        ),
    }


def freeze_reviewed_reference(root: Path, output: Path, reviewer: str = "Polun") -> dict[str, Any]:
    """Write a new portable snapshot of every photo the named reviewer has started.

    Includes all COCO competitors and all eye slots on those photographs. Only
    reviewed, visible, sided, measurement-Pass masks become numeric references.
    Source masks, metadata and histories remain untouched; RLE is lossless.
    """
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise Phase1Error("Snapshot output already exists; refusing to overwrite.")
    if not reviewer.strip():
        raise Phase1Error("Reviewer name must not be empty.")
    preserved: dict[str, str] = {}

    def preserve(path: Path) -> None:
        preserved[_relative(root, path)] = sha256_file(path)

    imports: dict[tuple[str, int, int, str], int] = {}
    for path in sorted((root / "outputs/phase3/eye_draft_jobs").glob("*/manifest.json")):
        preserve(path)
        for row in json.loads(path.read_text(encoding="utf-8"))["saved"]:
            key = row["batch"], *_identity(row)
            imports[key] = max(imports.get(key, 0), int(row["annotation_version"]))
    sessions = [root / "configs/annotation_workspace.toml"]
    sessions.extend(sorted((root / "outputs/phase3/eye_annotation_batches").glob("*/session.toml")))
    images: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    datasets: list[dict[str, str]] = []
    seen_images: set[int] = set()
    for session_path in sessions:
        session = read_session(session_path)
        preparation = verify_session(session)
        dataset = _dataset(session)
        if dataset.source_sha256 != preparation["dataset_annotations_sha256"]:
            raise Phase1Error("COCO annotations differ from the prepared session.")
        targets = load_targets(session.proposal, dataset)
        _check_scope(session, targets)
        for path in (
            session.source,
            session.phase1_config,
            session.proposal,
            session.protocol_document,
        ):
            preserve(path)
        for path in sorted(session.ground_truth.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                preserve(path)
        csv_path = session.ground_truth / "eye_mask_annotations.csv"
        before_csv = sha256_file(csv_path)
        rows = load_annotations(
            session.ground_truth, targets, protocol_version=session.protocol_version
        )
        if sha256_file(csv_path) != before_csv:
            raise Phase1Error(
                "Annotation CSV changed during snapshot; retry after saving finishes."
            )
        latest: dict[tuple[int, int, str], dict[str, str]] = {}
        for row in rows:
            key = _identity(row)
            if key not in latest or int(row["annotation_version"]) > int(
                latest[key]["annotation_version"]
            ):
                latest[key] = row
        batch = "demo" if session_path == sessions[0] else session_path.parent.name
        datasets.append({"batch": batch, "annotations_sha256": dataset.source_sha256})
        started = {
            key[0] for key, row in latest.items() if _human_saved(row, batch, imports, reviewer)
        }
        for image_id in dict.fromkeys(int(t["image_id"]) for t in targets):
            if image_id not in started:
                continue
            if image_id in seen_images:
                raise Phase1Error("Duplicate photograph across reviewed workspaces.")
            seen_images.add(image_id)
            image_targets = [t for t in targets if int(t["image_id"]) == image_id]
            first = image_targets[0]
            source_image = session.images / first["file_name"]
            if sha256_file(source_image) != first["image_sha256"]:
                raise Phase1Error("Source photograph hash changed.")
            preserve(source_image)
            width, height = int(first["image_width"]), int(first["image_height"])
            exported_targets = []
            for target in image_targets:
                annotation = dataset.annotations[int(target["annotation_id"])]
                exported_targets.append(
                    {
                        **target,
                        "segmentation": annotation.segmentation,
                        "category_id": annotation.category_id,
                        "iscrowd": annotation.iscrowd,
                        "area": annotation.area,
                    }
                )
                for side in ("left", "right"):
                    key = image_id, int(target["annotation_id"]), side
                    row = latest.get(key)
                    human = _human_saved(row, batch, imports, reviewer)
                    reviewed = bool(
                        human and row and row["mask_review_status"] in ("Accepted", "Rejected")
                    )
                    eligible = bool(
                        reviewed
                        and row
                        and row["mask_review_status"] == "Accepted"
                        and row["visibility"] == "Visible"
                        and row["side_confirmed"] == "Yes"
                        and row["measurement_quality"] == "Pass"
                        and row["mask_path"]
                    )
                    source_mask = (
                        session.ground_truth / row["mask_path"]
                        if row and row["mask_path"]
                        else None
                    )
                    mask_rle = (
                        _encode_mask(source_mask, width, height)
                        if eligible and source_mask
                        else None
                    )
                    reason = (
                        ""
                        if eligible
                        else (
                            "not_annotated"
                            if row is None
                            else "not_human_reviewed"
                            if not reviewed
                            else "review_rejected"
                            if row["mask_review_status"] == "Rejected"
                            else "not_measurement_eligible"
                        )
                    )
                    cases.append(
                        {
                            **(row or {}),
                            "batch": batch,
                            "image_id": image_id,
                            "annotation_id": key[1],
                            "eye_side": side,
                            "animal_category": target["animal_category"],
                            "annotation_version": int(row["annotation_version"]) if row else None,
                            "saved": row is not None,
                            "human_saved": human,
                            "human_reviewed": reviewed,
                            "eligible": eligible,
                            "eligibility_reason": reason,
                            "mask_review_status": row["mask_review_status"]
                            if row
                            else "not_annotated",
                            "exclusion_reason": row["exclusion_reason"] if row else "",
                            "reference_mask_rle": mask_rle,
                            "reference_mask_sha256": sha256_file(source_mask)
                            if source_mask
                            else None,
                            "source_mask": _relative(root, source_mask) if source_mask else None,
                            "preannotation_version": imports.get((batch, *key)),
                        }
                    )
            images.append(
                {
                    "image_id": image_id,
                    "batch": batch,
                    "source_image": _relative(root, source_image),
                    "file_name": first["file_name"],
                    "width": width,
                    "height": height,
                    "image_sha256": first["image_sha256"],
                    "coco_url": f"https://images.cocodataset.org/val2017/{first['file_name']}",
                    "targets": exported_targets,
                }
            )
    if not images:
        raise Phase1Error("No owner-saved annotation photographs were found.")
    for relative, digest in preserved.items():
        if sha256_file(root / relative) != digest:
            raise Phase1Error(f"Source changed during snapshot: {relative}")
    record = {
        "schema_version": 1,
        "protocol_version": V2_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reviewer": reviewer,
        "dataset_role": "reviewed_development_validation",
        "independent_blind_ground_truth": False,
        "selection_policy": (
            "All saved owner-reviewed photographs at snapshot time; include all eye slots "
            "and COCO competitors. Numeric references require Visible/Yes/Accepted/Pass "
            "and a nonempty original-size mask."
        ),
        "dataset_sources": datasets,
        "images": images,
        "cases": cases,
        "summary": _summary(images, cases),
        "source_artifacts_sha256": preserved,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "reference.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return record
