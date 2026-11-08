"""Schema-preserving CSV review updates for a local macOS/Linux application."""

import csv
import fcntl
import os
import tempfile
from pathlib import Path
from typing import Any

from animal_eye_metrology.analysis import REVIEW_FIELDS, REVIEW_MANUAL_FIELDS
from animal_eye_metrology.errors import Phase1Error

CHOICES = {
    "left_eye_visibility": ("Yes", "No", "Uncertain"),
    "right_eye_visibility": ("Yes", "No", "Uncertain"),
    "head_pose": ("Front", "Side", "Other"),
    "occlusion": ("None", "Partial", "Heavy"),
    "measurement_eligibility": ("Yes", "No", "Uncertain"),
}
IDENTITY_FIELDS = tuple(field for field in REVIEW_FIELDS if field not in REVIEW_MANUAL_FIELDS)
IMAGE_SELECTION_FIELDS = ("image_id", "selection_status", "review_notes")
SELECTION_STATUSES = ("Selected", "Rejected", "Pending")


def review_key(row: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(row["image_id"]), int(row["annotation_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise Phase1Error("Review records need valid image_id and annotation_id.") from error


def is_reviewed(row: dict[str, str]) -> bool:
    """Completion of required choices, independent of eligibility or visibility outcome."""
    return all(row.get(field) in options for field, options in CHOICES.items())


def load_reviews(path: Path, template: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Merge a valid saved subset into the complete template; never silently drop records."""
    expected = {
        review_key(row): {field: str(row[field]) for field in REVIEW_FIELDS} for row in template
    }
    if len(expected) != len(template):
        raise Phase1Error("The review template contains duplicate instance keys.")
    if not path.exists():
        return list(expected.values())
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise Phase1Error("Review CSV schema differs from eye_visibility_review.csv.")
        seen = set()
        for row in reader:
            key = review_key(row)
            if key in seen:
                raise Phase1Error(f"Duplicate review record: {key}")
            if key not in expected:
                raise Phase1Error(f"Review instance is not a current candidate: {key}")
            if any(row.get(field) != expected[key][field] for field in IDENTITY_FIELDS):
                raise Phase1Error(f"Review identity or category metadata changed: {key}")
            if set(row) != set(REVIEW_FIELDS) or any(value is None for value in row.values()):
                raise Phase1Error(f"Malformed review row: {key}")
            expected[key] = dict(row)
            seen.add(key)
    return list(expected.values())


def _write_atomic(
    path: Path,
    rows: list[dict[str, str]],
    fields: tuple[str, ...] = REVIEW_FIELDS,
) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=".review-",
            suffix=".csv",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialize_reviews(path: Path, template: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Create a working copy once, keeping the source CSV and existing labels untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_reviews(path, template)
        if not path.exists():
            _write_atomic(path, rows)
    return rows


def save_review(
    path: Path,
    template: list[dict[str, Any]],
    key: tuple[int, int],
    updates: dict[str, str],
    *,
    expected: dict[str, str],
) -> dict[str, str]:
    """Reload under a file lock, update one key, and reject stale edits to that same key."""
    if set(updates) - set(REVIEW_MANUAL_FIELDS):
        raise Phase1Error("Only human review fields can be updated.")
    if any(not isinstance(value, str) for value in updates.values()):
        raise Phase1Error("Human review values must be strings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_reviews(path, template)
        index = {review_key(row): row for row in rows}
        if key not in index:
            raise Phase1Error("Selected review instance is not in the candidate list.")
        current = index[key]
        if current != expected:
            raise Phase1Error("此實例已被另一個視窗更新；請重新載入此實例後再儲存。")
        updated = current | updates
        if not is_reviewed(updated):
            raise Phase1Error("請完成左右眼、姿態、遮擋與 Eligibility 選項；可選 Uncertain。")
        current.update(updates)
        _write_atomic(path, rows)
        return dict(current)


def load_image_selections(path: Path, image_ids: list[int]) -> list[dict[str, str]]:
    """An absent image choice remains Pending, independent of any instance review."""
    expected = {
        key: {"image_id": str(key), "selection_status": "Pending", "review_notes": ""}
        for key in sorted(set(image_ids))
    }
    if not path.exists():
        return list(expected.values())
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != IMAGE_SELECTION_FIELDS:
            raise Phase1Error(
                "Image Selection CSV schema must be image_id,selection_status,review_notes."
            )
        seen = set()
        for row in reader:
            try:
                key = int(row["image_id"])
            except (TypeError, ValueError) as error:
                raise Phase1Error("Invalid Image Selection image_id.") from error
            if key not in expected or key in seen:
                raise Phase1Error(f"Unknown or duplicate Image Selection image_id: {key}")
            if (
                set(row) != set(IMAGE_SELECTION_FIELDS)
                or row.get("selection_status") not in SELECTION_STATUSES
                or row.get("review_notes") is None
            ):
                raise Phase1Error(f"Malformed Image Selection record: {key}")
            expected[key] = dict(row)
            seen.add(key)
    return list(expected.values())


def initialize_image_selections(path: Path, image_ids: list[int]) -> list[dict[str, str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_image_selections(path, image_ids)
        if not path.exists():
            _write_atomic(path, rows, IMAGE_SELECTION_FIELDS)
    return rows


def save_image_selection(
    path: Path,
    image_ids: list[int],
    image_id: int,
    status: str,
    notes: str,
    *,
    expected: dict[str, str],
) -> dict[str, str]:
    if status not in SELECTION_STATUSES or not isinstance(notes, str):
        raise Phase1Error("Invalid image selection status or notes.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_image_selections(path, image_ids)
        current = next((row for row in rows if int(row["image_id"]) == image_id), None)
        if current is None:
            raise Phase1Error("Image is not a COCO multi-animal candidate.")
        if current != expected:
            raise Phase1Error("這張圖片已由另一個視窗更新，請重新載入圖片選擇。")
        current.update(selection_status=status, review_notes=notes)
        _write_atomic(path, rows, IMAGE_SELECTION_FIELDS)
        return dict(current)


def image_eye_condition(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Read-only screening hint; never classify unreviewed instances as failures."""
    qualifying = [
        review_key(row)[1]
        for row in rows
        if is_reviewed(row)
        and all(
            row[field] == "Yes"
            for field in ("left_eye_visibility", "right_eye_visibility", "measurement_eligibility")
        )
    ]
    unreviewed = sum(not is_reviewed(row) for row in rows)
    uncertain = sum(
        is_reviewed(row)
        and any(
            row[field] == "Uncertain"
            for field in ("left_eye_visibility", "right_eye_visibility", "measurement_eligibility")
        )
        for row in rows
    )
    if len(qualifying) >= 2:
        condition = "Ready"
    elif unreviewed or uncertain or not rows:
        condition = "Pending"
    else:
        condition = "Insufficient"
    return {
        "condition": condition,
        "qualifying_annotation_ids": qualifying,
        "qualifying_count": len(qualifying),
        "unreviewed_count": unreviewed,
        "uncertain_count": uncertain,
        "total_instances": len(rows),
    }
