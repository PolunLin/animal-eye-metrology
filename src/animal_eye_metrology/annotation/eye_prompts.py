"""Human point/box prompts only; never read eye masks or derive prompts from Ground Truth."""

import csv
import fcntl
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animal_eye_metrology.annotation.review_store import _write_atomic
from animal_eye_metrology.core.errors import Phase1Error

FIELDS = (
    "image_id",
    "annotation_id",
    "eye_side",
    "prompt_version",
    "point_x",
    "point_y",
    "box_x0",
    "box_y0",
    "box_x1",
    "box_y1",
    "prompt_source",
    "reviewer",
    "annotated_at_utc",
    "image_sha256",
)


def validate_prompt(row: dict[str, Any], target: dict[str, str]) -> None:
    if (
        int(row["image_id"]) != int(target["image_id"])
        or int(row["annotation_id"]) != int(target["annotation_id"])
        or row["eye_side"] not in ("left", "right")
        or row["prompt_source"] != "manual_original_image"
        or row["image_sha256"] != target["image_sha256"]
        or not row["reviewer"].strip()
        or int(row["prompt_version"]) < 1
    ):
        raise Phase1Error("Prompt identity, source, version or reviewer is invalid.")
    values = [float(row[k]) for k in ("point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1")]
    px, py, x0, y0, x1, y1 = values
    width, height = int(target["image_width"]), int(target["image_height"])
    if not (
        all(math.isfinite(v) for v in values)
        and 0 <= x0 < px < x1 <= width
        and 0 <= y0 < py < y1 <= height
    ):
        raise Phase1Error("Draw an in-image box and an interior positive point.")


def load_prompts(directory: Path, targets: list[dict[str, str]]) -> list[dict[str, str]]:
    path = directory / "prompts.csv"
    if not path.exists():
        return []
    index = {(int(t["image_id"]), int(t["annotation_id"])): t for t in targets}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise Phase1Error("Prompt CSV schema differs; refusing to overwrite.")
        rows = list(reader)
    seen = set()
    for row in rows:
        try:
            key = (
                int(row["image_id"]),
                int(row["annotation_id"]),
                row["eye_side"],
                int(row["prompt_version"]),
            )
            if key in seen:
                raise ValueError("duplicate version")
            validate_prompt(row, index[key[:2]])
            datetime.fromisoformat(row["annotated_at_utc"])
        except (KeyError, ValueError, TypeError) as error:
            raise Phase1Error(f"Invalid prompt CSV: {error}") from error
        seen.add(key)
    return rows


def latest_prompt(rows: list[dict[str, str]], key: tuple[int, int, str]) -> dict[str, str] | None:
    matches = [
        r for r in rows if (int(r["image_id"]), int(r["annotation_id"]), r["eye_side"]) == key
    ]
    return max(matches, key=lambda r: int(r["prompt_version"]), default=None)


def save_prompt(
    directory: Path,
    targets: list[dict[str, str]],
    key: tuple[int, int, str],
    geometry: dict[str, Any],
    reviewer: str,
    expected_version: int,
) -> dict[str, str]:
    target = next(
        (t for t in targets if (int(t["image_id"]), int(t["annotation_id"])) == key[:2]), None
    )
    if target is None:
        raise Phase1Error("Animal identity is not in the approved prompt target list.")
    if not geometry.get("point") or not geometry.get("box"):
        raise Phase1Error("Save requires one box and one positive point.")
    point, box = geometry["point"], geometry["box"]
    row = dict(
        zip(
            FIELDS,
            (
                *key,
                expected_version + 1,
                *point,
                *box,
                "manual_original_image",
                reviewer.strip(),
                datetime.now(UTC).isoformat(),
                target["image_sha256"],
            ),
            strict=True,
        )
    )
    validate_prompt(row, target)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".prompts.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_prompts(directory, targets)
        latest = latest_prompt(rows, key)
        if (int(latest["prompt_version"]) if latest else 0) != expected_version:
            raise Phase1Error("Another window saved a newer prompt; reload before saving.")
        saved = {k: str(v) for k, v in row.items()}
        _write_atomic(directory / "prompts.csv", [*rows, saved], FIELDS)
    return saved


def crop_for_prompt(row: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(row[k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1")]
    side = max(32.0, 4 * max(x1 - x0, y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (
        max(0, math.floor(cx - side / 2)),
        max(0, math.floor(cy - side / 2)),
        min(width, math.ceil(cx + side / 2)),
        min(height, math.ceil(cy + side / 2)),
    )
