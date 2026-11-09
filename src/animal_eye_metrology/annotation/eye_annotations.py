"""Manual eye masks: original-coordinate polygons and append-only CSV revisions."""

import csv
import fcntl
import json
import math
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from animal_eye_metrology.annotation.review_store import _write_atomic
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.data.coco import CocoDataset
from animal_eye_metrology.paths import PROJECT_ROOT

V2_PROTOCOL_VERSION = "eye-visible-eyelid-v2"
PROTOCOL_VERSION = V2_PROTOCOL_VERSION
PROTOCOL_DOCUMENTS = {V2_PROTOCOL_VERSION: "eye_mask_protocol_v2.md"}
SUPPORTED_PROTOCOLS = frozenset(PROTOCOL_DOCUMENTS)
FIELDS = (
    "image_id",
    "annotation_id",
    "eye_side",
    "mask_path",
    "visibility",
    "side_confirmed",
    "mask_review_status",
    "measurement_quality",
    "exclusion_reason",
    "reviewer",
    "protocol_version",
    "annotation_version",
    "annotated_at_utc",
)
CHOICES = {
    "visibility": ("Visible", "Partial", "NotVisible", "Uncertain"),
    "side_confirmed": ("Yes", "No", "Unknown"),
    "mask_review_status": ("Pending", "Accepted", "Rejected", "Uncertain"),
    "measurement_quality": ("Pass", "Fail", "Unknown"),
}


def validate_protocol_version(protocol_version: str) -> str:
    """Only explicitly approved definitions may identify an annotation store."""
    if not isinstance(protocol_version, str) or protocol_version not in SUPPORTED_PROTOCOLS:
        raise Phase1Error(f"不支援的眼部標註規範：{protocol_version!r}")
    return protocol_version


def _source_targets(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            key: row[key]
            for key in (
                "image_id",
                "annotation_id",
                "file_name",
                "image_width",
                "image_height",
                "image_sha256",
            )
        }
        for row in targets
    ]


def _check_store_protocol(
    output: Path, targets: list[dict[str, str]], protocol_version: str
) -> None:
    """Reject cross-protocol access even for an empty, initialized CSV."""
    path = output / "source_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["protocol_version"] != protocol_version or manifest[
            "targets"
        ] != _source_targets(targets):
            raise Phase1Error("標註來源／規範與既有工作目錄不一致，請使用新的輸出目錄。")
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, Phase1Error):
            raise
        raise Phase1Error(f"無法核對眼部標註來源 manifest：{error}") from error


def load_targets(path: Path, dataset: CocoDataset) -> list[dict[str, str]]:
    """Use the approved proposal identities; never infer eye positions from animal masks."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    if not rows:
        raise Phase1Error("眼睛試標清單為空。")
    for row in rows:
        try:
            key = int(row["image_id"]), int(row["annotation_id"])
            image, ann = dataset.images[key[0]], dataset.annotations[key[1]]
            valid = (
                key not in seen
                and ann.image_id == image.id
                and not ann.iscrowd
                and ann.category_id in dataset.animal_categories
                and row["file_name"] == image.file_name
                and int(row["image_width"]) == image.width
                and int(row["image_height"]) == image.height
                and row["animal_category"] == dataset.categories[ann.category_id].name
                and json.loads(row["coco_bbox_xywh"]) == list(ann.bbox)
                and json.loads(row["proposed_eye_sides"]) == ["left", "right"]
                and len(row["image_sha256"]) == 64
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Phase1Error("試標清單缺少有效的 COCO 身份或來源資料。") from error
        if not valid:
            raise Phase1Error(f"試標身份重複或與 COCO 不符：{key}")
        seen.add(key)
    return rows


def empty_geometry(width: int, height: int, bbox: tuple[float, ...]) -> dict[str, Any]:
    x, y, w, h = bbox
    return {
        "contours": [],
        "active": [],
        "operation": "add",
        "view": [max(0, x - 3), max(0, y - 3), min(width, x + w + 3), min(height, y + h + 3)],
    }


def validate_geometry(geometry: Any, width: int, height: int) -> None:
    if not isinstance(geometry, dict):
        raise Phase1Error("輪廓資料格式不正確。")
    contours, active = geometry.get("contours"), geometry.get("active")
    if not isinstance(contours, list) or not isinstance(active, list) or len(contours) > 100:
        raise Phase1Error("輪廓資料格式或數量不正確。")
    view = geometry.get("view", [])
    if (
        not isinstance(view, list)
        or len(view) != 4
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in view)
        or not (0 <= view[0] < view[2] <= width and 0 <= view[1] < view[3] <= height)
    ):
        raise Phase1Error("放大區域必須位於原圖內。")
    if geometry.get("operation") not in ("add", "erase"):
        raise Phase1Error("無效的描邊操作。")
    for points in [active, *[c.get("points") for c in contours if isinstance(c, dict)]]:
        if not isinstance(points, list) or len(points) > 1024:
            raise Phase1Error("每條輪廓最多 1024 個頂點。")
        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                or not (0 <= point[0] <= width and 0 <= point[1] <= height)
            ):
                raise Phase1Error("頂點不是有效的原圖座標。")
    for contour in contours:
        if (
            not isinstance(contour, dict)
            or contour.get("operation") not in ("add", "erase")
            or not isinstance(contour.get("points"), list)
            or len(contour["points"]) < 3
        ):
            raise Phase1Error("封閉輪廓至少需要三個頂點。")
        points = contour["points"]
        area = sum(
            a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1], strict=True)
        )
        if abs(area) < 1e-9:
            raise Phase1Error("輪廓面積為零，請重新描邊。")


def rasterize(geometry: dict[str, Any], width: int, height: int) -> Any:
    """Even-odd polygon fill at original pixel centers; include exact boundary centers."""
    import numpy as np
    from PIL import Image

    validate_geometry(geometry, width, height)
    output = np.zeros((height, width), dtype=bool)
    for contour in geometry["contours"]:
        points = contour["points"]
        x0 = max(0, math.floor(min(p[0] for p in points)))
        y0 = max(0, math.floor(min(p[1] for p in points)))
        x1 = min(width, math.ceil(max(p[0] for p in points)))
        y1 = min(height, math.ceil(max(p[1] for p in points)))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
        xx += 0.5
        yy += 0.5
        inside, boundary = np.zeros(xx.shape, bool), np.zeros(xx.shape, bool)
        for a, b in zip(points, points[1:] + points[:1], strict=True):
            dx, dy = b[0] - a[0], b[1] - a[1]
            cross = (xx - a[0]) * dy - (yy - a[1]) * dx
            boundary |= (
                (abs(cross) < 1e-9)
                & (xx >= min(a[0], b[0]) - 1e-9)
                & (xx <= max(a[0], b[0]) + 1e-9)
                & (yy >= min(a[1], b[1]) - 1e-9)
                & (yy <= max(a[1], b[1]) + 1e-9)
            )
            if dy:
                inside ^= ((a[1] > yy) != (b[1] > yy)) & (xx < a[0] + (yy - a[1]) * dx / dy)
        region = inside | boundary
        if contour["operation"] == "add":
            output[y0:y1, x0:x1] |= region
        else:
            output[y0:y1, x0:x1] &= ~region
    return Image.fromarray(output.astype("uint8") * 255)


def initialize_store(
    output: Path,
    targets: list[dict[str, str]],
    dataset_sha256: str,
    *,
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    validate_protocol_version(protocol_version)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol_version": protocol_version,
        "dataset_sha256": dataset_sha256,
        "targets": _source_targets(targets),
        "coordinate_convention": "top-left image edge (0,0); pixel centers (column+.5,row+.5)",
        "mask_encoding": "original-size grayscale PNG, values 0/255; even-odd pixel-center fill",
    }
    with (output / ".annotations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / "source_manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise Phase1Error("標註來源／規範與既有工作目錄不一致，請使用新的輸出目錄。")
        else:
            if (output / "eye_mask_annotations.csv").exists():
                raise Phase1Error("既有標註缺少來源 manifest，請先人工核對。")
            path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")
        if not (output / "eye_mask_annotations.csv").exists():
            _write_atomic(output / "eye_mask_annotations.csv", [], FIELDS)


def _identity(row: dict[str, str]) -> tuple[int, int, str]:
    return int(row["image_id"]), int(row["annotation_id"]), row["eye_side"]


def _revision_path(output: Path, row: dict[str, str]) -> Path:
    return (
        output
        / "revisions"
        / (
            f"{row['image_id']}_{row['annotation_id']}_{row['eye_side']}_v{row['annotation_version']}.json"
        )
    )


def load_annotations(
    output: Path,
    targets: list[dict[str, str]],
    *,
    protocol_version: str = PROTOCOL_VERSION,
) -> list[dict[str, str]]:
    from PIL import Image

    validate_protocol_version(protocol_version)
    target_index = {(int(t["image_id"]), int(t["annotation_id"])): t for t in targets}
    path = output / "eye_mask_annotations.csv"
    if path.exists() or (output / "source_manifest.json").exists():
        _check_store_protocol(output, targets, protocol_version)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise Phase1Error("Eye Mask CSV schema 不符，不能自動修正或覆寫。")
        rows = list(reader)
    seen = set()
    for row in rows:
        try:
            key = _identity(row)
            target = target_index[key[:2]]
            revision = int(row["annotation_version"])
            if (
                set(row) != set(FIELDS)
                or any(v is None for v in row.values())
                or key[2] not in ("left", "right")
                or revision < 1
                or (*key, revision) in seen
                or row["protocol_version"] != protocol_version
            ):
                raise ValueError("Invalid revision identity")
            for field, choices in CHOICES.items():
                if row[field] not in choices and not (field == "visibility" and not row[field]):
                    raise ValueError(f"Invalid {field}")
            metadata = json.loads(_revision_path(output, row).read_text())
            if metadata["record"] != row or metadata["source_sha256"] != target["image_sha256"]:
                raise ValueError("Revision metadata differs")
            validate_geometry(
                metadata["geometry"], int(target["image_width"]), int(target["image_height"])
            )
            if row["mask_path"]:
                mask = (output / row["mask_path"]).resolve()
                if not mask.is_relative_to((output / "masks").resolve()):
                    raise ValueError("Mask outside output masks directory")
                if sha256_file(mask) != metadata["mask_sha256"]:
                    raise ValueError("Mask hash differs")
                with Image.open(mask) as opened:
                    if (
                        opened.mode != "L"
                        or opened.size != (int(target["image_width"]), int(target["image_height"]))
                        or any(
                            count
                            for value, count in enumerate(opened.histogram())
                            if value not in (0, 255)
                        )
                    ):
                        raise ValueError("Mask encoding or original dimensions differ")
            seen.add((*key, revision))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise Phase1Error(f"Eye Mask 標註或版本檔案損壞：{error}") from error
    return rows


def latest_annotation(
    rows: list[dict[str, str]], key: tuple[int, int, str]
) -> dict[str, str] | None:
    candidates = [row for row in rows if _identity(row) == key]
    return max(candidates, key=lambda row: int(row["annotation_version"]), default=None)


def load_geometry(output: Path, row: dict[str, str]) -> dict[str, Any]:
    return json.loads(_revision_path(output, row).read_text())["geometry"]


def save_annotation(
    output: Path,
    targets: list[dict[str, str]],
    key: tuple[int, int, str],
    geometry: dict[str, Any],
    updates: dict[str, str],
    *,
    expected_version: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, str]:
    """Publish files before the atomic CSV commit; never overwrite prior revisions."""
    validate_protocol_version(protocol_version)
    _check_store_protocol(output, targets, protocol_version)
    target = next(
        (t for t in targets if (int(t["image_id"]), int(t["annotation_id"])) == key[:2]), None
    )
    if target is None or key[2] not in ("left", "right"):
        raise Phase1Error("眼睛不在試標清單內。")
    required = {*CHOICES, "exclusion_reason", "reviewer"}
    if set(updates) != required or any(not isinstance(v, str) for v in updates.values()):
        raise Phase1Error("審核欄位不完整。")
    for field, choices in CHOICES.items():
        if updates[field] not in choices and not (field == "visibility" and not updates[field]):
            raise Phase1Error(f"無效審核選項：{field}")
    mask = rasterize(geometry, int(target["image_width"]), int(target["image_height"]))
    nonempty = mask.getbbox() is not None
    status = updates["mask_review_status"]
    if status != "Pending" and (not updates["reviewer"].strip() or not updates["visibility"]):
        raise Phase1Error("審核結果需要 Reviewer 與可見性；草稿可保留空白。")
    if status == "Accepted" and (
        not nonempty
        or geometry["active"]
        or updates["side_confirmed"] != "Yes"
        or updates["visibility"] not in ("Visible", "Partial")
    ):
        raise Phase1Error("接受標註需完成非空輪廓、確認解剖左右，並完成可見性判讀。")
    if updates["measurement_quality"] == "Pass" and (
        status != "Accepted"
        or updates["visibility"] != "Visible"
        or updates["side_confirmed"] != "Yes"
        or not nonempty
    ):
        raise Phase1Error("量測 Pass 需接受完整可見眼部 Mask，並確認解剖左右。")
    if (status == "Rejected" or updates["measurement_quality"] == "Fail") and not updates[
        "exclusion_reason"
    ].strip():
        raise Phase1Error("Rejected／Fail 需要填寫原因。")
    if updates["visibility"] == "NotVisible" and nonempty:
        raise Phase1Error("NotVisible 不應同時保存非空眼睛 Mask，請先清除輪廓或修正判讀。")
    with (output / ".annotations.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_annotations(output, targets, protocol_version=protocol_version)
        current = latest_annotation(rows, key)
        if (int(current["annotation_version"]) if current else 0) != expected_version:
            raise Phase1Error("此眼睛已由另一個視窗更新；請先重新載入，原紀錄未被覆寫。")
        revisions = output / "revisions"
        revisions.mkdir(exist_ok=True)
        # A process crash can leave unreferenced files; skip their version rather than replace them.
        prefix = f"{key[0]}_{key[1]}_{key[2]}_v"
        used = [int(p.stem.removeprefix(prefix)) for p in revisions.glob(prefix + "*.json")]
        mask_dir = output / "masks" / str(key[0])
        mask_prefix = f"{key[1]}_{key[2]}_v"
        used += [
            int(p.stem.removeprefix(mask_prefix)) for p in mask_dir.glob(mask_prefix + "*.png")
        ]
        revision = max([expected_version, *used]) + 1
        relative = f"masks/{key[0]}/{key[1]}_{key[2]}_v{revision}.png" if nonempty else ""
        row = {
            "image_id": str(key[0]),
            "annotation_id": str(key[1]),
            "eye_side": key[2],
            "mask_path": relative,
            **updates,
            "protocol_version": protocol_version,
            "annotation_version": str(revision),
            "annotated_at_utc": datetime.now(UTC).isoformat(),
        }
        created = []
        try:
            if relative:
                mask_path = output / relative
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                with mask_path.open("xb") as handle:
                    created.append(mask_path)
                    mask.save(handle, format="PNG")
            metadata = {
                "record": row,
                "source_sha256": target["image_sha256"],
                "geometry": geometry,
                "mask_sha256": sha256_file(output / relative) if relative else None,
                "coordinate_convention": "pixel centers (column+.5,row+.5)",
                "tool": {
                    "python": platform.python_version(),
                    "Pillow": version("Pillow"),
                    "backend_sha256": sha256_file(Path(__file__)),
                    "canvas_sha256": sha256_file(PROJECT_ROOT / "app/assets/eye_canvas.js")
                    if (PROJECT_ROOT / "app/assets/eye_canvas.js").exists()
                    else None,
                },
            }
            revision_path = _revision_path(output, row)
            with revision_path.open("x", encoding="utf-8") as handle:
                created.append(revision_path)
                json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            _write_atomic(output / "eye_mask_annotations.csv", [*rows, row], FIELDS)
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
    return row
