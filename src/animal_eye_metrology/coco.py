"""Read COCO instances without modifying annotations or accessing the network."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from animal_eye_metrology.errors import Phase1Error


@dataclass(frozen=True)
class Category:
    id: int
    name: str
    supercategory: str


@dataclass(frozen=True)
class ImageRecord:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class Annotation:
    id: int
    image_id: int
    category_id: int
    iscrowd: int
    bbox: tuple[float, float, float, float]
    area: float
    segmentation: list[list[float]] | dict[str, Any]


@dataclass(frozen=True)
class CocoDataset:
    categories: dict[int, Category]
    images: dict[int, ImageRecord]
    annotations: dict[int, Annotation]
    source_sha256: str
    info: dict[str, Any]

    @property
    def animal_categories(self) -> dict[int, Category]:
        return {
            key: cat
            for key, cat in sorted(self.categories.items())
            if cat.supercategory == "animal"
        }


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Phase1Error(f"{label} must be an integer >= {minimum}.")
    return value


def _number(value: Any, label: str, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise Phase1Error(f"{label} must be a finite number.")
    try:
        result = float(value)
    except OverflowError as error:
        raise Phase1Error(f"{label} must be a finite number.") from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise Phase1Error(f"{label} must be finite" + (" and nonnegative." if nonnegative else "."))
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase1Error(f"{label} must be a nonempty string.")
    return value


def _records(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = data.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise Phase1Error(f"'{key}' must be a list of objects.")
    return rows


def _unique_id(row: dict[str, Any], existing: dict[int, Any], kind: str) -> int:
    value = _integer(row.get("id"), f"{kind}.id")
    if value in existing:
        raise Phase1Error(f"Duplicate {kind} ID: {value}")
    return value


def _validate_compressed_rle(encoded: str, pixel_count: int, label: str) -> None:
    """Check signed, delta-coded runs before passing data to the native mask decoder.

    Encoding reference: cocodataset/cocoapi common/maskApi.c, rleToString/rleFrString.
    This validates run lengths only; mask rasterization remains with pycocotools.
    """
    runs: list[int] = []
    cursor, total = 0, 0
    while cursor < len(encoded):
        value, shift = 0, 0
        while True:
            if cursor >= len(encoded) or shift >= 35:
                raise Phase1Error(f"{label}: truncated or oversized compressed RLE run.")
            char = ord(encoded[cursor]) - 48
            cursor += 1
            value |= (char & 31) << shift
            shift += 5
            if not char & 32:
                if char & 16:
                    value |= -1 << shift
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0 or value > min(pixel_count, 2**32 - 1):
            raise Phase1Error(f"{label}: compressed RLE run length is out of range.")
        total += value
        if total > pixel_count:
            raise Phase1Error(f"{label}: compressed RLE runs exceed the image.")
        runs.append(value)
    if total != pixel_count:
        raise Phase1Error(f"{label}: compressed RLE run lengths do not cover the image.")


def _segmentation(value: Any, image: ImageRecord, label: str) -> Any:
    if isinstance(value, list) and value:
        for polygon in value:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                raise Phase1Error(f"{label}: polygons need at least three coordinate pairs.")
            for coordinate in polygon:
                _number(coordinate, label)
        return value
    if isinstance(value, dict):
        size, counts = value.get("size"), value.get("counts")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(item) is not int for item in size)
            or size != [image.height, image.width]
        ):
            raise Phase1Error(f"{label}: RLE size must match image dimensions.")
        if isinstance(counts, list) and counts:
            for count in counts:
                _integer(count, f"{label}.counts")
            if sum(counts) != image.height * image.width:
                raise Phase1Error(f"{label}: RLE run lengths do not cover the image.")
        elif isinstance(counts, str) and counts:
            if any(not 48 <= ord(char) <= 111 for char in counts):
                raise Phase1Error(f"{label}: invalid compressed RLE characters.")
            _validate_compressed_rle(counts, image.height * image.width, label)
        else:
            raise Phase1Error(f"{label}: missing or invalid RLE counts.")
        return value
    raise Phase1Error(f"{label}: expected nonempty COCO polygons or RLE.")


def _reject_constant(value: str) -> NoReturn:
    raise Phase1Error(f"Non-finite JSON constant is not allowed: {value}")


def load_coco(
    path: Path, *, expected_animal_categories: int = 10, expected_sha256: str = ""
) -> CocoDataset:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and digest != expected_sha256.lower():
        raise Phase1Error("Annotation SHA-256 does not match the configured reference.")
    try:
        data = json.loads(payload, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise Phase1Error(f"Invalid annotation JSON: {error}") from error
    if not isinstance(data, dict):
        raise Phase1Error("The COCO root must be an object.")
    categories: dict[int, Category] = {}
    for row in _records(data, "categories"):
        key = _unique_id(row, categories, "category")
        categories[key] = Category(
            key,
            _text(row.get("name"), "category.name"),
            _text(row.get("supercategory"), "category.supercategory"),
        )
    animals = {key for key, cat in categories.items() if cat.supercategory == "animal"}
    if len(animals) != expected_animal_categories:
        raise Phase1Error(
            f"Expected {expected_animal_categories} animal categories in metadata; "
            f"found {len(animals)}. Check the input file."
        )
    images: dict[int, ImageRecord] = {}
    for row in _records(data, "images"):
        key = _unique_id(row, images, "image")
        name = _text(row.get("file_name"), "image.file_name")
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise Phase1Error(f"Unsafe image file_name: {name}")
        images[key] = ImageRecord(
            key,
            name,
            _integer(row.get("width"), "image.width", 1),
            _integer(row.get("height"), "image.height", 1),
        )
    annotations: dict[int, Annotation] = {}
    for row in _records(data, "annotations"):
        key = _unique_id(row, annotations, "annotation")
        image_id = _integer(row.get("image_id"), f"annotation {key}.image_id")
        category_id = _integer(row.get("category_id"), f"annotation {key}.category_id")
        if image_id not in images or category_id not in categories:
            raise Phase1Error(f"Annotation {key} refers to an unknown image or category.")
        crowd = _integer(row.get("iscrowd"), f"annotation {key}.iscrowd")
        if crowd not in (0, 1):
            raise Phase1Error(f"Annotation {key}.iscrowd must be 0 or 1.")
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise Phase1Error(f"Annotation {key}.bbox must be [x, y, width, height].")
        x, y, width, height = [
            _number(item, f"annotation {key}.bbox", index >= 2) for index, item in enumerate(bbox)
        ]
        _number(width * height, f"annotation {key}.bbox area")
        area = _number(row.get("area"), f"annotation {key}.area", True)
        segmentation = _segmentation(
            row.get("segmentation"), images[image_id], f"annotation {key}.segmentation"
        )
        annotations[key] = Annotation(
            key, image_id, category_id, crowd, (x, y, width, height), area, segmentation
        )
    info = data.get("info", {})
    if not isinstance(info, dict):
        raise Phase1Error("COCO info must be an object if present.")
    return CocoDataset(categories, images, annotations, digest, info)
