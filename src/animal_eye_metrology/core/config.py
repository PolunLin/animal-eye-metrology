"""Small, strict TOML configuration with portable path resolution."""

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.errors import Phase1Error


@dataclass(frozen=True)
class ContactSheetConfig:
    enabled: bool = False
    columns: int = 2
    rows: int = 3
    panel_width: int = 320
    panel_height: int = 240
    max_images: int = 0


@dataclass(frozen=True)
class AnalysisConfig:
    source: Path
    dataset_name: str
    annotations: Path
    images: Path
    output: Path
    expected_animal_categories: int
    expected_annotations_sha256: str
    contact_sheets: ContactSheetConfig

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        for field in ("source", "annotations", "images", "output"):
            record[field] = str(record[field])
        return record


def _section(data: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise Phase1Error(f"Configuration section [{name}] must be a table.")
    unknown = set(value) - allowed
    if unknown:
        raise Phase1Error(f"Unknown settings in [{name}]: {sorted(unknown)}")
    return value


def _text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise Phase1Error(f"Configuration setting '{key}' must be a nonempty string.")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise Phase1Error(f"Configuration setting '{name}' must be an integer >= {minimum}.")
    return value


def load_config(path: Path) -> AnalysisConfig:
    source = path.expanduser().resolve()
    with source.open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) - {"dataset", "output", "contact_sheets"}:
        raise Phase1Error("Unknown top-level configuration section.")
    dataset = _section(
        data,
        "dataset",
        {
            "name",
            "annotations",
            "images",
            "expected_animal_categories",
            "expected_annotations_sha256",
        },
    )
    output = _section(data, "output", {"directory"})
    contact = _section(data, "contact_sheets", set(ContactSheetConfig.__dataclass_fields__))
    settings = asdict(ContactSheetConfig()) | contact
    if type(settings["enabled"]) is not bool:
        raise Phase1Error("contact_sheets.enabled must be true or false.")
    for name in ("columns", "rows", "panel_width", "panel_height", "max_images"):
        _integer(settings[name], name, minimum=0 if name == "max_images" else 1)
    if min(settings["panel_width"], settings["panel_height"]) < 64:
        raise Phase1Error("Contact sheet panel dimensions must be at least 64 pixels.")
    digest = dataset.get("expected_annotations_sha256", "")
    if not isinstance(digest, str) or (
        digest and (len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest))
    ):
        raise Phase1Error("expected_annotations_sha256 must be empty or 64 hexadecimal characters.")

    def local_path(value: str) -> Path:
        candidate = Path(value).expanduser()
        return (source.parent / candidate).resolve()

    config = AnalysisConfig(
        source=source,
        dataset_name=_text(dataset, "name"),
        annotations=local_path(_text(dataset, "annotations")),
        images=local_path(_text(dataset, "images")),
        output=local_path(_text(output, "directory")),
        expected_animal_categories=_integer(
            dataset.get("expected_animal_categories", 10), "expected_animal_categories"
        ),
        expected_annotations_sha256=digest.lower(),
        contact_sheets=ContactSheetConfig(**settings),
    )
    return config
