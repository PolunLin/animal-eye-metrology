"""CSV exports and a provenance manifest, published only after successful completion."""

import csv
import hashlib
import json
import platform
import shlex
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from animal_eye_metrology import __version__
from animal_eye_metrology.analysis import (
    AREA_FIELDS,
    CANDIDATE_FIELDS,
    CATEGORY_FIELDS,
    INSTANCE_FIELDS,
    REVIEW_FIELDS,
    AnalysisResult,
)
from animal_eye_metrology.coco import CocoDataset
from animal_eye_metrology.config import AnalysisConfig
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.verification import sha256_file
from animal_eye_metrology.visualization import render_contact_sheets


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_output(config: AnalysisConfig) -> None:
    output = config.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise Phase1Error(
            f"Output must be new or empty: {output}. "
            "Choose a new directory to preserve earlier results and manual reviews."
        )
    if config.annotations.is_relative_to(output) or output == config.annotations:
        raise Phase1Error("Output cannot contain or replace the annotation input.")
    if output.is_relative_to(config.images) or config.images.is_relative_to(output):
        raise Phase1Error("Output and the image directory must be separate.")


def _environment() -> dict[str, Any]:
    versions = {}
    for package in ("pillow", "pycocotools", "numpy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
        "optional_packages": versions,
        "source_sha256": {
            path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def write_outputs(
    dataset: CocoDataset,
    analysis: AnalysisResult,
    config: AnalysisConfig,
    argv: list[str],
) -> dict[str, Any]:
    validate_output(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".phase1-", dir=config.output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        _write_csv(stage / "category_statistics.csv", CATEGORY_FIELDS, analysis.categories)
        _write_csv(stage / "animal_instances.csv", INSTANCE_FIELDS, analysis.instances)
        _write_csv(stage / "area_summary.csv", AREA_FIELDS, analysis.area_summary)
        for name, rows in analysis.candidates.items():
            _write_csv(stage / f"{name}.csv", CANDIDATE_FIELDS, rows)
        _write_csv(stage / "eye_visibility_review.csv", REVIEW_FIELDS, analysis.review)
        contact = (
            render_contact_sheets(dataset, analysis, config, stage)
            if config.contact_sheets.enabled
            else {"enabled": False, "verified_images": [], "note": "Images were not inspected."}
        )
        manifest = {
            "schema_version": 1,
            "dataset": {
                "declared_name": config.dataset_name,
                "info": dataset.info,
                "total_images": len(dataset.images),
                "total_annotations": len(dataset.annotations),
                "annotations_sha256": dataset.source_sha256,
                "expected_sha256": config.expected_annotations_sha256 or None,
                "reference_digest_matched": True if config.expected_annotations_sha256 else None,
                "authenticity_verified": False,
            },
            "configuration": config.as_record(),
            "configuration_sha256": hashlib.sha256(config.source.read_bytes()).hexdigest(),
            "execution": {
                "argv": argv,
                "working_directory": str(Path.cwd()),
                "reproduce": shlex.join([sys.executable, "-m", "animal_eye_metrology", *argv]),
                "random_seed": None,
            },
            "environment": _environment(),
            "method": {
                "animal_metadata": "supercategory == animal; no fixed IDs or names",
                "eligible_instance": "animal annotation with iscrowd == 0",
                "candidate_threshold": "at least two eligible animal instances",
                "groups_may_overlap": True,
                "area_population": "eligible animal instances, including single-animal images",
                "bbox_area": "width * height, without clipping",
                "segmentation_area": "original annotation area, not recomputed from masks",
                "area_fraction": "area / (image width * image height)",
                "quantiles": "linear interpolation at (n-1)*q; empty groups are blank",
                "ordering": "image ID then annotation ID; categories by category ID",
                "eye_labels": "blank; manual review protocol pending approval",
                "model": None,
                "measurement": None,
            },
            "summary": analysis.summary(),
            "contact_sheets": contact,
            "artifacts_sha256": {
                str(path.relative_to(stage)): sha256_file(path)
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
        }
        (stage / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        # Renaming the staging directory avoids leaving partial results after a failed run.
        stage.rename(config.output)
    return manifest
