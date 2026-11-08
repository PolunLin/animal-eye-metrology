"""Audit upper bounds on COCO study supply without selecting or inferring images."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animal_eye_metrology.coco import load_coco
from animal_eye_metrology.config import load_config
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.verification import sha256_file

LOGGER = logging.getLogger(__name__)
STRATA = ("multi_animal", "single_animal", "background")


def supply_bounds(
    strata: dict[int, str],
    exposed: set[int],
    protected: set[int],
    available: set[int],
    targets: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Return source ceilings, not a split or evidence of image/group integrity."""
    population = set(strata)
    if set(targets) != set(STRATA) or any(
        type(value) is not int or value < 0 for value in targets.values()
    ):
        raise Phase1Error("Supply targets must contain nonnegative counts for all three strata.")
    if set(strata.values()) - set(STRATA):
        raise Phase1Error("Unknown image stratum.")
    if (exposed | protected | available) - population:
        raise Phase1Error("Supply inputs contain image IDs outside the annotation metadata.")
    result = {}
    for label in STRATA:
        members = {key for key, value in strata.items() if value == label}
        eligible = members & available - protected
        holdout = eligible - exposed
        target = targets[label]
        result[label] = {
            "population": len(members),
            "nonzero_local_files": len(members & available),
            "known_exposed": len(members & exposed),
            "protected": len(members & protected),
            "development_before_holdout_upper_bound": len(eligible),
            "holdout_upper_bound": len(holdout),
            "target_per_split": target,
            "development_minimum_shortfall_before_holdout": max(0, target - len(eligible)),
            "holdout_minimum_shortfall": max(0, target - len(holdout)),
            "combined_minimum_shortfall": max(0, 2 * target - len(eligible)),
        }
    return result


def collect_exposed(snapshot: dict[str, Any]) -> set[int]:
    """Use full image histories rather than only the multi-animal intersection."""
    exposed: set[int] = set()
    for field in (
        "pilot100_ids",
        "selected_image_ids",
        "selection_decision_ids",
        "contact_sheet_sample_ids",
        "partially_or_fully_eye_reviewed_ids",
        "eye_method_development_ids",
        "inference_coco_ids",
    ):
        exposed.update(snapshot[field])
    for source in snapshot["phase2_prediction_sources"].values():
        exposed.update(source["all_image_ids"])
    if any(type(key) is not int or key < 0 for key in exposed):
        raise Phase1Error("Exposure snapshot contains invalid image IDs.")
    return exposed


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if config.annotations.name != "instances_val2017.json" or config.images.name != "val2017":
        raise Phase1Error("This approved study accepts COCO val2017 only.")
    snapshot_path = args.exposure_snapshot.resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not snapshot["source_unchanged"]:
        raise Phase1Error("Exposure snapshot did not pass its source stability check.")
    source_paths = {root / path for path in snapshot["source_sha256"]}
    source_paths |= {config_path, config.annotations, snapshot_path, Path(__file__).resolve()}
    before = {str(path.relative_to(root)): sha256_file(path) for path in sorted(source_paths)}
    for relative, digest in snapshot["source_sha256"].items():
        if before[relative] != digest:
            raise Phase1Error(f"Exposure source changed after the snapshot: {relative}")
    dataset = load_coco(
        config.annotations,
        expected_animal_categories=config.expected_animal_categories,
        expected_sha256=config.expected_annotations_sha256,
    )
    counts: Counter[int] = Counter(
        row.image_id
        for row in dataset.annotations.values()
        if row.category_id in dataset.animal_categories and not row.iscrowd
    )
    strata = {
        key: "multi_animal"
        if counts[key] >= 2
        else "single_animal"
        if counts[key]
        else "background"
        for key in dataset.images
    }
    multi = {key for key, value in strata.items() if value == "multi_animal"}
    exposed = collect_exposed(snapshot)
    if multi != set(snapshot["candidate_ids"]):
        raise Phase1Error("Current annotation candidates differ from the exposure snapshot.")
    unexposed_multi = multi - exposed
    if unexposed_multi != set(snapshot["no_known_exposure_candidate_ids"]):
        raise Phase1Error("Exposure snapshot has inconsistent unseen-candidate sets.")
    eye_references = set(args.eye_reference_id)
    protected = unexposed_multi | eye_references
    available = {
        key
        for key, image in dataset.images.items()
        if (config.images / image.file_name).is_file()
        and (config.images / image.file_name).stat().st_size > 0
    }
    targets = dict(zip(STRATA, args.per_split, strict=True))
    bounds = supply_bounds(strata, exposed, protected, available, targets)
    insufficient = any(
        row["holdout_minimum_shortfall"] or row["combined_minimum_shortfall"]
        for row in bounds.values()
    )
    rows = [
        {
            "source_split": "val2017",
            "image_id": key,
            "stratum": strata[key],
            "non_crowd_animal_count": counts[key],
            "nonzero_local_file": key in available,
            "known_exposed": key in exposed,
            "eye_candidate_protected": key in unexposed_multi,
            "eye_reference_protected": key in eye_references,
            "development_source_candidate": key in available and key not in protected,
            "holdout_source_candidate": key in available and key not in protected | exposed,
        }
        for key in sorted(dataset.images)
    ]
    after = {str(path.relative_to(root)): sha256_file(path) for path in sorted(source_paths)}
    if before != after:
        raise Phase1Error("One or more audit inputs changed during collection.")
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "insufficient_supply"
        if insufficient
        else "upper_bounds_sufficient_pending_review",
        "source_split": "val2017",
        "python_version": platform.python_version(),
        "execution_argv": [sys.executable, *sys.argv],
        "annotations_sha256": dataset.source_sha256,
        "source_sha256": before,
        "source_unchanged": before == after,
        "bounds": bounds,
        "protected_unexposed_multi_ids": sorted(unexposed_multi),
        "protected_eye_reference_ids": sorted(eye_references),
        "formal_split_created": False,
        "images_decoded": False,
        "image_digest_or_near_duplicate_review_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "limitation": (
            "Counts are source ceilings before holdout allocation, decoding and group review. "
            "These exclusions can only reduce the supply; this audit cannot authorize inference."
        ),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    (output / "provenance").mkdir()
    shutil.copyfile(snapshot_path, output / "inputs/exposure_snapshot.json")
    shutil.copyfile(Path(__file__), output / "provenance/segmentation_supply.py")
    with (output / "candidate_supply.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result["artifact_sha256"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    with (output / "supply_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exposure-snapshot", type=Path, required=True)
    parser.add_argument("--eye-reference-id", type=int, action="append", required=True)
    parser.add_argument("--per-split", type=int, nargs=3, default=(100, 50, 50))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = audit(args)
    except (OSError, ValueError, KeyError, Phase1Error) as error:
        LOGGER.error(json.dumps({"status": "audit_failed", "error": str(error)}))
        raise SystemExit(1) from error
    LOGGER.info(json.dumps({"status": result["status"], "bounds": result["bounds"]}))


if __name__ == "__main__":
    main()
