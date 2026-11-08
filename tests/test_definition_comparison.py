"""Invented geometry verifies definition changes without inventing research ground truth."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from animal_eye_metrology.annotation.eye_annotations import (  # noqa: E402
    PROTOCOL_VERSION,
    V2_PROTOCOL_VERSION,
    empty_geometry,
    initialize_store,
    save_annotation,
)
from animal_eye_metrology.core.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.core.verification import sha256_file  # noqa: E402
from animal_eye_metrology.measurement.measurement_baseline import COORDINATES, derive, write_csv  # noqa: E402
from animal_eye_metrology.measurement.metrology import build_measurements  # noqa: E402
from animal_eye_metrology.research import definition_comparison as comparison  # noqa: E402
from animal_eye_metrology.research.eye_comparison import distance_statistics, eye_statistics  # noqa: E402
from animal_eye_metrology.research.phase2 import write_json  # noqa: E402


def _hashes(directory: Path, excluded: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and str(path.relative_to(directory)) not in excluded
    }


def _reference(path: Path, protocol: str) -> None:
    frozen = path / "frozen"
    (frozen / "images").mkdir(parents=True)
    targets = []
    for image_id in (1, 2, 3):
        filename = f"{image_id:012d}.png"
        Image.new("RGB", (64, 32), "gray").save(frozen / "images" / filename)
        for animal in (1, 2):
            targets.append(
                {
                    "image_id": str(image_id),
                    "annotation_id": str(image_id * 10 + animal),
                    "file_name": filename,
                    "image_width": "64",
                    "image_height": "32",
                    "image_sha256": sha256_file(frozen / "images" / filename),
                    "animal_category": "SYNTHETIC_TEST_ONLY",
                }
            )
    initialize_store(frozen, targets, "synthetic-dataset", protocol_version=protocol)
    review = {
        "visibility": "Visible",
        "side_confirmed": "Yes",
        "mask_review_status": "Accepted",
        "measurement_quality": "Pass",
        "reviewer": "SYNTHETIC QA ONLY",
        "exclusion_reason": "",
    }
    for target in targets:
        for side in ("left", "right"):
            x = (int(target["annotation_id"]) % 10 - 1) * 20 + (2 if side == "left" else 9)
            width = 3 if protocol == V2_PROTOCOL_VERSION and side == "left" else 2
            height = 3 if protocol == V2_PROTOCOL_VERSION and side == "right" else 2
            geometry = empty_geometry(64, 32, (0, 0, 64, 32))
            geometry["contours"] = [
                {
                    "operation": "add",
                    "points": [[x, 4], [x + width, 4], [x + width, 4 + height], [x, 4 + height]],
                }
            ]
            save_annotation(
                frozen,
                targets,
                (int(target["image_id"]), int(target["annotation_id"]), side),
                geometry,
                review,
                expected_version=0,
                protocol_version=protocol,
            )
    write_csv(frozen / "proposed_instances.csv", targets)
    write_json(
        frozen / "snapshot_manifest.json",
        {
            "coordinate_convention": COORDINATES,
            "protocol_version": protocol,
            "dataset_annotations_sha256": "synthetic-dataset",
            "eye_count": 12,
            "files_sha256": _hashes(frozen, {"snapshot_manifest.json"}),
        },
    )
    eyes, distances = derive(frozen)
    write_csv(path / "eye_centers.csv", eyes)
    write_csv(path / "measurements.csv", distances)
    write_json(
        path / "run_manifest.json",
        {
            "snapshot_sha256": sha256_file(frozen / "snapshot_manifest.json"),
            "artifacts_sha256": _hashes(path, {"run_manifest.json"}),
        },
    )


@pytest.fixture
def comparison_config(tmp_path: Path) -> Path:
    for name, protocol in (("reference", PROTOCOL_VERSION), ("v2", V2_PROTOCOL_VERSION)):
        _reference(tmp_path / name, protocol)
    eyes1, measurements1 = derive(tmp_path / "reference/frozen")
    eyes2, _ = derive(tmp_path / "v2/frozen")
    new = comparison.index(eyes2, comparison.EYE_KEY)
    summaries, results, measurements, predictions = {}, {}, {}, {}
    sources = {
        "A_r100": "automatic",
        "A_r050": "ablation",
        "B_r100": "ablation",
        "auto_box_k050_point": "auto_box",
    }
    for condition, source in sources.items():
        directory = tmp_path / source
        directory.mkdir(exist_ok=True)
        output = []
        records = []
        for number, reference in enumerate(eyes1):
            identity = {field: reference[field] for field in comparison.EYE_KEY}
            relative = f"masks/{condition}/{number}.png"
            old_mask = comparison._binary(tmp_path / "reference" / reference["mask_path"])
            predicted = old_mask.copy()
            if condition == "A_r050":
                predicted = comparison._binary(
                    tmp_path
                    / "v2"
                    / new[comparison.key(reference, comparison.EYE_KEY)]["mask_path"]
                )
                if number == 0:
                    predicted = None
            if condition == "auto_box_k050_point" and number == 1:
                predicted = np.zeros_like(old_mask)
            if predicted is not None:
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(predicted * 255).save(destination)
            row = {
                **identity,
                "prediction_id": f"p{reference['annotation_id']}",
                "animal_category": "SYNTHETIC_TEST_ONLY",
                "visibility_status": "unverified",
                "failure_reason": "synthetic_missing" if predicted is None else "",
                "predicted_mask_path": relative if predicted is not None else "",
                **eye_statistics(predicted, old_mask),
            }
            output.append(row)
            records.append(
                {
                    **identity,
                    "condition_id": condition,
                    "prediction_id": row["prediction_id"],
                    "point_xy": [2.5, 4.5],
                    "mask_path": row["predicted_mask_path"],
                }
            )
        centers = [
            {
                **{field: row[field] for field in comparison.EYE_KEY},
                "annotation_version": 1,
                "centroid_x_px": row["predicted_cx_px"],
                "centroid_y_px": row["predicted_cy_px"],
            }
            for row in output
        ]
        dists = comparison.index(build_measurements(centers), comparison.DISTANCE_KEY)
        measured = [
            {
                **{field: ref[field] for field in comparison.DISTANCE_KEY},
                "visibility_status": "unverified",
                **distance_statistics(
                    dists[comparison.key(ref, comparison.DISTANCE_KEY)]["distance_px"],
                    ref["distance_px"],
                ),
            }
            for ref in measurements1
        ]
        summaries[condition] = {
            "condition_id": condition,
            "point_source": "manual" if condition.startswith("B") else "hrnet",
            "box_source": "geometry" if source == "auto_box" else "none",
            **comparison._summary(output, measured),
        }
        results[condition], measurements[condition], predictions[condition] = (
            output,
            measured,
            records,
        )
    snapshot = sha256_file(tmp_path / "reference/frozen/snapshot_manifest.json")
    for source, conditions in (
        ("automatic", ["A_r100"]),
        ("ablation", ["A_r100", "A_r050", "B_r100"]),
        ("auto_box", ["auto_box_k050_point"]),
    ):
        directory = tmp_path / source
        if source == "automatic":
            values = {"eyes": results["A_r100"], "measurements": measurements["A_r100"]}
            write_json(directory / "segmentation.json", {"results": predictions["A_r100"]})
        else:
            values = {
                "eye_metrics.csv": [
                    dict(row, condition_id=condition)
                    for condition in conditions
                    for row in results[condition]
                ],
                "measurement_comparison.csv": [
                    dict(row, condition_id=condition)
                    for condition in conditions
                    for row in measurements[condition]
                ],
                "condition_summary.csv": [summaries[c] for c in conditions],
                "localization_metrics.csv": [],
                "promotion_checks.csv": [],
            }
            write_json(
                directory / "predictions.json", [row for c in conditions for row in predictions[c]]
            )
        write_json(directory / "evaluation_values.json", values)
        write_json(
            directory / "run_manifest.json",
            {
                "primary_snapshot_sha256": snapshot,
                "files_sha256": _hashes(directory, {"run_manifest.json"}),
            },
        )
    results_config = tmp_path / "results.toml"
    results_config.write_text(
        "\n".join(
            f'{source}_run = "{source}"'
            for source in ("automatic", "ablation", "auto_box", "reference")
        )
        + f'\nprimary_snapshot_sha256 = "{snapshot}"\n'
    )
    config = tmp_path / "definition.toml"
    config.write_text(
        'results_config = "results.toml"\nv2_reference = "v2"\noutput_directory = "comparison"\n'
        "expected_images = 3\nexpected_instances = 6\nexpected_eyes = 12\n"
    )
    return config


def test_reuses_masks_and_separates_target_shift_from_model_errors(comparison_config: Path) -> None:
    cfg = comparison.read_settings(comparison_config)
    before = {p: p.read_bytes() for p in comparison_config.parent.rglob("*") if p.is_file()}
    assert comparison.check_ready(cfg)["ready"] is True
    inputs = comparison.load_inputs(cfg)
    tables = comparison.evaluate(inputs)
    assert len(tables["eye_comparison.csv"]) == 48
    assert len(tables["distance_comparison.csv"]) == 36
    baseline = next(r for r in tables["condition_summary.csv"] if r["condition_id"] == "A_r100")
    assert baseline["v1_mean_iou_all_targets_failure_as_zero"] == 1
    assert baseline["v2_mean_iou_all_targets_failure_as_zero"] == pytest.approx(2 / 3)
    assert baseline["v1_mean_centroid_error_px"] == 0
    assert baseline["v2_mean_centroid_error_px"] == pytest.approx(0.5)
    missing = next(r for r in tables["condition_summary.csv"] if r["condition_id"] == "A_r050")
    assert missing["v2_mean_iou_all_targets_failure_as_zero"] == pytest.approx(11 / 12)
    assert (
        missing["v2_inference_failed"],
        missing["v2_centroid_defined"],
        missing["v2_binocular_defined"],
    ) == (1, 11, 5)
    empty = next(
        r for r in tables["condition_summary.csv"] if r["condition_id"].startswith("auto_box")
    )
    assert empty["v2_empty_masks"] == 1
    assert empty["method_role"] == "primary_not_promoted"
    human = next(r for r in tables["condition_summary.csv"] if r["condition_id"] == "B_r100")
    assert human["human_prompts_remain_from_v1_experiment"] is True
    for row in tables["distance_comparison.csv"]:
        assert row["v1_predicted_distance_px"] == row["v2_predicted_distance_px"]
    assert all(r["centroid_shift_px"] == 0.5 for r in tables["reference_eye_changes.csv"])
    assert {p: p.read_bytes() for p in before} == before


def test_publication_is_separate_versioned_and_never_overwrites(comparison_config: Path) -> None:
    cfg = comparison.read_settings(comparison_config)
    output = comparison.compare(cfg)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["model_inference"] is False
    assert manifest["candidate_selection_changed"] is False
    assert manifest["baseline_promoted"] is False
    for relative, digest in manifest["files_sha256"].items():
        assert sha256_file(output / relative) == digest
    with pytest.raises(Phase1Error, match="overwrite"):
        comparison.compare(cfg)


def test_missing_v2_never_creates_fake_comparison(comparison_config: Path, capsys) -> None:
    text = comparison_config.read_text().replace(
        'v2_reference = "v2"', 'v2_reference = "missing-v2"'
    )
    comparison_config.write_text(text)
    before = set(comparison_config.parent.rglob("*"))
    assert comparison.main(["--config", str(comparison_config), "--check"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "awaiting_v2_reference"
    assert set(comparison_config.parent.rglob("*")) == before
    assert comparison.main(["--config", str(comparison_config)]) == 2
    assert not (comparison_config.parent / "comparison").exists()


def test_source_tampering_wrong_protocol_and_alias_outputs_are_rejected(
    comparison_config: Path,
) -> None:
    cfg = comparison.read_settings(comparison_config)
    bad_output = cfg | {"output_directory": comparison_config.parent / "v2/subdirectory"}
    with pytest.raises(Phase1Error, match="separate"):
        comparison.check_ready(bad_output)
    bad_reference = cfg | {"v2_reference": comparison_config.parent / "reference"}
    with pytest.raises(Phase1Error, match="protocol"):
        comparison.load_inputs(bad_reference)
    inputs = comparison.load_inputs(cfg)
    path = comparison_config.parent / "automatic/masks/A_r100/0.png"
    Image.new("L", (64, 32), 0).save(path)
    with pytest.raises(ValueError, match="雜湊"):
        comparison.evaluate(inputs)


def test_identity_and_image_mismatch_do_not_become_smaller_evaluation(
    comparison_config: Path,
) -> None:
    cfg = comparison.read_settings(comparison_config)
    inputs = comparison.load_inputs(cfg)
    with pytest.raises(Phase1Error, match="identities"):
        comparison.validate_scope(replace(inputs, v2_eyes=inputs.v2_eyes[:-1]), cfg)
    snapshot = json.loads(inputs.v2.asset_path("frozen/snapshot_manifest.json").read_text())
    snapshot["files_sha256"]["images/000000000001.png"] = "0" * 64
    (inputs.v2.directory / "frozen/snapshot_manifest.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="雜湊"):
        comparison.validate_scope(inputs, cfg)


def test_changed_source_manifest_prevents_publication(comparison_config: Path, monkeypatch) -> None:
    cfg = comparison.read_settings(comparison_config)
    original = comparison.evaluate

    def change_after_evaluation(inputs):
        result = original(inputs)
        path = inputs.bundle.runs["automatic"].directory / "run_manifest.json"
        data = json.loads(path.read_text())
        data["modified_during_run"] = True
        path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(comparison, "evaluate", change_after_evaluation)
    with pytest.raises(Phase1Error, match="manifest changed"):
        comparison.compare(cfg)
    assert not cfg["output_directory"].exists()
