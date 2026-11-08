"""Invented geometry and temporary annotations only; not animal research results."""

import csv
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from animal_eye_metrology.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.metrology import (  # noqa: E402
    build_measurements,
    euclidean_distance,
    mask_centroid,
    mask_overlap,
)


def test_mask_overlap_known_sets_empty_masks_and_invalid_inputs() -> None:
    a = np.array([[1, 1, 0, 0]])
    b = np.array([[0, 1, 1, 0]])
    overlap = mask_overlap(a, b)
    assert overlap == {"intersection_px": 1, "union_px": 3, "iou": 1 / 3, "dice": 0.5}
    assert mask_overlap(b, a) == overlap
    assert mask_overlap(a, a)["iou"] == mask_overlap(a, a)["dice"] == 1
    assert mask_overlap(a, np.array([[0, 0, 1, 1]]))["iou"] == 0
    empty = np.zeros_like(a)
    assert mask_overlap(empty, a)["iou"] == mask_overlap(empty, a)["dice"] == 0
    assert mask_overlap(empty, empty)["iou"] is None
    assert mask_overlap(empty, empty)["dice"] is None
    for invalid in (np.array([[255]]), np.array([[np.nan]]), np.zeros((0, 1))):
        with pytest.raises(Phase1Error):
            mask_overlap(invalid, invalid)
    with pytest.raises(Phase1Error, match="same original-image"):
        mask_overlap(a, np.zeros((2, 2)))


def test_centroid_uses_original_pixel_centers_and_translates_correctly() -> None:
    mask = np.zeros((9, 12), dtype=np.uint8)
    mask[1:3, 2:6] = 1
    assert mask_centroid(mask) == (4.0, 2.0)
    shifted = np.zeros_like(mask)
    shifted[4:6, 4:8] = 1
    assert mask_centroid(shifted) == (6.0, 5.0)
    assert mask_centroid(np.array([[1]], dtype=bool)) == (0.5, 0.5)
    # A disconnected mask's centroid can lie outside foreground; do not snap it.
    assert mask_centroid(np.array([[1, 0, 1]])) == (1.5, 0.5)


def test_empty_masks_invalid_encodings_and_undefined_distances() -> None:
    assert mask_centroid(np.zeros((3, 5))) is None
    assert euclidean_distance(None, (2, 3)) is None
    assert euclidean_distance((1, 2), (1, 2)) == 0
    assert euclidean_distance((0, 0), (3, 4)) == 5
    assert euclidean_distance((3, 4), (0, 0)) == 5
    for invalid in (
        np.array([[255]]),
        np.array([[float("nan")]]),
        np.zeros((0, 2)),
        np.zeros((2, 2, 1)),
    ):
        with pytest.raises(Phase1Error):
            mask_centroid(invalid)
    with pytest.raises(Phase1Error):
        euclidean_distance((float("inf"), 0), (0, 0))


def test_eye_comparison_known_displacement_and_undefined_measurements() -> None:
    from animal_eye_metrology.eye_comparison import distance_statistics, eye_statistics

    ref, pred = np.zeros((6, 6), dtype=np.uint8), np.zeros((6, 6), dtype=np.uint8)
    ref[0, 0], pred[4, 3] = 1, 1
    result = eye_statistics(pred, ref)
    assert result["centroid_error_px"] == 5
    assert result["iou"] == result["dice"] == 0
    assert result["centroid_dx_px"] == 3 and result["centroid_dy_px"] == 4
    empty = eye_statistics(np.zeros_like(ref), ref)
    assert empty["iou"] == 0 and empty["centroid_error_px"] is None
    missing = eye_statistics(None, ref)
    assert missing["iou"] is None and missing["prediction_status"] == "inference_failed"
    distance = distance_statistics(3, 4)
    assert distance["signed_error_px"] == -1 and distance["absolute_error_px"] == 1
    assert distance["relative_absolute_error_percent"] == 25
    assert distance_statistics(None, 4)["absolute_error_px"] is None
    assert distance_statistics(0, 0)["relative_absolute_error_percent"] is None
    assert distance_statistics(0, 0)["measurement_status"] == "Defined"
    with pytest.raises(Phase1Error, match="nonempty reference"):
        eye_statistics(pred, np.zeros_like(ref))


def test_summary_keeps_failed_and_empty_targets_in_denominators() -> None:
    from animal_eye_metrology.eye_comparison import MODELS, distance_statistics, summarize

    eyes, distances = [], []
    for model in MODELS:
        for status in ("nonempty", "empty", "inference_failed"):
            eyes.append(
                {
                    "model": model,
                    "image_id": 10,
                    "prediction_status": status,
                    "iou": 1 if status == "nonempty" else (0 if status == "empty" else None),
                    "dice": 1 if status == "nonempty" else (0 if status == "empty" else None),
                    "centroid_error_px": 0 if status == "nonempty" else None,
                }
            )
        for predicted in (3, None):
            distances.append(
                {
                    "model": model,
                    "measurement_type": "binocular",
                    **distance_statistics(predicted, 4),
                }
            )
    for row in summarize(eyes, distances):
        assert row["eyes_expected"] == 3 and row["defined_centers"] == 1
        assert row["inference_failed"] == row["empty_predictions"] == 1
        assert row["mean_iou_all_targets_failure_as_zero"] == 1 / 3
        assert row["mean_centroid_error_px"] == 0
        assert row["geometrically_complete_images"] == 0
        assert row["binocular_expected"] == 2 and row["binocular_defined"] == 1
        assert row["binocular_mae_px"] == row["binocular_rmse_px"] == 1
        assert row["binocular_mean_signed_error_px"] == -1


def invented_centers() -> list[dict]:
    rows = []
    for image_id in (10, 20, 30):
        for annotation_id, offset in ((1, 0), (2, 6)):
            for side, point in (("left", (0.5 + offset, 0.5)), ("right", (3.5 + offset, 4.5))):
                rows.append(
                    {
                        "image_id": image_id,
                        "annotation_id": annotation_id,
                        "eye_side": side,
                        "annotation_version": 7,
                        "centroid_x_px": point[0],
                        "centroid_y_px": point[1],
                    }
                )
    return rows


def test_pairs_use_same_image_same_animal_binocular_and_distinct_right_eyes() -> None:
    centers = invented_centers()
    result = build_measurements(centers[::-1])
    assert len(result) == 9
    binocular = [r for r in result if r["measurement_type"] == "binocular"]
    pairs = [r for r in result if r["measurement_type"] == "right_eye_pair"]
    assert len(binocular) == 6 and all(r["distance_px"] == 5 for r in binocular)
    assert len(pairs) == 3 and all(r["distance_px"] == 6 for r in pairs)
    assert all(
        r["annotation_id_a"] == r["annotation_id_b"]
        and (r["eye_side_a"], r["eye_side_b"]) == ("left", "right")
        for r in binocular
    )
    assert all(
        r["annotation_id_a"] < r["annotation_id_b"]
        and r["eye_side_a"] == r["eye_side_b"] == "right"
        for r in pairs
    )
    assert {r["image_id"] for r in pairs} == {10, 20, 30}
    with pytest.raises(Phase1Error, match="Duplicate"):
        build_measurements([*centers, centers[0]])
    with pytest.raises(Phase1Error, match="Both left/right"):
        build_measurements(centers[1:])
    centers[0] = centers[0] | {"centroid_x_px": None, "centroid_y_px": None}
    undefined = [r for r in build_measurements(centers) if r["distance_px"] is None]
    assert len(undefined) == 1 and undefined[0]["measurement_status"] == "Undefined"


@pytest.fixture
def synthetic_baseline(local_config: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, object]:
    from animal_eye_metrology import measurement_baseline
    from animal_eye_metrology.cli import main
    from animal_eye_metrology.coco import load_coco
    from animal_eye_metrology.eye_annotations import initialize_store, save_annotation
    from animal_eye_metrology.measurement_baseline import read_settings
    from animal_eye_metrology.verification import sha256_file

    assert main(["analyze", "--config", str(local_config)]) == 0
    root = local_config.parent
    synthetic_docs = root / "synthetic_docs"
    synthetic_docs.mkdir()
    for name in ("phase3a_eye_annotation_protocol.md", "phase3b_annotation_tool.md"):
        (synthetic_docs / name).write_text(
            "SYNTHETIC TEST DOCUMENT ONLY; NOT AN APPROVED RESEARCH PROTOCOL OR GROUND TRUTH.\n"
        )
    monkeypatch.setattr(measurement_baseline, "DOCS_ROOT", synthetic_docs)
    data = load_coco(root / "annotations.json")
    (root / "images").mkdir()
    for image_id in (10, 20, 50):
        Image.new("RGB", (100, 80), "gray").save(root / "images" / f"{image_id:012d}.png")
    targets = []
    for image_id, animals in ((10, (1001, 1002)), (20, (2001, 2002)), (50, (5001, 5002))):
        for annotation_id in animals:
            annotation = data.annotations[annotation_id]
            targets.append(
                {
                    "image_id": str(image_id),
                    "annotation_id": str(annotation_id),
                    "animal_category": data.categories[annotation.category_id].name,
                    "file_name": f"{image_id:012d}.png",
                    "image_width": "100",
                    "image_height": "80",
                    "coco_bbox_xywh": json.dumps(annotation.bbox),
                    "proposed_eye_sides": '["left","right"]',
                    "image_sha256": sha256_file(root / "images" / f"{image_id:012d}.png"),
                }
            )
    proposal = root / "proposal.csv"
    with proposal.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)
    output = root / "phase3/live_gt"
    initialize_store(output, targets, data.source_sha256)
    for target in targets:
        for side, x in (("left", 3), ("right", 7)):
            geometry = {
                "view": [0, 0, 100, 80],
                "operation": "add",
                "active": [],
                "contours": [
                    {"operation": "add", "points": [[x, 3], [x + 2, 3], [x + 2, 5], [x, 5]]}
                ],
            }
            save_annotation(
                output,
                targets,
                (int(target["image_id"]), int(target["annotation_id"]), side),
                geometry,
                {
                    "visibility": "Visible",
                    "side_confirmed": "Yes",
                    "mask_review_status": "Accepted",
                    "measurement_quality": "Pass",
                    "reviewer": "SYNTHETIC QA ONLY",
                    "exclusion_reason": "",
                },
                expected_version=0,
            )
    cfg = root / "measurement.toml"
    cfg.write_text(
        '[data]\nphase1_config="phase1.toml"\nproposal_csv="proposal.csv"\n'
        'ground_truth_directory="phase3/live_gt"\n[snapshot]\n'
        'protocol_version="eye-visible-aperture-v1"\nexpected_images=3\n'
        "expected_instances=6\nexpected_eyes=12\n[output]\n"
        'directory="phase3/frozen_baseline"\n'
    )
    return read_settings(cfg)


def test_baseline_requires_the_original_protocol_document(synthetic_baseline) -> None:
    from animal_eye_metrology import measurement_baseline

    cfg, data = synthetic_baseline
    protocol = measurement_baseline.PROTOCOL_DOCUMENTS[cfg["snapshot"]["protocol_version"]]
    (measurement_baseline.DOCS_ROOT / protocol).unlink()
    with pytest.raises(Phase1Error, match="Approved protocol document is missing"):
        measurement_baseline.run(cfg, data)
    assert not cfg["output"]["directory"].exists()


def test_freeze_replays_after_live_gt_changes_and_rejects_tampering(synthetic_baseline) -> None:
    from animal_eye_metrology.eye_annotations import (
        load_annotations,
        load_geometry,
        save_annotation,
    )
    from animal_eye_metrology.measurement_baseline import run, verify
    from animal_eye_metrology.phase2 import read_csv
    from animal_eye_metrology.verification import sha256_file

    cfg, data = synthetic_baseline
    source = cfg["data"]["ground_truth_directory"]
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    baseline = run(cfg, data)
    assert all(Path(p).read_bytes() == content for p, content in before.items())
    assert verify(baseline)["measurements_recomputed"] == 9
    assert len(read_csv(baseline / "frozen/eye_mask_annotations.csv")) == 12
    assert len(list((baseline / "overlays").glob("*.png"))) == 3
    for center in read_csv(baseline / "eye_centers.csv"):
        assert sha256_file(baseline / center["mask_path"]) == center["mask_sha256"]
    with pytest.raises(Phase1Error, match="already exists"):
        run(cfg, data)
    targets = read_csv(cfg["data"]["proposal_csv"])
    row = load_annotations(source, targets)[0]
    save_annotation(
        source,
        targets,
        (10, 1001, "left"),
        load_geometry(source, row),
        {
            k: row[k]
            for k in (
                "visibility",
                "side_confirmed",
                "mask_review_status",
                "measurement_quality",
                "exclusion_reason",
                "reviewer",
            )
        },
        expected_version=1,
    )
    assert verify(baseline)["uses_live_ground_truth"] is False
    mask = next((baseline / "frozen/masks").rglob("*.png"))
    mask.write_bytes(b"SYNTHETIC CORRUPTION")
    with pytest.raises(Phase1Error, match="hash differs"):
        verify(baseline)


def test_latest_pending_version_never_falls_back_to_older_accepted(synthetic_baseline) -> None:
    from animal_eye_metrology.eye_annotations import (
        load_annotations,
        load_geometry,
        save_annotation,
    )
    from animal_eye_metrology.measurement_baseline import run
    from animal_eye_metrology.phase2 import read_csv

    cfg, data = synthetic_baseline
    source = cfg["data"]["ground_truth_directory"]
    targets = read_csv(cfg["data"]["proposal_csv"])
    row = load_annotations(source, targets)[0]
    save_annotation(
        source,
        targets,
        (10, 1001, "left"),
        load_geometry(source, row),
        {
            "visibility": "Visible",
            "side_confirmed": "Yes",
            "mask_review_status": "Pending",
            "measurement_quality": "Unknown",
            "reviewer": "",
            "exclusion_reason": "",
        },
        expected_version=1,
    )
    with pytest.raises(Phase1Error, match="no history fallback"):
        run(cfg, data)
    assert not cfg["output"]["directory"].exists()
