"""Synthetic annotation stores prove v2 isolation without changing v1 references."""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("PIL.Image")
pytest.importorskip("numpy")

from test_eye_annotations import geometry, review, target  # noqa: E402
from test_metrology import synthetic_baseline  # noqa: E402, F401 - imported pytest fixture

from animal_eye_metrology.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.eye_annotations import (  # noqa: E402
    FIELDS,
    PROTOCOL_VERSION,
    V2_PROTOCOL_VERSION,
    initialize_store,
    load_annotations,
    load_geometry,
    save_annotation,
)
from animal_eye_metrology.measurement_baseline import read_settings, run, verify  # noqa: E402
from animal_eye_metrology.phase2 import read_csv  # noqa: E402
from animal_eye_metrology.verification import sha256_file  # noqa: E402


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_v2_store_cannot_be_read_or_written_as_v1_even_when_empty(tmp_path: Path) -> None:
    targets = [target()]
    initialize_store(tmp_path, targets, "synthetic-dataset", protocol_version=V2_PROTOCOL_VERSION)
    before = _files(tmp_path)
    assert load_annotations(tmp_path, targets, protocol_version=V2_PROTOCOL_VERSION) == []
    with pytest.raises(Phase1Error, match="來源／規範"):
        load_annotations(tmp_path, targets)
    with pytest.raises(Phase1Error, match="來源／規範"):
        initialize_store(tmp_path, targets, "synthetic-dataset")
    with pytest.raises(Phase1Error, match="來源／規範"):
        save_annotation(
            tmp_path, targets, (10, 1001, "left"), geometry(), review(), expected_version=0
        )
    assert _files(tmp_path) == before

    row = save_annotation(
        tmp_path,
        targets,
        (10, 1001, "left"),
        geometry(),
        review(),
        expected_version=0,
        protocol_version=V2_PROTOCOL_VERSION,
    )
    assert row["protocol_version"] == V2_PROTOCOL_VERSION
    assert load_annotations(tmp_path, targets, protocol_version=V2_PROTOCOL_VERSION) == [row]
    metadata = json.loads((tmp_path / "revisions/10_1001_left_v1.json").read_text())
    assert metadata["record"] == row


def test_v1_default_remains_and_v2_access_cannot_replace_it(tmp_path: Path) -> None:
    targets = [target()]
    initialize_store(tmp_path, targets, "synthetic-dataset")
    row = save_annotation(
        tmp_path, targets, (10, 1001, "left"), geometry(), review(), expected_version=0
    )
    before = _files(tmp_path)
    assert row["protocol_version"] == PROTOCOL_VERSION == "eye-visible-aperture-v1"
    with pytest.raises(Phase1Error, match="來源／規範"):
        save_annotation(
            tmp_path,
            targets,
            (10, 1001, "left"),
            geometry(),
            review(),
            expected_version=1,
            protocol_version=V2_PROTOCOL_VERSION,
        )
    assert _files(tmp_path) == before
    assert load_annotations(tmp_path, targets) == [row]


def test_mixed_protocol_revision_is_rejected_even_when_csv_and_revision_agree(
    tmp_path: Path,
) -> None:
    targets = [target()]
    initialize_store(tmp_path, targets, "synthetic-dataset", protocol_version=V2_PROTOCOL_VERSION)
    row = save_annotation(
        tmp_path,
        targets,
        (10, 1001, "left"),
        geometry(),
        review(),
        expected_version=0,
        protocol_version=V2_PROTOCOL_VERSION,
    )
    row["protocol_version"] = PROTOCOL_VERSION
    with (tmp_path / "eye_mask_annotations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    metadata_path = tmp_path / "revisions/10_1001_left_v1.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["record"] = row
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(Phase1Error, match="Invalid revision identity"):
        load_annotations(tmp_path, targets, protocol_version=V2_PROTOCOL_VERSION)


def test_unknown_protocol_creates_no_store(tmp_path: Path) -> None:
    output = tmp_path / "unsupported"
    with pytest.raises(Phase1Error, match="不支援"):
        initialize_store(output, [target()], "synthetic-dataset", protocol_version="eye-v999")
    assert not output.exists()


def test_v2_freeze_and_replay_use_v2_protocol_without_changing_v1(
    synthetic_baseline,  # noqa: F811 - imported pytest fixture
) -> None:
    cfg, data = synthetic_baseline
    v1_baseline = run(cfg, data)
    v1_source = cfg["data"]["ground_truth_directory"]
    before_source, before_baseline = _files(v1_source), _files(v1_baseline)
    targets = read_csv(cfg["data"]["proposal_csv"])
    source_manifest = json.loads((v1_source / "source_manifest.json").read_text())
    source_v2 = v1_source.with_name("live_gt_v2")
    initialize_store(
        source_v2,
        targets,
        source_manifest["dataset_sha256"],
        protocol_version=V2_PROTOCOL_VERSION,
    )
    # Reusing invented rectangles here verifies unchanged arithmetic, not actual relabeling.
    for row in load_annotations(v1_source, targets):
        save_annotation(
            source_v2,
            targets,
            (int(row["image_id"]), int(row["annotation_id"]), row["eye_side"]),
            load_geometry(v1_source, row),
            review(),
            expected_version=0,
            protocol_version=V2_PROTOCOL_VERSION,
        )
    config_v2 = cfg["source"].with_name("measurement_v2.toml")
    config_v2.write_text(
        cfg["source"]
        .read_text()
        .replace(PROTOCOL_VERSION, V2_PROTOCOL_VERSION)
        .replace("phase3/live_gt", "phase3/live_gt_v2")
        .replace("phase3/frozen_baseline", "phase3/frozen_baseline_v2")
    )
    v2_cfg, v2_data = read_settings(config_v2)
    with pytest.raises(Phase1Error, match="protocol and manifest are required"):
        run(v2_cfg, v2_data)
    assert not v2_cfg["output"]["directory"].exists()
    prepared_protocol = source_v2.parent / "protocol.md"
    prepared_protocol.write_text("# Synthetic v2 protocol fixture; not research annotation data.\n")
    (source_v2.parent / "preparation_manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": V2_PROTOCOL_VERSION,
                "files_sha256": {"protocol.md": sha256_file(prepared_protocol)},
            }
        )
    )
    v2_baseline = run(v2_cfg, v2_data)
    assert verify(v2_baseline)["eyes_recomputed"] == 12
    assert verify(v2_baseline)["measurements_recomputed"] == 9
    v2_centers = read_csv(v2_baseline / "eye_centers.csv")
    assert {row["protocol_version"] for row in v2_centers} == {V2_PROTOCOL_VERSION}
    assert (v2_baseline / "provenance/eye_annotation_protocol_v2.md").is_file()
    assert (v2_baseline / "frozen/protocol.md").read_bytes() == prepared_protocol.read_bytes()
    assert (v2_baseline / "provenance/eye_annotation_protocol_v2.md").read_bytes() == (
        prepared_protocol.read_bytes()
    )
    assert not (v2_baseline / "provenance/phase3a_eye_annotation_protocol.md").exists()
    assert not (v2_baseline / "provenance/phase3b_annotation_tool.md").exists()
    summary = json.loads((v2_baseline / "summary.json").read_text())
    assert "eyelid_region" in summary["reference_type"]
    for old, new in zip(read_csv(v1_baseline / "eye_centers.csv"), v2_centers, strict=True):
        for field in ("mask_area_px", "centroid_x_px", "centroid_y_px"):
            assert old[field] == new[field]
    assert _files(v1_source) == before_source
    assert _files(v1_baseline) == before_baseline
    assert verify(v1_baseline)["measurements_recomputed"] == 9
    prepared_protocol.write_text("SYNTHETIC SOURCE DRIFT AFTER FREEZE")
    assert verify(v2_baseline)["measurements_recomputed"] == 9
    v2_cfg["output"]["directory"] = v2_baseline.with_name("must_not_publish")
    with pytest.raises(Phase1Error, match="Prepared v2 annotation protocol"):
        run(v2_cfg, v2_data)
    assert not v2_cfg["output"]["directory"].exists()
