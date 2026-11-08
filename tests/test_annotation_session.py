"""Independent relabeling preparation uses invented images, never research ground truth."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

from animal_eye_metrology.annotation_session import (  # noqa: E402
    AnnotationSession,
    main,
    prepare,
    read_session,
    status,
    verify_session,
)
from animal_eye_metrology.coco import load_coco  # noqa: E402
from animal_eye_metrology.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.eye_annotations import (  # noqa: E402
    FIELDS,
    V2_PROTOCOL_VERSION,
    empty_geometry,
    load_annotations,
    load_targets,
    save_annotation,
)
from animal_eye_metrology.verification import sha256_file  # noqa: E402


@pytest.fixture
def annotation_session(local_config: Path) -> AnnotationSession:
    root = local_config.parent
    image_directory = root / "images"
    image_directory.mkdir()
    image_path = image_directory / "000000000010.png"
    Image.new("RGB", (100, 80), "gray").save(image_path)
    dataset = load_coco(root / "annotations.json")
    rows = [
        {
            "image_id": "10",
            "annotation_id": str(annotation_id),
            "animal_category": "test_species_0",
            "file_name": image_path.name,
            "image_width": "100",
            "image_height": "80",
            "coco_bbox_xywh": json.dumps(list(dataset.annotations[annotation_id].bbox)),
            "proposed_eye_sides": '["left","right"]',
            "image_sha256": sha256_file(image_path),
        }
        for annotation_id in (1001, 1002)
    ]
    proposal = root / "proposal.csv"
    with proposal.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = root / "protocol.md"
    protocol.write_text(f"# SYNTHETIC TEST PROTOCOL ONLY\n{V2_PROTOCOL_VERSION}\n")
    config = root / "session.toml"
    config.write_text(
        'phase1_config = "phase1.toml"\n'
        'proposal_csv = "proposal.csv"\n'
        'directory = "phase3/relabel_session"\n'
        'protocol_document = "protocol.md"\n'
        f'protocol_version = "{V2_PROTOCOL_VERSION}"\n'
        "expected_images = 1\nexpected_instances = 2\nexpected_eyes = 4\n"
    )
    return read_session(config)


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_prepare_copies_only_sources_and_status_waits_for_humans(
    annotation_session: AnnotationSession, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    session = annotation_session
    source_before = _files(session.source.parent)
    manifest = prepare(session)
    assert manifest["initial_annotation_count"] == 0
    assert manifest["model_predictions_loaded"] is False
    assert manifest["prior_masks_loaded"] is False
    assert manifest["independent_validation_set"] is False
    assert (session.images / "000000000010.png").read_bytes() == source_before[
        "images/000000000010.png"
    ]
    assert session.proposal.read_bytes() == session.source_proposal.read_bytes()
    assert (
        session.directory / "protocol.md"
    ).read_bytes() == session.protocol_document.read_bytes()
    with (session.ground_truth / "eye_mask_annotations.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(FIELDS)
        assert list(reader) == []
    assert not (session.ground_truth / "masks").exists()
    assert not (session.ground_truth / "revisions").exists()
    with (session.directory / "relabel_scope.csv").open(newline="") as handle:
        scope = list(csv.DictReader(handle))
    assert len(scope) == 4
    assert {row["scope_status"] for row in scope} == {"scheduled_not_annotated"}
    assert all("reviewer" not in row and "mask_path" not in row for row in scope)
    for relative, value in source_before.items():
        assert (session.source.parent / relative).read_bytes() == value
    prepared_before = _files(session.directory)
    result = status(session)
    assert result["saved_eyes"] == result["eligible_eyes"] == 0
    assert result["expected_eyes"] == 4
    assert result["ready_for_freeze"] is False
    assert all(row["review_status"] == "not_annotated" for row in result["cases"])
    monkeypatch.setattr(
        "sys.argv", ["annotation_session", "--config", str(session.source), "--status"]
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["ready_for_freeze"] is False
    assert _files(session.directory) == prepared_before


def test_preparation_refuses_overwrite_and_uses_frozen_sources_afterward(
    annotation_session: AnnotationSession,
) -> None:
    session = annotation_session
    prepare(session)
    prepared_before = _files(session.directory)
    with pytest.raises(Phase1Error, match="拒絕覆寫"):
        prepare(session)
    assert _files(session.directory) == prepared_before
    # Prepared sources stay independent of later edits to the original proposal/images.
    session.source_proposal.write_text("SYNTHETIC CHANGED SOURCE PROPOSAL")
    (session.source.parent / "images/000000000010.png").write_bytes(b"changed original")
    assert status(session)["saved_eyes"] == 0
    assert _files(session.directory) == prepared_before


@pytest.mark.parametrize("change", ["scope", "image_hash", "protocol_version"])
def test_invalid_input_fails_without_creating_a_session(
    annotation_session: AnnotationSession, change: str
) -> None:
    session = annotation_session
    if change == "scope":
        session = replace(session, expected_images=2)
    elif change == "image_hash":
        (session.source.parent / "images/000000000010.png").write_bytes(b"changed original")
    else:
        session.protocol_document.write_text("# SYNTHETIC wrong protocol\n")
    with pytest.raises(Phase1Error):
        prepare(session)
    assert not session.directory.exists()


@pytest.mark.parametrize("change", ["copied_image", "protocol_document", "config", "dataset"])
def test_saved_input_or_source_definition_drift_is_rejected(
    annotation_session: AnnotationSession, change: str
) -> None:
    session = annotation_session
    prepare(session)
    if change == "copied_image":
        (session.images / "000000000010.png").write_bytes(b"corrupt frozen source")
    elif change == "protocol_document":
        session.protocol_document.write_text(f"{V2_PROTOCOL_VERSION}\nSYNTHETIC NEW RULE\n")
    elif change == "config":
        session.source.write_text(session.source.read_text() + "# changed preparation config\n")
    else:
        path = session.source.parent / "annotations.json"
        contents = json.loads(path.read_text())
        contents["info"]["description"] += " changed"
        path.write_text(json.dumps(contents))
    with pytest.raises(Phase1Error):
        status(session)


def test_manifest_cannot_reference_outside_session(annotation_session: AnnotationSession) -> None:
    session = annotation_session
    prepare(session)
    path = session.directory / "preparation_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files_sha256"]["../../protocol.md"] = sha256_file(session.protocol_document)
    path.write_text(json.dumps(manifest))
    with pytest.raises(Phase1Error, match="超出工作區"):
        verify_session(session)


def test_config_rejects_source_containment_and_non_phase3_destinations(
    annotation_session: AnnotationSession,
) -> None:
    config = annotation_session.source
    original = config.read_text()
    config.write_text(original.replace('"phase3/relabel_session"', '"outside"'))
    with pytest.raises(Phase1Error, match="Phase 3"):
        read_session(config)
    config.write_text(original.replace('"proposal.csv"', '"phase3/relabel_session/proposal.csv"'))
    with pytest.raises(Phase1Error, match="不能包含來源"):
        read_session(config)


def test_reviewed_uncertain_case_is_saved_but_never_freeze_eligible(
    annotation_session: AnnotationSession,
) -> None:
    session = annotation_session
    prepare(session)
    dataset = load_coco(session.source.parent / "annotations.json")
    targets = load_targets(session.proposal, dataset)
    geometry = empty_geometry(100, 80, (1, 2, 10, 20))
    save_annotation(
        session.ground_truth,
        targets,
        (10, 1001, "left"),
        geometry,
        {
            "visibility": "Uncertain",
            "side_confirmed": "Unknown",
            "mask_review_status": "Uncertain",
            "measurement_quality": "Unknown",
            "reviewer": "SYNTHETIC QA ONLY",
            "exclusion_reason": "Synthetic unclear boundary",
        },
        expected_version=0,
        protocol_version=V2_PROTOCOL_VERSION,
    )
    report = status(session)
    assert report["saved_eyes"] == 1
    assert report["eligible_eyes"] == 0
    assert report["ready_for_freeze"] is False
    assert report["cases"][0]["review_status"] == "Uncertain"
    assert report["cases"][0]["reason"] == "Synthetic unclear boundary"
    assert (
        load_annotations(session.ground_truth, targets, protocol_version=V2_PROTOCOL_VERSION)[0][
            "mask_path"
        ]
        == ""
    )
    assert not (session.ground_truth / "masks").exists()
