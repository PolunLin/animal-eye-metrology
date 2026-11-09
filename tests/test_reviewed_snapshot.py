"""Synthetic owner-review snapshots preserve provenance and source bytes."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
mask_utils = pytest.importorskip("pycocotools.mask")

from animal_eye_metrology.annotation import reviewed_snapshot as subject  # noqa: E402
from animal_eye_metrology.core.errors import Phase1Error  # noqa: E402
from animal_eye_metrology.core.verification import sha256_file  # noqa: E402


@pytest.fixture
def synthetic_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = tmp_path / "configs/annotation_workspace.toml"
    config.parent.mkdir()
    config.write_text("synthetic configuration")
    base = tmp_path / "workspace"
    gt = base / "ground_truth"
    gt.mkdir(parents=True)
    image_dir = base / "images"
    image_dir.mkdir()
    proposal = base / "proposal.csv"
    proposal.write_text("synthetic proposal")
    csv_path = gt / "eye_mask_annotations.csv"
    csv_path.write_text("synthetic revision history")
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:5, 4:9] = 255
    Image.fromarray(mask).save(gt / "eye.png")
    targets = []
    for image_id, annotation_ids in ((10, (101, 102)), (20, (201,))):
        source = image_dir / f"{image_id:012d}.png"
        Image.new("RGB", (16, 12), "gray").save(source)
        for annotation_id in annotation_ids:
            targets.append(
                {
                    "image_id": str(image_id),
                    "annotation_id": str(annotation_id),
                    "animal_category": "cat",
                    "file_name": source.name,
                    "image_width": "16",
                    "image_height": "12",
                    "image_sha256": sha256_file(source),
                    "coco_bbox_xywh": "[0,0,16,12]",
                }
            )
    session = SimpleNamespace(
        source=config,
        phase1_config=config,
        proposal=proposal,
        protocol_document=config,
        ground_truth=gt,
        images=image_dir,
        protocol_version=subject.V2_PROTOCOL_VERSION,
    )
    dataset = SimpleNamespace(
        source_sha256="synthetic-coco-hash",
        annotations={
            int(t["annotation_id"]): SimpleNamespace(
                segmentation=[[0, 0, 15, 0, 15, 11]], category_id=17, iscrowd=0, area=100
            )
            for t in targets
        },
    )

    def row(image, annotation, side, version=1, status="Pending"):
        return {
            "image_id": str(image),
            "annotation_id": str(annotation),
            "eye_side": side,
            "annotation_version": str(version),
            "reviewer": "Polun",
            "mask_path": "eye.png",
            "visibility": "Visible",
            "side_confirmed": "Yes",
            "measurement_quality": "Pass",
            "mask_review_status": status,
            "exclusion_reason": "",
            "protocol_version": subject.V2_PROTOCOL_VERSION,
        }

    rows = [
        row(10, 101, "left"),
        row(10, 101, "right", status="Accepted"),
        row(10, 102, "left", status="Rejected"),
        row(20, 201, "left"),
    ]
    imports = [
        {
            "batch": "demo",
            **{k: r[k] for k in ("image_id", "annotation_id", "eye_side", "annotation_version")},
        }
        for r in (rows[0], rows[3])
    ]
    jobs = tmp_path / "outputs/phase3/eye_draft_jobs/synthetic"
    jobs.mkdir(parents=True)
    (jobs / "manifest.json").write_text(json.dumps({"saved": imports}))
    monkeypatch.setattr(subject, "read_session", lambda path: session)
    monkeypatch.setattr(
        subject, "verify_session", lambda s: {"dataset_annotations_sha256": dataset.source_sha256}
    )
    monkeypatch.setattr(subject, "_dataset", lambda s: dataset)
    monkeypatch.setattr(subject, "load_targets", lambda p, d: targets)
    monkeypatch.setattr(subject, "_check_scope", lambda s, t: None)
    monkeypatch.setattr(subject, "load_annotations", lambda *a, **kw: rows)
    return tmp_path, gt, rows, mask


def test_freeze_excludes_unreviewed_prefills_and_keeps_competitors(synthetic_review):
    root, gt, rows, mask = synthetic_review
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    record = subject.freeze_reviewed_reference(root, root / "snapshot")
    assert len(record["images"]) == 1
    assert len(record["images"][0]["targets"]) == 2
    assert len(record["cases"]) == 4
    assert record["summary"]["human_saved_eyes"] == 2
    assert record["summary"]["reviewed_eyes"] == 2
    assert record["summary"]["eligible_eyes"] == 1
    assert record["summary"]["fully_reviewed_photos"] == 0
    pending = next(
        c for c in record["cases"] if c["annotation_id"] == 101 and c["eye_side"] == "left"
    )
    assert not pending["human_saved"] and not pending["eligible"]
    eligible = next(c for c in record["cases"] if c["eligible"])
    assert np.array_equal(mask_utils.decode(eligible["reference_mask_rle"]) != 0, mask != 0)
    assert eligible["reference_mask_sha256"] == sha256_file(gt / "eye.png")
    assert all(p.read_bytes() == content for p, content in before.items())
    assert json.loads((root / "snapshot/reference.json").read_text()) == record


def test_latest_human_revision_supersedes_prefill(synthetic_review):
    root, _, rows, _ = synthetic_review
    rows.insert(0, {**rows[0], "annotation_version": "2", "mask_review_status": "Accepted"})
    record = subject.freeze_reviewed_reference(root, root / "snapshot")
    assert record["summary"]["eligible_eyes"] == 2
    assert record["summary"]["binocular_pairs"] == 1
    assert record["summary"]["right_right_pairs"] == 0
    assert record["summary"]["human_saved_eyes"] == 3


def test_snapshot_refuses_overwrite_and_empty_eligible_mask(synthetic_review):
    root, gt, _, mask = synthetic_review
    output = root / "snapshot"
    output.mkdir()
    with pytest.raises(Phase1Error, match="overwrite"):
        subject.freeze_reviewed_reference(root, output)
    Image.fromarray(np.zeros_like(mask)).save(gt / "eye.png")
    with pytest.raises(Phase1Error, match="must not be empty"):
        subject.freeze_reviewed_reference(root, root / "invalid")
    assert not (root / "invalid").exists()


def test_concurrent_csv_change_aborts_without_publishing(synthetic_review, monkeypatch):
    root, gt, rows, _ = synthetic_review

    def changed(*args, **kwargs):
        (gt / "eye_mask_annotations.csv").write_text("changed while loading")
        return rows

    monkeypatch.setattr(subject, "load_annotations", changed)
    with pytest.raises(Phase1Error, match="changed during snapshot"):
        subject.freeze_reviewed_reference(root, root / "snapshot")
    assert not (root / "snapshot").exists()
