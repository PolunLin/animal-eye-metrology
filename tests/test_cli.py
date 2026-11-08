"""Minimal output integration and data protection checks."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from animal_eye_metrology.analysis import REVIEW_FIELDS, REVIEW_MANUAL_FIELDS
from animal_eye_metrology.cli import main
from animal_eye_metrology.config import load_config


def test_annotation_only_cli_produces_traceable_reproducible_artifacts(local_config: Path) -> None:
    source = local_config.parent / "annotations.json"
    before = source.read_bytes()
    assert main(["analyze", "--config", str(local_config)]) == 0
    output = local_config.parent / "result"
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["dataset"]["annotations_sha256"] == hashlib.sha256(before).hexdigest()
    with (output / "multi_any.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["image_id"]) for row in rows] == [10, 20, 50]
    assert json.loads(rows[-1]["annotation_ids"]) == [5001, 5002, 5003]
    with (output / "eye_visibility_review.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or []) == REVIEW_FIELDS
        assert all(row[field] == "" for row in reader for field in REVIEW_MANUAL_FIELDS)
    second = local_config.parent / "second"
    assert main(["analyze", "--config", str(local_config), "--output", str(second)]) == 0
    assert all(
        path.read_bytes() == (second / path.name).read_bytes() for path in output.glob("*.csv")
    )
    assert source.read_bytes() == before


def test_existing_manual_review_is_never_overwritten(local_config: Path) -> None:
    assert main(["analyze", "--config", str(local_config)]) == 0
    review = local_config.parent / "result" / "eye_visibility_review.csv"
    review.write_text("user-entered review; keep this")
    assert main(["analyze", "--config", str(local_config)]) == 2
    assert review.read_text() == "user-entered review; keep this"


def test_missing_data_fails_without_creating_outputs_or_downloading(local_config: Path) -> None:
    (local_config.parent / "annotations.json").unlink()
    assert main(["analyze", "--config", str(local_config)]) == 2
    assert not (local_config.parent / "result").exists()


def test_failed_visualization_does_not_publish_partial_results(
    local_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_render(*args: object, **kwargs: object) -> None:
        raise OSError("Simulated local image read failure")

    monkeypatch.setattr("animal_eye_metrology.outputs.render_contact_sheets", fail_render)
    assert main(["analyze", "--config", str(local_config), "--contact-sheets"]) == 2
    assert not (local_config.parent / "result").exists()
    assert not list(local_config.parent.glob(".phase1-*"))


def test_output_cannot_be_inside_image_inputs(local_config: Path) -> None:
    assert (
        main(
            [
                "analyze",
                "--config",
                str(local_config),
                "--output",
                str(local_config.parent / "images" / "results"),
            ]
        )
        == 2
    )
    assert not (local_config.parent / "images").exists()


def test_paths_are_relative_to_config_not_cwd(
    local_config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    config = load_config(local_config)
    assert config.annotations == local_config.parent / "annotations.json"
    assert config.output == local_config.parent / "result"


@pytest.mark.parametrize("kind", ["wrong_dimensions", "corrupt"])
def test_invalid_image_rejects_run(local_config: Path, kind: str) -> None:
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("pycocotools.mask")
    config = load_config(local_config)
    config.images.mkdir()
    path = config.images / "000000000010.png"
    if kind == "wrong_dimensions":
        image_module.new("RGB", (99, 80)).save(path)
    else:
        path.write_bytes(b"not a real image")
    assert main(["analyze", "--config", str(local_config), "--contact-sheets"]) == 2
    assert not config.output.exists()
