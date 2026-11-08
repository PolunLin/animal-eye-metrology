"""Delivery checks detect missing assets and false sample geometry without private outputs."""

import hashlib
import importlib.util
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("delivery_check", ROOT / "scripts/check_delivery.py")
assert SPEC is not None and SPEC.loader is not None
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


@pytest.fixture
def shareable_copy(tmp_path: Path) -> Path:
    # Copy shareable source directories so new README links to code or demo
    # notebooks work, while datasets, weights, outputs and venvs stay excluded.
    for directory in (
        ".github",
        "app",
        "configs",
        "docs",
        "examples",
        "notebooks",
        "requirements",
        "scripts",
        "src",
        "templates",
        "tests",
    ):
        source = ROOT / directory
        if source.is_dir():
            shutil.copytree(
                source,
                tmp_path / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    for relative in delivery.REQUIRED_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return tmp_path


def status(result: dict[str, Any], name: str) -> bool:
    return next(check["passed"] for check in result["checks"] if check["name"] == name)


def update_sample_digest(root: Path) -> None:
    """Keep bytes/provenance consistent to independently exercise the geometry check."""
    path = root / "examples/README.md"
    digest = hashlib.sha256((root / "examples/measurements.csv").read_bytes()).hexdigest()
    text = re.sub(r"(CSV SHA-256\s*\|\s*`)[0-9a-f]{64}", r"\g<1>" + digest, path.read_text())
    path.write_text(text)


def test_shareable_files_pass_without_datasets_weights_or_outputs(shareable_copy: Path) -> None:
    result = delivery.check_delivery(shareable_copy)
    assert result["status"] == "passed", result
    assert result["model_inference_performed"] is False
    assert result["accuracy_evaluated"] is False
    assert not (shareable_copy / "outputs").exists()


def test_missing_environment_example_and_broken_links_fail_independently(
    shareable_copy: Path,
) -> None:
    (shareable_copy / ".env.example").unlink()
    with (shareable_copy / "README.md").open("a") as handle:
        handle.write("\n[Missing guide](docs/missing.md)\n")
    result = delivery.check_delivery(shareable_copy)
    assert result["status"] == "failed"
    assert status(result, "required_files") is False
    assert status(result, "readme_local_links") is False
    assert status(result, "recorded_csv") is True


@pytest.mark.parametrize("change", ["bytes", "geometry", "qualification", "empty"])
def test_recorded_sample_detects_tampering_and_invalid_geometry(
    shareable_copy: Path, change: str
) -> None:
    path = shareable_copy / "examples/measurements.csv"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if change == "bytes":
        path.write_text("\n".join(lines) + "\n")
    elif change == "empty":
        path.write_text(lines[0] + "\n")
        update_sample_digest(shareable_copy)
    else:
        cells = lines[1].split(",")
        fields = lines[0].split(",")
        if change == "geometry":
            cells[fields.index("distance_px")] = "9999"
        else:
            cells[fields.index("qualification")] = "verified"
        lines[1] = ",".join(cells)
        path.write_text("\n".join(lines) + "\n")
        update_sample_digest(shareable_copy)
    result = delivery.check_delivery(shareable_copy)
    assert result["status"] == "failed"
    assert status(result, "recorded_csv") is False
