"""Package boundaries preserve historical imports without duplicating module state."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from animal_eye_metrology.paths import (
    MODULE_LOCATIONS,
    PACKAGE_ROOT,
    SOURCE_ROOT,
    module_source,
    package_sources,
)


def subprocess_environment() -> dict[str, str]:
    return os.environ | {"PYTHONPATH": str(SOURCE_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}


def test_root_import_does_not_load_application_or_model_dependencies(tmp_path: Path) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; import animal_eye_metrology; "
            "print(json.dumps(sorted(set(sys.modules) & {'torch','fastapi','streamlit','numpy'})))",
        ],
        cwd=tmp_path,
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert json.loads(process.stdout) == []


def test_published_modules_resolve_without_legacy_or_private_packages() -> None:
    for name, location in MODULE_LOCATIONS.items():
        canonical = importlib.import_module(
            "animal_eye_metrology." + location.removesuffix(".py").replace("/", ".")
        )
        assert Path(canonical.__file__).resolve() == module_source(name)
    assert not (PACKAGE_ROOT / "research").joinpath("__init__.py").exists()
    assert not (PACKAGE_ROOT / "_compat.py").exists()


@pytest.mark.parametrize(
    "module",
    [
        "animal_eye_metrology",
        "animal_eye_metrology.inference.inference_worker",
        "animal_eye_metrology.inference.eye_pose_worker",
    ],
)
def test_delivery_commands_work_from_an_unrelated_directory(module: str, tmp_path: Path) -> None:
    process = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=tmp_path,
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "usage:" in process.stdout.lower()


def test_source_collection_keeps_nested_packages_and_same_named_initializers(
    tmp_path: Path,
) -> None:
    for group in ("one", "two"):
        package = tmp_path / group
        package.mkdir()
        (package / "__init__.py").write_text('"""Synthetic source fixture."""\n')
    sources = package_sources(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in sources] == [
        "one/__init__.py",
        "two/__init__.py",
    ]
    with pytest.raises(ValueError, match="Unknown project module"):
        module_source("../../private")
