"""Training source relocation contracts; never execute an optimizer or real export."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from animal_eye_metrology.paths import PROJECT_ROOT, SOURCE_ROOT
from animal_eye_metrology.research.training import read_settings


def command_environment() -> dict[str, str]:
    return os.environ | {"PYTHONPATH": str(SOURCE_ROOT), "PYTHONDONTWRITEBYTECODE": "1"}


def files_in(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ["-m", "animal_eye_metrology.training", "--help"],
        ["-m", "animal_eye_metrology.research.training", "--help"],
        ["-m", "animal_eye_metrology.training_data", "--help"],
        ["-m", "animal_eye_metrology.data.training_data", "--help"],
        ["-m", "animal_eye_metrology.segmentation_supply", "--help"],
        ["-m", "animal_eye_metrology.data.segmentation_supply", "--help"],
        [str(PROJECT_ROOT / "scripts/export_training_weights.py"), "--help"],
    ],
)
def test_training_help_commands_preserve_existing_artifacts(
    arguments: list[str], tmp_path: Path
) -> None:
    # Deliberately invalid sentinel bytes prove --help never opens or rewrites checkpoints.
    for name in ("last.pt", "best.pt", "native_weights.pt", "epochs.json", "fitting.json"):
        (tmp_path / name).write_bytes(f"SYNTHETIC IMMUTABLE FIXTURE: {name}\n".encode())
    before = files_in(tmp_path)
    process = subprocess.run(
        [sys.executable, *arguments],
        cwd=tmp_path,
        env=command_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "usage:" in process.stdout.lower()
    assert files_in(tmp_path) == before


def test_training_configuration_still_resolves_from_its_file_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    checkpoint_bytes = b"SYNTHETIC SENTINEL; NOT A REAL CHECKPOINT\n"
    model_config = configs / "models.toml"
    model_config.write_text(
        '[[models]]\nname="synthetic"\ncheckpoint="../weights/source.pt"\n'
        f'sha256="{hashlib.sha256(checkpoint_bytes).hexdigest()}"\n'
    )
    training_config = configs / "training.toml"
    training_config.write_text(
        'model_config="models.toml"\noutput="../outputs/frozen"\n'
        "epochs=3\nbatch_size=2\nimage_size=640\nthreads=4\n"
    )
    checkpoint = tmp_path / "weights/source.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(checkpoint_bytes)
    before = files_in(tmp_path)
    configuration = read_settings(training_config)
    legacy = importlib.import_module("animal_eye_metrology.training")
    canonical = importlib.import_module("animal_eye_metrology.research.training")
    assert legacy is canonical
    assert legacy.read_settings(training_config) == configuration
    assert configuration["source"] == training_config
    assert configuration["model_config"] == model_config
    assert configuration["output"] == tmp_path / "outputs/frozen"
    assert configuration["models"][0]["checkpoint"] == checkpoint
    assert files_in(tmp_path) == before


def test_training_entrypoints_do_not_import_model_dependencies(tmp_path: Path) -> None:
    script = (
        "import importlib,json,runpy,sys; "
        "modules=[importlib.import_module(name) for name in "
        "['animal_eye_metrology.training', 'animal_eye_metrology.training_data', "
        "'animal_eye_metrology.segmentation_supply']]; "
        f"runpy.run_path({str(PROJECT_ROOT / 'scripts/export_training_weights.py')!r}, "
        "run_name='aem_export_import_contract'); "
        "print(json.dumps(sorted(set(sys.modules) & {'torch','cv2','ultralytics'})))"
    )
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=command_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert json.loads(process.stdout) == []
