"""Validate interprocess exclusion and release without loading models."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from animal_eye_metrology.execution_lock import InferenceBusyError, inference_execution_lock


def test_second_open_is_rejected_and_lock_file_is_reused(tmp_path: Path) -> None:
    output = tmp_path / "new" / "runs"
    with inference_execution_lock(output):
        inode = (output / ".inference.lock").stat().st_ino
        with pytest.raises(InferenceBusyError, match="Another inference"):
            with inference_execution_lock(output):
                pytest.fail("A second caller must not enter the protected operation.")
    with inference_execution_lock(output):
        assert (output / ".inference.lock").stat().st_ino == inode


def test_operation_exception_releases_lock_without_masking_original_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Synthetic model failure"):
        with inference_execution_lock(tmp_path):
            raise ValueError("Synthetic model failure")
    with inference_execution_lock(tmp_path):
        assert (tmp_path / ".inference.lock").is_file()


def test_real_subprocess_cannot_enter_until_parent_releases_lock(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ | {"PYTHONPATH": str(source)}
    code = """
import sys
from pathlib import Path
from animal_eye_metrology.execution_lock import InferenceBusyError, inference_execution_lock
try:
    with inference_execution_lock(Path(sys.argv[1])):
        pass
except InferenceBusyError:
    raise SystemExit(42)
"""

    def child() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)],
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )

    with inference_execution_lock(tmp_path):
        busy = child()
        assert busy.returncode == 42, busy.stderr
    released = child()
    assert released.returncode == 0, released.stderr


def test_filesystem_error_is_not_reported_as_busy(tmp_path: Path) -> None:
    output = tmp_path / "not-a-directory"
    output.write_text("synthetic fixture")
    with pytest.raises(FileExistsError):
        with inference_execution_lock(output):
            pytest.fail("An invalid directory must not enter the protected operation.")
