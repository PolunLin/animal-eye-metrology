"""Checkpoint preparation is explicit and never overwrites mismatched existing bytes."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


def script():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_models.py"
    spec = importlib.util.spec_from_file_location("aem_prepare_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verification_is_offline_and_distinguishes_missing_mismatch_and_ready(
    tmp_path, monkeypatch
):
    module = script()
    expected = hashlib.sha256(b"approved synthetic bytes").hexdigest()
    monkeypatch.setattr(
        module,
        "MODELS",
        tuple(
            {"file": name, "url": "https://invalid.test/model", "sha256": expected}
            for name in ("ready.pt", "wrong.pt", "missing.pt")
        ),
    )
    (tmp_path / "ready.pt").write_bytes(b"approved synthetic bytes")
    (tmp_path / "wrong.pt").write_bytes(b"different bytes")
    rows = module.prepare(tmp_path)
    assert [row["status"] for row in rows] == ["ready", "mismatch", "missing"]
    # Even explicit download mode refuses to overwrite existing mismatched bytes.
    monkeypatch.setattr(module, "MODELS", module.MODELS[:2])
    assert module.prepare(tmp_path, download=True)[1]["status"] == "mismatch"
    assert (tmp_path / "wrong.pt").read_bytes() == b"different bytes"


def test_download_failure_removes_partial_file_without_creating_checkpoint(tmp_path, monkeypatch):
    module = script()
    monkeypatch.setattr(
        module,
        "MODELS",
        (
            {
                "file": "model.pt",
                "url": "https://invalid.test/model",
                "sha256": "0" * 64,
            },
        ),
    )

    def fail(*args, **kwargs):
        raise OSError("synthetic network failure")

    monkeypatch.setattr(module.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="synthetic network failure"):
        module.prepare(tmp_path, download=True)
    assert list(tmp_path.iterdir()) == []
