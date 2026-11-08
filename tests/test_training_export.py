"""Synthetic state-serialization checks without importing or forwarding real models."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SPEC = importlib.util.spec_from_file_location(
    "aem_training_export",
    Path(__file__).resolve().parents[1] / "scripts/export_training_weights.py",
)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_state_verification_includes_buffers_and_exact_fp32_bytes():
    original = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "num_batches_tracked": torch.tensor(9, dtype=torch.int64),
    }
    reloaded = {key: tensor.clone() for key, tensor in original.items()}
    fingerprints = exporter.verify_state_dict(original, reloaded)
    assert (
        fingerprints["weight"]["sha256"]
        == hashlib.sha256(original["weight"].numpy().tobytes()).hexdigest()
    )
    assert fingerprints["num_batches_tracked"]["dtype"] == "torch.int64"
    reloaded["num_batches_tracked"] += 1
    with pytest.raises(ValueError, match="changed tensor"):
        exporter.verify_state_dict(original, reloaded)


def test_state_verification_rejects_lossy_casts_keys_shapes_and_nonfinite_values():
    original = {"weight": torch.tensor([[0.123456791]], dtype=torch.float32)}
    for changed in (
        {"weight": original["weight"] + 0.001},
        {"weight": original["weight"].reshape(1)},
        {"weight": original["weight"].to(torch.int64)},
    ):
        with pytest.raises(ValueError, match="changed tensor"):
            exporter.verify_state_dict(original, changed)
    with pytest.raises(ValueError, match="FP32"):
        exporter.verify_state_dict(original, {"weight": original["weight"].half()})
    with pytest.raises(ValueError, match="key mismatch"):
        exporter.verify_state_dict(original, {"unexpected": original["weight"]})
    with pytest.raises(ValueError, match="nonfinite"):
        exporter.state_fingerprints({"weight": torch.tensor([float("nan")])})


def test_native_payload_keeps_head_object_version_and_source_metadata():
    network = torch.nn.Linear(2, 80)
    original = {
        "version": "8.4.0",
        "model": object(),
        "ema": object(),
        "optimizer": {"state": "invented"},
        "license": "upstream-license",
    }
    payload = exporter.native_yolo_payload(original, network, {"best_epoch": 3})
    assert payload["model"] is network
    assert payload["model"].out_features == 80
    assert payload["version"] == original["version"]
    assert payload["license"] == original["license"]
    assert payload["ema"] is payload["optimizer"] is payload["scaler"] is None
    assert original["optimizer"] == {"state": "invented"}
    assert "aem_native_export" not in original


def test_existing_native_weights_or_verification_cannot_be_overwritten(tmp_path):
    weights = tmp_path / "native_weights.pt"
    verification = tmp_path / "native_export_verification.json"
    exporter.require_unused_destinations(weights, verification)
    verification.write_text("previous-verification")
    with pytest.raises(FileExistsError, match="overwrite"):
        exporter.require_unused_destinations(weights, verification)
    verification.unlink()
    weights.symlink_to(tmp_path / "absent-target")
    with pytest.raises(FileExistsError, match="overwrite"):
        exporter.require_unused_destinations(weights, verification)


def test_torchvision_head_sizes_remain_91_without_prediction_calls():
    state = {
        "roi_heads.box_predictor.cls_score.weight": torch.zeros(91, 2),
        "roi_heads.box_predictor.bbox_pred.weight": torch.zeros(364, 2),
        "roi_heads.mask_predictor.mask_fcn_logits.weight": torch.zeros(91, 2, 1, 1),
    }
    assert exporter._torchvision_head(state) == 91
    state["roi_heads.box_predictor.cls_score.weight"] = torch.zeros(10, 2)
    with pytest.raises(ValueError, match="91-class"):
        exporter._torchvision_head(state)
