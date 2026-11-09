"""Export completed fine-tuning states in native formats, with exact reload checks.

This command only loads and serializes local models on one CPU thread. It never
performs a forward pass, fuses layers, downloads assets, or updates parameters.
YOLO26 requires the same isolated Ultralytics 8.4.0 environment as its training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.research.training import deny_network, read_settings


def state_fingerprints(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Fingerprint every named parameter and buffer; reject float quantization."""
    import torch

    if not isinstance(state, Mapping) or not state:
        raise ValueError("Expected a nonempty tensor state dictionary.")
    result = {}
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("State keys must be strings and all values must be tensors.")
        tensor = tensor.detach().cpu().contiguous()
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32:
                raise ValueError(f"FP32 export requires float32 tensors: {name} is {tensor.dtype}.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Refusing nonfinite model state: {name}.")
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    return result


def verify_state_dict(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Require identical keys, dtypes, shapes and tensor bytes after native reload."""
    expected_fingerprints = state_fingerprints(expected)
    actual_fingerprints = state_fingerprints(actual)
    if expected_fingerprints.keys() != actual_fingerprints.keys():
        missing = sorted(expected_fingerprints.keys() - actual_fingerprints.keys())
        extra = sorted(actual_fingerprints.keys() - expected_fingerprints.keys())
        raise ValueError(f"State key mismatch: missing={missing}, extra={extra}.")
    changed = [
        name
        for name in expected_fingerprints
        if expected_fingerprints[name] != actual_fingerprints[name]
    ]
    if changed:
        raise ValueError(f"Native reload changed tensor state: {changed}.")
    return actual_fingerprints


def native_yolo_payload(
    original: Mapping[str, Any], network: Any, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain source metadata/version and the unchanged model object in FP32."""
    payload = dict(original)
    payload.update(model=network, ema=None, optimizer=None, scaler=None)
    payload["aem_native_export"] = dict(provenance)
    return payload


def require_unused_destinations(weights: Path, verification: Path) -> None:
    """Do not replace previous weights, verification, or dangling output symlinks."""
    for path in (weights, verification):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing native export: {path}")


def _names(network: Any) -> dict[int, str]:
    names = network.names
    result = dict(names) if isinstance(names, Mapping) else dict(enumerate(names))
    if set(result) != set(range(80)) or len(set(result.values())) != 80:
        raise ValueError("Native YOLO export requires the unchanged 80-class COCO names.")
    return result


def _torchvision_head(state: Mapping[str, Any]) -> int:
    expected = {
        "roi_heads.box_predictor.cls_score.weight": 91,
        "roi_heads.box_predictor.bbox_pred.weight": 91 * 4,
        "roi_heads.mask_predictor.mask_fcn_logits.weight": 91,
    }
    if any(key not in state or state[key].shape[0] != size for key, size in expected.items()):
        raise ValueError("TorchVision export must preserve the original 91-class predictors.")
    return 91


def export_training_weights(config_path: Path, model_name: str) -> dict[str, Any]:
    """Export a completed run's selected best state and verify its native reload."""
    cfg = read_settings(Path(config_path))
    model_cfg = next((row for row in cfg["models"] if row["name"] == model_name), None)
    if model_cfg is None:
        raise ValueError(f"Unknown configured model: {model_name}")
    directory = cfg["output"] / model_name
    weights = directory / "native_weights.pt"
    verification = directory / "native_export_verification.json"
    require_unused_destinations(weights, verification)
    best_path = directory / "best.pt"
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete":
        raise ValueError("Only completed training runs can produce a native export.")
    best_digest = sha256_file(best_path)
    if best_digest != result["best_checkpoint_sha256"]:
        raise ValueError("Selected best.pt differs from the completed training result.")
    if sha256_file(model_cfg["checkpoint"]) != model_cfg["sha256"]:
        raise ValueError("Original source checkpoint differs from its pinned SHA-256.")
    source_paths = {
        cfg["source"],
        cfg["model_config"],
        best_path,
        result_path,
        model_cfg["checkpoint"],
        Path(__file__).resolve(),
    }
    source_before = {str(path): sha256_file(path) for path in sorted(source_paths)}
    cache = directory.parent / "cache" / f"{model_name}_native_export"
    os.environ.update(
        YOLO_CONFIG_DIR=str(cache),
        YOLO_OFFLINE="true",
        YOLO_AUTOINSTALL="false",
        MPLCONFIGDIR=str(cache / "matplotlib"),
        WANDB_DISABLED="true",
    )
    deny_network()
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    if best["source_checkpoint_sha256"] != model_cfg["sha256"]:
        raise ValueError("Best state belongs to a different source checkpoint.")
    if best["epoch"] != result["best_epoch"]:
        raise ValueError("Best epoch and completed training result disagree.")
    expected = best["model_state"]
    state_fingerprints(expected)  # Validate FP32 and finiteness before serializing anything.
    context = {
        "model": model_name,
        "best_epoch": best["epoch"],
        "best_checkpoint_sha256": best_digest,
        "source_checkpoint_sha256": model_cfg["sha256"],
        "training_configuration_sha256": sha256_file(cfg["source"]),
        "format": "FP32 inference artifact; optimizer/scaler state omitted",
        "original_checkpoint_metadata_retained": True,
    }
    backend = model_cfg["backend"]
    source_version = None
    expected_names = None
    if backend == "ultralytics":
        import ultralytics
        from ultralytics import YOLO

        if ultralytics.__version__ != model_cfg["package_version"]:
            raise ValueError("Use the exact isolated Ultralytics version from the model config.")
        wrapper = YOLO(str(model_cfg["checkpoint"]), task="segment")
        network = wrapper.model.cpu().float()
        expected_names = _names(network)
        original_head = network.model[-1]
        if getattr(original_head, "nc", None) != 80:
            raise ValueError("Source YOLO head is not the expected original 80-class head.")
        network.load_state_dict(expected, strict=True)
        if network.model[-1] is not original_head or _names(network) != expected_names:
            raise ValueError("Loading best state unexpectedly replaced the original head/names.")
        verify_state_dict(expected, network.state_dict())
        source_version = wrapper.ckpt.get("version")
        payload = native_yolo_payload(wrapper.ckpt, network, context)
        class_count = 80
        package = {"name": "ultralytics", "version": ultralytics.__version__}
    elif backend == "torchvision":
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

        if version("torchvision") != model_cfg["package_version"]:
            raise ValueError("TorchVision version differs from the configured training version.")
        original = torch.load(model_cfg["checkpoint"], map_location="cpu", weights_only=True)
        class_count = _torchvision_head(original)
        _torchvision_head(expected)
        network = (
            maskrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                num_classes=91,
                min_size=cfg["image_size"],
                max_size=cfg["maskrcnn_max_size"],
            )
            .cpu()
            .float()
        )
        network.load_state_dict(original, strict=True)
        original_head = network.roi_heads.box_predictor
        network.load_state_dict(expected, strict=True)
        if network.roi_heads.box_predictor is not original_head:
            raise ValueError("Loading best state replaced the original TorchVision predictor.")
        verify_state_dict(expected, network.state_dict())
        payload = {key: tensor.detach().cpu().clone() for key, tensor in expected.items()}
        package = {"name": "torchvision", "version": version("torchvision")}
    else:
        raise ValueError(f"Unsupported configured backend: {backend}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=".native_weights.", suffix=".pt", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        if backend == "ultralytics":
            reloaded = YOLO(str(temporary), task="segment")
            actual = reloaded.model.cpu().state_dict()
            if _names(reloaded.model) != expected_names or reloaded.model.model[-1].nc != 80:
                raise ValueError("Native YOLO reload changed class names or head size.")
            if reloaded.ckpt.get("version") != source_version:
                raise ValueError("Native export changed the original checkpoint version metadata.")
            if reloaded.ckpt.get("ema") is not None or reloaded.ckpt.get("optimizer") is not None:
                raise ValueError(
                    "Native inference export unexpectedly contains EMA/optimizer state."
                )
        else:
            actual = torch.load(temporary, map_location="cpu", weights_only=True)
            _torchvision_head(actual)
            network.load_state_dict(actual, strict=True)
        fingerprints = verify_state_dict(expected, actual)
        source_after = {str(path): sha256_file(path) for path in sorted(source_paths)}
        if source_before != source_after:
            raise ValueError("An export source changed during native conversion/reload.")
        require_unused_destinations(weights, verification)
        os.link(temporary, weights)  # Atomic installation that cannot replace an existing file.
        report = {
            "status": "verified",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model": model_name,
            "backend": backend,
            "package": package,
            "torch_version": torch.__version__,
            "device": "cpu",
            "threads": 1,
            "best_epoch": best["epoch"],
            "tensor_count": len(fingerprints),
            "all_state_tensors_bitwise_equal": True,
            "all_floating_tensors_fp32": True,
            "class_count": class_count,
            "names": expected_names,
            "head_architecture_unchanged": True,
            "source_version_metadata": source_version,
            "native_weights_sha256": sha256_file(weights),
            "best_checkpoint_sha256": best_digest,
            "source_sha256": source_before,
            "sources_unchanged": True,
            "forward_passes": 0,
            "optimizer_steps": 0,
            "model_fused": False,
            "network_calls_disabled": True,
            "state_tensors": fingerprints,
            "scope": (
                "serialization and native reload only; no new inference or accuracy evaluation"
            ),
        }
        with verification.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return report
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    result = export_training_weights(args.config, args.model)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "model",
                    "tensor_count",
                    "native_weights_sha256",
                    "class_count",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
