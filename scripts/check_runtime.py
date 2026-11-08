#!/usr/bin/env python3
"""Check pinned isolated runtimes and local weights; optionally load all three models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKER = "AEM_RUNTIME_CHECK="
APP_PROBE = r"""
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path
import fastapi
import python_multipart
import numpy
import streamlit
import torch
import torchvision
import uvicorn
from animal_eye_metrology.inference_pipeline import load_config, preflight
from animal_eye_metrology.sam_runtime import load_predictor, settings, validate_source
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

cfg = load_config(Path(sys.argv[1]))
checks = preflight(cfg)
assert sys.version_info[:2] == (3, 11), "Python 3.11 is required."
assert torch.__version__.split('+')[0] == '2.9.1', "Unexpected app Torch version."
assert torchvision.__version__.split('+')[0] == '0.24.1', "Unexpected torchvision version."
assert metadata.version('ultralytics') == '8.4.0', "Unexpected detector package version."
assert torch.version.cuda is None, "This runtime must use a CPU-only Torch build."
from torchvision.ops import nms
nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
sam_cfg = settings(cfg['sam_runtime'])
sam_source = validate_source(sam_cfg, sam_cfg['models']['sam2_1_tiny'])
loaded = []
if sys.argv[2] == 'load':
    assert all(row['status'] == 'ready' for row in checks), "Preflight failed; models not loaded."
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    detector = YOLO(str(cfg['detector_checkpoint']), task='segment')
    assert detector.task == 'segment'
    loaded.append('YOLO26s-seg')
    del detector
    predictor = load_predictor('sam2_1_tiny', cfg['sam_checkpoint'])
    assert predictor is not None
    loaded.append('SAM 2.1 Tiny')
    del predictor
result = {
    'status': 'ready' if all(row['status'] == 'ready' for row in checks) else 'failed',
    'python': platform.python_version(),
    'versions': {name: metadata.version(name) for name in (
        'torch', 'torchvision', 'ultralytics', 'SAM-2', 'numpy', 'Pillow',
        'streamlit', 'fastapi', 'uvicorn', 'python-multipart')},
    'preflight': checks, 'sam_source': sam_source, 'models_loaded': loaded,
    'inference_performed': False, 'accuracy_evaluated': False,
}
print('AEM_RUNTIME_CHECK=' + json.dumps(result))
"""
POSE_PROBE = r"""
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
import tomllib
from pathlib import Path
import cv2
import mmcv
import mmdet
import mmengine
import mmpose
import numpy
import torch
from mmcv.ops import nms
from mmpose.apis import init_model
from mmpose.utils import register_all_modules
from animal_eye_metrology.eye_pose_worker import CONFIG

config = Path(sys.argv[1]).resolve()
cfg = tomllib.loads(config.read_text())
def path(name):
    return config.parent / cfg[name]
def sha256(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()
assert sys.version_info[:2] == (3, 11), "Python 3.11 is required."
assert torch.__version__.split('+')[0] == '2.2.2', "Unexpected pose Torch version."
assert numpy.__version__ == '1.26.4', "Unexpected pose NumPy version."
assert mmcv.__version__ == '2.1.0', "Unexpected MMCV version."
assert mmdet.__version__ == '3.3.0', "Unexpected MMDetection version."
assert mmengine.__version__ == '0.10.7', "Unexpected MMEngine version."
assert torch.version.cuda is None, "Pose runtime must use a CPU-only Torch build."
nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
manifest = json.loads(path('pose_source_manifest').read_text())
assert manifest['commit'] == cfg['mmpose_commit'], "MMPose manifest commit differs."
module = Path(mmpose.__file__).parent
for relative, expected in manifest['files_sha256'].items():
    assert sha256(module / relative) == expected, f'MMPose runtime source differs: {relative}'
checkpoint = path('pose_checkpoint')
assert sha256(checkpoint) == cfg['pose_checkpoint_sha256'], "Pose checkpoint differs."
packaged_config = module / '.mim' / CONFIG
assert packaged_config.is_file(), "Packaged AP-10K model config is missing."
loaded = []
if sys.argv[2] == 'load':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    register_all_modules()
    model = init_model(str(packaged_config), str(checkpoint), device='cpu',
        cfg_options={'model.backbone.init_cfg': None, 'model.test_cfg.output_heatmaps': True})
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    names = model.dataset_meta['keypoint_id2name']
    assert model.dataset_meta['dataset_name'] == 'ap10k'
    assert [names[i] for i in range(3)] == ['L_Eye', 'R_Eye', 'Nose']
    assert len(names) == 17 and list(model.dataset_meta['flip_indices'])[:3] == [1, 0, 2]
    loaded.append('HRNet-W32/AP-10K')
result = {
    'status': 'ready', 'python': platform.python_version(),
    'versions': {name: metadata.version(name) for name in (
        'torch', 'torchvision', 'numpy', 'mmcv', 'mmengine', 'mmdet', 'mmpose')},
    'mmpose_source': {'commit': manifest['commit'],
        'verified_packaged_files': len(manifest['files_sha256'])},
    'checkpoint_sha256': sha256(checkpoint), 'models_loaded': loaded,
    'inference_performed': False, 'accuracy_evaluated': False,
}
print('AEM_RUNTIME_CHECK=' + json.dumps(result))
"""


def configured_path(config: Path, values: dict[str, Any], field: str) -> Path:
    """Keep virtual-environment interpreter symlinks intact."""
    return Path(os.path.abspath(config.parent / values[field]))


def run_probe(
    name: str,
    interpreter: Path,
    config: Path,
    code: str,
    source_paths: list[Path],
    cache: Path,
    load_models: bool,
    timeout: int,
) -> dict[str, Any]:
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        return {"status": "failed", "error": f"Missing executable: {interpreter}"}
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(map(str, source_paths)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": "0",
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "YOLO_CONFIG_DIR": str(cache / "ultralytics"),
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    command = [str(interpreter), "-c", code, str(config), "load" if load_models else "imports"]
    try:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if process.returncode:
            return {
                "status": "failed",
                "returncode": process.returncode,
                "error": (process.stderr or process.stdout)[-6000:],
            }
        lines = [
            line[len(MARKER) :] for line in process.stdout.splitlines() if line.startswith(MARKER)
        ]
        if not lines:
            return {"status": "failed", "error": f"{name} returned no structured check result."}
        result: dict[str, Any] = json.loads(lines[-1])
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"status": "failed", "error": str(error)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            os.environ.get("AEM_INFERENCE_CONFIG", PROJECT_ROOT / "configs/inference.runtime.toml")
        ),
    )
    parser.add_argument("--load-models", action="store_true", help="Load real checkpoints on CPU.")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".runtime/check-cache")
    parser.add_argument("--timeout", type=int, default=900, help="Per-runtime timeout in seconds.")
    args = parser.parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive.")
        config = args.config.resolve()
        values = tomllib.loads(config.read_text(encoding="utf-8"))
        source = PROJECT_ROOT / "src"
        app_paths = [configured_path(config, values, "detector_source"), source]
        probes = {
            "app": run_probe(
                "app",
                configured_path(config, values, "python"),
                config,
                APP_PROBE,
                app_paths,
                args.cache_dir / "app",
                args.load_models,
                args.timeout,
            ),
            "pose": run_probe(
                "pose",
                configured_path(config, values, "pose_python"),
                config,
                POSE_PROBE,
                [source],
                args.cache_dir / "pose",
                args.load_models,
                args.timeout,
            ),
        }
        ready = all(result.get("status") == "ready" for result in probes.values())
        print(
            json.dumps(
                {
                    "status": "ready" if ready else "failed",
                    "config": str(config),
                    "mode": "model_load" if args.load_models else "imports_sources_and_weights",
                    "runtimes": probes,
                    "accuracy_evaluated": False,
                },
                indent=2,
            )
        )
        return 0 if ready else 1
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
