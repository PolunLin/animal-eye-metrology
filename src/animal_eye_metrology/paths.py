"""Stable project paths and module locations, without importing model dependencies."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SOURCE_ROOT.parent

# Source locations for worker execution and reproducibility records.
MODULE_LOCATIONS: dict[str, str] = {
    "errors": "core/errors.py",
    "execution_lock": "core/execution_lock.py",
    "verification": "core/verification.py",
    "eye_pose_worker": "inference/eye_pose_worker.py",
    "inference_artifacts": "inference/inference_artifacts.py",
    "inference_capabilities": "inference/inference_capabilities.py",
    "inference_pipeline": "inference/inference_pipeline.py",
    "inference_worker": "inference/inference_worker.py",
    "eye_geometry": "measurement/eye_geometry.py",
    "metrology": "measurement/metrology.py",
    "sam_runtime": "models/sam_runtime.py",
    "inference_view": "ui/inference_view.py",
}


def module_source(name: str) -> Path:
    """Locate a known source file by its module basename."""
    key = name.removesuffix(".py")
    if key in MODULE_LOCATIONS:
        return PACKAGE_ROOT / MODULE_LOCATIONS[key]
    if key in {"__init__", "__main__", "api", "cli", "paths"}:
        return PACKAGE_ROOT / f"{key}.py"
    raise ValueError(f"Unknown project module: {name}")


def package_sources(package_root: Path = PACKAGE_ROOT) -> list[Path]:
    """Collect every Python source with stable ordering across responsibility folders."""
    return sorted(package_root.rglob("*.py"))
