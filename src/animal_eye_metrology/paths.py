"""Stable project paths and module locations, without importing model dependencies."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SOURCE_ROOT.parent

# One source of truth for lazy legacy imports and research source provenance.
MODULE_LOCATIONS: dict[str, str] = {
    "ablation_metrics": "research/ablation_metrics.py",
    "ablation_visualization": "ui/ablation_visualization.py",
    "analysis": "data/analysis.py",
    "annotation_session": "annotation/annotation_session.py",
    "annotation_view": "ui/annotation_view.py",
    "auto_box_metrics": "research/auto_box_metrics.py",
    "auto_box_visualization": "ui/auto_box_visualization.py",
    "auto_eye_box": "research/auto_eye_box.py",
    "auto_eye_localization": "research/auto_eye_localization.py",
    "auto_eye_visualization": "ui/auto_eye_visualization.py",
    "canvas_view": "ui/canvas_view.py",
    "coco": "data/coco.py",
    "config": "core/config.py",
    "definition_comparison": "research/definition_comparison.py",
    "errors": "core/errors.py",
    "execution_lock": "core/execution_lock.py",
    "eye_annotations": "annotation/eye_annotations.py",
    "eye_comparison": "research/eye_comparison.py",
    "eye_geometry": "measurement/eye_geometry.py",
    "eye_poc": "research/eye_poc.py",
    "eye_pose_worker": "inference/eye_pose_worker.py",
    "eye_prompts": "annotation/eye_prompts.py",
    "inference_artifacts": "inference/inference_artifacts.py",
    "inference_capabilities": "inference/inference_capabilities.py",
    "inference_pipeline": "inference/inference_pipeline.py",
    "inference_view": "ui/inference_view.py",
    "inference_worker": "inference/inference_worker.py",
    "measurement_baseline": "measurement/measurement_baseline.py",
    "metrology": "measurement/metrology.py",
    "outputs": "data/outputs.py",
    "phase2": "research/phase2.py",
    "phase2_pilot": "research/phase2_pilot.py",
    "prompt_roi_ablation": "research/prompt_roi_ablation.py",
    "prompt_view": "ui/prompt_view.py",
    "results_definitions": "ui/results_definitions.py",
    "results_overlay": "ui/results_overlay.py",
    "results_view": "ui/results_view.py",
    "results_workspace": "ui/results_workspace.py",
    "review_sample": "annotation/review_sample.py",
    "review_store": "annotation/review_store.py",
    "sam_runtime": "models/sam_runtime.py",
    "segmentation": "models/segmentation.py",
    "verification": "core/verification.py",
    "visualization": "ui/visualization.py",
}


def module_source(name: str) -> Path:
    """Locate a known source file by its original module name or basename."""
    key = name.removesuffix(".py")
    if key in MODULE_LOCATIONS:
        return PACKAGE_ROOT / MODULE_LOCATIONS[key]
    if key in {"__init__", "__main__", "api", "cli", "paths", "_compat"}:
        return PACKAGE_ROOT / f"{key}.py"
    raise ValueError(f"Unknown project module: {name}")


def package_sources(package_root: Path = PACKAGE_ROOT) -> list[Path]:
    """Collect every Python source with stable ordering across responsibility folders."""
    return sorted(package_root.rglob("*.py"))
