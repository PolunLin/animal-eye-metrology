"""Shared SAM predictor and source checks; application runtime has no research-data paths."""

import json
import tomllib
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.errors import Phase1Error


def settings(path: Path) -> dict[str, Any]:
    """Read the fixed SAM 2.1 Tiny application configuration."""
    path = path.resolve()
    cfg = tomllib.loads(path.read_text())
    if cfg.get("runtime_kind") != "application":
        raise Phase1Error("SAM requires the application runtime configuration.")
    spec = cfg.get("models", {}).get("sam2_1_tiny", {})
    if (cfg.get("device"), cfg.get("threads"), cfg.get("seed")) != ("cpu", 4, 42):
        raise Phase1Error("Application SAM requires CPU, four threads and seed 42.")
    if set(cfg.get("models", {})) != {"sam2_1_tiny"} or spec != {
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "distribution": "SAM-2",
        "commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
    }:
        raise Phase1Error("Application SAM differs from the approved SAM 2.1 Tiny baseline.")
    cfg["source"] = path
    cfg["weights_directory"] = (path.parent / cfg["weights_directory"]).resolve()
    return cfg


def load_predictor(name: str, checkpoint: Path) -> Any:
    if name == "sam2_1_tiny":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(checkpoint),
            device="cpu",
            apply_postprocessing=False,
        )
        return SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
    raise Phase1Error("Only SAM 2.1 Tiny is included in the delivery.")


def validate_source(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Verify the fixed official source commit installed by bootstrap."""
    package = distribution(spec["distribution"])
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") == spec["commit"]:
        return {"commit": spec["commit"], "method": "pinned_vcs_install"}
    raise Phase1Error("Application SAM must be installed from its pinned VCS commit.")
