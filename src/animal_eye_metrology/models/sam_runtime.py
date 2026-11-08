"""Shared SAM predictor and source checks; application runtime has no research-data paths."""

import json
import subprocess
import tomllib
from importlib.metadata import distribution
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file


def settings(path: Path) -> dict[str, Any]:
    """Read the minimal application runtime or preserve the historical POC contract."""
    path = path.resolve()
    cfg = tomllib.loads(path.read_text())
    if cfg.get("runtime_kind") != "application":
        from animal_eye_metrology.research.eye_poc import settings as legacy_settings

        return legacy_settings(path)
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
    if name == "sam_vit_b":
        from segment_anything import SamPredictor, sam_model_registry
    else:
        from mobile_sam import SamPredictor, sam_model_registry
    model = sam_model_registry["vit_b" if name == "sam_vit_b" else "vit_t"](
        checkpoint=str(checkpoint)
    )
    model.to(device="cpu", dtype=__import__("torch").float32)
    model.eval()
    return SamPredictor(model)


def validate_source(cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Verify either a pinned VCS install or the same commit from a sparse local checkout."""
    package = distribution(spec["distribution"])
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") == spec["commit"]:
        return {"commit": spec["commit"], "method": "pinned_vcs_install"}
    if "output_directory" not in cfg:
        raise Phase1Error("Application SAM must be installed from its pinned VCS commit.")
    manifest_path = cfg["output_directory"] / "vendor/source_manifest.json"
    sources = json.loads(manifest_path.read_text())
    source = sources[spec["distribution"]]
    directory = Path(source["directory"]).resolve()
    installed_url = urlparse(direct.get("url", ""))
    if (
        source["commit"] != spec["commit"]
        or installed_url.scheme != "file"
        or Path(unquote(installed_url.path)).resolve() != directory
    ):
        raise Phase1Error("Installed official source commit differs from configuration.")
    revision = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    if revision != spec["commit"]:
        raise Phase1Error("Sparse checkout revision differs.")
    for relative, digest in source["files_sha256"].items():
        if sha256_file(directory / relative) != digest:
            raise Phase1Error(f"Official source changed: {relative}")
        # CUDA build sources are not packaged when SAM2_BUILD_CUDA=0. Check all
        # Python/config runtime files against the official checkout instead.
        if relative.startswith(source["module"] + "/") and Path(relative).suffix in (
            ".py",
            ".yaml",
            ".yml",
            ".json",
        ):
            installed = Path(package.locate_file(relative))
            if sha256_file(installed) != digest:
                raise Phase1Error(f"Installed model source differs: {relative}")
    return {
        "commit": revision,
        "method": "verified_sparse_checkout_and_installed_file_hashes",
        "manifest_sha256": sha256_file(manifest_path),
    }
