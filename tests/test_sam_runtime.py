"""Minimal application SAM configuration preserves the frozen predictor policy."""

from pathlib import Path

import pytest

from animal_eye_metrology import eye_poc, sam_runtime
from animal_eye_metrology.errors import Phase1Error

ROOT = Path(__file__).resolve().parents[1]


def test_application_has_no_research_paths_and_legacy_uses_same_predictor():
    cfg = sam_runtime.settings(ROOT / "configs/sam2_runtime.toml")
    assert set(cfg["models"]) == {"sam2_1_tiny"}
    assert cfg["weights_directory"] == ROOT / "weights/eye_poc"
    assert "output_directory" not in cfg and "proposal_csv" not in cfg
    assert eye_poc.load_predictor is sam_runtime.load_predictor
    assert eye_poc.validate_source is sam_runtime.validate_source
    legacy = sam_runtime.settings(ROOT / "configs/eye_segmentation_poc.toml")
    assert set(legacy["models"]) == {"sam2_1_tiny", "sam_vit_b", "mobilesam"}


def test_application_rejects_accidental_model_or_execution_policy_change(tmp_path):
    original = (ROOT / "configs/sam2_runtime.toml").read_text()
    for text in (
        original.replace("threads = 4", "threads = 8"),
        original.replace("sam2.1_hiera_tiny.pt", "another.pt"),
    ):
        config = tmp_path / "runtime.toml"
        config.write_text(text)
        with pytest.raises(Phase1Error):
            sam_runtime.settings(config)
