"""Minimal application SAM configuration preserves the frozen predictor policy."""

from pathlib import Path

import pytest

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.models import sam_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_application_has_only_the_final_model_and_no_research_paths():
    cfg = sam_runtime.settings(ROOT / "configs/sam2_runtime.toml")
    assert set(cfg["models"]) == {"sam2_1_tiny"}
    assert cfg["weights_directory"] == ROOT / "weights/eye_poc"
    assert "output_directory" not in cfg and "proposal_csv" not in cfg


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
