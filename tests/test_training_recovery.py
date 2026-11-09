"""Keep fresh assessment behind an immutable development-only selection."""

import json
from pathlib import Path

import pytest

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.research.training_recovery import finalize


def test_fresh_evaluation_requires_all_three_completed_development_runs(tmp_path: Path) -> None:
    source = tmp_path / "settings.toml"
    source.write_text("seed = 42\n")
    models = [{"name": str(index)} for index in range(3)]
    for index in range(3):
        directory = tmp_path / str(index)
        directory.mkdir()
        result = {
            "status": "trained_pending_fresh_evaluation" if index < 2 else "training",
            "best_epoch": 1,
            "best_checkpoint_sha256": "unused",
            "best_development_mask_ap": 0.5,
            "recommended_candidate_from_development": "pretrained",
        }
        (directory / "result.json").write_text(json.dumps(result))
    with pytest.raises(Phase1Error, match="All three"):
        finalize({"output": tmp_path, "source": source, "models": models}, "0")
    assert not (tmp_path / "selection_frozen.json").exists()


def test_fresh_evaluation_rejects_settings_changed_after_selection(tmp_path: Path) -> None:
    source = tmp_path / "settings.toml"
    source.write_text("seed = 42\n")
    (tmp_path / "selection_frozen.json").write_text(
        json.dumps({"config_sha256": sha256_file(source)})
    )
    source.write_text("seed = 43\n")
    with pytest.raises(Phase1Error, match="Settings changed"):
        finalize({"output": tmp_path, "source": source}, "unused")
