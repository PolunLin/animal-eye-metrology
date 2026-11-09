"""Keep fresh assessment behind an immutable development-only selection."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.research import training_recovery
from animal_eye_metrology.research.training_recovery import finalize


@pytest.fixture
def pending_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Three invented completed runs; no real models, data or evaluations."""
    source = tmp_path / "settings.toml"
    source.write_text("image_size = 640\n")
    model_config = tmp_path / "models.toml"
    model_config.write_text("# Synthetic model configuration\n")
    data = tmp_path / "data"
    data.mkdir()
    for name in ("fitting", "development", "holdout"):
        (data / f"{name}.json").write_text(json.dumps({"synthetic_split": name}))
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "status": "frozen",
                "artifact_sha256": {path.name: sha256_file(path) for path in data.glob("*.json")},
            }
        )
    )
    sources = {
        str(path): sha256_file(path)
        for path in (
            source,
            model_config,
            data / "manifest.json",
            data / "fitting.json",
            data / "development.json",
        )
    }
    # Historical code provenance is retained without requiring current code identity.
    sources[str(tmp_path / "historical_training.py")] = "historical-code-digest"
    models = [{"name": name, "sha256": f"synthetic-{name}"} for name in ("a", "b", "c")]
    output = tmp_path / "retrain"
    output.mkdir()
    for model in models:
        directory = output / model["name"]
        directory.mkdir()
        (directory / "best.pt").write_bytes(b"synthetic checkpoint; never loaded")
        (directory / "run_manifest.json").write_text(
            json.dumps(
                {
                    "model": model["name"],
                    "source_checkpoint_sha256": model["sha256"],
                    "source_sha256": sources,
                }
            )
        )
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "status": "trained_pending_fresh_evaluation",
                    "best_epoch": 1,
                    "best_checkpoint_sha256": sha256_file(directory / "best.pt"),
                    "best_development_mask_ap": 0.5,
                    "recommended_candidate_from_development": "pretrained",
                }
            )
        )
    monkeypatch.setattr(
        training_recovery, "initialize", Mock(side_effect=AssertionError("Must fail before models"))
    )
    monkeypatch.setattr(
        training_recovery, "evaluate", Mock(side_effect=AssertionError("No real evaluation"))
    )
    return {
        "source": source,
        "model_config": model_config,
        "data_directory": data,
        "output": output,
        "previous_output": tmp_path / "previous",
        "models": models,
    }


def assert_no_assessment_started(cfg: dict[str, Any]) -> None:
    assert not (cfg["output"] / "selection_frozen.json").exists()
    training_recovery.initialize.assert_not_called()
    training_recovery.evaluate.assert_not_called()


@pytest.mark.parametrize("model", ["a", "b", "c"])
def test_first_freeze_rejects_any_runs_different_training_config(
    pending_recovery: dict[str, Any], model: str
) -> None:
    cfg = pending_recovery
    path = cfg["output"] / model / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_sha256"][str(cfg["source"])] = "different-training-config"
    path.write_text(json.dumps(manifest))
    with pytest.raises(Phase1Error, match=f"recorded run: {model}"):
        finalize(cfg, "a")
    assert_no_assessment_started(cfg)


@pytest.mark.parametrize(
    "source", ["source", "model_config", "manifest.json", "fitting.json", "development.json"]
)
def test_first_freeze_rejects_changed_configuration_or_training_data(
    pending_recovery: dict[str, Any], source: str
) -> None:
    cfg = pending_recovery
    path = cfg[source] if source in cfg else cfg["data_directory"] / source
    path.write_text(path.read_text() + "\n")
    with pytest.raises(Phase1Error, match="configuration or data differs"):
        finalize(cfg, "a")
    assert_no_assessment_started(cfg)


@pytest.mark.parametrize("failure", ["missing", "invalid_json", "missing_source_digest"])
def test_first_freeze_requires_every_training_manifest(
    pending_recovery: dict[str, Any], failure: str
) -> None:
    cfg = pending_recovery
    path = cfg["output"] / "b" / "run_manifest.json"
    if failure == "missing":
        path.unlink()
    elif failure == "invalid_json":
        path.write_text("not JSON")
    else:
        manifest = json.loads(path.read_text())
        del manifest["source_sha256"][str(cfg["model_config"])]
        path.write_text(json.dumps(manifest))
    with pytest.raises(Phase1Error):
        finalize(cfg, "a")
    assert_no_assessment_started(cfg)


@pytest.mark.parametrize("field", ["model", "source_checkpoint_sha256"])
def test_first_freeze_rejects_mismatched_model_identity(
    pending_recovery: dict[str, Any], field: str
) -> None:
    cfg = pending_recovery
    path = cfg["output"] / "b" / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "another-model"
    path.write_text(json.dumps(manifest))
    with pytest.raises(Phase1Error, match="recorded run: b"):
        finalize(cfg, "a")
    assert_no_assessment_started(cfg)


def test_first_freeze_requires_holdout_bound_by_training_data_manifest(
    pending_recovery: dict[str, Any],
) -> None:
    cfg = pending_recovery
    (cfg["data_directory"] / "holdout.json").write_text('{"synthetic_split": "replaced"}')
    with pytest.raises(Phase1Error, match="dataset changed: holdout.json"):
        finalize(cfg, "a")
    assert_no_assessment_started(cfg)


def mock_assessment(monkeypatch: pytest.MonkeyPatch) -> Mock:
    network = SimpleNamespace(load_state_dict=Mock())
    monkeypatch.setattr(
        training_recovery,
        "initialize",
        Mock(return_value=(SimpleNamespace(model=network), {"name": "synthetic"})),
    )
    evaluator = Mock(return_value={"coco_segm": {"mask_ap": 0.5}})
    monkeypatch.setattr(training_recovery, "evaluate", evaluator)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(load=Mock(return_value={"model_state": {}}))
    )
    return evaluator


def test_matching_runs_finalize_sequentially_without_rewriting_freeze(
    pending_recovery: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = pending_recovery
    evaluator = mock_assessment(monkeypatch)
    finalize(cfg, "a")
    freeze = cfg["output"] / "selection_frozen.json"
    frozen_bytes = freeze.read_bytes()
    sources = json.loads(frozen_bytes)["source_sha256"]
    assert sources[str(cfg["data_directory"] / "holdout.json")] == sha256_file(
        cfg["data_directory"] / "holdout.json"
    )
    for name in ("b", "c"):
        finalize(cfg, name)
        assert freeze.read_bytes() == frozen_bytes
    assert evaluator.call_count == 9
    for name in ("a", "b", "c"):
        result = json.loads((cfg["output"] / name / "result.json").read_text())
        assert result["status"] == "complete"


@pytest.mark.parametrize("source", ["model_config", "holdout.json", "run_manifest.json"])
def test_new_freeze_rejects_changed_sources_before_next_models_initialization(
    pending_recovery: dict[str, Any], monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    cfg = pending_recovery
    mock_assessment(monkeypatch)
    finalize(cfg, "a")
    freeze = cfg["output"] / "selection_frozen.json"
    frozen_bytes = freeze.read_bytes()
    if source == "run_manifest.json":
        path = cfg["output"] / "b" / source
    else:
        path = cfg[source] if source in cfg else cfg["data_directory"] / source
    path.write_text(path.read_text() + "\n")
    training_recovery.initialize.reset_mock()
    training_recovery.evaluate.reset_mock()
    with pytest.raises(Phase1Error, match="Frozen evaluation source changed"):
        finalize(cfg, "b")
    training_recovery.initialize.assert_not_called()
    training_recovery.evaluate.assert_not_called()
    assert freeze.read_bytes() == frozen_bytes


def test_legacy_freeze_remains_usable_without_new_source_provenance(
    pending_recovery: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = pending_recovery
    freeze = cfg["output"] / "selection_frozen.json"
    freeze.write_text(
        json.dumps(
            {
                "config_sha256": sha256_file(cfg["source"]),
                "selections": {
                    "a": {"best_checkpoint_sha256": sha256_file(cfg["output"] / "a/best.pt")}
                },
            }
        )
    )
    frozen_bytes = freeze.read_bytes()
    for model in cfg["models"]:
        (cfg["output"] / model["name"] / "run_manifest.json").unlink()
    evaluator = mock_assessment(monkeypatch)
    finalize(cfg, "a")
    assert evaluator.call_count == 3
    assert freeze.read_bytes() == frozen_bytes


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
