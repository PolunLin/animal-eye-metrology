"""The clean runtime changes deployment paths, not the approved model or geometry policy."""

import tomllib
from pathlib import Path

import pytest

from animal_eye_metrology.inference_pipeline import load_config
from animal_eye_metrology.sam_runtime import settings

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
POLICY_FIELDS = (
    "detector_checkpoint_sha256",
    "pose_checkpoint_sha256",
    "sam_checkpoint_sha256",
    "mmpose_commit",
    "device",
    "seed",
    "threads",
    "supported_categories",
    "animal_score_threshold",
    "image_size",
    "confidence_floor",
    "nms_iou",
    "max_detections",
    "max_upload_bytes",
    "max_image_pixels",
    "stage_timeout_seconds",
)


@pytest.mark.parametrize("name", ["inference.runtime.toml", "inference.docker.toml"])
def test_clean_runtime_preserves_model_hashes_and_approved_execution_policy(name: str) -> None:
    original = load_config(CONFIGS / "inference.toml")
    clean = load_config(CONFIGS / name)
    assert {key: clean[key] for key in POLICY_FIELDS} == {
        key: original[key] for key in POLICY_FIELDS
    }
    for key in ("detector_checkpoint", "pose_checkpoint", "sam_checkpoint", "pose_source_manifest"):
        assert clean[key] == original[key]
    # Only result storage belongs in outputs; runtime code cannot depend on old experiment runs.
    for key in ("python", "pose_python", "detector_source", "sam_runtime"):
        assert ROOT / "outputs" not in clean[key].parents
    assert clean["python"] != clean["pose_python"]
    sam = settings(clean["sam_runtime"])
    assert sam["runtime_kind"] == "application"
    assert (
        not {"output_directory", "proposal_csv", "prompts_directory", "phase1_config"} & sam.keys()
    )


def test_clean_runtime_sam_uses_the_same_predictor_commit_and_checkpoint_as_research() -> None:
    with (CONFIGS / "eye_segmentation_poc.toml").open("rb") as handle:
        research = tomllib.load(handle)
    clean = settings(CONFIGS / "sam2_runtime.toml")
    assert clean["models"] == {"sam2_1_tiny": research["models"]["sam2_1_tiny"]}
    assert (clean["device"], clean["threads"], clean["seed"]) == ("cpu", 4, 42)
    assert clean["weights_directory"] == (ROOT / "weights/eye_poc").resolve()


def test_local_and_container_paths_keep_both_python_environments_separate() -> None:
    local = load_config(CONFIGS / "inference.runtime.toml")
    container = load_config(CONFIGS / "inference.docker.toml")
    assert local["python"] == ROOT / ".runtime/app/bin/python"
    assert local["pose_python"] == ROOT / ".runtime/pose/bin/python"
    assert container["python"] == Path("/opt/venvs/app/bin/python")
    assert container["pose_python"] == Path("/opt/venvs/pose/bin/python")


def test_api_defaults_to_clean_runtime_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from animal_eye_metrology.api import create_app

    monkeypatch.delenv("AEM_INFERENCE_CONFIG", raising=False)
    assert create_app().state.config_path == CONFIGS / "inference.runtime.toml"
    monkeypatch.setenv("AEM_INFERENCE_CONFIG", str(CONFIGS / "inference.docker.toml"))
    assert create_app().state.config_path == CONFIGS / "inference.docker.toml"
