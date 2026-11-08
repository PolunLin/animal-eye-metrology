"""HTTP contracts use synthetic images and saved fake results; no model or network calls."""

import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from animal_eye_metrology import api

RUN_ID = "20261108T010203Z_012345abcdef"
CONFIG = Path(__file__).resolve().parents[1] / "configs/inference.toml"


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, TestClient, dict]:
    cfg = api.pipeline.load_config(CONFIG)
    cfg["output_directory"] = tmp_path / "runs"
    cfg["max_upload_bytes"] = 1024
    cfg["max_image_pixels"] = 100
    monkeypatch.setattr(api.pipeline, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        api.pipeline,
        "preflight",
        lambda config: [
            {"component": "fake_runtime", "status": "ready", "detail": "Synthetic test"}
        ],
    )
    app = api.create_app(tmp_path / "inference.toml")
    calls: list = []

    def runner(data: bytes, filename: str, config: Path, conditions: tuple[str, ...]) -> Path:
        calls.append((data, filename, config, conditions))
        directory = cfg["output_directory"] / RUN_ID
        directory.mkdir(parents=True, exist_ok=True)
        result = {
            "run_id": RUN_ID,
            "status": "completed",
            "conditions": list(conditions),
            "measurements": [
                {"distance_px": 5.0, "status": "candidate", "qualification": "unverified"}
            ],
            "artifacts": {"measurements_csv": "measurements.csv"},
        }
        (directory / "result.json").write_text(json.dumps(result))
        (directory / "measurements.csv").write_text("distance_px,status\n5.0,candidate\n")
        return directory

    app.state.runner = runner
    cfg["test_calls"] = calls
    return app, TestClient(app), cfg


def png(size: tuple[int, int] = (4, 3)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_openapi_do_not_run_models(api_client) -> None:
    _, client, cfg = api_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["authentication"] == "none_local_demo"
    spec = client.get("/openapi.json").json()
    assert {"/health", "/predict", "/runs/{run_id}"}.issubset(spec["paths"])
    assert client.get("/docs").status_code == 200
    assert cfg["test_calls"] == []


def test_predict_download_and_read_preserve_candidate_semantics(api_client) -> None:
    _, client, cfg = api_client
    response = client.post(
        "/predict", files={"file": ("test.png", png(), "image/png")}, data={"compare_r050": "true"}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["conditions"] == ["A_r100", "A_r050"]
    assert result["measurements"][0]["qualification"] == "unverified"
    assert cfg["test_calls"][0][3] == ("A_r100", "A_r050")
    assert client.get(f"/runs/{RUN_ID}").json() == result
    artifact = client.get(f"/runs/{RUN_ID}/artifacts/measurements_csv")
    assert artifact.status_code == 200
    assert artifact.content == b"distance_px,status\n5.0,candidate\n"
    assert len(cfg["test_calls"]) == 1


@pytest.mark.parametrize("content,status", [(b"fake JPEG", 400), (b"x" * 1025, 413), (b"", 400)])
def test_invalid_or_large_images_do_not_run_models(api_client, content: bytes, status: int) -> None:
    _, client, cfg = api_client
    response = client.post("/predict", files={"file": ("test.jpg", content, "image/jpeg")})
    assert response.status_code == status
    assert cfg["test_calls"] == []
    assert not cfg["output_directory"].exists()


def test_dimensions_are_checked_from_content_and_default_condition_is_fixed(api_client) -> None:
    _, client, cfg = api_client
    response = client.post("/predict", files={"file": ("test.png", png((11, 10)), "image/png")})
    assert response.status_code == 400
    assert cfg["test_calls"] == []
    response = client.post("/predict", files={"file": ("any.name", png(), "text/plain")})
    assert response.status_code == 200
    assert cfg["test_calls"][0][3] == ("A_r100",)


def test_missing_runtime_reports_not_ready_without_model_call(api_client, monkeypatch) -> None:
    _, client, cfg = api_client
    monkeypatch.setattr(
        api.pipeline,
        "preflight",
        lambda config: [{"component": "weights", "status": "missing", "detail": "Unavailable"}],
    )
    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["detail"]["ready"] is False
    assert client.post("/predict", files={"file": ("test.png", png())}).status_code == 503
    assert cfg["test_calls"] == []


def test_busy_request_is_rejected_and_failed_runner_releases_lock(api_client) -> None:
    app, client, cfg = api_client
    app.state.inference_lock.acquire()
    try:
        assert client.post("/predict", files={"file": ("test.png", png())}).status_code == 409
    finally:
        app.state.inference_lock.release()
    assert cfg["test_calls"] == []

    def failure(*args: Any) -> Path:
        raise RuntimeError("Synthetic failure")

    app.state.runner = failure
    assert client.post("/predict", files={"file": ("test.png", png())}).status_code == 500
    assert not app.state.inference_lock.locked()


def test_failed_model_result_is_returned_without_promoting_it_to_success(api_client) -> None:
    app, client, cfg = api_client
    original = app.state.runner

    def failed_result(*args: Any) -> Path:
        directory = original(*args)
        result = json.loads((directory / "result.json").read_text())
        result.update(
            status="failed", measurements=[{"distance_px": None, "status": "unavailable"}]
        )
        (directory / "result.json").write_text(json.dumps(result))
        return directory

    app.state.runner = failed_result
    response = client.post("/predict", files={"file": ("test.png", png())})
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["measurements"][0]["distance_px"] is None
    assert len(cfg["test_calls"]) == 1


def test_shared_pipeline_lock_contention_is_http_409_and_local_lock_is_released(api_client) -> None:
    app, client, _ = api_client

    def busy(*args: Any) -> Path:
        raise api.InferenceBusyError("Synthetic other-process contention")

    app.state.runner = busy
    response = client.post("/predict", files={"file": ("test.png", png())})
    assert response.status_code == 409
    assert "other-process contention" in response.json()["detail"]
    assert not app.state.inference_lock.locked()


def test_run_ids_and_artifact_paths_cannot_escape_output_root(api_client, tmp_path) -> None:
    _, client, cfg = api_client
    assert client.get("/runs/not-a-run").status_code == 404
    assert client.get(f"/runs/{RUN_ID}").status_code == 404
    client.post("/predict", files={"file": ("test.png", png())})
    assert client.get(f"/runs/{RUN_ID}/artifacts/unknown").status_code == 404
    outside = tmp_path / "outside.json"
    outside.write_text("private")
    directory = cfg["output_directory"] / RUN_ID
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    result["artifacts"].update(escape="../../outside.json", absolute=str(outside), link="link.json")
    result_path.write_text(json.dumps(result))
    (directory / "link.json").symlink_to(outside)
    for key in ("escape", "absolute", "link"):
        assert client.get(f"/runs/{RUN_ID}/artifacts/{key}").status_code == 404
    result_path.unlink()
    result_path.symlink_to(outside)
    assert client.get(f"/runs/{RUN_ID}").status_code == 404


def test_symlinked_run_directory_cannot_escape_root(api_client, tmp_path) -> None:
    _, client, cfg = api_client
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.json").write_text('{"run_id":"outside"}')
    cfg["output_directory"].mkdir()
    (cfg["output_directory"] / RUN_ID).symlink_to(outside, target_is_directory=True)
    assert client.get(f"/runs/{RUN_ID}").status_code == 404


def test_invalid_configuration_and_corrupt_results_are_explicit_errors(
    api_client, monkeypatch
) -> None:
    _, client, cfg = api_client
    client.post("/predict", files={"file": ("test.png", png())})
    (cfg["output_directory"] / RUN_ID / "result.json").write_text("not JSON")
    assert client.get(f"/runs/{RUN_ID}").status_code == 500

    def invalid(path: Path) -> dict:
        raise ValueError("Synthetic invalid config")

    monkeypatch.setattr(api.pipeline, "load_config", invalid)
    assert client.get("/health").status_code == 503
    assert client.post("/predict", files={"file": ("test.png", png())}).status_code == 503
