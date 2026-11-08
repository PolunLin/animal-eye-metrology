"""Checkpoint preparation is explicit and never overwrites mismatched existing bytes."""

import hashlib
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest


def script():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_models.py"
    spec = importlib.util.spec_from_file_location("aem_prepare_models", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verification_is_offline_and_distinguishes_missing_mismatch_and_ready(
    tmp_path, monkeypatch
):
    module = script()
    expected = hashlib.sha256(b"approved synthetic bytes").hexdigest()
    monkeypatch.setattr(
        module,
        "MODELS",
        tuple(
            {"file": name, "url": "https://invalid.test/model", "sha256": expected}
            for name in ("ready.pt", "wrong.pt", "missing.pt")
        ),
    )
    (tmp_path / "ready.pt").write_bytes(b"approved synthetic bytes")
    (tmp_path / "wrong.pt").write_bytes(b"different bytes")
    rows = module.prepare(tmp_path)
    assert [row["status"] for row in rows] == ["ready", "mismatch", "missing"]
    # Even explicit download mode refuses to overwrite existing mismatched bytes.
    monkeypatch.setattr(module, "MODELS", module.MODELS[:2])
    assert module.prepare(tmp_path, download=True)[1]["status"] == "mismatch"
    assert (tmp_path / "wrong.pt").read_bytes() == b"different bytes"


def test_download_failure_removes_partial_file_without_creating_checkpoint(tmp_path, monkeypatch):
    module = script()
    monkeypatch.setattr(
        module,
        "MODELS",
        (
            {
                "file": "model.pt",
                "url": "https://invalid.test/model",
                "sha256": "0" * 64,
            },
        ),
    )

    def fail(*args, **kwargs):
        raise OSError("synthetic network failure")

    monkeypatch.setattr(module.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="synthetic network failure"):
        module.prepare(tmp_path, download=True)
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def downloadable_model(monkeypatch):
    module = script()
    content = b"approved synthetic checkpoint bytes"
    model = {
        "file": "model.pt",
        "url": "https://invalid.test/model.pt",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(module, "MODELS", (model,))
    delays = []
    monkeypatch.setattr(module.time, "sleep", delays.append)
    return module, model, content, delays


def test_504_then_approved_bytes_retries_and_logs_model_identity(
    tmp_path, monkeypatch, capsys, downloadable_model
):
    module, model, content, delays = downloadable_model
    requests = []

    def respond(request, **kwargs):
        requests.append(request.full_url)
        if len(requests) == 1:
            raise urllib.error.HTTPError(model["url"], 504, "Synthetic gateway timeout", {}, None)
        return io.BytesIO(content)

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    rows = module.prepare(tmp_path, download=True)
    assert rows[0]["status"] == "ready"
    assert (tmp_path / model["file"]).read_bytes() == content
    assert requests == [model["url"], model["url"]]
    assert delays == [1]
    assert list(tmp_path.glob("*.part")) == []
    captured = capsys.readouterr()
    assert captured.out == ""
    events = [json.loads(line) for line in captured.err.splitlines()]
    retry = next(event for event in events if event["event"] == "model_download_retry")
    assert (retry["file"], retry["url"], retry["attempt"], retry["http_status"]) == (
        model["file"],
        model["url"],
        1,
        504,
    )


def test_connection_failure_midstream_is_bounded_and_removes_each_owned_partial(
    tmp_path, monkeypatch, downloadable_model
):
    module, model, _, delays = downloadable_model
    attempts = []
    unowned = tmp_path / "model.pt.part"
    unowned.write_bytes(b"unrelated interrupted download")

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            data = super().read(size)
            if not data:
                raise ConnectionResetError("Synthetic connection reset after partial bytes")
            return data

    def respond(*args, **kwargs):
        # The previous attempt's temporary file was removed before retrying.
        assert list(tmp_path.glob(".model.pt.*.part")) == []
        attempts.append(1)
        return InterruptedResponse(b"partial synthetic bytes")

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    with pytest.raises(OSError, match=r"model.pt.*https://invalid.test/model.pt.*3/3"):
        module.prepare(tmp_path, download=True)
    assert len(attempts) == 3 and delays == [1, 2]
    assert not (tmp_path / model["file"]).exists()
    assert list(tmp_path.iterdir()) == [unowned]
    assert unowned.read_bytes() == b"unrelated interrupted download"


@pytest.mark.parametrize("failure_kind", ["404", "integrity"])
def test_permanent_http_or_integrity_failure_never_retries(
    tmp_path, monkeypatch, downloadable_model, failure_kind
):
    module, model, _, delays = downloadable_model
    requests = []

    def respond(*args, **kwargs):
        requests.append(1)
        if failure_kind == "404":
            raise urllib.error.HTTPError(model["url"], 404, "Synthetic not found", {}, None)
        return io.BytesIO(b"unapproved bytes")

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    expected_error = OSError if failure_kind == "404" else ValueError
    with pytest.raises(expected_error, match="model.pt.*https://invalid.test/model.pt"):
        module.prepare(tmp_path, download=True)
    assert requests == [1] and delays == []
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_created_during_download_is_never_overwritten(
    tmp_path, monkeypatch, downloadable_model
):
    module, model, content, delays = downloadable_model
    target = tmp_path / model["file"]

    def respond(*args, **kwargs):
        target.write_bytes(b"concurrent writer's existing bytes")
        return io.BytesIO(content)

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    with pytest.raises(OSError, match="FileExistsError"):
        module.prepare(tmp_path, download=True)
    assert target.read_bytes() == b"concurrent writer's existing bytes"
    assert delays == [] and list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
def test_other_transient_http_codes_retry_before_success(
    tmp_path, monkeypatch, downloadable_model, status
):
    module, model, content, delays = downloadable_model
    requests = []

    def respond(*args, **kwargs):
        requests.append(1)
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                model["url"], status, "Synthetic temporary failure", {}, None
            )
        return io.BytesIO(content)

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    assert module.prepare(tmp_path, download=True)[0]["status"] == "ready"
    assert len(requests) == 2 and delays == [1]


@pytest.mark.parametrize(
    "failure", [urllib.error.URLError("Synthetic DNS failure"), TimeoutError("Synthetic timeout")]
)
def test_url_and_timeout_failures_are_bounded(tmp_path, monkeypatch, downloadable_model, failure):
    module, model, _, delays = downloadable_model
    requests = []

    def respond(*args, **kwargs):
        requests.append(1)
        raise failure

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    with pytest.raises(OSError, match="model.pt.*https://invalid.test/model.pt.*3/3"):
        module.prepare(tmp_path, download=True)
    assert len(requests) == 3 and delays == [1, 2]
    assert list(tmp_path.iterdir()) == []


def test_yolo_transient_error_uses_official_asset_api_without_changing_spec(
    tmp_path, monkeypatch, capsys, downloadable_model
):
    module, model, content, delays = downloadable_model
    model = model | {
        "file": "yolo26s-seg.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
    }
    monkeypatch.setattr(module, "MODELS", (model,))
    fallback = module.OFFICIAL_TRANSPORT_FALLBACKS[model["file"]]
    requests = []

    def respond(request, **kwargs):
        requests.append((request.full_url, request.get_header("Accept")))
        if len(requests) == 1:
            raise urllib.error.HTTPError(model["url"], 504, "Synthetic gateway timeout", {}, None)
        return io.BytesIO(content)

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    rows = module.prepare(tmp_path, download=True)
    assert requests == [(model["url"], None), (fallback, "application/octet-stream")]
    assert rows[0]["url"] == model["url"] and rows[0]["sha256"] == model["sha256"]
    assert rows[0]["actual_sha256"] == model["sha256"] and rows[0]["status"] == "ready"
    assert (tmp_path / model["file"]).read_bytes() == content and delays == [1]
    captured = capsys.readouterr()
    assert captured.out == ""
    events = [json.loads(line) for line in captured.err.splitlines()]
    ready = next(event for event in events if event["event"] == "model_download_ready")
    assert ready["url"] == model["url"] and ready["request_url"] == fallback


@pytest.mark.parametrize("failure_kind", ["transient", "404", "integrity"])
def test_yolo_api_fallback_stays_bounded_and_cannot_bypass_permanent_failure(
    tmp_path, monkeypatch, downloadable_model, failure_kind
):
    module, model, _, delays = downloadable_model
    model = model | {"file": "yolo26s-seg.pt"}
    monkeypatch.setattr(module, "MODELS", (model,))
    fallback = module.OFFICIAL_TRANSPORT_FALLBACKS[model["file"]]
    requests = []

    def respond(request, **kwargs):
        requests.append(request.full_url)
        if failure_kind == "integrity":
            return io.BytesIO(b"unapproved bytes")
        code = 504 if failure_kind == "transient" else 404
        raise urllib.error.HTTPError(request.full_url, code, "Synthetic failure", {}, None)

    monkeypatch.setattr(module.urllib.request, "urlopen", respond)
    error_class = ValueError if failure_kind == "integrity" else OSError
    with pytest.raises(error_class) as caught:
        module.prepare(tmp_path, download=True)
    if failure_kind == "transient":
        assert requests == [model["url"], fallback, fallback] and delays == [1, 2]
        assert fallback in str(caught.value) and "3/3" in str(caught.value)
    else:
        assert requests == [model["url"]] and delays == []
    assert model["file"] in str(caught.value) and model["url"] in str(caught.value)
    assert list(tmp_path.iterdir()) == []
