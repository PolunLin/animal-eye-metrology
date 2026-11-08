"""Verify approved model files; download missing files only with explicit --download."""

import argparse
import hashlib
import http.client
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_ATTEMPTS = 3
RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
OFFICIAL_TRANSPORT_FALLBACKS = {
    "yolo26s-seg.pt": "https://api.github.com/repos/ultralytics/assets/releases/assets/340096333",
}
MODELS = (
    {
        "file": "yolo26s-seg.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
        "sha256": "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4",
    },
    {
        "file": "eye_pose/hrnet_w32_ap10k_256x256-18aac840_20211029.pth",
        "url": "https://download.openmmlab.com/mmpose/animal/hrnet/"
        "hrnet_w32_ap10k_256x256-18aac840_20211029.pth",
        "sha256": "18aac840eee49f190e1443d93c6c23784920d2fd90813a42d35c918ec23a36e9",
    },
    {
        "file": "eye_poc/sam2.1_hiera_tiny.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sha256": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    },
)


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _event(event: str, spec: dict[str, str], attempt: int, **details: Any) -> None:
    """Keep progress off stdout, whose successful rows remain machine-readable."""
    print(
        json.dumps(
            {
                "event": event,
                "file": spec["file"],
                "url": spec["url"],
                "attempt": attempt,
                "max_attempts": MAX_ATTEMPTS,
                **details,
            }
        ),
        file=sys.stderr,
        flush=True,
    )


def _retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_HTTP_CODES
    return isinstance(
        error,
        (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead),
    )


def _download(path: Path, spec: dict[str, str]) -> None:
    """Retry transient transport failures; never replace an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        temporary: Path | None = None
        failure: Exception | None = None
        request_url = (
            OFFICIAL_TRANSPORT_FALLBACKS.get(spec["file"], spec["url"])
            if attempt > 1
            else spec["url"]
        )
        _event("model_download_started", spec, attempt, request_url=request_url)
        try:
            headers = {"User-Agent": "AEM/0.1"}
            if request_url != spec["url"]:
                headers["Accept"] = "application/octet-stream"
            request = urllib.request.Request(request_url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".part",
                    delete=False,
                ) as output:
                    temporary = Path(output.name)
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            if sha256(temporary) != spec["sha256"]:
                raise ValueError("Downloaded checkpoint differs from approved bytes.")
            # The temporary and final file share a filesystem. link() fails if
            # another process created the checkpoint while this download ran.
            os.link(temporary, path)
            _event("model_download_ready", spec, attempt, request_url=request_url)
            return
        except (OSError, ValueError, http.client.IncompleteRead) as error:
            failure = error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        assert failure is not None
        detail = {
            "error_type": type(failure).__name__,
            "error": str(failure),
            "request_url": request_url,
        }
        if isinstance(failure, urllib.error.HTTPError):
            detail["http_status"] = failure.code
        if _retryable(failure) and attempt < MAX_ATTEMPTS:
            delay = 2 ** (attempt - 1)
            _event("model_download_retry", spec, attempt, delay_seconds=delay, **detail)
            time.sleep(delay)
            continue
        _event("model_download_failed", spec, attempt, **detail)
        message = (
            f"Model {spec['file']} from {spec['url']} (requested {request_url}) failed on attempt "
            f"{attempt}/{MAX_ATTEMPTS}: {type(failure).__name__}: {failure}"
        )
        if isinstance(failure, ValueError):
            raise ValueError(message) from failure
        raise OSError(message) from failure


def prepare(directory: Path, *, download: bool = False) -> list[dict[str, Any]]:
    """Never replace an existing checkpoint; checksums pin observed research artifacts."""
    rows = []
    for spec in MODELS:
        _event("model_check_started", spec, 0)
        path = directory / spec["file"]
        if not path.exists() and download:
            _download(path, spec)
        actual = sha256(path) if path.is_file() else None
        if actual is not None and actual != spec["sha256"]:
            _event("model_verification_failed", spec, 0, actual_sha256=actual)
        rows.append(
            dict(spec)
            | {
                "status": "ready"
                if actual == spec["sha256"]
                else "mismatch"
                if actual
                else "missing",
                "actual_sha256": actual,
                "reference": "observed_frozen_research_checkpoint_not_independent_upstream_digest",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "weights")
    parser.add_argument(
        "--download", action="store_true", help="Download only missing model files."
    )
    args = parser.parse_args()
    try:
        rows = prepare(args.directory, download=args.download)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(rows, indent=2))
    return 0 if all(row["status"] == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
