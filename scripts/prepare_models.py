"""Verify approved model files; download missing files only with explicit --download."""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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


def prepare(directory: Path, *, download: bool = False) -> list[dict[str, Any]]:
    """Never replace an existing checkpoint; checksums pin observed research artifacts."""
    rows = []
    for spec in MODELS:
        path = directory / spec["file"]
        if not path.exists() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            if temporary.exists():
                raise FileExistsError(f"Unfinished download exists: {temporary}")
            try:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "AEM/0.1"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    with temporary.open("xb") as output:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(chunk)
                if sha256(temporary) != spec["sha256"]:
                    raise ValueError(f"Downloaded checkpoint differs from approved bytes: {path}")
                temporary.rename(path)
            finally:
                temporary.unlink(missing_ok=True)
        actual = sha256(path) if path.is_file() else None
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
