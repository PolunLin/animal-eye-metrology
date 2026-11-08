#!/usr/bin/env python3
"""Check shareable delivery files and the recorded CSV without models or research outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

PUBLIC_DOCUMENTS = (
    "docs/index.md",
    "docs/setup.md",
    "docs/docker.md",
    "docs/api.md",
    "docs/architecture.md",
    "docs/methodology.md",
    "docs/notebook_demo.md",
)

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".env.example",
    ".dockerignore",
    "Dockerfile",
    "compose.yaml",
    ".github/workflows/ci.yml",
    "configs/inference.runtime.toml",
    "configs/inference.docker.toml",
    "configs/sam2_runtime.toml",
    "configs/eye_pose_source_manifest.json",
    "requirements/app-linux-amd64.txt",
    "requirements/app-macos-arm64.txt",
    "requirements/pose-linux-amd64.txt",
    "requirements/pose-macos-arm64.txt",
    "scripts/bootstrap.py",
    "scripts/prepare_models.py",
    "scripts/check_runtime.py",
    "scripts/smoke_api.py",
    "src/animal_eye_metrology/api.py",
    "src/animal_eye_metrology/inference_pipeline.py",
    "src/animal_eye_metrology/eye_pose_worker.py",
    "src/animal_eye_metrology/sam_runtime.py",
    "app/streamlit_app.py",
    "app/assets/eye_canvas.js",
    "app/assets/eye_prompt_canvas.js",
    *PUBLIC_DOCUMENTS,
    "examples/measurements.csv",
    "examples/input_manifest.csv",
    "examples/README.md",
)


def check_documentation_links(root: Path) -> None:
    """Check published documentation links without relying on local research archives."""
    broken = []
    for relative in ("README.md", *PUBLIC_DOCUMENTS, "examples/README.md"):
        document = root / relative
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^\s)]+)\)", text):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path = (document.parent / unquote(parsed.path)).resolve()
            if not path.is_relative_to(root.resolve()) or not path.exists():
                broken.append(f"{relative}: {target}")
                continue
            project_path = path.relative_to(root).as_posix()
            if project_path.startswith("docs/") and project_path not in PUBLIC_DOCUMENTS:
                broken.append(f"{relative}: {target} (local research archive)")
    if broken:
        raise ValueError(f"Unpublished, missing or out-of-repository documentation links: {broken}")


def check_recorded_csv(root: Path) -> None:
    """Validate the recorded bytes and distance arithmetic, never model accuracy."""
    sample = root / "examples/measurements.csv"
    provenance = (root / "examples/README.md").read_text(encoding="utf-8")
    match = re.search(r"CSV SHA-256\s*\|\s*`([0-9a-f]{64})`", provenance)
    if match is None or hashlib.sha256(sample.read_bytes()).hexdigest() != match[1]:
        raise ValueError("Recorded CSV SHA-256 does not match its provenance document.")
    with sample.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The example CSV must contain data rows, not only a header.")
    for row in rows:
        coordinates = [float(row[key]) for key in ("cx_a_px", "cy_a_px", "cx_b_px", "cy_b_px")]
        distance = float(row["distance_px"])
        if not all(math.isfinite(number) for number in [*coordinates, distance]):
            raise ValueError("The recorded CSV contains nonfinite geometry.")
        expected = math.hypot(coordinates[0] - coordinates[2], coordinates[1] - coordinates[3])
        if not math.isclose(distance, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("A recorded distance does not match its saved endpoints.")
        if (row["units"], row["status"], row["qualification"]) != ("px", "candidate", "unverified"):
            raise ValueError("The recorded example must preserve pixel candidate semantics.")
    if {row["measurement_type"] for row in rows} != {"binocular", "right_to_right"}:
        raise ValueError("The example must demonstrate both requested measurement types.")


def check_delivery(root: Path) -> dict[str, Any]:
    """Return independent, structured checks without installing or downloading anything."""
    root = root.resolve()
    checks: list[dict[str, Any]] = []
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    checks.append({"name": "required_files", "passed": not missing, "detail": missing})
    for name, operation in (
        ("documentation_local_links", lambda: check_documentation_links(root)),
        ("recorded_csv", lambda: check_recorded_csv(root)),
    ):
        try:
            operation()
            checks.append({"name": name, "passed": True, "detail": "Verified."})
        except (OSError, ValueError, KeyError, csv.Error) as error:
            checks.append({"name": name, "passed": False, "detail": str(error)})
    try:
        diagram = (root / "docs/architecture.md").read_text(encoding="utf-8")
        present = "```mermaid" in diagram
    except OSError:
        present = False
    checks.append({"name": "architecture_diagram", "passed": present, "detail": "Mermaid source."})
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "model_inference_performed": False,
        "accuracy_evaluated": False,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = check_delivery(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
