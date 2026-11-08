"""Exercise a running local API with an existing development image, without ground truth."""

import argparse
import csv
import hashlib
import io
import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image


def request(
    url: str, data: bytes | None = None, content_type: str | None = None
) -> tuple[int, bytes]:
    headers = {"Content-Type": content_type} if content_type else {}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=data, headers=headers), timeout=1800
        ) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def upload(base: str, image: bytes, name: str, *, compare: bool = False) -> tuple[int, bytes]:
    boundary = "aem_" + uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    ).encode() + image
    body += (
        f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="compare_r050"\r\n\r\n'
        f"{'true' if compare else 'false'}\r\n--{boundary}--\r\n"
    ).encode()
    return request(base + "/predict", body, f"multipart/form-data; boundary={boundary}")


def smoke(base: str, image: Path) -> dict[str, Any]:
    base = base.rstrip("/")
    status, raw = request(base + "/health")
    assert status == 200 and json.loads(raw)["ready"], "API is not ready."
    status, raw = request(base + "/openapi.json")
    assert status == 200 and "/predict" in json.loads(raw)["paths"]
    assert upload(base, b"invalid synthetic content", "invalid.png")[0] == 400
    image_bytes = image.read_bytes()
    status, raw = upload(base, image_bytes, image.name, compare=True)
    result = json.loads(raw)
    assert status == 200 and result["status"] in {"completed", "completed_with_issues"}, result
    assert result["conditions"] == ["A_r100", "A_r050"]
    assert result["input"]["original_sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert sum(animal["accepted"] for animal in result["animals"]) >= 2
    assert any(
        row["measurement_type"] == "binocular" and row["status"] == "candidate"
        for row in result["measurements"]
    )
    assert any(
        row["measurement_type"] == "right_to_right" and row["status"] == "candidate"
        for row in result["measurements"]
    )
    assert all(row["qualification"] == "unverified" for row in result["measurements"])
    status, saved = request(base + "/runs/" + result["run_id"])
    assert status == 200 and json.loads(saved) == result
    status, csv_bytes = request(base + f"/runs/{result['run_id']}/artifacts/measurements_csv")
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    assert status == 200 and len(rows) == len(result["measurements"])
    for saved_row, json_row in zip(rows, result["measurements"], strict=True):
        assert saved_row["condition_id"] == json_row["condition_id"]
        assert saved_row["measurement_type"] == json_row["measurement_type"]
        assert saved_row["status"] == json_row["status"]
        assert saved_row["qualification"] == json_row["qualification"]
        if json_row["distance_px"] is None:
            assert saved_row["distance_px"] == ""
        else:
            assert float(saved_row["distance_px"]) == json_row["distance_px"]
    png_key = next(key for key, value in result["artifacts"].items() if value.endswith(".png"))
    status, png = request(base + f"/runs/{result['run_id']}/artifacts/{png_key}")
    assert status == 200 and png.startswith(b"\x89PNG\r\n\x1a\n")
    white = io.BytesIO()
    Image.new("RGB", (640, 480), "white").save(white, format="PNG")
    status, raw = upload(base, white.getvalue(), "synthetic_no_animals.png")
    no_animals = json.loads(raw)
    assert status == 200 and no_animals["status"] == "no_animals", no_animals
    assert not no_animals["measurements"]
    return {
        "status": "passed",
        "run_id": result["run_id"],
        "synthetic_no_animals_run_id": no_animals["run_id"],
        "input_sha256": result["input"]["original_sha256"],
        "eyes": len(result["eyes"]),
        "candidate_distance_rows": len(result["measurements"]),
        "image_role": "existing_development_image_engineering_smoke",
        "independent_accuracy_evaluated": False,
        "checks": [
            "health",
            "openapi",
            "invalid_upload",
            "two_animals",
            "both_distance_types",
            "candidate_semantics",
            "saved_result",
            "csv_download",
            "png_download",
            "synthetic_no_animals",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = smoke(args.base_url, args.image)
    except (OSError, ValueError, AssertionError, KeyError, StopIteration) as error:
        result = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "image_role": "existing_development_image_engineering_smoke",
            "independent_accuracy_evaluated": False,
        }
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
