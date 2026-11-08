"""All fixtures are invented geometry for testing, never COCO observations."""

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def isolate_application_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fixtures select their own workspace/config, independent of Docker or host defaults."""
    for key in tuple(os.environ):
        if key.startswith("AEM_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Tests must not access the network or download data.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def synthetic_coco() -> dict[str, Any]:
    def annotation(
        key: int,
        image: int,
        category: int,
        crowd: int = 0,
        bbox: tuple[int, int, int, int] = (1, 2, 10, 20),
        area: int = 90,
    ) -> dict[str, Any]:
        x, y, width, _ = bbox
        mask_height = area / width
        return {
            "id": key,
            "image_id": image,
            "category_id": category,
            "iscrowd": crowd,
            "bbox": list(bbox),
            "area": area,
            "segmentation": [[x, y, x + width, y, x + width, y + mask_height, x, y + mask_height]],
        }

    return {
        "info": {"description": "SYNTHETIC TEST GEOMETRY; NOT COCO DATA OR RESEARCH RESULTS"},
        "categories": [
            {"id": 100 + 7 * index, "name": f"test_species_{index}", "supercategory": "animal"}
            for index in range(10)
        ]
        + [{"id": 9999, "name": "test_nonanimal", "supercategory": "other"}],
        "images": [
            {"id": image, "file_name": f"{image:012d}.png", "width": 100, "height": 80}
            for image in (10, 20, 30, 40, 50, 60, 70)
        ],
        "annotations": [
            annotation(1001, 10, 100),
            annotation(1002, 10, 100, bbox=(40, 20, 20, 20), area=300),
            annotation(2001, 20, 100),
            annotation(2002, 20, 107),
            annotation(3001, 30, 107),
            annotation(3002, 30, 107, crowd=1),
            annotation(4001, 40, 114, crowd=1),
            annotation(4002, 40, 114, crowd=1),
            annotation(5001, 50, 100),
            annotation(5002, 50, 100),
            annotation(5003, 50, 107),
            annotation(6001, 60, 100),
            annotation(6002, 60, 9999),
        ],
    }


@pytest.fixture
def local_config(tmp_path: Path, synthetic_coco: dict[str, Any]) -> Path:
    (tmp_path / "annotations.json").write_text(json.dumps(synthetic_coco), encoding="utf-8")
    config = tmp_path / "phase1.toml"
    config.write_text(
        '[dataset]\nname = "SYNTHETIC TEST ONLY"\nannotations = "annotations.json"\n'
        'images = "images"\nexpected_animal_categories = 10\n'
        '[output]\ndirectory = "result"\n',
        encoding="utf-8",
    )
    return config
