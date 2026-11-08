"""Synthetic pipeline contracts and failures, without running models or reading GT."""

import csv
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from animal_eye_metrology.inference import inference_pipeline as pipeline
from animal_eye_metrology.inference.inference_capabilities import (
    COCO_ANIMAL_CATEGORIES,
    category_capability,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/inference.runtime.toml"


def image_bytes(image: Image.Image, fmt: str = "PNG", **kwargs: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def animal(identifier: str, *, supported: bool = True, accepted: bool = True) -> dict:
    return {
        "prediction_id": identifier,
        "category": "dog" if supported else "synthetic_unknown_species",
        "accepted": accepted,
        "supported": supported,
        "bbox_xyxy": [10, 10, 90, 80],
        "score": 0.8,
        "mask_path": None,
        "mask_status": "completed",
        "status": "completed" if supported else "unsupported_category",
        "failure_reason": "" if supported else "unsupported_category",
    }


def pose(identifier: str, **updates: Any) -> dict:
    return {
        "prediction_id": identifier,
        "status": "passed",
        "keypoints_xy": [[40, 40], [30, 40]],
        "scores": [0.8, 0.9],
        "geometry": {"input_center": [50, 50], "input_scale": [100, 100]},
        **updates,
    }


def test_config_preserves_isolated_python_symlinks_and_baseline_scope() -> None:
    config = pipeline.load_config(CONFIG)
    assert config["pose_python"].parts[-4:] == (".runtime", "pose", "bin", "python")
    assert config["python"].parts[-4:] == (".runtime", "app", "bin", "python")
    assert config["supported_categories"] == list(COCO_ANIMAL_CATEGORIES)
    assert config["animal_score_threshold"] == 0.5


def test_configuration_rejects_accidental_return_to_three_category_scope(tmp_path: Path) -> None:
    text = CONFIG.read_text()
    prefix = "supported_categories = "
    old_line = next(line for line in text.splitlines() if line.startswith(prefix))
    narrowed = tmp_path / "inference.toml"
    narrowed.write_text(text.replace(old_line, prefix + '["dog", "bear", "giraffe"]'))
    with pytest.raises(ValueError, match="all ten COCO"):
        pipeline.load_config(narrowed)


@pytest.mark.parametrize("stage", ["detection", "localization", "segmentation"])
def test_workers_resolve_package_sources_after_folder_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    cfg = pipeline.load_config(CONFIG)
    directory = tmp_path / "run"
    directory.mkdir()
    output = directory / ("pose/pose_results.json" if stage == "localization" else f"{stage}.json")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def execute(command: list[str], **options: Any) -> pipeline.subprocess.CompletedProcess:
        calls.append((command, options))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"results": []}')
        return pipeline.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", execute)
    result = {"execution": []}
    assert pipeline._run_worker(cfg, directory, stage, {}, result) == {"results": []}
    command, options = calls[0]
    project = CONFIG.parents[1]
    assert Path(options["env"]["PYTHONPATH"].split(os.pathsep)[-1]) == project / "src"
    assert options["cwd"] == project
    if stage == "localization":
        assert command[:2] == [
            str(cfg["pose_python"]),
            str(project / "src/animal_eye_metrology/inference/eye_pose_worker.py"),
        ]
        assert Path(command[1]).is_file()
    else:
        assert command[:3] == [
            str(cfg["python"]),
            "-m",
            "animal_eye_metrology.inference.inference_worker",
        ]


@pytest.mark.parametrize("mode,pixel", [("RGBA", (0, 0, 0, 0)), ("LA", (0, 0))])
def test_normalize_transparency_on_white_preserves_image_size(mode: str, pixel: tuple) -> None:
    source = Image.new(mode, (11, 9), pixel)
    result, suffix = pipeline.normalize_image(
        image_bytes(source), {"max_upload_bytes": 10000, "max_image_pixels": 1000}
    )
    assert suffix == ".png"
    assert result.mode == "RGB" and result.size == (11, 9)
    assert result.getpixel((4, 3)) == (255, 255, 255)


def test_normalize_applies_exif_orientation_and_preserves_lossless_rgb_pixels() -> None:
    source = Image.new("RGB", (2, 3), (20, 50, 90))
    exif = source.getexif()
    exif[274] = 6
    cfg = {"max_upload_bytes": 10000, "max_image_pixels": 1000}
    oriented, suffix = pipeline.normalize_image(image_bytes(source, "JPEG", exif=exif), cfg)
    assert suffix == ".jpg" and oriented.size == (3, 2)
    unchanged, _ = pipeline.normalize_image(image_bytes(source), cfg)
    np.testing.assert_array_equal(np.asarray(unchanged), np.asarray(source))


@pytest.mark.parametrize("problem", ["empty", "invalid", "gif", "pixels", "bytes", "animated"])
def test_invalid_upload_has_no_image_resize_or_format_fallback(problem: str) -> None:
    source = Image.new("RGB", (11, 9))
    data = image_bytes(source)
    cfg = {"max_upload_bytes": 10000, "max_image_pixels": 1000}
    if problem == "empty":
        data = b""
    elif problem == "invalid":
        data = b"invalid PNG content"
    elif problem == "gif":
        data = image_bytes(source, "GIF")
    elif problem == "pixels":
        cfg["max_image_pixels"] = 50
    elif problem == "bytes":
        cfg["max_upload_bytes"] = 10
    else:
        data = image_bytes(
            source, save_all=True, append_images=[Image.new("RGB", source.size, "white")]
        )
    with pytest.raises(ValueError):
        pipeline.normalize_image(data, cfg)


def test_pose_preserves_named_sides_and_baseline_roi_rule() -> None:
    eyes = pipeline._eye_slots([animal("animal_0001")], ("A_r100",))
    cases = pipeline._apply_pose(eyes, [pose("animal_0001")], 100, 100)
    assert len(cases) == 2
    assert cases[0]["eye_side"] == "left" and cases[0]["point_xy"] == [40, 40]
    assert cases[1]["eye_side"] == "right" and cases[1]["point_xy"] == [30, 40]
    assert cases[0]["roi_xyxy"] == [24, 24, 56, 56]
    assert cases[1]["roi_xyxy"] == [14, 24, 46, 56]
    assert {eye["visibility_status"] for eye in eyes} == {"unverified"}
    assert {eye["side_assignment_status"] for eye in eyes} == {"unverified"}


@pytest.mark.parametrize("outputs", [[], [pose("animal_0001", scores=[0.8, 0])]])
def test_missing_pose_or_one_invalid_eye_blocks_both_sam_slots(outputs: list) -> None:
    eyes = pipeline._eye_slots([animal("animal_0001"), animal("animal_0002")], ("A_r100",))
    assert pipeline._apply_pose(eyes, outputs, 100, 100) == []
    assert len(eyes) == 4
    assert {eye["localization_status"] for eye in eyes} == {"failed"}
    assert {eye["segmentation_status"] for eye in eyes} == {"skipped"}
    assert all(eye["roi_xyxy"] is None for eye in eyes)


def test_unsupported_keeps_eye_slots_while_below_threshold_never_enters_pose() -> None:
    eyes = pipeline._eye_slots(
        [animal("animal_0001", supported=False), animal("animal_0002", accepted=False)],
        ("A_r100",),
    )
    assert len(eyes) == 2
    assert {eye["prediction_id"] for eye in eyes} == {"animal_0001"}
    assert pipeline._apply_pose(eyes, [], 100, 100) == []
    assert {eye["localization_status"] for eye in eyes} == {"unsupported_category"}


def test_saved_masks_have_original_grid_centroids_and_missing_output_stays_null(
    tmp_path: Path,
) -> None:
    eyes = pipeline._eye_slots([animal("animal_0001"), animal("animal_0002")], ("A_r100",))
    pipeline._apply_pose(eyes, [pose("animal_0001"), pose("animal_0002")], 100, 100)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[41, 42] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    Image.new("L", (100, 100)).save(tmp_path / "empty.png")
    first = {key: eyes[0][key] for key in ("prediction_id", "eye_side", "condition_id")}
    second = {key: eyes[1][key] for key in ("prediction_id", "eye_side", "condition_id")}
    third = {key: eyes[2][key] for key in ("prediction_id", "eye_side", "condition_id")}
    pipeline._apply_segmentation(
        eyes,
        [
            first | {"status": "completed", "mask_path": "mask.png"},
            second | {"status": "empty_mask", "mask_path": "empty.png"},
            third | {"status": "failed", "failure_reason": "synthetic_error"},
        ],
        tmp_path,
        100,
        100,
    )
    assert eyes[0]["centroid_x_px"] == 42.5 and eyes[0]["centroid_y_px"] == 41.5
    assert eyes[0]["mask_area_px"] == 1
    assert eyes[1]["segmentation_status"] == "empty_mask" and eyes[1]["mask_area_px"] == 0
    assert all(eye["centroid_x_px"] is None for eye in eyes[1:])
    assert eyes[2]["failure_reason"] == "synthetic_error"
    assert eyes[3]["failure_reason"] == "missing_segmentation_output"


@pytest.mark.parametrize("problem", ["wrong_size", "nonbinary", "escape"])
def test_unusable_mask_is_failed_not_a_measurement_endpoint(tmp_path: Path, problem: str) -> None:
    eyes = pipeline._eye_slots([animal("animal_0001")], ("A_r100",))
    pipeline._apply_pose(eyes, [pose("animal_0001")], 100, 100)
    if problem == "wrong_size":
        Image.new("L", (20, 20)).save(tmp_path / "mask.png")
    else:
        Image.new("L", (100, 100), 128).save(tmp_path / "mask.png")
    first = {key: eyes[0][key] for key in ("prediction_id", "eye_side", "condition_id")}
    pipeline._apply_segmentation(
        eyes,
        [
            first
            | {
                "status": "completed",
                "mask_path": "../outside.png" if problem == "escape" else "mask.png",
            }
        ],
        tmp_path,
        100,
        100,
    )
    assert eyes[0]["segmentation_status"] == "failed"
    assert "invalid_mask" in eyes[0]["failure_reason"]
    assert eyes[0]["centroid_x_px"] is None and eyes[0]["mask_path"] is None


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = pipeline.load_config(CONFIG)
    cfg["output_directory"] = tmp_path / "runs"
    cfg["pose_source_manifest"] = tmp_path / "manifest.json"
    cfg["pose_source_manifest"].write_text("{}")
    monkeypatch.setattr(pipeline, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        pipeline,
        "preflight",
        lambda _: [{"component": "synthetic", "status": "ready", "detail": "test"}],
    )
    return cfg


def test_all_ten_categories_reach_pose_and_sam_without_promoting_measurement_qualification(
    fake_runtime: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    animal_rows = [
        animal(f"animal_{index + 1:04d}") | {"category": category} | category_capability(category)
        for index, category in enumerate(COCO_ANIMAL_CATEGORIES)
    ]

    def worker(cfg, run_dir, stage, request, result):
        calls.append(stage)
        if stage == "detection":
            assert request["supported_categories"] == list(COCO_ANIMAL_CATEGORIES)
            return {"results": animal_rows}
        if stage == "localization":
            assert len(request["cases"]) == 10
            return {"results": [pose(row["prediction_id"]) for row in animal_rows]}
        assert len(request["cases"]) == 20
        predictions = []
        for index, case in enumerate(request["cases"]):
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[40:42, 30 + index : 32 + index] = 255
            path = f"mask{index}.png"
            Image.fromarray(mask).save(run_dir / path)
            predictions.append(case | {"status": "completed", "mask_path": path})
        bird_cases = [case for case in request["cases"] if case["experimental"]]
        assert len(bird_cases) == 2
        assert all(case["pose_domain"] == "out_of_training_domain" for case in bird_cases)
        return {"results": predictions}

    monkeypatch.setattr(pipeline, "_run_worker", worker)
    output = pipeline.run_inference(image_bytes(Image.new("RGB", (100, 100))), "test.png", CONFIG)
    result = pipeline.load_result(output)
    assert calls == ["detection", "localization", "segmentation"]
    assert result["status"] == "completed"
    assert {row["category"] for row in result["eyes"]} == set(COCO_ANIMAL_CATEGORIES)
    assert all(row["localization_status"] == "completed" for row in result["eyes"])
    assert all(row["segmentation_status"] == "completed" for row in result["eyes"])
    assert all(row["measurement_qualification"] == "unverified" for row in result["eyes"])
    assert all(row["qualification"] == "unverified" for row in result["measurements"])
    assert len(result["measurements"]) == 55
    assert result["capability_policy"]["accuracy_validated_for_all_categories"] is False
    assert result["capability_policy"]["experimental_categories"] == ["bird"]
    exported = list(csv.DictReader((output / "eyes.csv").open(encoding="utf-8-sig")))
    assert len(exported) == 20
    assert {row["category"] for row in exported if row["experimental"] == "True"} == {"bird"}


@pytest.mark.parametrize(
    "scenario", ["no_animals", "unsupported", "detector_failure", "pose_failure", "partial_sam"]
)
def test_orchestration_preserves_failures_and_never_needs_gt(
    fake_runtime: dict, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    calls = []

    def worker(cfg, run_dir, stage, request, result):
        calls.append(stage)
        assert "annotation_id" not in str(request) and "reference" not in str(request)
        if stage == "detection":
            if scenario == "detector_failure":
                raise RuntimeError("synthetic_detector_error")
            animals = (
                []
                if scenario == "no_animals"
                else [animal("animal_0001", supported=scenario != "unsupported")]
            )
            return {"results": animals}
        if stage == "localization":
            if scenario == "pose_failure":
                raise RuntimeError("synthetic_pose_error")
            assert all(
                set(case) == {"image_id", "prediction_id", "image", "image_sha256", "bbox_xyxy"}
                for case in request["cases"]
            )
            return {"results": [pose("animal_0001")]}
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:42, 39:41] = 255
        Image.fromarray(mask).save(run_dir / "testmask.png")
        return {
            "results": [
                request["cases"][0] | {"status": "completed", "mask_path": "testmask.png"},
                request["cases"][1] | {"status": "failed", "failure_reason": "synthetic_sam_error"},
            ]
        }

    monkeypatch.setattr(pipeline, "_run_worker", worker)
    output = pipeline.run_inference(image_bytes(Image.new("RGB", (100, 100))), "test.png", CONFIG)
    result = pipeline.load_result(output)
    assert (output / "run_manifest.json").is_file()
    assert all(
        (output / result["artifacts"][key]).is_file()
        for key in ("animals_csv", "eyes_csv", "measurements_csv")
    )
    if scenario == "no_animals":
        assert result["status"] == "no_animals" and result["measurements"] == []
        assert calls == ["detection"]
        for name in ("animals", "eyes", "measurements"):
            reader = csv.DictReader(
                io.StringIO((output / f"{name}.csv").read_text(encoding="utf-8-sig"))
            )
            assert reader.fieldnames and list(reader) == []
    elif scenario == "detector_failure":
        assert result["status"] == "failed"
        assert result["stages"]["detection"]["status"] == "failed"
        assert calls == ["detection"]
    else:
        assert len(result["eyes"]) == 2 and len(result["measurements"]) == 1
        assert result["measurements"][0]["distance_px"] is None
        assert result["measurements"][0]["status"] == "unavailable"
        assert result["measurements"][0]["qualification"] == "unverified"
        if scenario == "unsupported":
            assert calls == ["detection"] and result["status"] == "completed_with_issues"
        elif scenario == "pose_failure":
            assert calls == ["detection", "localization"] and result["status"] == "failed"
        else:
            assert calls == ["detection", "localization", "segmentation"]
            assert result["eyes"][0]["segmentation_status"] == "completed"
            assert result["eyes"][1]["segmentation_status"] == "failed"
            assert result["status"] == "completed_with_issues"


@pytest.mark.parametrize(
    "scenario",
    [
        "invalid_box",
        "instance_empty",
        "pose_partial",
        "sam_empty",
        "sam_missing",
        "sam_global_failure",
    ],
)
def test_partial_stage_failures_preserve_candidates_and_export_unavailable_pairs(
    fake_runtime: dict,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    calls = []

    def worker(cfg, run_dir, stage, request, result):
        calls.append(stage)
        assert "annotation_id" not in str(request) and "reference" not in str(request)
        if stage == "detection":
            animals = [animal("animal_0001")]
            if scenario == "invalid_box":
                animals[0].update(
                    accepted=False, status="failed", failure_reason="invalid_animal_box"
                )
            if scenario == "instance_empty":
                animals[0].update(
                    mask_status="empty_mask", mask_failure_reason="empty_instance_mask"
                )
            if scenario == "pose_partial":
                animals.append(animal("animal_0002"))
            return {"results": animals}
        if stage == "localization":
            assert all(
                set(case) == {"image_id", "prediction_id", "image", "image_sha256", "bbox_xyxy"}
                for case in request["cases"]
            )
            poses = [pose("animal_0001")]
            if scenario == "pose_partial":
                poses.append(pose("animal_0002", scores=[0.8, 0]))
            return {"results": poses}
        if scenario == "sam_global_failure":
            raise RuntimeError("synthetic_sam_global_error")
        predictions = []
        for index, case in enumerate(request["cases"]):
            if index == 1 and scenario == "sam_missing":
                continue
            mask = np.zeros((100, 100), dtype=np.uint8)
            if index != 1 or scenario != "sam_empty":
                mask[40:42, 39 + index * 10 : 41 + index * 10] = 255
            path = f"testmask{index}.png"
            Image.fromarray(mask).save(run_dir / path)
            predictions.append(
                case | {"status": "completed" if mask.any() else "empty_mask", "mask_path": path}
            )
        return {"results": predictions}

    monkeypatch.setattr(pipeline, "_run_worker", worker)
    output = pipeline.run_inference(image_bytes(Image.new("RGB", (100, 100))), "test.png", CONFIG)
    result = pipeline.load_result(output)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["research_reference_used"] is False
    assert manifest["measurement_qualification"] == "unverified"
    assert manifest["source_unchanged_during_run"] is True
    assert {
        "inference/inference_pipeline.py",
        "measurement/metrology.py",
        "core/__init__.py",
        "ui/__init__.py",
    } <= manifest["source_sha256"].keys()
    assert result["status"] == (
        "failed" if scenario == "sam_global_failure" else "completed_with_issues"
    )
    exported = list(
        csv.DictReader(io.StringIO((output / "measurements.csv").read_text(encoding="utf-8-sig")))
    )
    assert len(exported) == len(result["measurements"])
    assert all(row["qualification"] == "unverified" for row in exported)
    if scenario == "invalid_box":
        assert calls == ["detection"]
        assert result["stages"]["detection"]["status"] == "completed_with_issues"
        assert result["animals"][0]["status"] == "failed"
        assert result["eyes"] == [] and result["measurements"] == []
    elif scenario == "instance_empty":
        assert result["stages"]["detection"]["status"] == "completed_with_issues"
        assert result["measurements"][0]["distance_px"] == 10
        assert result["measurements"][0]["status"] == "candidate"
        assert result["animals"][0]["mask_status"] == "empty_mask"
    elif scenario == "pose_partial":
        assert result["stages"]["localization"]["status"] == "completed_with_issues"
        assert result["stages"]["measurement"]["status"] == "completed_with_issues"
        assert len(result["eyes"]) == 4 and len(result["measurements"]) == 3
        assert [row["distance_px"] for row in result["measurements"]] == [10, None, None]
        assert [row["status"] for row in result["measurements"]] == [
            "candidate",
            "unavailable",
            "unavailable",
        ]
        assert [row["distance_px"] for row in exported] == ["10.0", "", ""]
    else:
        assert calls == ["detection", "localization", "segmentation"]
        assert len(result["eyes"]) == 2 and len(result["measurements"]) == 1
        assert result["measurements"][0]["distance_px"] is None
        assert result["measurements"][0]["status"] == "unavailable"
        assert exported[0]["distance_px"] == ""
        if scenario == "sam_global_failure":
            assert all(row["segmentation_status"] == "failed" for row in result["eyes"])
        else:
            assert result["eyes"][0]["segmentation_status"] == "completed"
            assert result["eyes"][1]["segmentation_status"] == (
                "empty_mask" if scenario == "sam_empty" else "failed"
            )
            assert result["eyes"][1]["centroid_x_px"] is None
            if scenario == "sam_empty":
                assert result["eyes"][1]["mask_area_px"] == 0
            if scenario == "sam_missing":
                assert result["eyes"][1]["failure_reason"] == "missing_segmentation_output"
