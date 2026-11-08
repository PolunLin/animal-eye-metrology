"""Synthetic UI checks; these fixtures are not real inference results or annotations."""

import io
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("PIL.Image")

from PIL import Image, ImageDraw  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from animal_eye_metrology.ui import inference_view as view  # noqa: E402


class Upload(io.BytesIO):
    name = "synthetic.png"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    config = tmp_path / "inference.toml"
    config.write_text('purpose = "synthetic UI test"\n')
    monkeypatch.setenv("AEM_INFERENCE_CONFIG", str(config))
    run = tmp_path / "run"
    run.mkdir()
    image = Image.new("RGB", (80, 60), "white")
    image.save(run / "overlay.png")
    image.save(run / "input.png")
    for number, box in enumerate(((5, 12, 25, 40), (45, 15, 65, 44)), start=1):
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(box, fill=255)
        mask.save(run / f"animal_{number}.png")
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    upload = Upload(raw.getvalue())
    result = {
        "run_id": "synthetic-ui-only",
        "status": "completed_with_issues",
        "input": {
            "filename": "synthetic.png",
            "width": 80,
            "height": 60,
            "image_path": "input.png",
        },
        "conditions": ["A_r100"],
        "stages": {
            stage: {"status": "completed", "seconds": 0.1, "reason": ""} for stage in view.STAGES
        },
        "animals": [
            {
                "prediction_id": "animal_0001",
                "category": "dog",
                "score": 0.75,
                "accepted": True,
                "supported": True,
                "status": "completed",
                "failure_reason": "",
                "mask_status": "completed",
                "mask_path": "animal_1.png",
                "bbox_xyxy": [5, 12, 26, 41],
            },
            {
                "prediction_id": "animal_0002",
                "category": "cat",
                "score": 0.75,
                "accepted": True,
                "supported": False,
                "status": "unsupported_category",
                "failure_reason": "category_not_in_approved_scope",
                "mask_status": "completed",
                "mask_path": "animal_2.png",
                "bbox_xyxy": [45, 15, 66, 45],
            },
        ],
        "eyes": [
            {
                "prediction_id": "animal_0001",
                "eye_side": side,
                "condition_id": "A_r100",
                "localization_status": "completed",
                "segmentation_status": "completed" if side == "left" else "empty_mask",
                "point_x_px": 20,
                "point_y_px": 20,
                "point_score": 0.8,
                "roi_xyxy": [10, 10, 30, 30],
                "mask_area_px": 4 if side == "left" else 0,
                "centroid_x_px": 20.5 if side == "left" else None,
                "centroid_y_px": 20.5 if side == "left" else None,
                "failure_reason": "" if side == "left" else "no_foreground_pixels",
                "visibility_status": "unverified",
                "side_assignment_status": "unverified",
            }
            for side in ("left", "right")
        ],
        "measurements": [
            {
                "condition_id": "A_r100",
                "measurement_type": "binocular",
                "prediction_id_a": "animal_0001",
                "prediction_id_b": "animal_0001",
                "distance_px": None,
                "status": "unavailable",
                "qualification": "unverified",
                "reason": "right_eye_empty_mask",
            },
            {
                "condition_id": "A_r100",
                "measurement_type": "right_to_right",
                "prediction_id_a": "synthetic_a",
                "prediction_id_b": "synthetic_b",
                "distance_px": 12.25,
                "status": "candidate",
                "qualification": "unverified",
                "reason": "",
            },
        ],
        "artifacts": {
            "overlay": "overlay.png",
            "A_r100_overlay": "overlay.png",
            "animals_csv": "animals.csv",
            "eyes_csv": "eyes.csv",
            "measurements_csv": "measurements.csv",
        },
    }
    for name in ("animals", "eyes", "measurements"):
        (run / f"{name}.csv").write_text("synthetic,only\n1,2\n")
    (run / "result.json").write_text(json.dumps(result))
    calls: list[dict[str, Any]] = []

    def infer(data, filename, config_path, conditions, progress):
        calls.append({"data": data, "filename": filename, "conditions": conditions})
        progress("detection", "synthetic execution only")
        result["conditions"] = list(conditions)
        (run / "result.json").write_text(json.dumps(result))
        return run

    monkeypatch.setattr(view.pipeline, "load_config", lambda _: {})
    monkeypatch.setattr(
        view.pipeline,
        "preflight",
        lambda _: [{"component": "synthetic resource", "status": "ready", "detail": "fixture"}],
    )
    monkeypatch.setattr(view.pipeline, "run_inference", infer)
    monkeypatch.setattr(view.pipeline, "load_result", lambda _: result)
    monkeypatch.setattr(view.st, "file_uploader", lambda *args, **kwargs: upload)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from animal_eye_metrology.ui.inference_view import inference_workspace\n"
        f"inference_workspace(Path({str(tmp_path)!r}))\n",
        default_timeout=20,
    )
    return {"app": app, "result": result, "calls": calls, "config": config, "run": run}


def table(app: AppTest, column: str):
    return next(item.value for item in app.dataframe if column in item.value.columns)


def test_upload_run_retains_baseline_and_does_not_rerun_on_display(workspace):
    app = workspace["app"].run()
    assert not app.exception
    assert not any(item.key == "inference_compare" for item in app.checkbox)
    app.button(key="inference_run").click().run()
    assert not app.exception
    assert not app.error
    assert workspace["calls"][0]["conditions"] == ("A_r100",)
    assert len(app.get("download_button")) == 8
    app.run()
    assert len(workspace["calls"]) == 1
    assert any("候選量測" in item.value for item in app.warning)
    assert not any("IoU" in str(item.value.columns) for item in app.dataframe)


def test_changed_image_hides_old_result(workspace, monkeypatch):
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    monkeypatch.setattr(view.st, "file_uploader", lambda *args, **kwargs: Upload(b"changed"))
    app.run()
    assert not app.exception
    assert any("請重新執行" in item.value for item in app.info)
    assert not any("本次推論結果" == item.value for item in app.subheader)
    assert len(workspace["calls"]) == 1


def test_missing_eye_and_unsupported_animal_remain_visible(workspace):
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    eyes = table(app, "分割狀態")
    assert list(eyes["分割狀態"]) == ["執行完成", "空遮罩"]
    assert eyes.iloc[1]["重心 x px"] != 0
    assert eyes.iloc[1]["原因"] == "no_foreground_pixels"
    animals = table(app, "類別")
    assert animals.iloc[1]["偵測／實例狀態"] == "本次未啟用眼部流程"
    distances = table(app, "量測狀態")
    assert list(distances["量測狀態"]) == ["無法計算", "候選量測"]
    assert set(distances["量測資格"]) == {"尚未驗證"}
    assert distances.iloc[0]["原因"] == "right_eye_empty_mask"


def test_missing_resources_disable_execution(workspace, monkeypatch):
    monkeypatch.setattr(
        view.pipeline,
        "preflight",
        lambda _: [{"component": "model", "status": "missing", "detail": "missing weight"}],
    )
    app = workspace["app"].run()
    assert not app.exception
    assert app.button(key="inference_run").disabled
    assert any("不會自動下載" in item.value for item in app.error)
    assert not workspace["calls"]


def test_invalid_upload_error_clears_stale_result(workspace, monkeypatch):
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()

    def invalid(*args, **kwargs):
        raise ValueError("invalid_image_bytes")

    monkeypatch.setattr(view.pipeline, "run_inference", invalid)
    app.button(key="inference_run").click().run()
    assert not app.exception
    assert any("invalid_image_bytes" in item.value for item in app.error)
    assert not any("本次推論結果" == item.value for item in app.subheader)


def test_no_animals_and_failed_stages_are_explicit(workspace):
    result = workspace["result"]
    result.update(status="no_animals", eyes=[], measurements=[], animals=[])
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    assert not app.exception
    assert any("不代表圖片一定沒有動物" in item.value for item in app.info)
    result["status"] = "failed"
    result["stages"]["detection"] = {"status": "failed", "reason": "worker_timeout", "seconds": 1.0}
    app.run()
    assert not app.exception
    assert any("推論未完成" in item.value for item in app.error)
    assert table(app, "階段").iloc[0]["原因"] == "worker_timeout"


def test_result_assets_cannot_escape_the_run_directory(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not a run asset")
    (root / "alias").symlink_to(outside)
    inside = root / "asset.txt"
    inside.write_text("asset")
    assert view.artifact_path(root, "asset.txt") == inside
    assert view.artifact_path(root, "../outside.txt") is None
    assert view.artifact_path(root, "alias") is None
    assert view.artifact_path(root, str(outside)) is None


def test_signature_includes_configuration_and_conditions(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("seed = 42\n")
    signature = view.request_signature(b"image", ("A_r100",), config)
    assert signature != view.request_signature(b"new image", ("A_r100",), config)
    assert signature != view.request_signature(b"image", ("A_r100", "A_r050"), config)
    config.write_text("seed = 43\n")
    assert signature != view.request_signature(b"image", ("A_r100",), config)


def test_animal_contour_display_switches_without_inference_or_file_changes(workspace):
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    assert not app.exception
    assert not app.error
    assert app.radio(key="inference_animals_mode_synthetic-ui-only").value == "輪廓疊圖"
    assert not app.checkbox(key="inference_animals_boxes_synthetic-ui-only").value
    before = {p: p.read_bytes() for p in workspace["run"].rglob("*") if p.is_file()}
    saved = json.dumps(workspace["result"], sort_keys=True)
    app.radio(key="inference_animals_mode_synthetic-ui-only").set_value("實例遮罩").run()
    app.checkbox(key="inference_animals_boxes_synthetic-ui-only").check().run()
    app.selectbox(key="inference_animals_selected_synthetic-ui-only").set_value("animal_0002").run()
    assert not app.exception
    assert len(workspace["calls"]) == 1
    assert {p: p.read_bytes() for p in workspace["run"].rglob("*") if p.is_file()} == before
    assert json.dumps(workspace["result"], sort_keys=True) == saved
    assert any("動物輪廓（實例分割）" == item.value for item in app.subheader)
    titles = [item.value for item in app.subheader]
    assert titles.index("動物輪廓（實例分割）") < titles.index("眼部分割與候選距離")


@pytest.mark.parametrize("problem", ["missing", "empty", "wrong_size", "bad_source"])
def test_unavailable_animal_mask_does_not_replace_contour_with_box_or_hide_eyes(workspace, problem):
    row = workspace["result"]["animals"][0]
    if problem == "missing":
        (workspace["run"] / row["mask_path"]).unlink()
    elif problem == "empty":
        Image.new("L", (80, 60), 0).save(workspace["run"] / row["mask_path"])
        row["mask_status"] = "empty_mask"
    elif problem == "wrong_size":
        Image.new("L", (8, 6), 255).save(workspace["run"] / row["mask_path"])
    else:
        workspace["result"]["input"]["image_path"] = "missing_input.png"
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    assert not app.exception
    assert any(
        "不會以偵測框代替輪廓" in item.value or "無法繪製動物輪廓" in item.value
        for item in app.warning
    )
    assert len(table(app, "分割狀態")) == 2
    assert len(table(app, "量測狀態")) == 2
    assert len(workspace["calls"]) == 1


def test_raw_animal_mask_download_preserves_original_bytes(workspace, monkeypatch):
    mask_path = workspace["run"] / "animal_1.png"
    with Image.open(mask_path) as mask:
        mask.point(lambda pixel: 1 if pixel else 0).save(mask_path)
    original = mask_path.read_bytes()
    downloaded = []
    download_button = view.st.download_button

    def capture(label, data, *args, **kwargs):
        if label == "下載此動物原始二值遮罩 PNG":
            downloaded.append(data)
        return download_button(label, data, *args, **kwargs)

    monkeypatch.setattr(view.st, "download_button", capture)
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    assert not app.exception
    assert not app.error
    assert downloaded == [original]
    assert mask_path.read_bytes() == original


def test_coco_ten_category_caption_keeps_domain_limit_explicit(workspace):
    app = workspace["app"].run()
    captions = "\n".join(item.value for item in app.caption)
    assert "COCO 全部 10 類" in captions
    assert "鳥、貓、狗、馬、羊、牛、象、熊、斑馬、長頸鹿" in captions
    assert "AP-10K 訓練範圍外" in captions
    assert "所有類別的眼部正確性與量測資格仍待驗證" in captions
    assert "COCO 以外的所有物種" in captions


def test_bird_domain_warning_and_candidate_metadata_are_preserved(workspace):
    result = workspace["result"]
    result["animals"] = result["animals"][:1]
    result["animals"][0].update(
        category="bird", pose_domain="out_of_training_domain", experimental=True
    )
    for row in result["eyes"]:
        row.update(category="bird", pose_domain="out_of_training_domain", experimental=True)
    for row in result["measurements"]:
        row.update(
            pose_domain_a="out_of_training_domain",
            pose_domain_b="ap10k_mammal_mapping",
            experimental=True,
        )
    before = json.dumps(result, sort_keys=True)
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    assert not app.exception
    assert any("鳥類眼部輸出屬 AP-10K 訓練範圍外試驗" in item.value for item in app.warning)
    assert table(app, "類別").iloc[0]["眼點模型範圍"] == "AP-10K 訓練範圍外"
    assert set(table(app, "預測眼別")["域外試驗"]) == {"是（域外試驗）"}
    distances = table(app, "量測狀態")
    assert set(distances["量測資格"]) == {"尚未驗證"}
    assert set(distances["域外試驗"]) == {"是（域外試驗）"}
    assert distances.iloc[1]["距離 px"] == 12.25
    assert json.dumps(result, sort_keys=True) == before


def test_old_three_category_run_explains_skipped_eyes_without_reclassifying(workspace):
    result = workspace["result"]
    result["config"] = {"supported_categories": ["dog", "bear", "giraffe"]}
    result["animals"] = result["animals"][1:]
    result["animals"][0]["category"] = "horse"
    for row in result["eyes"]:
        row.update(
            prediction_id="animal_0002",
            localization_status="unsupported_category",
            segmentation_status="skipped",
            failure_reason="unsupported_category",
            roi_xyxy=None,
            mask_area_px=None,
            centroid_x_px=None,
            centroid_y_px=None,
        )
    before = json.dumps(result, sort_keys=True)
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    assert not app.exception
    warnings = "\n".join(item.value for item in app.warning)
    assert "舊版三類範圍的保存結果" in warnings
    assert "馬的眼部流程被跳過" in warnings
    assert "本次沒有產生任何眼部遮罩" in warnings
    assert "當次物種範圍未啟用 2 筆" in warnings
    assert table(app, "非空眼部遮罩").iloc[0]["非空眼部遮罩"] == 0
    assert table(app, "類別").iloc[0]["眼點模型範圍"] == "舊結果未記錄"
    assert set(table(app, "預測眼別")["域外試驗"]) == {"舊結果未記錄"}
    assert json.dumps(result, sort_keys=True) == before
    assert len(workspace["calls"]) == 1


def test_eye_summary_explains_invalid_pose_and_empty_masks_before_overlays(workspace):
    result = workspace["result"]
    result["animals"] = result["animals"][:1]
    result["eyes"][0].update(
        localization_status="failed",
        segmentation_status="skipped",
        failure_reason="eye_outside_image",
        mask_area_px=None,
        centroid_x_px=None,
        centroid_y_px=None,
    )
    app = workspace["app"].run()
    app.button(key="inference_run").click().run()
    assert not app.exception
    warnings = "\n".join(item.value for item in app.warning)
    assert "本次沒有產生任何眼部遮罩" in warnings
    assert "眼點定位失敗或未通過有效性檢查 1 筆" in warnings
    assert "SAM 輸出空遮罩 1 筆" in warnings
    summary = table(app, "非空眼部遮罩").iloc[0]
    assert summary["預期逐眼結果"] == 2
    assert summary["定位完成"] == 1
    assert summary["空遮罩"] == 1
    assert summary["失敗／未執行"] == 1
    elements = list(app.main)
    summary_position = next(
        i
        for i, element in enumerate(elements)
        if element.type == "markdown" and "眼部處理摘要" in element.value
    )
    contour_position = next(
        i
        for i, element in enumerate(elements)
        if element.type == "subheader" and element.value == "動物輪廓（實例分割）"
    )
    assert summary_position < contour_position
