"""Synthetic checks for geometric-box coverage, paired errors and predeclared gates."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from animal_eye_metrology import ablation_metrics as shared
from animal_eye_metrology import auto_box_metrics as metrics
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.eye_comparison import eye_statistics
from animal_eye_metrology.metrology import build_measurements, mask_centroid


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Four invented eye masks; no actual project image or annotation."""
    (tmp_path / "inputs").mkdir()
    reference = tmp_path / "evaluation_inputs/reference"
    (reference / "frozen").mkdir(parents=True)
    (tmp_path / "evaluation_inputs/baseline").mkdir()
    conditions = [
        {
            "condition_id": f"auto_box_k{round(gamma * 100):03d}_{group}",
            "group": group,
            "gamma": gamma,
            "roi_factor": 1.0,
            "point_source": "hrnet" if group == "point" else "none",
            "box_source": "geometry",
            "is_primary": gamma == 0.5 and group == "point",
        }
        for gamma in (0.3, 0.5, 0.7)
        for group in ("point", "only")
    ]
    refs, masks = [], {}
    for aid, y in ((1, 2), (2, 8)):
        for side, x in (("left", 2), ("right", 6)):
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[y, x : x + 2] = 1
            center = mask_centroid(mask)
            name = f"{aid}_{side}.png"
            Image.fromarray(mask * 255).save(reference / "frozen" / name)
            refs.append(
                {
                    "image_id": 10,
                    "annotation_id": aid,
                    "eye_side": side,
                    "animal_category": "SYNTHETIC",
                    "annotation_version": 1,
                    "centroid_x_px": center[0],
                    "centroid_y_px": center[1],
                    "mask_path": f"frozen/{name}",
                }
            )
            masks[(aid, side)] = mask
    monkeypatch.setattr(metrics, "verify", lambda _: {"status": "passed"})
    monkeypatch.setattr(metrics, "derive", lambda _: (refs, build_measurements(refs)))
    baseline = {
        "eye_metrics.csv": [],
        "measurement_comparison.csv": [],
        "condition_summary.csv": [],
    }
    for cid in metrics.BASELINES:
        descriptor = {
            "condition_id": cid,
            "group": cid[0],
            "roi_factor": 0.5 if cid == "A_r050" else 1.0,
            "point_source": "hrnet",
            "box_source": "none",
        }
        local_eyes = []
        for i, ref in enumerate(refs):
            aid, side = ref["annotation_id"], ref["eye_side"]
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[i // 2, i % 2] = 1
            stats = eye_statistics(mask, masks[(aid, side)])
            local_eyes.append(
                {
                    **descriptor,
                    **{k: ref[k] for k in shared.IDENTITY},
                    "case_id": f"p{aid}_{side}",
                    "prediction_id": f"p{aid}",
                    "animal_category": "SYNTHETIC",
                    "point_x": ref["centroid_x_px"],
                    "point_y": ref["centroid_y_px"],
                    "point_score": 0.9,
                    "point_in_target_mask": True,
                    "point_in_opposite_mask": False,
                    "point_offset_from_mask_centroid_px": 0.0,
                    "roi_x0": 0,
                    "roi_y0": 0,
                    "roi_x1": 12,
                    "roi_y1": 12,
                    "roi_clipped_to_image": False,
                    "point_outside_prompt_box": False,
                    "reference_pixels_outside_roi": 0,
                    "mask_touches_roi_boundary": True,
                    "candidate_iou_gap": 1,
                    "candidate_best_iou": 1,
                    "area_ratio": 0.5,
                    **stats,
                }
            )
        distances = shared.measurements(local_eyes, build_measurements(refs), descriptor)
        baseline["eye_metrics.csv"].extend(local_eyes)
        baseline["measurement_comparison.csv"].extend(distances)
        baseline["condition_summary.csv"].append(
            shared.summarize(local_eyes, distances, descriptor)
        )
    (tmp_path / "evaluation_inputs/baseline/evaluation_values.json").write_text(
        json.dumps(baseline)
    )
    (tmp_path / "evaluation_inputs/animal_matches.csv").write_text(
        "image_id,prediction_id,annotation_id,match_status,confidence_threshold,iou_threshold\n"
        "10,p1,1,TP,0.5,0.5\n10,p2,2,TP,0.5,0.5\n"
    )
    predictions = []
    for condition in conditions:
        for ref in refs:
            aid, side = ref["annotation_id"], ref["eye_side"]
            gt = masks[(aid, side)]
            x, y = ref["centroid_x_px"], ref["centroid_y_px"]
            positive = condition["group"] == "point"
            cid = condition["condition_id"]
            row = {
                **condition,
                "image_id": 10,
                "prediction_id": f"p{aid}",
                "eye_side": side,
                "case_id": f"p{aid}_{side}",
                "hrnet_point_xy": [x, y],
                "hrnet_score": 0.9,
                "point_xy": [x, y] if positive else None,
                "point_labels": [1] if positive else None,
                "point_coords_in_roi": [[x, y]] if positive else None,
                "box_xyxy": [x - 1, y - 0.5, x + 1, y + 0.5],
                "box_side_length_px": 2.0,
                "box_clipped": False,
                "roi_xyxy": [0, 0, 12, 12],
                "human_assisted": False,
                "status": "passed",
                "failure_reason": "",
            }
            if (cid, aid, side) == ("auto_box_k030_point", 1, "left"):
                row.update(status="failed", failure_reason="synthetic failure")
            else:
                selected = gt.copy()
                if (cid, aid, side) == ("auto_box_k030_only", 2, "right"):
                    selected[:] = 0
                if (cid, aid, side) == ("auto_box_k070_only", 1, "left"):
                    selected[:] = 0
                    selected[0, 0] = 1
                directory = tmp_path / cid
                directory.mkdir(exist_ok=True)
                name = f"{aid}_{side}"
                Image.fromarray(selected * 255).save(directory / f"{name}.png")
                np.save(
                    directory / f"{name}.npy",
                    np.stack([selected, gt, np.zeros_like(gt)]).astype(float) * 2 - 1,
                )
                row.update(
                    mask_path=f"{cid}/{name}.png",
                    logits_path=f"{cid}/{name}.npy",
                    candidate_scores=[0.9, 0.8, 0.7],
                    selected_candidate=0,
                )
            predictions.append(row)
    (tmp_path / "inputs/conditions.json").write_text(json.dumps(conditions))
    (tmp_path / "predictions.json").write_text(json.dumps(predictions))
    return tmp_path


def test_half_open_box_uses_pixel_centers_and_preserves_prediction() -> None:
    reference = np.zeros((3, 4), dtype=np.uint8)
    reference[0, :3] = 1
    predicted = reference.copy()
    before = predicted.copy()
    row = metrics.box_statistics([0.5, 0.5, 2.5, 1.5], reference, predicted)
    assert row["reference_pixels_in_box"] == 2
    assert row["reference_pixels_outside_box"] == 1
    assert row["reference_box_coverage_fraction"] == 2 / 3
    assert row["reference_fully_in_box"] is False
    assert row["prediction_pixels_outside_box"] == 1
    assert row["prediction_outside_box_fraction"] == 1 / 3
    assert row["box_area_px2"] == 2
    assert np.array_equal(predicted, before)
    empty = metrics.box_statistics([0.5, 0.5, 2.5, 1.5], reference, np.zeros_like(reference))
    failed = metrics.box_statistics([0.5, 0.5, 2.5, 1.5], reference, None)
    assert empty["prediction_pixels_outside_box"] == 0
    assert empty["prediction_outside_box_fraction"] is None
    assert failed["prediction_pixels_outside_box"] is None


def test_compute_retains_failures_and_separates_geometry_from_positive_prompt(
    experiment: Path,
) -> None:
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in experiment.rglob("*")
        if p.is_file()
    }
    tables = metrics.compute(experiment)
    expected = {
        "eye_metrics.csv": 24,
        "candidate_metrics.csv": 72,
        "measurement_comparison.csv": 18,
        "condition_summary.csv": 6,
        "baseline_summary.csv": 4,
        "localization_metrics.csv": 4,
        "contrast_summary.csv": 19,
        "paired_eye_deltas.csv": 76,
        "paired_distance_deltas.csv": 57,
    }
    for name, count in expected.items():
        assert len(tables[name]) == count
    only = [r for r in tables["eye_metrics.csv"] if r["group"] == "only"]
    assert all(not r["had_positive_prompt"] and r["prompt_point_x"] is None for r in only)
    assert all(r["geometry_point_in_target_mask"] is True for r in only)
    assert all("point_x" not in r for r in tables["eye_metrics.csv"])
    summaries = {r["condition_id"]: r for r in tables["condition_summary.csv"]}
    failed = summaries["auto_box_k030_point"]
    empty = summaries["auto_box_k030_only"]
    assert failed["eyes_expected"] == 4 and failed["inference_failed"] == 1
    assert failed["mean_iou_all_targets_failure_as_zero"] == 0.75
    assert failed["binocular_defined"] == 1
    assert empty["empty_masks"] == 1 and empty["right_eye_pair_defined"] == 0
    undefined = [
        r for r in tables["measurement_comparison.csv"] if r["measurement_status"] == "Undefined"
    ]
    assert len(undefined) == 3
    assert all(
        r["predicted_distance_px"] is None and r["absolute_error_px"] is None for r in undefined
    )
    selected = next(
        r
        for r in tables["eye_metrics.csv"]
        if r["condition_id"] == "auto_box_k070_only"
        and r["annotation_id"] == 1
        and r["eye_side"] == "left"
    )
    assert selected["iou"] == 0 and selected["candidate_best_iou"] == 1
    assert selected["prediction_outside_box_fraction"] == 1
    assert all(r["passed"] for r in tables["promotion_checks.csv"])
    after = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in experiment.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_paired_comparisons_keep_direction_and_do_not_hide_undefined(experiment: Path) -> None:
    tables = metrics.compute(experiment)
    rows = {r["contrast_id"]: r for r in tables["contrast_summary.csv"]}
    comparison = rows[f"{metrics.PRIMARY}_minus_A_r100"]
    assert comparison["mean_delta_iou_all_targets_failure_as_zero"] == 1
    assert comparison["binocular_mean_delta_absolute_error_px"] == -3
    assert comparison["right_eye_pair_mean_delta_absolute_error_px"] == -5
    assert rows["auto_box_k030_only_minus_auto_box_k030_point"]["binocular_pairs_defined"] == 0


def test_promotion_gate_cannot_hide_new_zero_or_replace_primary(experiment: Path) -> None:
    tables = metrics.compute(experiment)
    primary = next(r for r in tables["condition_summary.csv"] if r["is_primary"])
    baseline = tables["baseline_summary.csv"][0]
    pairs = tables["paired_eye_deltas.csv"]
    target = next(
        r
        for r in pairs
        if r["target_condition"] == metrics.PRIMARY and r["baseline_condition"] == "A_r100"
    )
    target["new_zero_overlap"] = True
    gates = {r["check_id"]: r for r in metrics.promotion_checks(primary, baseline, pairs)}
    assert gates["no_new_zero_overlap"]["passed"] is False
    assert gates["all_research_gates_passed"]["passed"] is False
    primary["binocular_mae_px"] = None
    gates = {r["check_id"]: r for r in metrics.promotion_checks(primary, baseline, pairs)}
    assert gates["binocular_mae_improved"]["passed"] is False
    primary["condition_id"] = "auto_box_k070_only"
    with pytest.raises(Phase1Error, match="predeclared"):
        metrics.promotion_checks(primary, baseline, pairs)


@pytest.mark.parametrize(
    "mutation", ["missing", "positive_in_only", "roi_changed", "point_changed", "selected_changed"]
)
def test_rejects_design_or_policy_changes(experiment: Path, mutation: str) -> None:
    path = experiment / "predictions.json"
    rows = json.loads(path.read_text())
    row = next(r for r in rows if r["condition_id"] == "auto_box_k050_only")
    if mutation == "missing":
        rows.remove(row)
    elif mutation == "positive_in_only":
        row["point_xy"] = row["hrnet_point_xy"]
    elif mutation == "roi_changed":
        row["roi_xyxy"][2] -= 1
    elif mutation == "point_changed":
        row["hrnet_point_xy"][0] += 1
    else:
        row["selected_candidate"] = 1
    path.write_text(json.dumps(rows))
    with pytest.raises(Phase1Error):
        metrics.compute(experiment)
