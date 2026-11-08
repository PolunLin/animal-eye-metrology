"""Synthetic QA for fixed denominators, independent conditions and diagnostic candidate ranking."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.measurement.metrology import build_measurements, mask_centroid
from animal_eye_metrology.research import ablation_metrics as metrics


@pytest.fixture
def experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Four invented eyes across two animals; no real image or research annotation."""
    (tmp_path / "inputs").mkdir()
    reference = tmp_path / "evaluation_inputs/reference"
    (reference / "frozen").mkdir(parents=True)
    conditions = []
    for group, (point_source, box_source) in metrics.GROUPS.items():
        for factor in metrics.FACTORS:
            conditions.append(
                {
                    "condition_id": f"{group}_r{round(factor * 100):03d}",
                    "group": group,
                    "roi_factor": factor,
                    "point_source": point_source,
                    "box_source": box_source,
                }
            )
    refs = []
    masks = {}
    for aid, y in ((1, 2), (2, 8)):
        for side, x in (("left", 2), ("right", 6)):
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[y, x : x + 2] = 1
            name = f"{aid}_{side}.png"
            Image.fromarray(mask * 255).save(reference / "frozen" / name)
            center = mask_centroid(mask)
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
    predictions = []
    for condition in conditions:
        for ref in refs:
            aid, side = ref["annotation_id"], ref["eye_side"]
            gt = masks[(aid, side)]
            cid = condition["condition_id"]
            row = {
                **condition,
                "image_id": 10,
                "annotation_id": aid,
                "prediction_id": f"p{aid}",
                "eye_side": side,
                "case_id": f"10_{aid}_{side}",
                "point_xy": [0.5, 0.5]
                if condition["point_source"] == "hrnet"
                else [ref["centroid_x_px"], ref["centroid_y_px"]],
                "point_score": 0.95 if condition["point_source"] == "hrnet" else None,
                "box_xyxy": [0, 0, 12, 12] if condition["box_source"] == "manual" else None,
                "roi_xyxy": [0, 0, 12, 12],
                "roi_clipped_to_image": False,
                "point_outside_prompt_box": False,
                "status": "passed",
                "failure_reason": "",
            }
            if (cid, aid, side) == ("A_r150", 1, "left"):
                row.update(status="failed", failure_reason="SYNTHETIC FAILURE")
            else:
                selected = gt.copy()
                if condition["group"] == "A":
                    selected[:] = 0
                    selected[0, 0] = 1
                if (cid, aid, side) == ("C_r050", 1, "right"):
                    selected[:] = 0
                directory = tmp_path / cid
                directory.mkdir(exist_ok=True)
                stem = f"{aid}_{side}"
                Image.fromarray(selected * 255).save(directory / f"{stem}.png")
                logits = np.stack([selected, gt, np.zeros_like(gt)]).astype(float) * 2 - 1
                np.save(directory / f"{stem}.npy", logits)
                row.update(
                    mask_path=f"{cid}/{stem}.png",
                    logits_path=f"{cid}/{stem}.npy",
                    candidate_scores=[0.9, 0.8, 0.7],
                    selected_candidate=0,
                )
            predictions.append(row)
    (tmp_path / "inputs/conditions.json").write_text(json.dumps(conditions))
    (tmp_path / "predictions.json").write_text(json.dumps(predictions))
    return tmp_path


def by_condition(rows: list[dict], cid: str) -> dict:
    return next(row for row in rows if row["condition_id"] == cid)


def test_evaluation_preserves_denominators_and_never_uses_gt_best_for_main_result(
    experiment: Path,
) -> None:
    before = {
        str(p.relative_to(experiment)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in experiment.rglob("*")
        if p.is_file()
    }
    result = metrics.compute(experiment)
    assert len(result["eye_metrics.csv"]) == 48
    assert len(result["candidate_metrics.csv"]) == 144
    assert len(result["measurement_comparison.csv"]) == 36
    assert len(result["condition_summary.csv"]) == 12
    assert len(result["image_summary.csv"]) == 12
    assert len(result["localization_metrics.csv"]) == 8
    summary = by_condition(result["condition_summary.csv"], "A_r100")
    assert summary["eyes_expected"] == 4
    assert summary["mean_iou_all_targets_failure_as_zero"] == 0
    assert summary["mean_candidate_iou_gap"] == 1
    assert summary["candidate_best_iou_all_targets_failure_as_zero"] == 1
    assert summary["binocular_mae_px"] == 4
    assert summary["right_eye_pair_mae_px"] == 6
    failed = by_condition(result["condition_summary.csv"], "A_r150")
    assert failed["inference_failed"] == 1 and failed["centroid_defined"] == 3
    assert failed["binocular_expected"] == 2 and failed["binocular_defined"] == 1
    assert failed["candidate_gap_defined"] == 3
    empty = by_condition(result["condition_summary.csv"], "C_r050")
    assert empty["empty_masks"] == 1 and empty["inference_failed"] == 0
    assert empty["binocular_defined"] == 1 and empty["right_eye_pair_defined"] == 0
    # Undefined measurements remain absent; they never receive distance zero.
    undefined = [
        r for r in result["measurement_comparison.csv"] if r["measurement_status"] == "Undefined"
    ]
    assert len(undefined) == 3
    assert all(
        r["predicted_distance_px"] is None and r["absolute_error_px"] is None for r in undefined
    )
    # Localization is independent of the ROI/box settings and is not duplicated 12 times.
    assert sum(r["point_in_target_mask"] for r in result["localization_metrics.csv"]) == 4
    after = {
        str(p.relative_to(experiment)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in experiment.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_paired_effects_and_interactions_have_explicit_direction_and_valid_pairs(
    experiment: Path,
) -> None:
    result = metrics.compute(experiment)
    assert len(result["paired_eye_deltas.csv"]) == 80
    assert len(result["paired_distance_deltas.csv"]) == 60
    assert len(result["contrast_summary.csv"]) == 20
    contrast = next(
        r for r in result["contrast_summary.csv"] if r["contrast_id"] == "point_B_minus_A_r100"
    )
    assert contrast["mean_delta_iou_all_targets_failure_as_zero"] == 1
    assert contrast["iou_improved"] == 4 and contrast["iou_worsened"] == 0
    assert contrast["binocular_mean_delta_absolute_error_px"] == -4
    assert contrast["right_eye_pair_mean_delta_absolute_error_px"] == -6
    pair = next(
        r for r in result["contrast_summary.csv"] if r["contrast_id"] == "box_C_minus_A_r050"
    )
    assert pair["centroid_pairs_defined"] == 3
    assert pair["right_eye_pair_pairs_defined"] == 0
    interaction = next(r for r in result["interaction_summary.csv"] if r["roi_factor"] == 1)
    assert interaction["mean_iou_difference_in_differences_all_targets_failure_as_zero"] == -1
    assert interaction["binocular_mean_absolute_error_difference_in_differences_px"] == 4


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "point_changed", "selected_changed"])
def test_rejects_incomplete_design_or_changed_prediction_policy(
    experiment: Path,
    mutation: str,
) -> None:
    path = experiment / "predictions.json"
    rows = json.loads(path.read_text())
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "point_changed":
        rows[4]["point_xy"] = [1.5, 1.5]
    else:
        rows[0]["selected_candidate"] = 1
    path.write_text(json.dumps(rows))
    with pytest.raises(Phase1Error):
        metrics.compute(experiment)


def test_rejects_mask_that_does_not_match_selected_logits(experiment: Path) -> None:
    Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(experiment / "A_r050/1_left.png")
    with pytest.raises(Phase1Error, match="reproduce"):
        metrics.compute(experiment)
