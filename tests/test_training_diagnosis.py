"""Meaningful guards and known L2 geometry for controlled training diagnosis."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from animal_eye_metrology.research.training_diagnosis import (  # noqa: E402
    guard_data_paths,
    guard_lr_only,
    relative_parameter_drift,
)


def test_lr_guard_rejects_confounded_momentum_freeze_or_mask_changes():
    common = {"momentum": 0.9, "freeze_blocks": 10, "mask_sampling": "nearest", "seed": 42}
    candidates = {
        "legacy": common | {"learning_rate": 0.001},
        "lower": common | {"learning_rate": 0.0001},
    }
    guard_lr_only(candidates)
    for key, value in (("momentum", 0.8), ("freeze_blocks", 0), ("mask_sampling", "linear")):
        with pytest.raises(ValueError, match="other than learning_rate"):
            guard_lr_only(candidates | {"lower": candidates["lower"] | {key: value}})
    with pytest.raises(ValueError, match="positive and finite"):
        guard_lr_only(candidates | {"lower": common | {"learning_rate": float("nan")}})


def test_relative_l2_uses_pooled_energy_and_retains_zero_reference_case():
    initial = {
        "weight": torch.tensor([3.0, 4.0]),
        "bias": torch.zeros(1),
        "frozen": torch.tensor([12.0]),
    }
    current = {
        "weight": torch.tensor([6.0, 8.0]),
        "bias": torch.tensor([2.0]),
        "frozen": initial["frozen"].clone(),
    }
    drift = relative_parameter_drift(
        initial, current, {"overall": list(initial), "head": ["weight"], "bn_affine": ["bias"]}
    )
    assert drift["head"]["absolute_l2"] == 5
    assert drift["head"]["relative_l2"] == 1
    assert drift["overall"]["reference_l2"] == 13
    assert drift["overall"]["relative_l2"] == pytest.approx(29**0.5 / 13)
    assert drift["bn_affine"]["absolute_l2"] == 2
    assert drift["bn_affine"]["relative_l2"] is None
    with pytest.raises(ValueError, match="unique known"):
        relative_parameter_drift(initial, current, {"bad": ["weight", "weight"]})


def test_probe_cannot_substitute_holdout_for_development():
    guard_data_paths(Path("data/fitting.json"), Path("data/development.json"))
    with pytest.raises(ValueError, match="fitting/development"):
        guard_data_paths(Path("data/fitting.json"), Path("data/holdout.json"))
