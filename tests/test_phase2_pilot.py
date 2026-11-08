"""Sampling reproducibility and population-weighted time estimates, without models."""

import pytest

from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.phase2_pilot import clip_box_roundoff, forecast, sample_ids


def test_sampling_is_order_independent_unique_and_does_not_mutate_global_rng() -> None:
    import random

    population = list(range(5000))
    state = random.getstate()
    drawn = sample_ids(population, 100, 42)
    assert drawn == sample_ids(population[::-1], 100, 42)
    assert len(set(drawn)) == 100
    assert set(drawn) <= set(population)
    assert random.getstate() == state
    assert drawn != sample_ids(population, 100, 43)
    with pytest.raises(Phase1Error):
        sample_ids([1, 1], 2, 42)


def test_runtime_weighting_uses_full_population_instead_of_pilot_mix() -> None:
    rows = [
        {"group": "animal", "latency_seconds": 3},
        {"group": "background", "latency_seconds": 1},
    ]
    result = forecast(rows, {"animal": 1000, "background": 4000})
    assert result["linear_seconds"] == 10000
    assert result["population_weighted_seconds"] == 7000
    assert result["strata"]["animal"]["mean_seconds"] == 3
    missing = forecast(rows[:1], {"animal": 1000, "background": 4000})
    assert missing["population_weighted_seconds"] is None
    with pytest.raises(Phase1Error):
        forecast([], {"animal": 1000, "background": 4000})


def test_box_roundoff_allows_float_resize_error_but_rejects_wrong_coordinates() -> None:
    import numpy as np

    boxes = np.array([[1, 2, 640, 426.0000305175781, 0.9, 20]])
    clipped, count = clip_box_roundoff(boxes, 640, 426)
    assert count == 1
    assert clipped[0, 3] == 426
    assert boxes[0, 3] > 426  # Never modify the model array in place.
    np.testing.assert_array_equal(clipped[0, [0, 1, 2, 4, 5]], boxes[0, [0, 1, 2, 4, 5]])
    boxes[0, 3] = 426.1
    with pytest.raises(Phase1Error, match="roundoff"):
        clip_box_roundoff(boxes, 640, 426)
