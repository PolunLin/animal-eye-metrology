"""Scientific data protection checks using invented image IDs only."""

import pytest

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.data.segmentation_supply import supply_bounds


def test_protecting_all_unexposed_images_leaves_no_independent_holdout() -> None:
    strata = {1: "multi_animal", 2: "multi_animal", 3: "multi_animal", 4: "background"}
    result = supply_bounds(
        strata,
        exposed={1},
        protected={2, 3},
        available=set(strata),
        targets={"multi_animal": 2, "single_animal": 0, "background": 0},
    )["multi_animal"]
    assert result["development_before_holdout_upper_bound"] == 1
    assert result["holdout_upper_bound"] == 0
    assert result["development_minimum_shortfall_before_holdout"] == 1
    assert result["holdout_minimum_shortfall"] == 2


def test_two_splits_cannot_both_claim_the_same_available_images() -> None:
    result = supply_bounds(
        {1: "multi_animal", 2: "multi_animal", 3: "multi_animal"},
        exposed=set(),
        protected=set(),
        available={1, 2},
        targets={"multi_animal": 2, "single_animal": 0, "background": 0},
    )["multi_animal"]
    assert result["holdout_minimum_shortfall"] == 0
    assert result["combined_minimum_shortfall"] == 2


def test_unknown_image_ids_and_negative_targets_are_rejected() -> None:
    targets = {"multi_animal": 1, "single_animal": 0, "background": 0}
    with pytest.raises(Phase1Error, match="outside"):
        supply_bounds({1: "multi_animal"}, {2}, set(), {1}, targets)
    with pytest.raises(Phase1Error, match="nonnegative"):
        supply_bounds({}, set(), set(), set(), targets | {"multi_animal": -1})
