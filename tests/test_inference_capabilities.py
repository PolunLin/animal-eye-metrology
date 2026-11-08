"""Policy scope is distinct from measured accuracy; these tests do not benchmark models."""

import pytest

from animal_eye_metrology.inference.inference_capabilities import (
    AP10K_MAMMAL_CATEGORIES,
    COCO_ANIMAL_CATEGORIES,
    capability_policy,
    category_capability,
    validate_category_scope,
)


def test_ten_categories_have_one_explicit_experimental_domain() -> None:
    assert len(COCO_ANIMAL_CATEGORIES) == 10
    assert len(AP10K_MAMMAL_CATEGORIES) == 9
    for category in COCO_ANIMAL_CATEGORIES:
        capability = category_capability(category)
        assert capability["supported"]
        assert capability["experimental"] is (category == "bird")
        assert capability["pose_domain"] == (
            "out_of_training_domain" if category == "bird" else "ap10k_mammal_mapping"
        )
    policy = capability_policy()
    assert policy["accuracy_validated_for_all_categories"] is False
    assert policy["measurement_qualification"] == "unverified"


@pytest.mark.parametrize("category", ["rabbit", "fish", "lizard", "person", "unknown"])
def test_unknown_category_is_not_claimed_as_supported(category: str) -> None:
    assert category_capability(category) == {
        "supported": False,
        "pose_domain": "unsupported",
        "experimental": False,
        "capability_reason": "category_outside_coco_animal_scope",
    }


@pytest.mark.parametrize(
    "scope",
    [
        [],
        ["dog", "bear", "giraffe"],
        [*COCO_ANIMAL_CATEGORIES, "bird"],
        ["rabbit", *COCO_ANIMAL_CATEGORIES[1:]],
        "all",
        [1] * 10,
    ],
)
def test_invalid_or_narrowed_category_scope_cannot_silently_change_execution(scope: object) -> None:
    with pytest.raises(ValueError, match="all ten COCO"):
        validate_category_scope(scope)


def test_scope_order_is_not_a_hidden_gate_and_policy_returns_independent_values() -> None:
    validate_category_scope(list(reversed(COCO_ANIMAL_CATEGORIES)))
    first = capability_policy()
    first["enabled_categories"].clear()
    first["category_capabilities"]["bird"]["experimental"] = False
    second = capability_policy()
    assert len(second["enabled_categories"]) == 10
    assert second["category_capabilities"]["bird"]["experimental"] is True
