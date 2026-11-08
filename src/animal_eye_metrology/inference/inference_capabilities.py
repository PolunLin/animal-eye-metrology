"""Single source for enabled COCO categories and the HRNet training-domain distinction.

Enabling inference is not a claim of localization, segmentation, side-assignment,
visibility, or measurement accuracy. Bird inference deliberately tests transfer
outside the AP-10K mammal training domain and remains explicitly experimental.
"""

from typing import Any

COCO_ANIMAL_CATEGORIES = (
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
)
AP10K_MAMMAL_CATEGORIES = frozenset(COCO_ANIMAL_CATEGORIES) - {"bird"}
CAPABILITY_POLICY_VERSION = "coco10_ap10k_bird_experimental_v1"
AP10K_REPOSITORY = "https://github.com/AlexTheBad/AP-10K"
AP10K_PAPER = "https://arxiv.org/abs/2108.12617"


def category_capability(category: str) -> dict[str, Any]:
    """Return execution scope independently from empirical model correctness."""
    if category == "bird":
        return {
            "supported": True,
            "pose_domain": "out_of_training_domain",
            "experimental": True,
            "capability_reason": "bird_not_in_ap10k_mammal_training_domain",
        }
    if category in AP10K_MAMMAL_CATEGORIES:
        return {
            "supported": True,
            "pose_domain": "ap10k_mammal_mapping",
            "experimental": False,
            "capability_reason": "dataset_category_mapping_only_accuracy_unverified",
        }
    return {
        "supported": False,
        "pose_domain": "unsupported",
        "experimental": False,
        "capability_reason": "category_outside_coco_animal_scope",
    }


def validate_category_scope(categories: object) -> None:
    """Require all ten COCO animal labels once; never silently narrow the upload scope."""
    if (
        not isinstance(categories, list)
        or len(categories) != len(COCO_ANIMAL_CATEGORIES)
        or not all(isinstance(category, str) for category in categories)
        or set(categories) != set(COCO_ANIMAL_CATEGORIES)
    ):
        raise ValueError("Inference scope must contain all ten COCO animal categories once.")


def capability_policy() -> dict[str, Any]:
    """Serializable provenance for each new run; historical run files stay unchanged."""
    return {
        "version": CAPABILITY_POLICY_VERSION,
        "enabled_categories": list(COCO_ANIMAL_CATEGORIES),
        "experimental_categories": ["bird"],
        "category_capabilities": {
            category: category_capability(category) for category in COCO_ANIMAL_CATEGORIES
        },
        "measurement_qualification": "unverified",
        "accuracy_validated_for_all_categories": False,
        "ap10k_sources": [AP10K_REPOSITORY, AP10K_PAPER],
    }
