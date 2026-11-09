"""Fresh assessment groups must not reuse any previously used group member."""

import pytest

pytest.importorskip("pycocotools")

from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.data.training_recovery_data import select_unused_groups


def test_used_nonrepresentative_member_excludes_whole_group_and_order_is_stable():
    groups = [
        {"group_id": "a", "members": [{"id": 1}, {"id": 2}]},
        {"group_id": "b", "members": [{"id": 3}]},
        {"group_id": "c", "members": [{"id": 4}]},
    ]
    chosen = select_unused_groups(groups, {2}, 2)
    assert {group["group_id"] for group in chosen} == {"b", "c"}
    assert chosen == select_unused_groups(groups[::-1], {2}, 2)
    with pytest.raises(Phase1Error, match="Insufficient"):
        select_unused_groups(groups, {2}, 3)
