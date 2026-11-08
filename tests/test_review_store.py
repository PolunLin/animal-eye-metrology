"""Protect human work across restarts, stale windows, and duplicate CSV records."""

import csv
from pathlib import Path

import pytest

from animal_eye_metrology.analysis import REVIEW_FIELDS, analyze
from animal_eye_metrology.coco import load_coco
from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.review_store import (
    image_eye_condition,
    initialize_image_selections,
    initialize_reviews,
    is_reviewed,
    load_image_selections,
    load_reviews,
    review_key,
    save_image_selection,
    save_review,
)

DECISIONS = {
    "left_eye_visibility": "Yes",
    "right_eye_visibility": "Uncertain",
    "head_pose": "Front",
    "occlusion": "Partial",
    "measurement_eligibility": "Uncertain",
}


def test_csv_roundtrip_preserves_schema_other_records_and_unedited_fields(
    local_config: Path,
) -> None:
    source = local_config.parent / "annotations.json"
    before = source.read_bytes()
    template = analyze(load_coco(source)).review
    path = local_config.parent / "working.csv"
    initial = initialize_reviews(path, template)
    first = save_review(
        path,
        template,
        review_key(initial[0]),
        DECISIONS
        | {
            "notes": "Synthetic test notes, with comma\nand newline",
            "eye_resolution": "manual width pending",
        },
        expected=initial[0],
    )
    save_review(path, template, review_key(first), {"notes": "Edited once"}, expected=first)
    loaded = load_reviews(path, template)
    assert len(loaded) == len({review_key(row) for row in loaded}) == 7
    assert loaded[0]["notes"] == "Edited once"
    assert loaded[0]["eye_resolution"] == "manual width pending"
    assert is_reviewed(loaded[0]) and not is_reviewed(loaded[1])
    assert loaded[1:] == initial[1:]
    with path.open(newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames or ()) == REVIEW_FIELDS
    assert source.read_bytes() == before


def test_stale_windows_merge_different_instances_and_reject_same_instance(
    local_config: Path,
) -> None:
    template = analyze(load_coco(local_config.parent / "annotations.json")).review
    path = local_config.parent / "working.csv"
    viewed = initialize_reviews(path, template)
    save_review(path, template, review_key(viewed[0]), DECISIONS, expected=viewed[0])
    save_review(
        path,
        template,
        review_key(viewed[1]),
        DECISIONS | {"notes": "Second window"},
        expected=viewed[1],
    )
    before = path.read_bytes()
    with pytest.raises(Phase1Error, match="另一個視窗"):
        save_review(path, template, review_key(viewed[0]), DECISIONS, expected=viewed[0])
    assert path.read_bytes() == before
    assert sum(is_reviewed(row) for row in load_reviews(path, template)) == 2


def test_image_selection_roundtrip_never_changes_existing_eye_review(local_config: Path) -> None:
    template = analyze(load_coco(local_config.parent / "annotations.json")).review
    eye_path = local_config.parent / "working.csv"
    old = initialize_reviews(eye_path, template)
    save_review(eye_path, template, review_key(old[0]), DECISIONS, expected=old[0])
    eye_bytes = eye_path.read_bytes()
    path = local_config.parent / "image_selection.csv"
    ids = [10, 20, 50]
    initial = initialize_image_selections(path, ids)
    assert all(row["selection_status"] == "Pending" for row in initial)
    saved = save_image_selection(
        path, ids, 10, "Selected", "Keep, with note\nsecond line", expected=initial[0]
    )
    save_image_selection(path, ids, 20, "Rejected", "Not suitable", expected=initial[1])
    saved = save_image_selection(path, ids, 10, "Pending", saved["review_notes"], expected=saved)
    save_image_selection(path, ids, 10, "Selected", saved["review_notes"], expected=saved)
    loaded = load_image_selections(path, ids)
    assert [row["selection_status"] for row in loaded] == ["Selected", "Rejected", "Pending"]
    assert len(loaded) == len({row["image_id"] for row in loaded}) == 3
    assert loaded[0]["review_notes"] == "Keep, with note\nsecond line"
    assert eye_path.read_bytes() == eye_bytes


def test_image_selection_rejects_stale_same_image_and_merges_other_images(
    local_config: Path,
) -> None:
    path = local_config.parent / "image_selection.csv"
    ids = [10, 20]
    viewed = initialize_image_selections(path, ids)
    save_image_selection(path, ids, 10, "Selected", "First", expected=viewed[0])
    save_image_selection(path, ids, 20, "Selected", "Second", expected=viewed[1])
    before = path.read_bytes()
    with pytest.raises(Phase1Error, match="另一個視窗"):
        save_image_selection(path, ids, 10, "Rejected", "Stale", expected=viewed[0])
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["duplicate", "status", "schema"])
def test_invalid_image_selection_csv_is_preserved(local_config: Path, damage: str) -> None:
    path = local_config.parent / "image_selection.csv"
    path.write_text("image_id,selection_status,review_notes\n10,Pending,\n")
    if damage == "duplicate":
        path.write_text(path.read_text() + "10,Selected,duplicate\n")
    elif damage == "status":
        path.write_text(path.read_text().replace("Pending", "unknown"))
    else:
        path.write_text("image_id,annotation_id\n10,1001\n")
    before = path.read_bytes()
    with pytest.raises(Phase1Error):
        initialize_image_selections(path, [10, 20])
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("critical_values", "expected"),
    [
        ([("", "", ""), ("", "", "")], "Pending"),
        ([("No", "Yes", "No"), ("", "", "")], "Pending"),
        ([("Yes", "Yes", "Yes"), ("Uncertain", "Yes", "Yes")], "Pending"),
        ([("No", "Yes", "No"), ("Yes", "No", "No")], "Insufficient"),
        ([("Yes", "Yes", "Yes"), ("Yes", "Yes", "Yes"), ("", "", "")], "Ready"),
    ],
)
def test_image_eye_condition_keeps_unknowns_pending_and_allows_early_success(
    critical_values: list[tuple[str, str, str]],
    expected: str,
) -> None:
    rows = [
        DECISIONS
        | {
            "image_id": "10",
            "annotation_id": str(index),
            "left_eye_visibility": values[0],
            "right_eye_visibility": values[1],
            "measurement_eligibility": values[2],
        }
        for index, values in enumerate(critical_values)
    ]
    assert image_eye_condition(rows)["condition"] == expected


@pytest.mark.parametrize("damage", ["duplicate", "metadata", "schema"])
def test_bad_saved_csv_is_rejected_without_overwriting(local_config: Path, damage: str) -> None:
    template = analyze(load_coco(local_config.parent / "annotations.json")).review
    path = local_config.parent / "working.csv"
    rows = initialize_reviews(path, template)
    fields = list(REVIEW_FIELDS)
    if damage == "duplicate":
        rows.append(rows[0])
    elif damage == "metadata":
        rows[0]["animal_category"] = "wrong_species"
    else:
        fields = fields[:-1]
        rows = [{key: value for key, value in row.items() if key in fields} for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_bytes()
    with pytest.raises(Phase1Error):
        initialize_reviews(path, template)
    assert path.read_bytes() == before
