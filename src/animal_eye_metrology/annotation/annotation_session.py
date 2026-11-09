"""Prepare an independent human relabeling session; never generate eye annotations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from animal_eye_metrology.annotation.eye_annotations import (
    V2_PROTOCOL_VERSION,
    initialize_store,
    latest_annotation,
    load_annotations,
    load_targets,
)
from animal_eye_metrology.core.config import load_config
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.data.coco import CocoDataset, load_coco

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnnotationSession:
    source: Path
    phase1_config: Path
    source_proposal: Path
    directory: Path
    protocol_document: Path
    protocol_version: str
    expected_images: int
    expected_instances: int
    expected_eyes: int

    @property
    def proposal(self) -> Path:
        return self.directory / "inputs/proposed_instances.csv"

    @property
    def ground_truth(self) -> Path:
        return self.directory / "ground_truth"

    @property
    def images(self) -> Path:
        return self.directory / "images"


def read_session(path: Path) -> AnnotationSession:
    path = path.expanduser().resolve()
    with path.open("rb") as handle:
        settings = tomllib.load(handle)
    paths = {"phase1_config", "proposal_csv", "directory", "protocol_document"}
    counts = {"expected_images", "expected_instances", "expected_eyes"}
    if set(settings) != paths | counts | {"protocol_version"}:
        raise Phase1Error("獨立重標設定欄位不符。")
    if settings["protocol_version"] != V2_PROTOCOL_VERSION:
        raise Phase1Error("此獨立重標工作區只接受已核准的 v2 規範。")
    if any(type(settings[key]) is not int or settings[key] < 1 for key in counts):
        raise Phase1Error("重標預期數量必須為正整數。")
    if settings["expected_eyes"] != 2 * settings["expected_instances"]:
        raise Phase1Error("重標範圍需包含每隻動物的兩個眼側槽位。")
    if any(not isinstance(settings[key], str) or not settings[key] for key in paths):
        raise Phase1Error("重標路徑不得為空。")
    resolved = {key: (path.parent / settings[key]).resolve() for key in paths}
    data = load_config(resolved["phase1_config"])
    destination = resolved["directory"]
    phase3 = (data.output.parent / "phase3").resolve()
    if destination == phase3 or not destination.is_relative_to(phase3):
        raise Phase1Error("請使用獨立的 Phase 3 重標目錄。")
    if any(
        source.is_relative_to(destination)
        for source in [path, *resolved.values()]
        if source != destination
    ):
        raise Phase1Error("重標目錄不能包含來源設定、清單或規範。")
    return AnnotationSession(
        path,
        resolved["phase1_config"],
        resolved["proposal_csv"],
        destination,
        resolved["protocol_document"],
        settings["protocol_version"],
        *(settings[key] for key in ("expected_images", "expected_instances", "expected_eyes")),
    )


def _dataset(session: AnnotationSession) -> CocoDataset:
    data = load_config(session.phase1_config)
    digest = data.expected_annotations_sha256
    manifest = data.output / "run_manifest.json"
    if not digest and manifest.exists():
        digest = json.loads(manifest.read_text())["dataset"]["annotations_sha256"]
    return load_coco(
        data.annotations,
        expected_sha256=digest,
        expected_animal_categories=data.expected_animal_categories,
    )


def _check_scope(session: AnnotationSession, targets: list[dict[str, str]]) -> None:
    if (
        len(targets) != session.expected_instances
        or len({r["image_id"] for r in targets}) != session.expected_images
        or 2 * len(targets) != session.expected_eyes
    ):
        raise Phase1Error("圖片／動物／眼睛槽位數與已核准清單範圍不同。")


def verify_session(session: AnnotationSession) -> dict[str, Any]:
    """Check immutable prepared inputs only; live human revisions are separate."""
    path = session.directory / "preparation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["protocol_version"] != session.protocol_version or manifest[
        "config_sha256"
    ] != sha256_file(session.source):
        raise Phase1Error("重標工作區的設定或規範版本不符。")
    for name, digest in manifest["files_sha256"].items():
        candidate = (session.directory / name).resolve()
        if Path(name).is_absolute() or not candidate.is_relative_to(session.directory):
            raise Phase1Error("重標來源路徑超出工作區。")
        if not candidate.is_file() or sha256_file(candidate) != digest:
            raise Phase1Error(f"重標輸入缺失或雜湊不符：{name}")
    if sha256_file(session.protocol_document) != manifest["files_sha256"]["protocol.md"]:
        raise Phase1Error("v2 規範已變更，請先核對版本；不能混用新的規則與已準備工作區。")
    return manifest


def prepare(session: AnnotationSession) -> dict[str, Any]:
    if session.directory.exists():
        raise Phase1Error("重標目錄已存在，拒絕覆寫；請使用 --status 繼續既有工作。")
    dataset = _dataset(session)
    data = load_config(session.phase1_config)
    targets = load_targets(session.source_proposal, dataset)
    _check_scope(session, targets)
    protocol = session.protocol_document.read_text(encoding="utf-8")
    if session.protocol_version not in protocol:
        raise Phase1Error("規範文件未標明相同 protocol_version。")
    session.directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".eye-relabel-", dir=session.directory.parent
    ) as temporary:
        stage = Path(temporary) / "prepared"
        (stage / "inputs").mkdir(parents=True)
        (stage / "images").mkdir()
        shutil.copyfile(session.source_proposal, stage / "inputs/proposed_instances.csv")
        shutil.copyfile(session.protocol_document, stage / "protocol.md")
        for target in targets:
            filename = target["file_name"]
            source = (data.images / filename).resolve()
            if Path(filename).name != filename or not source.is_relative_to(data.images):
                raise Phase1Error("原圖檔名必須位於指定資料目錄。")
            if sha256_file(source) != target["image_sha256"]:
                raise Phase1Error(f"原圖與核准清單雜湊不一致：{filename}")
            destination = stage / "images" / filename
            if not destination.exists():
                shutil.copyfile(source, destination)
        cases = [
            {
                "order": index * 2 + offset + 1,
                "image_id": target["image_id"],
                "annotation_id": target["annotation_id"],
                "animal_category": target["animal_category"],
                "eye_side": side,
                "protocol_version": session.protocol_version,
                "scope_status": "scheduled_not_annotated",
            }
            for index, target in enumerate(targets)
            for offset, side in enumerate(("left", "right"))
        ]
        with (stage / "relabel_scope.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
        # Only an empty header + source identity; never placeholders for accepted GT.
        initialize_store(
            stage / "ground_truth",
            targets,
            dataset.source_sha256,
            protocol_version=session.protocol_version,
        )
        inputs = {
            p.relative_to(stage).as_posix(): sha256_file(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file() and not p.is_relative_to(stage / "ground_truth")
        }
        manifest = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "protocol_version": session.protocol_version,
            "config_sha256": sha256_file(session.source),
            "dataset_annotations_sha256": dataset.source_sha256,
            "expected_images": session.expected_images,
            "expected_instances": session.expected_instances,
            "expected_eyes": session.expected_eyes,
            "initial_annotation_count": 0,
            "model_predictions_loaded": False,
            "prior_masks_loaded": False,
            "independent_validation_set": False,
            "files_sha256": inputs,
        }
        (stage / "preparation_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stage.rename(session.directory)
    return manifest


def status(session: AnnotationSession) -> dict[str, Any]:
    manifest = verify_session(session)
    dataset = _dataset(session)
    if dataset.source_sha256 != manifest["dataset_annotations_sha256"]:
        raise Phase1Error("COCO 來源與重標準備時不同。")
    targets = load_targets(session.proposal, dataset)
    _check_scope(session, targets)
    rows = load_annotations(
        session.ground_truth, targets, protocol_version=session.protocol_version
    )
    cases = []
    for target in targets:
        for side in ("left", "right"):
            key = int(target["image_id"]), int(target["annotation_id"]), side
            row = latest_annotation(rows, key)
            eligible = bool(
                row
                and row["visibility"] == "Visible"
                and row["side_confirmed"] == "Yes"
                and row["mask_review_status"] == "Accepted"
                and row["measurement_quality"] == "Pass"
                and row["mask_path"]
            )
            cases.append(
                {
                    "image_id": key[0],
                    "annotation_id": key[1],
                    "eye_side": side,
                    "saved": row is not None,
                    "eligible_for_freeze": eligible,
                    "review_status": row["mask_review_status"] if row else "not_annotated",
                    "reason": row["exclusion_reason"] if row else "等待人工描繪與審核",
                }
            )
    eligible_count = sum(row["eligible_for_freeze"] for row in cases)
    return {
        "protocol_version": session.protocol_version,
        "expected_eyes": len(cases),
        "saved_eyes": sum(row["saved"] for row in cases),
        "eligible_eyes": eligible_count,
        "ready_for_freeze": eligible_count == len(cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        session = read_session(args.config)
        result = prepare(session) if args.prepare else status(session)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        LOGGER.error(
            json.dumps(
                {"event": "annotation_session_failed", "reason": str(error)}, ensure_ascii=False
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
