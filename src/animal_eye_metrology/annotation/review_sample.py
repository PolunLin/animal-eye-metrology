"""Prepare a reproducible manual-review sample using the existing COCO renderer."""

import argparse
import csv
import json
import os
import platform
import random
import shlex
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.config import ContactSheetConfig, load_config
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import sha256_file
from animal_eye_metrology.data.analysis import REVIEW_MANUAL_FIELDS, analyze
from animal_eye_metrology.data.coco import load_coco
from animal_eye_metrology.paths import PACKAGE_ROOT, package_sources
from animal_eye_metrology.ui.visualization import render_contact_sheets

CATEGORIES = ("zebra", "giraffe", "horse", "dog")


def sample_images(
    candidates: list[dict[str, Any]],
    category_ids: dict[str, int],
    *,
    seed: int = 42,
    per_category: int = 15,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Sample images without replacement; one RNG follows category insertion order."""
    rng = random.Random(seed)
    populations, selected = {}, {}
    for name, category_id in category_ids.items():
        population = sorted(
            {
                int(row["image_id"])
                for row in candidates
                if json.loads(row["category_counts"]).get(str(category_id), 0) >= 2
            }
        )
        populations[name] = population
        selected[name] = rng.sample(population, min(per_category, len(population)))
    return populations, selected


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


GUIDE = """# Phase 1 — Manual Visual Review

## Scope and Sampling

本批只準備人工審核，沒有模型推論、眼部 Ground Truth 或最終類別決定。
每類母體是至少兩個同類非 crowd 實例的影像；以 image ID 排序後抽樣，
使用一個 random.Random(42)，依 zebra → giraffe → horse → dog 順序，每類不放回抽最多 15 張。
抽出後按 image ID 排序顯示。沒有依眼睛清晰度、面積或姿態先挑圖，也沒有替換難圖。
母體、抽出順序、影像 ID、來源雜湊與實際繪圖設定都在 review_manifest.json。
這是四類等額的可行性試審，不能直接當作十類動物或全資料集的無偏估計。

## Review Steps

1. 開啟各類別 contact_sheets，左側是完整原圖，右側是動物 bbox／mask。
   每個實例有自己的顏色、#annotation_id 與下方圖例；引線連回 bbox 左上角。
   顯示全部非 crowd 動物，包含非目標物種，避免誤認實例。圖例不是眼睛標籤。
2. 在 sampled_images.csv 找 image_id、sample_category、target_annotation_ids
   與 original_image_path。
   original_image_path 以 review_manifest.json 的 working_directory（專案根目錄）為基準。
   必須開啟原始影像並以原始像素判讀；contact sheet 縮圖僅供索引。
3. 編輯本目錄的 eye_visibility_review.csv，或先另存人工工作副本。
   以 (image_id, annotation_id) 找動物，不改 ID、類別或計數，不覆寫上層原始審查表。
   所有人工欄位初始留白；空白是未審，不是不可見或不合格。
4. 左／右眼以動物本身的解剖學左右判讀，不能把畫面右邊當成右眼。
   側身、背向或無法判斷左右時記不確定，不推測被遮住的眼睛。
5. 分開記錄「是否能辨識眼睛」與「邊界是否足以可靠標註」。
   眼睛可見不等於能準確分割；COCO 動物 mask 也不是眼部 Ground Truth。
6. 補上 reviewer、review_protocol_version，將難例、遮擋部位、標註困難記在 notes。
   先用少量影像試審統一尺度；正式比較前確認可見性分級、眼部範圍與解析度量法。

## Existing CSV Schema

沿用既有 schema，不新增或改名以下欄位：

| 本次需求 | 現有欄位 | 填寫方式 |
| --- | --- | --- |
| category | animal_category（另有 category_id） | 已由原始資料填妥，不改寫 |
| left_eye_visible | left_eye_visibility | 人工記清楚／部分可見／不可見／不確定；分級需統一 |
| right_eye_visible | right_eye_visibility | 同上，以動物解剖學右眼為準 |
| head_pose | head_pose | 人工記錄正面、斜面、側面、背向或不確定 |
| occlusion | occlusion | 記錄眼睛或頭部的遮擋與程度，不只記身體遮擋 |
| eye_resolution | eye_resolution | 記原始像素下每眼可見區域寬×高、量法及不確定性；不依縮圖猜像素 |
| eligible | measurement_eligibility | 記錄初步可行性及不確定原因，避免把可見直接當成可準確分割 |
| exclusion_reason | exclusion_reason | 如側臉、遮擋、低解析度、左右無法辨識、邊界不清；可複選文字 |

建議 measurement_eligibility 用文字分開記錄 intra（同動物雙眼）與 cross_right（跨動物右眼）可行性，
不把未核准的最低像素門檻或眼睛中心／mask 定義當成既定規範。
本批未自動填入上述判讀，也沒有自動計算合格比例。

## Later Comparison

每張樣本影像也可能包含其他動物，人工 CSV 保留這些非 crowd 實例。
各類別的比較對象必須由 sampled_images.csv 的 sample_category 與 target_annotation_ids 界定，
不能把伴隨物種混入目標類別分母，也不能把全部審查列直接當成四類的目標實例數。

- 實例指標：目標實例中雙眼清楚可辨的數量／已完成審核的目標實例數，同報未完成數。
- 最重要的影像指標：同張圖至少兩個目標實例雙眼清楚可辨的影像數／已完成必要實例審核的樣本影像數。
  各類別共抽最多 15 張，不能把未審完的圖當成失敗；應同報全部抽樣張數與未完成張數。
- 雙眼可見與邊界可標註性、解析度及遮擋需要分開彙整，不能僅看一個 eligible 值。
- 單獨跨動物右眼距離只需要兩個動物的右眼可定位；至少兩個動物皆雙眼清楚，
  才同時提供兩種距離任務所需的完整眼部資訊。中心與分割 Ground Truth 仍需另行確認。
- 模型技術可行性需要後續文獻／官方實作查核；本批沒有執行或評估模型。

完成人工審核後再彙整比較，不自行選定最終動物類別或進入 Phase 2。
"""


def prepare(config_path: Path, destination: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    source = config.output
    output = (destination or source / "visual_review_seed42").resolve()
    if not output.is_relative_to(source) or output == source:
        raise Phase1Error("Review output must be a new subdirectory of the Phase 1 output.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise Phase1Error("Refusing to overwrite an existing review bundle.")
    run = json.loads((source / "run_manifest.json").read_text())
    for name, expected in run["artifacts_sha256"].items():
        if sha256_file(source / name) != expected:
            raise Phase1Error(f"Source output changed: {name}; preserve and reconcile it first.")
    dataset = load_coco(
        config.annotations,
        expected_animal_categories=config.expected_animal_categories,
        expected_sha256=run["dataset"]["annotations_sha256"],
    )
    result = analyze(dataset)
    if result.summary() != run["summary"] or len(dataset.images) != run["dataset"]["total_images"]:
        raise Phase1Error("Current annotation statistics differ from the original run.")
    _, candidate_rows = read_csv(source / "multi_same_category.csv")
    expected_candidates = result.candidates["multi_same_category"]
    if candidate_rows != [
        {key: str(value) for key, value in row.items()} for row in expected_candidates
    ]:
        raise Phase1Error("Candidate CSV does not match current annotations.")
    category_ids = {
        name: next(cat.id for cat in dataset.animal_categories.values() if cat.name == name)
        for name in CATEGORIES
    }
    populations, selected = sample_images(candidate_rows, category_ids)
    union = {key for ids in selected.values() for key in ids}
    review_fields, review_rows = read_csv(source / "eye_visibility_review.csv")
    sampled_review = [row for row in review_rows if int(row["image_id"]) in union]
    if any(row[field] for row in sampled_review for field in REVIEW_MANUAL_FIELDS):
        raise Phase1Error(
            "Source review contains human labels; preserve them before preparing a blank batch."
        )
    expected_review = {(key, ann.id) for key in union for ann in result.eligible_by_image[key]}
    actual_review = {(int(row["image_id"]), int(row["annotation_id"])) for row in sampled_review}
    if actual_review != expected_review or len(sampled_review) != len(expected_review):
        raise Phase1Error("Review rows do not match sampled instances.")
    original_hashes = {path.name: sha256_file(path) for path in source.iterdir() if path.is_file()}
    sheets = ContactSheetConfig(
        enabled=True, columns=1, rows=3, panel_width=640, panel_height=480, max_images=0
    )
    manifest: dict[str, Any] = {
        "dataset": run["dataset"],
        "reconfirmed_summary": result.summary(),
        "working_directory": str(Path.cwd()),
        "python_version": platform.python_version(),
        "seed": 42,
        "per_category_limit": 15,
        "category_order": list(CATEGORIES),
        "sampling_method": (
            "one random.Random(42); sorted image-ID population; sample without replacement; "
            "category order fixed; display sorted IDs"
        ),
        "execution": shlex.join(
            [
                sys.executable,
                "-m",
                "animal_eye_metrology.annotation.review_sample",
                "--config",
                str(config.source),
                "--output",
                str(output),
            ]
        ),
        "eye_based_preselection": False,
        "eye_review_sample": True,
        "manual_results_filled": False,
        "model": None,
        "render_settings": asdict(sheets),
        "instance_legend": True,
        "unique_sample_images": len(union),
        "category_image_assignments": sum(map(len, selected.values())),
        "manual_review_rows": len(sampled_review),
        "source_artifacts_sha256": original_hashes,
        "source_code_sha256": {
            path.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(path)
            for path in package_sources()
        },
        "configuration_sha256": sha256_file(config.source),
        "categories": {},
    }
    with tempfile.TemporaryDirectory(prefix=".visual-review-", dir=source) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        image_list = []
        for name in CATEGORIES:
            ids = sorted(selected[name])
            directory = stage / name
            directory.mkdir()
            subset = replace(
                result,
                candidates={
                    **result.candidates,
                    "multi_any": [
                        row for row in result.candidates["multi_any"] if row["image_id"] in ids
                    ],
                },
            )
            render_config = replace(config, contact_sheets=sheets)
            contact = render_contact_sheets(
                dataset, subset, render_config, directory, instance_legend=True
            )
            contact.update(
                {
                    "selection": "preselected_random_sample_without_replacement",
                    "random_seed": 42,
                    "eye_review_sample": True,
                    "selection_note": (
                        "sample_images selected the input list; "
                        "renderer applied no further sampling"
                    ),
                }
            )
            if contact["selected_image_ids"] != ids:
                raise Phase1Error("Rendered images differ from random sample.")
            target_instances = 0
            for index, key in enumerate(ids):
                anns = result.eligible_by_image[key]
                target_ids = [ann.id for ann in anns if ann.category_id == category_ids[name]]
                target_instances += len(target_ids)
                page = contact["pages"][index // (sheets.rows * sheets.columns)]
                image_list.append(
                    {
                        "sample_category": name,
                        "category_id": category_ids[name],
                        "image_id": key,
                        "file_name": dataset.images[key].file_name,
                        "target_animal_count": len(target_ids),
                        "animal_count": len(anns),
                        "target_annotation_ids": json.dumps(target_ids),
                        "annotation_ids": json.dumps([ann.id for ann in anns]),
                        "seed": 42,
                        "draw_rank": selected[name].index(key) + 1,
                        "display_order": index + 1,
                        "contact_sheet": f"{name}/{page}",
                        "original_image_path": os.path.relpath(
                            config.images / dataset.images[key].file_name
                        ),
                    }
                )
            manifest["categories"][name] = {
                "category_id": category_ids[name],
                "population_image_ids": populations[name],
                "population_images": len(populations[name]),
                "draw_order": selected[name],
                "sampled_image_ids": ids,
                "sample_images": len(ids),
                "target_instances": target_instances,
                "all_instances_in_sampled_images": sum(
                    len(result.eligible_by_image[key]) for key in ids
                ),
                "contact_sheets": contact,
            }
        write_csv(stage / "sampled_images.csv", list(image_list[0]), image_list)
        write_csv(stage / "eye_visibility_review.csv", review_fields, sampled_review)
        table = (
            "\n## Batch Counts\n\n"
            "| 類別 | 母體影像 | 抽樣影像 | 目標實例 | 圖內全部實例 |\n"
            "| --- | ---: | ---: | ---: | ---: |\n"
        )
        for name, record in manifest["categories"].items():
            table += (
                f"| {name} | {record['population_images']} | {record['sample_images']} "
                f"| {record['target_instances']} | {record['all_instances_in_sampled_images']} |\n"
            )
        (stage / "REVIEW_GUIDE.md").write_text(GUIDE + table, encoding="utf-8")
        manifest["artifacts_sha256"] = {
            str(path.relative_to(stage)): sha256_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        (stage / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name, expected in original_hashes.items():
            if sha256_file(source / name) != expected:
                raise Phase1Error(f"Original analysis artifact changed: {name}")
        stage.rename(output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    try:
        manifest = prepare(options.config, options.output)
    except (OSError, ValueError, StopIteration) as error:
        print(
            json.dumps({"event": "review_preparation_failed", "error": str(error)}), file=sys.stderr
        )
        return 2
    print(
        json.dumps(
            {
                "event": "review_prepared",
                "unique_images": manifest["unique_sample_images"],
                "review_rows": manifest["manual_review_rows"],
                "seed": manifest["seed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
