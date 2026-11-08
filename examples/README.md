# Recorded Measurement CSV

[`measurements.csv`](measurements.csv) 是既有新圖推論 run 的完整距離 CSV，直接複製原 bytes。
未重新推論、篩選成功列、加入 GT、四捨五入或改寫資格。
它用於檢查 CSV、兩種距離與候選狀態，是開發圖輸出，不是獨立測試集或新環境執行成績。

## Provenance

| 項目 | 來源 |
| --- | --- |
| Run | `20261108T075732Z_92c42f74bb73` |
| Original artifact | `outputs/inference/20261108T075732Z_92c42f74bb73/measurements.csv` |
| Image | COCO 2017 validation，image_id `125405`；640×428，兩隻狗的模型預測 |
| Conditions | A_r100／A_r050；各 2 筆雙眼＋1 筆跨右眼，共 6 列 |
| Pipeline | YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny；CPU、seed 42、4 threads |
| Measurement definition | 原圖二值眼遮罩的像素中心平均重心，再取歐氏距離，單位 px |
| Reference definition | 無；新圖推論未讀 v1／v2 GT，沒有誤差或準確度欄位 |
| Eligibility | 全部 candidate／unverified，左右與可見性未驗證 |
| CSV SHA-256 | `4a8ea8409a1d63bc5c3efd35b446c5ce17bcbf5f860ef686f5d1132a8b41bf11` |
| Source result.json SHA-256 | `269dade32d6dc3ace244507155b349294635a85caee3b79f5aaf59161ac833ae` |
| Source run_manifest.json SHA-256 | `db28c4f065d7b3630130c0112b6b89e1c5803e0796dd27c49e2f91870a1ff768` |
| Original JPEG SHA-256 | `aeb172b088be9b334a97fcd4df1fe89b5730bc5836765a23c82f2b8b4ff3d03b` |

以上列出可核對的來源識別碼；完整 run 與執行報告僅在本機保存，不納入 Git。
原圖、遮罩及完整 run 不在這個範例包；沒有圖時不能只看數字驗收眼部分割。
模型、重心公式與評估界線見 [Methodology](../docs/methodology.md)。

## Fields

| 欄位 | 意義 |
| --- | --- |
| `condition_id` | ROI 條件；同圖兩條件不是兩個獨立樣本 |
| `measurement_type` | binocular＝同隻雙眼；right_to_right＝同圖不同動物右眼 |
| `prediction_id_a/b` | 此 run 的預測 ID，不是 COCO annotation_id |
| `eye_side_a/b` | 模型解剖學通道，不按畫面 x 重新排序 |
| `cx_a_px/cy_a_px/cx_b_px/cy_b_px` | 正規化原圖上的遮罩重心 |
| `distance_px`、`units` | 重心像素距離，不是瞳孔距離或實體長度 |
| `status`、`qualification` | 幾何可計算與量測資格分開，candidate 不代表合格 |
| `reason` | 缺失端點原因；這六列皆可計算，故留空 |

此 run 早於十類域外試驗政策，原 CSV 沒有 experimental／pose_domain 欄位；保留原樣。
新 schema 以 [API Guide](../docs/api.md) 及當次結果為準。CSV 是 UTF-8 BOM，Python 使用 `utf-8-sig` 讀取。

## Read and Verify

以下只檢查 bytes 與已保存端點的公式，不執行模型，不證明定位／分割正確：

```sh
python3.11 - <<'PY'
import csv
import hashlib
import math
from pathlib import Path

path = Path("examples/measurements.csv")
assert hashlib.sha256(path.read_bytes()).hexdigest() == (
    "4a8ea8409a1d63bc5c3efd35b446c5ce17bcbf5f860ef686f5d1132a8b41bf11"
)
with path.open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 6
for row in rows:
    distance = math.hypot(
        float(row["cx_a_px"]) - float(row["cx_b_px"]),
        float(row["cy_a_px"]) - float(row["cy_b_px"]),
    )
    assert math.isclose(distance, float(row["distance_px"]), abs_tol=1e-12)
    assert row["status"] == "candidate" and row["qualification"] == "unverified"
print("Verified 6 recorded candidate distances; this is not accuracy evaluation.")
PY
```
