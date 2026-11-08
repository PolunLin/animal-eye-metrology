# Manual Eye Mask Measurement Baseline

## Result and Scope

依使用者核准，凍結第一輪 3 張圖片、6 隻動物、12 隻眼睛的最新 Accepted／Pass 標註，
建立人工 Mask 重心的二維像素距離參考。本次產生 12 個重心與 9 筆距離：
6 組同隻動物雙眼距離、3 組同圖兩隻動物右眼距離，均為 Valid。

這些數值由人工可見眼部 Mask 衍生，不是眼睛模型成績、獨立 Landmark GT、
瞳孔／眼球解剖中心，或經校正的實際長度。沒有執行眼睛模型或重標實驗。

## Frozen Annotation Versions

快照位於 outputs/phase3/manual_measurement_baseline_v1/frozen/。
擷取時來源工作 CSV 共 54 個歷史版本，凍結清單僅含每眼最新的 12 個版本。
每筆均有 Visible、側別 Yes、Accepted、Pass、Reviewer 與非空原圖 Mask。
若最新版本未完成，程式停止，不回退到較早的 Accepted 版本。

| Image ID | 動物 | Annotation ID | Left 版本 | Right 版本 |
| --- | --- | --- | ---: | ---: |
| 125405 | dog | 13692 | 6 | 8 |
| 125405 | dog | 17207 | 9 | 7 |
| 20247 | bear | 587181 | 4 | 3 |
| 20247 | bear | 588042 | 3 | 1 |
| 333069 | giraffe | 599488 | 5 | 4 |
| 333069 | giraffe | 599888 | 3 | 1 |

凍結時在現有標註檔案鎖內核對並複製：12 張 PNG、相對應的 12 份版本 JSON、
選定版本 CSV、來源 manifest、試標清單，以及三張原圖。
PNG／JPEG 保持原始 bytes，原始工作目錄不改寫；快照每檔保存 SHA-256。
完整 snapshot_manifest.json 的 digest 亦寫入每列重心、距離及 run_manifest.json，
識別的是這批固定版本，不隨之後新增人工修訂而改變。

COCO 原始註記 digest 與 Phase 1 紀錄相符；未獨立驗證上游真實性，也未將此小樣本當獨立測試集。
三張圖片沿用既有 Phase 3A／3B 範圍，沒有重新抽樣，random seed 不適用。

## Method

輸入為原圖尺寸的 L-mode PNG，值只允許 0／255；載入後轉為二值 M。
座標原點為原圖左上外框，x 向右、y 向下，第 row=r、column=c 個像素中心是 (c+0.5,r+0.5)。

```text
N = sum(M)
cx = sum((column + 0.5) * M) / N
cy = sum((row + 0.5) * M) / N
d_px = sqrt((cx_a - cx_b)^2 + (cy_a - cy_b)^2)
```

重心直接由完整原圖 Mask 的前景像素計算，不需依賴標註時的顯示倍率。
計算使用 float64，途中不四捨五入；CSV 保存計算值，圖與表僅顯示到三位小數。
小數位數是顯示格式，不代表已證明的量測精度。

數學函式對空 Mask 回傳未定義中心／距離，不填 0；兩個相同有效重心的距離則可合法為 0。
本次凍結只接受非空已通過標註，因此沒有未定義輸出。

| 配對 | 規則 |
| --- | --- |
| binocular | 同 image_id、同 annotation_id 的解剖 left 與 right |
| right_eye_pair | 同 image_id、不同 annotation_id 的 right；每個無序動物配對只產生一次 |

依 ID 配對，不按畫面左右或預測順序重指定側別；不跨圖片連線。
三張圖片原始尺寸不同，距離皆為各自原圖上的 px，不將它們當成實際身體尺寸比較。

## Measured Values

| Image ID | 動物 A／B 的 Annotation ID | A 雙眼距離 px | B 雙眼距離 px | A 右眼至 B 右眼 px |
| --- | --- | ---: | ---: | ---: |
| 125405 | 13692／17207 | 34.043 | 25.166 | 275.428 |
| 20247 | 587181／588042 | 28.368 | 25.061 | 385.235 |
| 333069 | 599488／599888 | 32.844 | 23.325 | 145.726 |

這裡的 A／B 只依 annotation_id 排序，方便讀圖，不代表前後、左右或物種等級。
完整端點座標、左右眼、各眼版本與 snapshot digest 均在 measurements.csv；
每眼 Mask 面積為 13–42 個原圖像素，邊界變動可能影響重心，尚未估計人工再標不確定性。

## Artifacts and Visual Review

產物集中於 outputs/phase3/manual_measurement_baseline_v1/。

| 檔案／目錄 | 內容 |
| --- | --- |
| frozen/snapshot_manifest.json | 來源 digest、12 個固定版本、快照每檔 digest |
| frozen/eye_mask_annotations.csv | 選定的 12 筆原有審核列，原始 Schema 與欄位值不變 |
| frozen/masks/、frozen/revisions/ | 12 個原始 Mask bytes 與對應版本 metadata |
| frozen/images/ | 3 張原圖副本，與來源 bytes 一致 |
| frozen/source_manifest.json、frozen/proposed_instances.csv | 來源座標慣例、試標身份、尺寸與影像 digest |
| eye_centers.csv | 12 個原圖重心、前景面積、版本及 Mask／snapshot digest |
| measurements.csv | 9 筆距離、各端點側別／版本／座標、單位與狀態 |
| summary.json、run_manifest.json | 數量、方法、環境、執行命令、設定與產物 digest |
| provenance/ | 此次計算／繪圖程式、原始設定檔及 3A／3B 操作規範副本 |
| overlays/*.png | 原圖／人工眼部 Mask 與重心／重心距離三欄對照 |

eye_centers.csv 的 mask_path 相對於 baseline 目錄，包含 frozen/ 前綴；
frozen/eye_mask_annotations.csv 則沿用原有相對於 frozen/ 的 mask_path。
原始歷史版本仍保留在 ground_truth/，不將其全部混入此固定版本 CSV。

請開啟以下原圖對照檢查身份與連線：

- [125405 dogs](../outputs/phase3/manual_measurement_baseline_v1/overlays/000000125405.png)
- [20247 bears](../outputs/phase3/manual_measurement_baseline_v1/overlays/000000020247.png)
- [333069 giraffes](../outputs/phase3/manual_measurement_baseline_v1/overlays/000000333069.png)

中欄標記重心，L 為黃色、R 為藍色；右欄黃色為雙眼連線、綠色為兩動物右眼連線。
圖下列出 A／B 對應的動物 ID、各眼版本、重心與距離；圖上的標記只是顯示，沒有更改原始 Mask。
三張 PNG 已實際開啟檢視，圖例與原圖範圍完整。

## Execution and Reproducibility

本次從 repository 根目錄執行：

```sh
.venv/bin/python -m animal_eye_metrology.measurement_baseline \
  --config configs/measurement_baseline.toml

.venv/bin/python -m animal_eye_metrology.measurement_baseline \
  --verify outputs/phase3/manual_measurement_baseline_v1
```

建立命令拒絕覆寫既有基準。若未來核准新標註版本，複製設定檔並將 output 改成新的目錄。
唯讀 --verify 只讀取固定快照與基準產物，重新計算 12 個重心與 9 筆距離，不讀目前的人工工作 CSV，
也不需要原始 COCO JSON／原圖目錄。它同時核對保存的檔案 digest 與 CSV 逐列計算值。
provenance/ 中的設定副本用於稽核，原始相對路徑仍以 configs/ 為基準；重現現成基準請用 --verify。

本次唯讀重算通過，snapshot_sha256 為：

```text
3a3b8b98fba72ce79fd8416138ecd62c26051e528844f9d0ff70fdcacc22af13
```

版本資訊包含 Python、NumPy、Pillow 與平台；同環境重算相符，跨平台／版本的逐位元一致性未驗證。
快照 digest 用於完整性與版本辨識，不是標註解剖學正確性的獨立認證。

## Implementation Files

| 檔案 | 變更 |
| --- | --- |
| src/animal_eye_metrology/metrology.py | 新增二值 Mask 重心、歐氏距離與依身份配對 |
| src/animal_eye_metrology/measurement_baseline.py | 新增版本凍結、CSV 輸出與唯讀重算驗證 |
| src/animal_eye_metrology/visualization.py | 沿用既有繪圖工具，新增三欄量測對照圖 |
| configs/measurement_baseline.toml | 新增來源、筆數與輸出目錄設定 |
| tests/test_metrology.py | 新增五個幾何與凍結流程測試 |
| README.md | 更新目前進度與驗證指令 |
| docs/manual_measurement_baseline.md | 新增本次方法、結果與操作紀錄 |

## Tests and Integrity

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests app
.venv/bin/ruff format --check src/animal_eye_metrology/metrology.py \
  src/animal_eye_metrology/measurement_baseline.py \
  src/animal_eye_metrology/visualization.py tests/test_metrology.py
```

87 tests passed，53 個既有上游類型的 pycocotools／NumPy 棄用警告；Ruff 通過。
新增五個測試涵蓋已知重心、單像素／空 Mask、3–4–5 距離、平移、同圖正確配對、重複／缺眼、
凍結後來源修改仍可重算、重複執行不覆寫、快照損壞檢出，以及最新 Pending 不回退歷史版本。
測試只用人工幾何與合成標註，不能當成動物模型評估。

本次執行核對 305 個原有受保護檔案，digest 全部不變，包括人工標註與歷史版本、
Phase 1／2／3A 產物、官方註記、AGENTS.md、Phase 01.md、pyproject.toml 與 uv.lock。
沒有新增相依套件、眼睛模型、UI、訓練、Commit 或 Push。

## Next Review

先由負責人檢查三張對照圖中的眼側、動物身份與連線。
下一步可在獨立目錄盲重標 125405 的四眼，估計同一人再標時 Mask／重心／距離的變動。
既有歷史修訂不能當成已完成獨立再標。本次尚未執行該實驗，也未啟動眼睛模型 POC。
