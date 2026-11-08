# Four-Eye Repeatability Results

## Findings

已完成圖片 125405、兩隻狗、四眼的同人再標比較。兩輪均由 Polun 審核，
選定版本全部 Visible、side_confirmed=Yes、Accepted、Pass，皆有非空、完整封閉的原圖 Mask。
四眼完整配對，沒有遺漏、跨圖或交換解剖眼側。沒有修改兩輪人工結果或替使用者選擇較正確的一輪。

平均 IoU 為 0.652、Dice 為 0.775，平均重心位移 0.447 px，最大 0.819 px。
兩筆雙眼距離的變化為 −0.349、−0.949 px；兩動物右眼距離變化 +0.331 px。
這是四眼的描述性再標差異，不是模型準確率，也沒有預先核准的通過門檻。

## Frozen Inputs

第一輪使用已驗收的 manual_measurement_baseline_v1 固定快照，不重新讀取第一輪工作 CSV 的新版本。
第二輪沿用 measurement_baseline 程式，在檔案鎖內選取最新版本並核對 CSV、版本 JSON、
PNG 尺寸／編碼／digest、原圖與規範，再另建 measurement_baseline_v1，未覆寫工作目錄。
最初讀取時第二輪仍是 Pending／Unknown；使用者重新保存後已再次讀取，正式比較僅使用下列 Accepted／Pass 版本。

| Image ID | Annotation ID | 解剖眼側 | 第一輪版本 | 第二輪版本 |
| --- | --- | --- | ---: | ---: |
| 125405 | 13692 | left | 6 | 3 |
| 125405 | 13692 | right | 8 | 3 |
| 125405 | 17207 | left | 9 | 3 |
| 125405 | 17207 | right | 7 | 2 |

| 項目 | 值 |
| --- | --- |
| Dataset | COCO 2017 validation；沿用 Phase 1 固定來源 |
| Annotation JSON SHA-256 | e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f |
| Protocol | eye-visible-aperture-v1 |
| 第一輪 snapshot SHA-256 | 3a3b8b98fba72ce79fd8416138ecd62c26051e528844f9d0ff70fdcacc22af13 |
| 第二輪 snapshot SHA-256 | a154caecdb43a9b471c2164bfceda3b71bf54950293e2dc72c418411eed42b98 |
| Sampling | 負責人核准重標原圖 125405 四眼；沒有隨機抽樣，seed 不適用 |

第二輪來源 CSV 當時共 11 筆歷史紀錄，僅凍結四個最新版本，不以較早版本替代。
原有 trial_manifest.json 保留準備當時的紀錄，其 prepared 狀態不是即時進度；完成結果以本文件及新產物為準。

## Mask and Centroid Comparison

| 動物 | 眼側 | 面積第一輪 → 第二輪（px） | 交集／聯集（px） | IoU | Dice | 重心位移（px） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 13692 | left | 13 → 10 | 10／13 | 0.769 | 0.870 | 0.186 |
| 13692 | right | 15 → 7 | 6／16 | 0.375 | 0.545 | 0.568 |
| 17207 | left | 15 → 14 | 13／16 | 0.812 | 0.897 | 0.214 |
| 17207 | right | 14 → 19 | 13／20 | 0.650 | 0.788 | 0.819 |

| 動物／眼側 | 第一輪重心（x, y）px | 第二輪重心（x, y）px | Δx, Δy（px） |
| --- | --- | --- | --- |
| 13692 left | 285.577, 209.115 | 285.600, 209.300 | +0.023, +0.185 |
| 13692 right | 251.567, 207.633 | 251.929, 208.071 | +0.362, +0.438 |
| 17207 left | 509.500, 61.500 | 509.286, 61.500 | −0.214, 0.000 |
| 17207 right | 484.357, 60.429 | 485.079, 60.816 | +0.722, +0.387 |

13692 右眼面積減少 8 px（相對第一輪 −53.3%），IoU 最低。
17207 右眼重心位移最大，但不是 IoU 最低的眼睛。
因此 Mask 輪廓一致性、重心一致性與距離一致性應分別報告；不能用其中一項代替其餘指標。
數字無法單獨判定哪一輪邊界較正確，這需要原圖與標註規範的人工作業確認。

## Distance Comparison

變化一律定義為第二輪減第一輪；百分比以第一輪距離為分母，不稱為真實誤差。

| 量測 | 第一輪（px） | 第二輪（px） | 有號變化（px） | 絕對變化（px） | 相對變化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13692 雙眼 | 34.043 | 33.694 | −0.349 | 0.349 | −1.024% |
| 17207 雙眼 | 25.166 | 24.216 | −0.949 | 0.949 | −3.772% |
| 13692／17207 右眼 | 275.428 | 275.760 | +0.331 | 0.331 | +0.120% |

右眼配對距離的相對變化較小，不能推論其端點標註更準確；它也受到原始距離較長與端點位移方向影響。
表格顯示三位小數，CSV 保留 float64 計算值，不代表已驗證到 0.001 px 的量測精度。

## Method

```text
IoU = |M1 ∩ M2| / |M1 ∪ M2|
Dice = 2 |M1 ∩ M2| / (|M1| + |M2|)
centroid = mean(column + 0.5), mean(row + 0.5) over foreground pixels
centroid displacement = sqrt((cx2-cx1)^2 + (cy2-cy1)^2)
signed distance change = D2 - D1
absolute distance change = |D2 - D1|
relative signed change percent = 100 (D2 - D1) / D1
```

Mask 在同一張原圖像素格直接比較，不插值、不對齊平移、不縮放後再算。
解剖眼側與 animal annotation ID 直接配對；雙眼與兩動物右眼的配對沿用既有 metrology。
空 Mask 重心／距離未定義，兩個空 Mask 的 IoU／Dice 也未定義，不能當完美一致。
本次四眼均非空、完整可測；第一輪為零的距離不計相對百分比。

## Artifacts and Reproduction

| 路徑 | 內容 |
| --- | --- |
| outputs/phase3/blind_repeat_v1/measurement_baseline_v1/ | 第二輪四眼快照、4 重心、3 距離與原圖對照圖 |
| outputs/phase3/blind_repeat_v1/comparison_v1/eye_comparison.csv | 逐眼面積／交聯集／IoU／Dice／重心／版本／審核者／時間／digest |
| outputs/phase3/blind_repeat_v1/comparison_v1/distance_comparison.csv | 三筆距離、有號與絕對變化、相對變化、來源 digest |
| outputs/phase3/blind_repeat_v1/comparison_v1/mask_comparison.png | 逐眼第一輪／第二輪／疊圖，顏色與重心標記及相同裁切範圍 |
| outputs/phase3/blind_repeat_v1/comparison_v1/analyze.py | 此次比較計算與繪圖程式，支援唯讀重算 |
| outputs/phase3/blind_repeat_v1/comparison_v1/configuration.json | 兩輪固定快照的相對路徑與核准身份 |
| outputs/phase3/blind_repeat_v1/comparison_v1/summary.json、run_manifest.json | 摘要、公式、版本、執行指令與產物 SHA-256 |
| outputs/phase3/blind_repeat_v1/comparison_v1/provenance/ | 計算模組與本輪設定副本 |

[查看逐眼 Mask 比較圖](../outputs/phase3/blind_repeat_v1/comparison_v1/mask_comparison.png)。
金色是第一輪、藍色是第二輪、白色是兩輪交集，十字為重心。像素以 nearest-neighbor 放大，沒有增添影像細節。
該 PNG 已實際開啟確認四列身份、圖例、原圖局部與數字可讀。

從 repository 根目錄執行；建立命令拒絕覆寫既有成果：

```sh
.venv/bin/python -m animal_eye_metrology.measurement_baseline \
  --config configs/blind_repeat_measurement.toml
.venv/bin/python outputs/phase3/blind_repeat_v1/comparison_v1/analyze.py
```

重現既有數值請使用唯讀命令，不必重新從工作 CSV 擷取標註：

```sh
.venv/bin/python -m animal_eye_metrology.measurement_baseline \
  --verify outputs/phase3/blind_repeat_v1/measurement_baseline_v1
.venv/bin/python outputs/phase3/blind_repeat_v1/comparison_v1/analyze.py --verify
```

比較重算同時驗證兩輪固定基準的 digest，重新計算四眼與三筆距離後逐列核對 CSV。
仍需保留兩輪基準目錄；不使用目前的人工標註工作 CSV。

## Changed Files and Validation

| 檔案 | 變更 |
| --- | --- |
| configs/blind_repeat_measurement.toml | 新增第二輪 1 圖／2 動物／4 眼的凍結設定 |
| src/animal_eye_metrology/metrology.py | 新增二值 Mask 交集／聯集／IoU／Dice，含空值與輸入檢查 |
| tests/test_metrology.py | 新增已知集合、對稱、相同／無交集、空 Mask、尺寸與編碼錯誤的數學測試 |
| docs/blind_repeat_results.md | 本次結果與完整執行／限制紀錄 |
| docs/blind_repeat_check.md、README.md | 加入已完成比較結果連結 |

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests app outputs/phase3/blind_repeat_v1/comparison_v1/analyze.py
.venv/bin/ruff format --check src/animal_eye_metrology/metrology.py \
  tests/test_metrology.py outputs/phase3/blind_repeat_v1/comparison_v1/analyze.py
```

實際結果：88 passed，53 個既有 pycocotools／NumPy 棄用警告；Ruff check／format 通過。
第一輪 12 眼／9 距離基準、第二輪 4 眼／3 距離基準與比較 CSV 均唯讀重算通過。
使用者修正完成後、計算開始前記錄的 700 個既有檔案 SHA-256 全部不變。
保留人工 CSV／歷史輪廓／Mask、原有研究產物、官方註記、AGENTS.md 與 Phase 01.md，沒有新增相依套件、AI、訓練、Commit 或 Push。

## Interpretation and Next Step

這次僅一張圖片、同一人、四隻已看過的眼睛，不能代表其他動物、圖片或審核者。
兩輪選定版本的保存時間相差約 24–26 分鐘；這是保存紀錄間隔，不能當作已證實的盲化等待時間或實際描繪起始時間。
先前已看過基準對照圖，記憶與短間隔的影響無法排除。
本次不估計母體標準差／信賴區間、不宣稱真實中心準確度或物理長度誤差，也不將三筆相關距離當成獨立樣本。

建議先人工查看 13692 右眼的邊界差異，確認可見眼部範圍的規範是否需補充；
若要改 Ground Truth 定義，先經負責人核准，保留兩輪原值，不為提高一致性而直接取平均或挑較好看的 Mask。
確認後再提眼睛分割 POC 的模型與驗證計畫；本次不自動開始下一階段。
