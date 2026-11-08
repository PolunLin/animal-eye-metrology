# Phase 2 YOLO11n-seg CPU POC Results

## Scope and Outcome

依負責人更新後的 phase2_instance_segmentation.md 實作，於 2026-11-07 10:00:55 UTC
（Asia/Taipei 18:00:55）完成最終 run。15／15 張 Selected Images 成功推論與匯出，
評估全部 47 個非 crowd 動物 GT。實作與產物核對完成，等待驗收，不進入 Phase 3。
只有 YOLO11n-seg，沒有訓練、微調、其他模型、眼睛定位／分割、GT UI 或距離量測。

主要結果在 outputs/phase2/yolo11n_seg_cpu/；smoke_cpu_1／smoke_cpu_3 是小樣本檢查，
yolo11n_seg_cpu_initial 保留補齊診斷前的首次完整 run，未覆寫它。
兩次完整 CPU run 的 checkpoint／推論設定相同，505 筆 predictions_coco.json 逐位元組一致；
這不證明跨硬體、跨版本或 latency 也完全一致。

## Model and Environment

| 項目 | 實際設定／觀察 |
| --- | --- |
| 模型 | 官方 YOLO11n-seg，COCO pretrained；未訓練／微調 |
| 來源 | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt |
| Checkpoint 大小 | 6,182,636 bytes |
| SHA-256 | 55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152 |
| 來源核對 | 官方 release HTTPS 取得、ZIP CRC 通過、自算完整雜湊固定於設定；沒有獨立上游參考 digest |
| 系統 | macOS 26.5.1 arm64；CPU、4 threads、FP32、batch=1；未測 MPS |
| 版本 | Python 3.11.14、torch 2.9.1、torchvision 0.24.1、ultralytics 8.3.230 |
| 其他版本 | NumPy 2.4.6、Pillow 12.3.0、pycocotools 2.0.11；完整相依由 uv.lock 記錄 |
| 前後處理 | imgsz=640、rect=False、retina_masks=True；原圖 H×W 原生二值 Mask、原圖 bbox |
| AP 推論 | confidence floor=0.001、class-aware NMS IoU=0.7、全類 max_det=100，之後過濾 animal categories |
| 固定診斷 | confidence>=0.5、Mask IoU>=0.5；同圖、同類、COCOeval greedy 配對 |
| 隨機性 | seed=42；eval／inference mode，無 augment、翻轉或資料隨機抽樣 |

Ultralytics 程式與模型有 AGPL-3.0／Enterprise 授權選項，見
[官方授權](https://www.ultralytics.com/license)。本專案 MIT 宣告不覆蓋上游條款，
本次沒有自行變更本專案授權或散布 checkpoint。

## Quantitative Results

使用 pycocotools.COCOeval(iouType="segm")，IoU=.50:.05:.95、101 recall positions、
maxDets=[1,10,100]，保留官方 crowd／ignore，不另寫 AP evaluator。
過濾十類動物，但只有八類有 GT；bird／cat 的 per-category AP 為 null，
不將缺少 GT 的類別填 0 或當成完美。AP 是有效類別宏平均，Precision／Recall 是實例總數統計。

| 指標 | 實測值（0–1） |
| --- | ---: |
| Mask AP@[.50:.95] | 0.6269461509 |
| Mask AP50 | 0.8738841506 |
| Mask AP75 | 0.7560381217 |
| Mask AR@100 | 0.6775000000 |
| 固定操作點 Precision | 0.8250000000 |
| 固定操作點 Recall | 0.7021276596 |
| 配對成功實例的平均 Mask IoU | 0.8518049257 |
| 配對成功實例的平均 Dice | 0.9173029370 |

固定操作點有 40 個預測：TP=33、FP=7、FN=14；33+14=47 個 GT。
AP 匯出有 505 個動物預測，其中 465 筆低於診斷門檻，保留為 below_confidence，
不將它們計為固定操作點 FP。所有 prediction_id 均有 CSV 記錄，33 個 TP 對應不同 GT。
本批沒有 animal crowd／ignored prediction；其特殊配對由合成測試驗證。

COCO 匯出刻意不附 bbox，讓 loadRes 從 RLE 計算 segmentation area；
原圖 bbox 另存 prediction_instances.json。若附 bbox，loadRes 可能優先採 box area，
影響 Mask 面積分組。RLE 不經 polygon 簡化；Ultralytics 原生二值 Mask 沒有被假稱為機率圖。

### Detection and Mask Diagnostics

相同操作點另以官方 bbox IoU>=0.5 配對：TP=37、FP=3、FN=10。
14 個 Mask FN 中，有 4 個 GT 的框已配對但 Mask 未配對，CSV 標為
bbox_matched_but_mask_unmatched；另外 10 個框與 Mask 都未恢復，標為
not_recovered_by_bbox_or_mask。這是操作點差異，不證明單一失敗因果。
平均 Mask IoU／Dice 只描述 33 個成功配對，不能掩蓋未配對實例。

## Per Image Results

| Image ID | 非 crowd GT | confidence>=0.5 預測 | TP | FP | FN | pipeline ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5992 | 5 | 5 | 5 | 0 | 0 | 77.63 |
| 20247 | 2 | 2 | 2 | 0 | 0 | 39.71 |
| 41635 | 4 | 2 | 0 | 2 | 4 | 137.33 |
| 49269 | 2 | 2 | 2 | 0 | 0 | 40.39 |
| 52462 | 2 | 2 | 2 | 0 | 0 | 50.45 |
| 66561 | 2 | 2 | 2 | 0 | 0 | 46.74 |
| 71756 | 2 | 1 | 0 | 1 | 2 | 67.01 |
| 81766 | 3 | 3 | 3 | 0 | 0 | 60.11 |
| 125405 | 2 | 2 | 2 | 0 | 0 | 51.07 |
| 170739 | 3 | 3 | 3 | 0 | 0 | 136.20 |
| 180011 | 5 | 3 | 3 | 0 | 2 | 133.55 |
| 198960 | 3 | 4 | 3 | 1 | 0 | 86.90 |
| 205282 | 5 | 3 | 2 | 1 | 3 | 76.40 |
| 269196 | 5 | 4 | 2 | 2 | 3 | 93.09 |
| 333069 | 2 | 2 | 2 | 0 | 0 | 44.33 |
| 合計 | 47 | 40 | 33 | 7 | 14 | — |

13 張 Demonstration Candidates 的 32 隻既有雙眼候選中，動物 Mask 恢復 24 隻；
11／13 張至少恢復兩隻。未達此條件的是 71756（0／2）與 205282（1／3）。
這只衡量已知 annotation IDs 的動物分割配對，不驗證眼睛位置、Eye Mask、左右眼或距離。
不據此改動 Phase 1 選圖或人工標記。

## Inspected Success and Failure Cases

已實際檢視以下代表疊圖。由左至右為原圖、COCO GT、confidence>=0.5 預測；
以 GT#annotation_id 與 Pxxx 區分，不靠兩欄顏色推定配對關係。

| Image ID | 觀察 | 產物 |
| --- | --- | --- |
| 20247 | 兩隻熊皆成功配對，平均 IoU=0.9220，包含畫面邊緣的部分動物 | [overlay](../outputs/phase2/yolo11n_seg_cpu/overlays/000000020247.png) |
| 41635 | 高分預測為 sheep，GT 為 cow；2 FP、4 FN，外觀重疊不能取代類別正確性 | [overlay](../outputs/phase2/yolo11n_seg_cpu/overlays/000000041635.png) |
| 71756 | 夜間兩隻熊被合併成一個高分 Mask，未達個別 GT IoU；1 FP、2 FN | [overlay](../outputs/phase2/yolo11n_seg_cpu/overlays/000000071756.png) |
| 269196 | 密集羊群有漏掉及不完整 Mask；2 TP、2 FP、3 FN，其中兩個 FN 有框配對但 Mask 未通過 | [overlay](../outputs/phase2/yolo11n_seg_cpu/overlays/000000269196.png) |

這不構成「夜間必然失敗」或「所有牛都不適用」的類別結論，亦未據此修改門檻。

## Latency

CPU 最終 run 暖機 1 次，每張測一次，共 15 個樣本：pipeline median=67.01 ms、
P95=136.54 ms、累計=1.1409 s；checkpoint 建構／載入=29.02 ms（不含暖機）。
pipeline 包含讀圖／解碼、模型前後處理、CPU transfer、原圖座標驗證與 RLE 編碼，
不含資料集載入、雜湊、評估、疊圖與磁碟匯出。Ultralytics 單獨回報的 inference median=38.21 ms。
這是本機單次 POC，不能換算成通用部署吞吐量；MPS、峰值 RAM、正式多次性能基準未實測。

## Reproduction and Artifacts

設定：configs/phase2.toml；權重來源紀錄：weights/yolo11n-seg.source.json。
輸出包括 predictions_coco.json、prediction_instances.json、instance_matches.csv、metrics.json、
run_manifest.json、image_results.json、15 張 overlays PNG。來源／設定／程式／權重／產物雜湊、
類別對照與版本均記在 manifest；資料、權重與輸出不提交 Git。

```sh
uv sync --frozen --extra visualization --extra explorer --extra segmentation
# 權重先依 README 從官方 release 取得；推論程式只讀本機檔案。
.venv/bin/python -m animal_eye_metrology.phase2 --config configs/phase2.toml \
  --limit 1 --output outputs/phase2/smoke_review_1
.venv/bin/python -m animal_eye_metrology.phase2 --config configs/phase2.toml \
  --output outputs/phase2/yolo11n_seg_cpu_review
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

命令從 repository 根目錄執行；輸出須為新的 outputs/phase2 子目錄，拒絕覆寫現有 run。
全部圖片、評估、視覺化及來源核對完成後才發布；錯誤以 phase2_failed 和非零結束碼回報。

## Validation and Limits

73 項 pytest 通過（原 68 項加 5 項分割核心測試），無跳過；49 則已知上游解碼棄用警告未隱藏。
新增合成測試覆蓋類別／ID、原圖座標／RLE、IoU／Dice、重複／錯類／負樣本、crowd ignore、
AP 低分真陽性、空預測／無 GT，以及 bbox 配對不能隱藏 Mask 失敗。
另外核對全部 505 個實際 RLE 編解碼及原圖尺寸、預測／GT ID 唯一性、輸出及輸入雜湊。
Phase 1 資料、官方 Annotation、AGENTS.md、Phase 01.md 均未改寫。

15 張為人工選圖，不能作為全 COCO 或獨立泛化成績。官方 COCO train／val 配方分離不保證
checkpoint 完整訓練歷史可獨立稽核；val2017 是已知開發 benchmark，不宣稱零洩漏。
來源參考：[該版本的 COCO 設定](https://github.com/ultralytics/ultralytics/blob/v8.3.230/ultralytics/cfg/datasets/coco.yaml)。
41635、170739、180011、269196 的全類低分輸出達到 max_det=100；這是截斷，並非 100 隻動物。
AP 受 confidence floor、NMS、cap 與前後處理影響，不直接與其他 validator／官方全 80 類成績比較。
動物分割品質不等於眼部或距離準確度。

建議先審查錯類、合併與邊界失敗；是否比較 Mask R-CNN、擴大評估、測 MPS 或進入 Phase 3，
由負責人另行決定。本次沒有自動 Commit／Push。
