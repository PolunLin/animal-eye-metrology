# YOLO26s Selected Images Segmentation Review

## Outcome

YOLO26s-seg 已完成既有 15 張 Selected Images 的動物實例分割，產生 15 張
原圖／COCO GT／預測對照圖。沿用原有門檻，13 張測量候選中有 12 張至少配對到
兩隻既有合格動物，合計恢復 27／32 隻；舊 YOLO11n-seg 為 11／13 張、24／32 隻。

269196 羊群圖合併了多隻前景動物，只有 1／5 隻合格動物成功配對。
它仍保留於原始 13 張人工候選中，不因模型失敗改寫人工標記或選圖。
12 張通過的是動物實例配對條件；眼睛邊界、左右眼與量測能力仍待人工確認。
本次為負責人核准的暫定 YOLO26s POC，沒有選定最終模型。

## Dataset and Evaluation

| 項目 | 實際來源與設定 |
| --- | --- |
| Dataset | 本機 COCO 2017 val2017 原圖與官方 instances_val2017.json |
| 選圖 | image_selection.csv 全部 15 張 Selected，依 image_id 升冪；沒有重新抽樣 |
| 動物 GT | 47 個非 crowd 動物，8 類有 GT；bird／cat 無 GT，不納入有效類別 AP 平均 |
| 人工候選 | measurement_feasibility.csv 的 13 張、32 個既有 eligible_annotation_ids |
| 人工來源核對 | eye_visibility_review_working.csv；eye_visibility_review.csv 是保留的原始空白模板 |
| 模型 | 官方 yolo26s-seg.pt，Ultralytics 8.4.0 的 Segment26、end2end=True |
| 權重 SHA-256 | 3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4 |
| Runtime | macOS arm64、Python 3.11.14、PyTorch 2.9.1、CPU FP32、batch=1、實際 4 threads |
| 輸入 | 640×640 letterbox、rect=False、retina_masks=True，原圖 H×W Mask 與 bbox |
| 隨機性 | seed=42、無 augment；此 seed 用於初始化，不是另抽 15 張 |
| AP | confidence floor=0.001、全類 max_det=100；官方 COCOeval segm，IoU=.50:.05:.95、maxDets=[1,10,100] |
| 診斷門檻 | confidence≥0.5、Mask IoU≥0.5，同圖同類官方 greedy 配對，保留 crowd ignore |
| 合格動物恢復條件 | 配對為 TP，且 image_id＋annotation_id 位於原有合格清單 |

動物名稱對應官方 sparse COCO category ID，不以模型索引直接加常數；prediction_id
與 annotation_id 分開。YOLO26 使用原生 end-to-end head，傳入 iou=0.7 不代表
套用 YOLO11 的傳統 NMS。沒有 5,000 張推論、訓練、眼睛分割、距離測量或 GT UI。

## Comparison with Existing YOLO11 POC

兩模型為同一份 15 張清單、同一官方註記 digest、同一評估方法與門檻。
舊結果讀自 outputs/phase2/yolo11n_seg_cpu/，沒有覆寫或重新推論。

| 指標 | YOLO11n-seg 舊 POC | YOLO26s-seg 本次 |
| --- | ---: | ---: |
| Mask AP 0–100 | 62.69 | 75.20 |
| Mask AP50 0–100 | 87.39 | 96.53 |
| Mask AP75 0–100 | 75.60 | 89.91 |
| 正確配對 TP | 33 | 37 |
| 誤報 FP | 7 | 6 |
| 漏掉 FN | 14 | 10 |
| Precision | 82.50% | 86.05% |
| Recall | 70.21% | 78.72% |
| TP 平均 Mask IoU | 0.8518 | 0.8607 |
| 恢復既有合格動物 | 24／32 | 27／32 |
| 至少恢復兩隻合格動物的圖片 | 11／13 | 12／13 |

Mask AP 是綜合偵測與分割的分數，不是正確像素百分比；TP 平均 IoU 不包含漏掉的動物。
這批圖經人工挑選，不能作為全 COCO 或獨立泛化成績，也不能與另一份 100 張試跑
當作相同資料上的前後進步比較。框架版本分別為 YOLO11 的 8.3.230 與 YOLO26 的 8.4.0。

## Thirteen Candidate Images

數字為恢復的合格動物數／既有合格動物數。點擊 image_id 開啟三欄對照圖；
完整成功／漏配 Annotation ID 見
[candidate_comparison.csv](../outputs/phase2/yolo26s_seg_selected15_cpu/candidate_comparison.csv)。

| Image ID | 動物 | YOLO11n | YOLO26s | 至少兩隻條件 | 視覺檢查重點 |
| --- | --- | ---: | ---: | --- | --- |
| [20247](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000020247.png) | bear | 2／2 | 2／2 | 達到 | 頭部保留；後方臉較小，左側身體截斷來自原圖 |
| [49269](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000049269.png) | dog、horse | 2／2 | 2／2 | 達到 | 頭部分離；馬頭轉向與鬃毛需後續放大確認 |
| [52462](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000052462.png) | zebra | 2／2 | 2／2 | 達到 | 草遮下半身；臉部較小、條紋影響眼部對比 |
| [66561](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000066561.png) | sheep | 2／2 | 2／2 | 達到 | 頭部分離；黑色小羊臉部對比低 |
| [71756](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000071756.png) | bear | 0／2 | 2／2 | 新達到 | 兩熊這次分開，紅外線眼部反光仍需確認 |
| [81766](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000081766.png) | dog | 3／3 | 3／3 | 達到 | 長毛黑犬分數約 0.502，接近門檻，眼部細節需確認 |
| [125405](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000125405.png) | dog | 2／2 | 2／2 | 達到 | 頭部保留，後方犬較小 |
| [170739](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000170739.png) | elephant | 2／2 | 2／2 | 達到 | 合格象頭部保留；額外重疊預測需依 ID 區分 |
| [180011](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000180011.png) | giraffe | 2／3 | 3／3 | 達到 | 人與枝葉遮擋，另有 zebra 誤報；側轉頭需另確認 |
| [198960](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000198960.png) | sheep | 2／2 | 2／2 | 達到 | 黑白影像，遠處臉部細節較少 |
| [205282](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000205282.png) | giraffe | 1／3 | 2／3 | 新達到 | 仍漏 1414076；其他非合格動物不計入此條件 |
| [269196](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000269196.png) | sheep | 2／5 | 1／5 | 未達到 | 前景多隻羊合併同一 Mask，優先檢查實例區分 |
| [333069](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000333069.png) | giraffe | 2／2 | 2／2 | 達到 | 頭部保留，眼睛解析度需原圖確認 |

兩張由未達到變為達到，一張由達到變為未達到，淨增加一張。
269196 成功配對 64938，漏配 61850、65773、66542、1817822；205282 漏配 1414076。
本次沒有調低門檻來使失敗圖片通過。

另外兩張 Selected 為
[5992](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000005992.png)
與 [41635](../outputs/phase2/yolo26s_seg_selected15_cpu/overlays/000000041635.png)，
動物 TP／FP／FN 分別為 4／0／1 與 1／1／3；它們不在既有 13 張測量候選中。

## Visual Review Instructions

1. 優先看 269196、71756、81766、205282；舊 YOLO11 結果位於 yolo11n_seg_cpu/overlays/ 同名 PNG。
2. 中欄 GT# 是官方 annotation_id，右欄 Pxxx 是 prediction ID；依 instance_matches.csv 配對，
   不要以兩欄顏色認定相同動物。右欄只畫 confidence≥0.5，低分預測仍保留供 AP 評估。
3. 檢查不同動物是否分開、頭部是否被框／遮罩明顯漏掉，以及重疊 FP 是否會混淆選擇。
   整隻動物 Mask IoU≥0.5 不能保證眼部像素完整。
4. 放大原圖確認反光、長毛、側轉頭與小臉；本次沒有標左右眼或 Eye Mask。
   後續眼睛資料應保留原圖像素，不宜直接以動物 Mask 抹掉框內其他像素。
5. 本次 15 張技術觀察另存 visual_review_notes.csv；owner_review_status 均為 Pending。
   這份新檔不是原始眼睛人工標記或新的 ground truth，等候負責人確認。

working CSV 中 47 個 Selected 動物的核心審核欄位已填；eye_resolution 仍有 47 筆空白。
此欄視為未知，不自動填寫或視為不合格。放大不能補回解析度。
主要量測方法保留為 Eye Segmentation Mask → Mask Centroid → Euclidean Distance，
等待 Phase 3 的 GT 定義與實作核准。

## Runtime and Reproduction

本次正式 15 張逐張合計 1.2339 秒，median=73.36 ms、P95=114.76 ms。
包含讀圖／RGB 解碼、框架前後處理／推論、座標驗證與動物 RLE，
不含初始化、暖機、來源雜湊、評估與輸出。模型建構／載入 0.0379 秒、另 3 次暖機 0.3082 秒。
這不是完整指令總時間。舊 YOLO11 未記錄實際 threads，套件又會在初始化時重設 threads；
本次以 on_predict_start 恢復 4 threads、逐張驗證並記錄，因此不作兩批速度排名。

從 repository 根目錄執行，沿用既有權重與獨立 Ultralytics 8.4.0 目錄：

```sh
PYTHONPATH=outputs/phase2/pilot100/ultralytics_8_4_0 \
  .venv/bin/python -m animal_eye_metrology.phase2 \
  --config configs/phase2_yolo26s_selected.toml
```

已有輸出拒絕覆寫，重跑須用 --output 指定新的 outputs/phase2 子目錄。
權重來源、授權及訓練資料履歷限制見 [100-image pilot report](phase2_pilot100_results.md)。
COCO validation 是公開開發 benchmark，不能宣稱為零洩漏的獨立測試。

## Artifacts and Verification

產物集中於 outputs/phase2/yolo26s_seg_selected15_cpu/。

| 產物 | 內容 |
| --- | --- |
| overlays/*.png | 15 張原圖／GT／YOLO26s 三欄對照，Image ID、GT annotation_id、prediction ID、score、TP／FP／FN |
| candidate_recovery.csv | 13 張候選的合格 IDs、成功／漏配 IDs、至少兩隻條件 |
| candidate_comparison.csv | 依舊 instance_matches.csv 與同一 candidate_recovery 函式衍生的兩模型比較 |
| visual_review_notes.csv | 15 張已檢視對照圖的技術觀察，負責人確認狀態仍 Pending |
| predictions_coco.json、prediction_instances.json | 367 個低分門檻動物預測；分別供官方 Mask AP 與 bbox 檢查 |
| instance_matches.csv、metrics.json | 全部配對、COCO／固定門檻評估 |
| image_results.json、run_manifest.json | 逐張時間、模型版本、來源／產物雜湊與設定 |
| integrity_check.json、review_artifacts_manifest.json | 原有資料核對、review 產物的來源與 digest |

修改 phase2.py，沿用既有 runner 增加模型標籤、合格 ID 配對表及實際 threads 核對；
新增 configs/phase2_yolo26s_selected.toml、本報告，更新 README 與 tests/test_segmentation.py。
沒有新增套件、資料庫、視覺化系統、眼睛功能、Commit 或 Push。

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests app
.venv/bin/ruff format --check src/animal_eye_metrology/phase2.py tests/test_segmentation.py
```

77 tests passed，49 個既有 pycocotools／NumPy deprecation warnings；Ruff 通過。
新增測試驗證只有同圖且在合格清單內的 TP 能計入候選恢復，不計其他動物、其他圖、FN 或 FP。
另核對 367 個預測的唯一 ID、原圖 Mask 尺寸與 RLE 解碼、15 張 PNG 尺寸、13 張統計，
以及原始 manifest 的 21 個產物 digest。167 個受保護檔案不變，包括 Phase 1 人工資料、
官方註記、舊 POC、四模型 100 張結果、權重、AGENTS.md、Phase 01.md、pyproject.toml、uv.lock。

目前停在動物分割驗收。下一步先由負責人確認上述失敗與邊界案例，再核准
Phase 3 的眼睛 Ground Truth 定義；尚未開始眼睛模型或 Ground Truth UI。
