# Phase 2 — Four-model 100-image Compatibility and Timing Pilot

## Scope and Protocol

依專案負責人指示，比較 YOLO11n-seg、YOLO26n-seg、Mask R-CNN ResNet-50 FPN v2、
YOLO26s-seg，先各跑相同 100 張，確認相容性並估算完整 5,000 張耗時。
**沒有執行 5,000 張推論；沒有訓練、眼睛推論、Ground Truth UI 或距離量測。**
Phase 1 人工資料與原始 15 張 POC 另行保留。此次取樣不以人工 Selected、動物種類、
眼睛清楚程度或模型預測結果作為條件。

| 項目 | 實際設定 |
| --- | --- |
| Dataset | COCO 2017 val2017，全 5,000 張 |
| Sampling | `random.Random(42).sample(sorted(all_image_ids), 100)`，不放回均勻抽樣 |
| 執行順序 | 同一份 image_id 升冪，四模型依序、不同 Python process 執行 |
| 樣本組成 | 12 張含非 crowd 動物，88 張不含非 crowd 動物；其中 9 張多動物 |
| 樣本動物實例 | 73：sheep 39、zebra 13、cow 9、dog 5、giraffe 3、cat 2、bird 1、horse 1 |
| 完整母體組成 | 1,016 張含非 crowd 動物，3,984 張不含；10 種動物 |
| Hardware | Apple M4 Pro，24 GB RAM，14 CPU cores（10 performance + 4 efficiency） |
| Runtime | macOS 26.5.1 arm64，Python 3.11.14，CPU FP32，batch=1 |
| Threads | PyTorch intra-op=4、inter-op=1；逐張檢查預算沒有被套件改動 |
| Warmup | 每個模型先重複第一張 3 次，不納入正式 100 張計時 |
| Reproducibility | seed=42；deterministic algorithms 採 warn_only，不承諾跨版本逐 bit 相同 |
| Prediction export | 所有模型原圖座標、原圖 H×W Mask，animal name 對應 COCO sparse category ID |
| Shared output limits | confidence floor=0.001、每圖全類最多 100 預測、Mask threshold=0.5 |
| Evaluation smoke check | 重用官方 COCOeval segm；診斷 confidence=0.5、IoU=0.5；保留 crowd ignore |

樣本中 elephant 與 bear 沒有非 crowd GT，因此這 100 張不能支持十類完整效能結論。
沒有非 crowd 動物的圖片仍會完整推論，以涵蓋實際完整驗證集的處理負擔及誤報情境。
樣本、原圖及註記雜湊詳見 outputs/phase2/pilot100/sample_manifest.json 與 sampled_images.csv。

## Model Compatibility

| 模型 | 套件 | 原生輸入與後處理 |
| --- | --- | --- |
| YOLO11n-seg | Ultralytics 8.3.230 | 640×640 letterbox、Segment head、NMS IoU=0.7、retina_masks |
| YOLO26n-seg | Ultralytics 8.4.0，獨立 target 目錄 | 640×640 letterbox、Segment26 head、end2end=True、retina_masks |
| Mask R-CNN ResNet-50 FPN v2 | TorchVision 0.24.1 | short side=800、long side≤1333；box NMS IoU=0.5；原圖機率 Mask threshold=0.5 |
| YOLO26s-seg | Ultralytics 8.4.0，獨立 target 目錄 | 640×640 letterbox、Segment26 head、end2end=True、retina_masks |

共享 PyTorch 2.9.1；原有 pyproject.toml、uv.lock 與基礎環境沒有升級。
YOLO26 使用官方 end-to-end head 的原生推論分支；傳入的 iou=0.7 不代表套用了
YOLO11 的傳統 NMS。輸入尺寸、架構與框架不同，時間比較代表上述完整處理設定，
不能單獨歸因為模型架構優劣。

## Compatibility Findings

1. Ultralytics 初始化 CPU predictor 時會重設 threads。最初 YOLO11 診斷跑使用到 8 threads；
   已隔離保存至 yolo11n_seg_initial_threads8/，不納入比較。
   正式執行在初始化後恢復 4 threads，並於每張預測後驗證。
2. Mask R-CNN 首次試跑在第 43 張（image_id=219440）遇到 float32 resize roundoff：
   框的 y2=426.0000305175781，但原圖高為 426。
   第一次只成功 42 張，保存在 maskrcnn_v2_initial_roundoff_failure/ 及對應 log，
   不當成完整 100 張結果。修正只針對 TorchVision 輸出，容許 ≤0.0001 pixel 的邊界捨入誤差，
   clip 到原圖邊界；超過此容許值仍報錯。沒有更動 Mask、分數、類別或 COCO GT。
   新增幾何測試驗證微小誤差可修正、真正超界必須拒絕，再重新執行完整 100 張。
3. YOLO11 與 YOLO26n 使用修正前 runner，TorchVision 與 YOLO26s 使用修正後 runner；
   差異只影響 TorchVision 邊界捨入及追蹤欄位，YOLO 推論設定沒有改動。
   每份正式結果附實際執行 runner snapshot 與來源雜湊。

## Runtime Results

正式實測與估時表由完成後的 comparison.csv 整理如下；四模型皆成功 100／100。

| 模型 | 100 張逐張合計（秒） | 平均秒／張 | 5,000 張線性估計（分） | 5,000 張母體校正估計（分） | process peak RSS（GiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n-seg | 6.27 | 0.0627 | 5.22 | 5.26 | 2.37 |
| YOLO26n-seg | 6.37 | 0.0637 | 5.31 | 5.34 | 2.41 |
| Mask R-CNN ResNet-50 FPN v2 | 134.23 | 1.3423 | 111.86 | 112.13 | 3.30 |
| YOLO26s-seg | 8.57 | 0.0857 | 7.14 | 7.18 | 2.73 |

四模型各跑 5,000 張，依序執行的母體校正估計合計 129.91 分鐘，約 2 小時 10 分。
這是逐張處理部分的外推；排程可先預留 3 小時作緩衝，並另計全量評估／輸出需求。
此緩衝是工作安排建議，不是量測出的信賴區間。
本次 100 張的 setup 分別為 0.72／4.68／5.42／1.52 秒，
evaluation 分別為 0.81／0.88／0.93／0.88 秒（依表順序），均另存 manifest。
peak RSS 為含套件、模型及暖機的整個 process 峰值，非 GPU 使用量；MPS 未測。

### Timing Definition

正式逐張時間包含：讀檔與 RGB 解碼、模型原生前處理／推論／後處理、轉 CPU array、
原圖尺寸／座標檢查，以及動物 Mask COCO RLE 編碼。`native_predict_seconds` 另存於逐圖記錄，
也包含框架前後處理，不能當成單純神經網路 forward 時間。

逐張時間不包含：首次下載／安裝、Python／重型套件載入、來源雜湊、暖機、評估、
JSON／CSV 寫檔、圖表。setup_seconds 含資料／權重核對、模型載入及暖機，
從重型共用套件 import 完成後開始計時；不是冷啟動整個 Python process 的時間。
model_load_seconds 與 warmup_seconds 已包含在 setup_seconds，不可重複相加。
evaluation_seconds 包含建立 COCO GT index 及本次 100 張官方評估。

### Extrapolation

兩種估時都保存於 comparison.csv；均是估計，並非全量實測。

- 線性估計：`5,000 × mean(t_100)`。
- 母體比例校正：`1,016 × mean(t_animal_12) + 3,984 × mean(t_background_88)`。

此處按非 crowd 動物有／無分層校正樣本比例，沒有依速度或結果重新選圖。
主表採母體比例校正，線性結果供核對。兩者都只推估上述逐張處理部分；
不能把 100 張的 COCO evaluation 時間直接宣稱為 5,000 張評估時間。
單次試跑無信賴區間；長時間熱降頻、其他程式、影像複雜度及檔案快取都會影響實際耗時。
若未來要為 5,000 張全部輸出對照圖，圖表時間與磁碟成本應另外實測。

## Provenance, Licensing and Leakage

權重來源及完整 SHA-256 固定於 configs/phase2_pilot.toml，各模型 manifest 重複記錄。
下載後檢查 checkpoint ZIP CRC；Mask R-CNN 完整 digest 前綴另與上游檔名 73cbd019 相符。
YOLO 資產未提供本次可獨立核對的上游完整 checksum，因此本機 SHA-256 用於重跑一致性，
不宣稱已獨立證明資產真實性。

- [YOLO11 official documentation](https://docs.ultralytics.com/models/yolo11/)
- [YOLO26 official documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLO26 segmentation v8.4.0 source configuration](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/cfg/models/26/yolo26-seg.yaml)
- [TorchVision Mask R-CNN v2 official documentation](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
- [TorchVision model and weight source](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/detection/mask_rcnn.py)
- [Ultralytics licensing](https://www.ultralytics.com/license)
- [TorchVision license](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE)

Ultralytics 程式／模型適用其 AGPL-3.0／Enterprise 條款；TorchVision 程式為 BSD-3-Clause，
預訓練權重仍須依上游模型及資料條款處理。專案本身 MIT 不覆蓋上游授權。

官方 COCO 預訓練權重的來源可確認，但沒有獨立稽核完整 checkpoint 訓練歷史。
不能僅由模型名稱，保證特定 segmentation checkpoint 未用過 val2017；
也不能把 YOLO26 detection 的預訓練敘述直接當作這兩個 segmentation checkpoint 的履歷。
val2017 是公開且反覆用於開發／選模的 benchmark，後續完整 5,000 張結果應標為
COCO validation benchmark，不能宣稱是零資料洩漏的獨立測試。
此次只報相容性與估時，未依這 100 張調整模型或選出勝出模型。

## Artifacts and Reproduction

| 檔案／目錄 | 內容 |
| --- | --- |
| configs/phase2_pilot.toml | 四個模型來源、digest、100 張、seed 42、CPU 預算 |
| src/animal_eye_metrology/phase2_pilot.py | prepare／run／summarize，沿用 COCO、analysis、prediction normalization 與官方評估 |
| tests/test_phase2_pilot.py | 固定抽樣、分層估時、TorchVision 邊界捨入測試 |
| outputs/phase2/pilot100/sampled_images.csv | 同一份 100 圖、抽樣順序、尺寸、組別、動物數、原圖 digest |
| outputs/phase2/pilot100/sample_manifest.json | 母體／樣本數量、抽樣方法及來源 digest |
| outputs/phase2/pilot100/comparison.csv、comparison.json | 正式四模型的實測與估時 |
| outputs/phase2/pilot100/{model}/ | 預測、逐圖時間、診斷配對、官方評估、run_manifest、runner snapshot |
| outputs/phase2/pilot100/*.log | 各 process 的進度與錯誤紀錄 |
| outputs/phase2/pilot100/environment_base.txt、hardware.json | 基礎環境版本清單與無機器識別碼的硬體資料 |
| outputs/phase2/pilot100/protected_baseline.json、integrity_check.json | 受保護檔案執行前 digest 與執行後核對 |

README 的 Four-model Timing Pilot 一節列出實際使用命令，沒有新增框架或資料庫。
輸出拒絕覆寫；重新試跑請複製設定檔至 configs/、改 output 並傳 --config。
不得刪除／覆寫現有 sample、Phase 1 人工結果或原始 POC。

## Verification and Next Decision

| 驗證 | 結果 |
| --- | --- |
| 正式相容性試跑 | 四模型皆 100／100；同一 image_id 清單、順序及 4 threads |
| COCO prediction export | 4 份官方格式匯出及評估成功；2,336 個低分數門檻動物預測的 RLE 皆可解碼且原圖尺寸一致 |
| 來源完整性 | 70 個受保護檔案 SHA-256 不變，包括 Phase 1 人工資料、官方註記、原始 POC、AGENTS.md、Phase 01.md、pyproject.toml、uv.lock |
| 樣本完整性 | 100 張原圖於準備、各次正式執行前及結束後核對 SHA-256 一致 |
| Pytest | `.venv/bin/pytest -q`：76 passed，49 個既有 pycocotools／NumPy deprecation warnings；log 保存於輸出目錄 |
| Ruff | `.venv/bin/ruff check src tests app`：All checks passed |
| 新增測試 | 抽樣不受輸入順序影響且不改 global RNG；完整母體分層估時；浮點邊界修正與真正超界拒絕 |

正式 TorchVision 100 張共有 229 個框接受 ≤0.0001 pixel 的邊界捨入 clip
（含非動物框，不含暖機）；沒有動到 COCO 註記。
沿用既有 float32 xyxy→xywh 匯出後，有 1 個框的 y+h 重組誤差為 0.0000153 pixel；
完整性核對明確採 0.0001 pixel 數值容許值，Mask 尺寸與像素編碼仍完全在原圖座標。
完整數字見 integrity_check.json。這些低分數預測數量不能當成偵測成功數或人工標註數。

新增／修改檔案為 README.md、configs/phase2_pilot.toml、
src/animal_eye_metrology/phase2_pilot.py、tests/test_phase2_pilot.py、本報告。
weights/ 與 outputs/ 為本機產物，沿用既有 Git 忽略規則；沒有 Commit 或 Push。

完整 5,000 張四模型推論、完整評估方案及長時間執行安排等待專案負責人確認。
這次沒有選定最終模型，也沒有進入後續眼睛分割／測量階段。
