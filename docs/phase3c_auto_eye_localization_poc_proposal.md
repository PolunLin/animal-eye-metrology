# HRNet-W32 AP-10K Automatic Eye Localization POC Proposal

## Decision Summary

日期：2026-11-08（Asia/Taipei）。狀態：**Proposed，供專案負責人核准**。

HRNet-W32／AP-10K 適合進入自動眼睛定位的可行性驗證：有獨立左右眼輸出，
訓練資料涵蓋目前三張開發圖的 dog、giraffe 與 bear 對應熊類。
本機 CPU 與 Python 可運作，但 MMPose 相依套件尚未安裝，**目前不能宣稱模型已在本機相容或可直接執行**。

建議依序執行 A「獨立環境相容性試跑」與 B「三張既有圖的自動定位接 SAM」。
A 未通過就停止 B；兩者皆不訓練模型。第一輪固定使用現有 SAM 2.1 Hiera-Tiny，
目的是隔離定位管線問題，不代表選定最終分割模型。

| 核對項目 | 結果 | 對 POC 的影響 |
| --- | --- | --- |
| 眼睛欄位 | 0-based index 0=L_Eye、1=R_Eye；共 17 關鍵點 | 可自動提供帶側別的候選點 |
| 解剖左右 | 官方編號示意圖與同側肩、肢標記支持動物自身左右的解讀 | 不按畫面 x 座標重新排序；保留側別錯誤評估 |
| 精確中心定義 | 官方資料只稱 Left Eye／Right Eye，未給出本案眼裂 Mask 重心或瞳孔中心的等價規範 | 只作 SAM 提示，不直接取代量測端點 |
| 物種 | 54 個資料集類別、23 個 family 標籤；COCO 十種動物中九種有對應，bird 無 | 本輪限既有 dog／bear／giraffe 開發圖 |
| 可見性 | 原始標註可包含被遮住但能推估的點；此 HRNet 配置沒有獨立可見性分類頭 | 分數與非空 Mask 均不能證明眼睛可見 |
| 本機 | macOS arm64、Python 3.11.14、torch 2.9.1；CPU 張量運算通過 | 以 CPU FP32 做獨立相容性階段 |
| 部署風險 | MMPose 尚未安裝；官方匯入鏈包含 mmcv.ops | 不把 mmcv-lite 視為可直接替代的完整相依套件 |

機器可讀查核紀錄：[hrnet_ap10k_review_2026-11-08.json](research/hrnet_ap10k_review_2026-11-08.json)。

## Eye Definitions and Evidence

AP-10K 官方 README 的 1-based 編號是 1=Left Eye、2=Right Eye、3=Nose；
MMPose 轉成 0-based index，對應 0、1、2。左右眼各自有輸出通道，
資料集設定也明列 L_Eye 與 R_Eye 的水平翻轉交換關係。
來源：[AP-10K 定義](https://github.com/AlexTheBad/AP-10K/blob/181b1a04755e4dc6fe5616ef7a88496f47bfe228/README.md#keypoint-definition)、
[MMPose AP-10K metadata](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/configs/_base_/datasets/ap10k.py)。

本次實際檢視了[官方黑猩猩示意圖](https://github.com/AlexTheBad/AP-10K/blob/181b1a04755e4dc6fe5616ef7a88496f47bfe228/images/keypointDef.jpg)：
1 號左眼與 6／7／8 號左側上肢一致，支持解剖左右的解讀。
這是依官方圖與標記作的判讀；README 沒有額外寫出一條獨立的 observer-left／anatomical-left 文字定義。
實作仍須核對載入的 dataset_meta，不能讓缺失 metadata 靜默回退成人體 COCO 的順序。

AP-10K 要求標可見點，也允許標註能由體態、對稱性推估的遮擋點。
MMPose 的 COCO-style loader 將 visibility 以 `min(1, v)` 轉成訓練有效性，
使已標註的遮擋點與可見點都可能參與關鍵點學習。
本候選以熱圖峰值產生 keypoint score，並非經校準的「眼睛可見機率」。
來源：[官方標註說明](https://github.com/AlexTheBad/AP-10K/blob/181b1a04755e4dc6fe5616ef7a88496f47bfe228/README.md#3-how-were-the-keypoints-instructed-to-be-labeled)、
[loader](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/mmpose/datasets/datasets/base/base_coco_style_dataset.py)、
[MSRA heatmap decoder](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/mmpose/codecs/msra_heatmap.py)。

沿用專案 eye-visible-aperture-v1 與 Mask 重心量測規範，不新增 Landmark GT。
HRNet 點到既有 Mask 重心的差只能稱「對 Mask 重心的偏移」，不能改稱解剖眼球／瞳孔定位誤差。
對側臉、閉眼、遮擋的自動可見性判定仍是未解決模組；本輪輸出標為候選，不能直接授予 measurement eligibility。

## Species Coverage

從官方 repository 的 test fixture 讀出完整 categories 清單，共 54 個 category IDs、23 個 family 字串；
與 README 的規模一致。這份 fixture 用來核對類別 metadata，不代表已核查完整 train split 的逐類樣本數。
來源：[官方類別 JSON](https://github.com/AlexTheBad/AP-10K/blob/181b1a04755e4dc6fe5616ef7a88496f47bfe228/tests/data/ap10k/test_ap10k.json)。

| COCO 動物 | AP-10K 對應類別及 ID | 本案解讀 |
| --- | --- | --- |
| dog | dog 8 | 有對應；本輪使用 |
| bear | black bear 50、brown bear 51、panda 52、polar bear 53 | COCO bear 較粗，不能只憑其標籤判定 AP-10K 子類；本輪使用 |
| giraffe | giraffe 33 | 有對應；本輪使用 |
| cat | cat 24 | 有對應，本輪不擴圖 |
| horse | horse 21 | 同上 |
| sheep | sheep 6；另有 argali sheep 2 | 同上 |
| cow | cow 5 | 同上 |
| elephant | elephant 20 | 同上 |
| zebra | zebra 22 | 同上 |
| bird | 無 | 不在該哺乳動物訓練類別清單；不能宣稱已支援 |

HRNet 是對給定動物框預測 17 個關鍵點的模型，並不是 54 類物種分類器。
「資料集收錄某類」不等於「已驗證該類在本案小眼睛上準確」。
官方表格的整體 pose AP 也不是眼睛分割或距離量測準確度。

## Candidate Model and Local Compatibility

| 項目 | 擬固定來源或已查核設定 |
| --- | --- |
| Framework | MMPose v1.3.2，commit `5408bc76f5b848cf925a0d1857899011d8c5b497` |
| Config | `configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py` |
| Checkpoint | `hrnet_w32_ap10k_256x256-18aac840_20211029.pth` |
| Weight source | [OpenMMLab 官方 checkpoint](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.pth) |
| Pose input | 256×256，17 張 64×64 heatmaps；MSRAHeatmap decoder |
| Animal bbox preprocessing | 官方 GetBBoxCenterScale 預設 padding=1.25，再依官方 TopdownAffine 處理 |
| Inference options | 官方 `flip_test=True`、`shift_heatmap=True`；不在本輪調整 |
| Checkpoint integrity | 尚未下載，SHA-256 尚未建立；檔名短碼不當作完整 SHA-256 |

來源：[固定版 config](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py)、
[model zoo](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/configs/animal_2d_keypoint/topdown_heatmap/ap10k/hrnet_ap10k.md)、
[bbox transform](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/mmpose/datasets/transforms/common_transforms.py)。

| 本機唯讀檢查 | 2026-11-08 實際結果 |
| --- | --- |
| OS／architecture | macOS 26.5.1，arm64 |
| Python | 3.11.14 |
| PyTorch／TorchVision | 2.9.1／0.24.1 |
| NumPy／OpenCV／SciPy | 2.4.6／5.0.0.93／1.17.1 |
| MMPose、MMCV、MMCV-lite、MMEngine、MMDetection、xtcocotools | 均未安裝於既有 `.venv` |
| CPU smoke check | torch 匯入成功；`tensor([1., 2.]) * 2` 得到 `[2., 4.]` |
| CUDA | available=False |
| MPS | built=True、available=False；只描述本次執行程序的可見狀態 |
| Build tools | `/usr/bin/clang` 與 Xcode Command Line Tools 路徑存在；未編譯 MMCV |

官方提供 macOS／CPU 執行與 Apple Silicon 的 MMCV 建置路徑，並不等於測過此機器的完整套件組合。
因此本機相容性結論是「有可行部署路徑，仍需實際安裝、載入與 forward 驗證」。
來源：[MMPose 安裝](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/docs/en/installation.md)、
[MMCV macOS 建置](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-on-macos)。

| 依賴查核 | 結果與處置 |
| --- | --- |
| MMPose 1.3.2／MMEngine 0.10.7 | PyPI 有通用 Python wheel；這只證明套件檔可取得 |
| MMCV 2.2.0 | 查閱的 PyPI release 只有 source tarball；未窮盡其他 wheel index，不宣稱全球沒有 arm64 wheel |
| MMCV-lite 2.2.0 | 雖有通用 wheel，但 stock MMPose 會經 models→heads→EDPose 匯入 `mmcv.ops.MultiScaleDeformableAttention`，不能當作無條件替代方案 |
| xtcocotools 1.14.3 | 有 CPython 3.11 macOS universal2 wheel；仍須驗證與實際 NumPy 的二進位相容性 |
| 舊相依套件 | MMPose runtime 宣告 chumpy 等套件；Python 3.11、NumPy 2.x 的安裝及匯入風險需在隔離環境處理，未實測為失敗 |
| 現有環境保護 | 不在原 `.venv` 執行升降級；提議另建 `.venv-eye-pose`，透過 JSON／檔案交換資料 |

PyPI 來源：[mmpose](https://pypi.org/project/mmpose/1.3.2/)、[mmcv](https://pypi.org/project/mmcv/2.2.0/)、
[mmcv-lite](https://pypi.org/project/mmcv-lite/2.2.0/)、[mmengine](https://pypi.org/project/mmengine/0.10.7/)、
[xtcocotools](https://pypi.org/project/xtcocotools/1.14.3/)。
匯入鏈來源：[heads](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/mmpose/models/heads/__init__.py)、
[EDPose](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/mmpose/models/heads/transformer_heads/edpose_head.py)、
[runtime dependencies](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/requirements/runtime.txt)。

## Proposed POC Stages

### Stage A Compatibility Validation

| 步驟 | 工作與驗證 |
| --- | --- |
| 1 | 建立獨立 Python 3.11 環境；以 MMPose v1.3.2、MMCV 2.2.0、MMEngine 0.10.7 為起始候選。其餘相依版本經解析與匯入驗證後固定，這不是已驗證 lock |
| 2 | 在獨立環境解決相依性，必要時編譯 CPU ops；可調整該環境的 torch／NumPy 等版本並記錄，不更動原環境，不替換模型或偷偷修上游演算法 |
| 3 | 僅下載指定 HRNet checkpoint，建立 SHA-256 與來源紀錄；不下載完整 AP-10K／COCO，不額外載入人體或 ImageNet 權重 |
| 4 | 驗證官方註冊、AP-10K metadata、權重參數載入與 17 點輸出；舊 checkpoint 的官方轉換可用，但未解釋的 missing／unexpected keys 視為失敗 |
| 5 | 固定以 image 20247 的既有 `image_20247_prediction_001` 動物預測框做一次 CPU FP32 試跑；只看工程輸出是否有效，不把此試跑當獨立準確度樣本 |
| 6 | 驗證原圖→官方 affine→原圖的座標還原，以及水平翻轉的索引交換；記錄時間、記憶體與所有錯誤，凍結實際環境 |

seed=42、CPU FP32、batch=1、4 threads；保留官方 flip test，關閉訓練與隨機增強。
Stage A 驗證通過才能進 Stage B。若需要改模型、另取權重、修改上游核心程式或使用 Python<3.11，
回報具體原因供負責人選擇，不在失敗後默默改走其他架構。

### Stage B Automatic Prompts and Segmentation

```text
Frozen YOLO prediction boxes + original RGB
  -> HRNet-W32/AP-10K with native preprocessing
  -> named eye points and raw scores in original coordinates
  -> deterministic eye crops and positive point prompts
  -> existing SAM 2.1 Hiera-Tiny
  -> original-resolution masks and candidate centroid distances
  -> independent evaluation using frozen references
```

| 設定 | 第一輪提案 |
| --- | --- |
| Images | COCO 2017 val2017：20247、125405、333069；原三張目的性開發圖，不稱獨立測試 |
| Upstream | 重播凍結的 `outputs/phase2/yolo26s_seg_selected15_cpu/prediction_instances.json`；confidence≥0.5，沿用既有診斷門檻，不用 GT 挑框 |
| Candidate count | 唯讀核對上述條件得到 6 個預測動物，每圖 2 個；最多 12 個眼睛槽位，不預設全部成功 |
| Identity | 推論全程用 prediction_id；人工 annotation_id、既有 eligible 清單、眼側參考只在獨立評分使用 |
| Pose | 原始 RGB＋YOLO bbox 交由官方 top-down pipeline；不把動物 Mask 外塗黑，不另接 MMDetection |
| SAM | 原先固定 commit／checkpoint 的 SAM 2.1 Hiera-Tiny，沿用 CPU FP32、multimask=True、最高模型品質候選、logits>0、無補洞或平滑 |
| Prompt type | 每眼一個 HRNet 正點，第一輪不輸入眼睛框；`prompt_source=hrnet_ap10k` |
| ROI | 由預測雙眼距離 d 決定，邊長 L=max(32, 2d) 原圖 px；以各眼為中心取正方形，floor 起點／ceil 終點，裁至原圖範圍 |
| Validity | 兩眼座標與分數須有限、分數>0、座標在原圖與官方 padded animal crop 內、d>0；否則保留失敗原因，不補點、不另以 GT 產生尺度 |
| Score policy | 保存原始熱圖分數；上述>0只排除無效熱圖，不稱可靠性／可見性門檻，不搜尋最佳 score cutoff |
| Outputs | 每個候選點、ROI、Mask 與幾何結果都保留；`visibility_status=unverified`，不以非空 Mask 宣告可量測 |

ROI 的 32 px 下限沿用既有開發流程的最小上下文概念，2d 是本提案的固定尺度假設，尚未驗證最優。
第一輪不調參；雙眼距離預測錯誤會同時影響裁切尺度，須列為錯誤來源。
此規則需要兩個有效眼點，因此不涵蓋「只可靠定位到一隻眼睛仍繼續量測」的側臉流程。
若結果顯示尺度或單點歧義是主要問題，下一輪再提出頭部模型、眼睛框或不同 ROI 規則。

上下游 crop／resize／padding／affine 變換皆保存，不能只用固定比例放大，也不能重複套用 +0.5 像素偏移。
眼睛點偏到毛髮時，SAM 有可能產生完整但錯誤的 Mask；幾何檢查只辨識可計算性，不證明語義正確。

## Evaluation and Failure Accounting

沿用第一輪 12 眼凍結參考，snapshot SHA-256：
`3a3b8b98fba72ce79fd8416138ecd62c26051e528844f9d0ff70fdcacc22af13`。
GT 不進入定位、ROI 產生、候選選取、門檻調整或推論身份對應。

| 層次 | 指標與限制 |
| --- | --- |
| 動物恢復 | 在評分階段沿用同類、score≥0.5、animal mask IoU≥0.5 的既有配對規則；漏檢、重複與未配對預測都報告 |
| 提示定位 | 預測點是否落入對應 eye Mask；到 Mask 重心的原圖 px 偏移。沒有獨立 Landmark GT，不報瞳孔定位 PCK／AP |
| 身份與側別 | 先固定動物配對，再按命名眼側評分；診斷是否落到另一側眼。不得用 GT 對調左右後報主成績；模糊案例列 unresolved |
| Segmentation | 原圖完整格點的 IoU／Dice，分母保留 12 眼；GT 在自動 ROI 外仍計入 FN，不裁掉難例 |
| Geometry | Mask 重心偏差；同隻雙眼距離最多 6 筆、跨動物右眼距離最多 3 筆，分別報 MAE／RMSE 與有效筆數 |
| Missing outcomes | 推論失敗／無效點／無效 ROI／空 Mask 分開記錄。主摘要可另列全目標失敗以零分計入的 IoU／Dice；不可算中心與距離保持缺值 |
| Extra predictions | 未匹配動物／眼睛候選另報，不只展示匹配成功者；本批雖有 6 個符合門檻的動物預測，仍檢查配對結果 |
| Runtime | 分列 pose 載入、pose 推論（含 flip test）、ROI、SAM、輸出成本；重播 YOLO 不偽稱本輪重新跑過其耗時 |
| Baseline comparison | 與既有人工一框＋一點結果並列。但新流程同時改變提示來源、提示種類與 ROI，差異只能稱整體流程差異，不能全歸因於定位模型 |

這三圖原本經人工選為雙眼可見候選，因此無法驗證全自動可見性篩選或廣泛泛化。
未來需有閉眼、遮擋、側臉、小眼睛、擁擠案例及按原 image_id 分組的獨立留出集；本輪不自行擴增資料。
AP-10K／SAM 預訓練與 COCO 原圖重複情形未完整稽核，不宣稱嚴格 zero-shot 或零資料洩漏。

## Deliverables and Approval Scope

核准後產物集中至新目錄 `outputs/phase3/auto_eye_localization_poc_v1/`，既有產物不覆寫。

| 產物 | 內容 |
| --- | --- |
| compatibility／environment manifest | 實際依賴 lock、版本、上游 commit、權重 SHA-256、CPU smoke 輸出與載入診斷 |
| input manifest | 原圖、凍結 YOLO 預測、推論設定與評分參考的 digest；推論與評分輸入分開 |
| pose_predictions.json | 每動物 17 點、分數、prediction_id、AP-10K metadata、完整座標變換 |
| auto_prompts.json／masks／overlays | 預測衍生點、ROI、選取候選、原圖 Mask、診斷疊圖 |
| metrics／failure report | 12 眼與最多 9 筆距離、漏檢／錯配／無效值、逐例結果與限制 |
| reproducible runner | 實際可執行命令、配置、seed、所有新增來源與必要幾何測試；不預先提供不存在的 CLI |

必要測試聚焦 bbox xywh／xyxy 轉換、官方 affine 還原、翻轉側別交換、裁切邊界、錯誤分母與無效量測，
以及推論程式不讀眼部 GT。驗證通過後鎖定相依性，後續不得只寫未釘版本的 install 指令。

| 待核准決策 | 建議 |
| --- | --- |
| 執行範圍 | A 通過後執行 B；不訓練、不下載完整資料集、不擴圖 |
| 定位模型 | 指定 HRNet-W32／AP-10K checkpoint 與 MMPose v1.3.2 |
| 環境 | 新 `.venv-eye-pose`，CPU FP32；在其中解決依賴，不升降級原 `.venv` |
| 分割與提示 | 暫用既有 SAM 2.1 Hiera-Tiny；每眼一個預測正點＋上述固定 ROI 規則 |
| 科學結論邊界 | 自動產生分割與幾何候選的開發 POC；可見性及量測資格不宣告全自動解決 |
| 成功判準 | A 以可載入、有限輸出與座標正確為工程通過條件；B 完整報告結果，不臨時設定眼睛準確度合格門檻 |

依本次「方案給我確認」要求與 AGENTS.md 的模型／方法核准規則，先交付此提案，等待負責人決定是否執行 A→B。

## Review Execution Record

本次完成官方定義、類別 metadata、固定版本模型設定與匯入鏈的查核，以及本機環境與既有預測的唯讀檢查。
未安裝套件、下載 checkpoint、執行 HRNet、訓練、修改既有模型程式或人工參考。
官方標註示意圖及 PyPI release metadata 暫存於系統暫存目錄，未將資料集影像或二進位權重加入 Git。

| 已執行命令／檢查 | 結果 |
| --- | --- |
| `git status --short`、`rg --files`、`cat`／`sed`／`rg` | 核對專案規範、配置、既有報告與程式；保留原有未提交修改 |
| `.venv/bin/python`＋`importlib.metadata`／`platform` | 取得上表實際套件版本與 arm64 平台資訊 |
| `.venv/bin/python`＋torch CPU／CUDA／MPS probes | CPU 張量運算通過；CUDA=False，MPS built=True／available=False |
| `.venv/bin/python`＋JSON 篩選 | 既有三圖、score≥0.5 共 6 個 YOLO 動物預測，每圖 2 個 |
| `command -v uv`、`command -v clang`、`xcode-select -p` | uv、clang 與 Command Line Tools 路徑存在 |
| Python HTTPS metadata fetch | 官方示意圖與 5 個指定 PyPI release metadata 取得成功；一般沙箱 DNS 不通後，以核准的唯讀網路請求完成 |
| 文件與 JSON 驗證 | 檢查 54 類／23 family、關鍵點索引、相對連結及既有受保護檔案 digest |
| pytest／模型效能測試 | 本輪沒有程式功能變更，未重跑 pytest；沒有 HRNet 耗時或準確度數字 |

## Full AP-10K Category Metadata

下表保留官方原始英文類別與 family 字串，不替作者修正分類學命名。
論文也說明其 genus／species 分組採簡化定義，因此 54 是資料集類別數，不作嚴格生物分類認證。
來源：[官方類別 JSON](https://github.com/AlexTheBad/AP-10K/blob/181b1a04755e4dc6fe5616ef7a88496f47bfe228/tests/data/ap10k/test_ap10k.json)、
[論文 Data organization](https://arxiv.org/html/2108.12617)。

| Family label | Count | Category names |
| --- | ---: | --- |
| Bovidae | 6 | antelope, argali sheep, bison, buffalo, cow, sheep |
| Canidae | 4 | arctic fox, dog, fox, wolf |
| Castoridae | 1 | beaver |
| Cercopithecidae | 5 | alouatta, monkey, noisy night monkey, spider monkey, uakari |
| Cervidae | 2 | deer, moose |
| Cricetidae | 1 | hamster |
| Elephantidae | 1 | elephant |
| Equidae | 2 | horse, zebra |
| Felidae | 10 | bobcat, cat, cheetah, jaguar, king cheetah, leopard, lion, panther, snow leopard, tiger |
| Giraffidae | 1 | giraffe |
| Hippopotamidae | 1 | hippo |
| Hominidae | 3 | chimpanzee, gorilla, orangutan |
| Leporidae | 1 | rabbit |
| Mephitidae | 1 | skunk |
| Muridae | 2 | mouse, rat |
| Mustelidae | 2 | otter, weasel |
| Procyonidae | 1 | raccoon |
| Rhinocerotidae | 1 | rhino |
| Sciuridae | 2 | marmot, squirrel |
| Suidae | 1 | pig |
| Talpidae | 1 | mole |
| Ursidae | 4 | black bear, brown bear, panda, polar bear |
| Vespertilionidae | 1 | bat |
