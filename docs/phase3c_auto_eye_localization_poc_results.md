# Automatic Eye Localization POC Results

## Decision Summary

日期：2026-11-08（Asia/Taipei）。使用者已核准 A 通過後執行 B；**兩階段均已完成**。

| 層次 | 實際結果 | 解讀 |
| --- | --- | --- |
| A 相容性 | 官方 HRNet-W32/AP-10K 權重 strict load 全部對上；758 個官方程式／配置檔雜湊通過 | 此 macOS arm64／Python 3.11 CPU 組合可執行 |
| 動物配對 | 6/6 個既有 YOLO 預測與 6 個參考動物匹配；無額外預測 | 本輪重播已凍結預測，未重新計時 YOLO |
| 眼睛定位 | 12/12 點通過固定幾何檢查；9/12 落在命名側眼睛參考內；0/12 落在對側眼睛內 | 側別沿用 AP-10K 通道，未按畫面 x 重排或用 GT 對調 |
| 眼睛分割 | 12/12 非空；平均 IoU 0.478、Dice 0.575 | 可計算不代表語義正確；有毛髮／大片頭部分割 |
| 同隻雙眼距離 | 6/6 定義；MAE 2.076 px、RMSE 3.038 px | 沿用可見眼裂 Mask 重心端點 |
| 跨動物右眼距離 | 3/3 定義；MAE 3.766 px、RMSE 4.561 px | 候選幾何結果，非自動核准的可靠量測 |

本輪證明已核准流程能自動執行。準確度仍受單點歧義與小眼睛定位影響，不能宣稱量測資格、可見性或泛化已解決。

![12-eye diagnostic contact sheet](../outputs/phase3/auto_eye_localization_poc_v1/visualizations/all_12_eyes.png)

## Frozen Protocol

| 項目 | 固定設定 |
| --- | --- |
| Dataset | COCO 2017 val2017；20247、125405、333069，原先目的性挑選的 3 張開發圖 |
| Animals | bear、dog、giraffe 各 2；來自凍結 YOLO 預測，confidence ≥ 0.5 |
| Pose | HRNet-W32/AP-10K；17 點、256×256 輸入、17×64×64 熱圖；官方 padding 1.25、flip test、shift heatmap |
| MMPose | 1.3.2，commit `5408bc76f5b848cf925a0d1857899011d8c5b497` |
| Checkpoint | `hrnet_w32_ap10k_256x256-18aac840_20211029.pth`；SHA-256 `18aac840eee49f190e1443d93c6c23784920d2fd90813a42d35c918ec23a36e9` |
| Prompts | index 0=L_Eye、1=R_Eye；每眼 1 個預測正點、無眼框；原生座標不另加 0.5 |
| Eye ROI | L=max(32,2d)；由預測雙眼距離 d 決定，floor 起點、ceil 終點、裁至原圖 |
| Validity | 兩眼點與分數有限、分數>0、點在原圖及 padded animal crop 內、d>0；無 GT 或人工補點 |
| SAM | 既有固定 commit/checkpoint 的 SAM 2.1 Hiera-Tiny，三候選中取最高模型品質；logits>0；不補洞或平滑 |
| Execution | CPU FP32、batch=1、threads=4、seed=42；未訓練 |
| Reference | `eye-visible-aperture-v1`；原 12 眼人工參考 snapshot SHA-256 `3a3b8b98fba72ce79fd8416138ecd62c26051e528844f9d0ff70fdcacc22af13` |
| Evaluation | 推論只用 prediction_id；完成預測後，才用凍結動物配對及眼睛 GT 獨立評分 |

AP-10K 的左右眼定義、54 類 metadata、可見性限制與來源見[已核准提案](phase3c_auto_eye_localization_poc_proposal.md)。
模型不輸出本案眼裂 Mask 重心或獨立可見性判斷。定位偏移因此稱為「對參考 Mask 重心的偏移」，不是瞳孔 Landmark 誤差。
距離單位為原圖 px，未作相機或實體尺度校正。

## Local Compatibility

| 元件／檢查 | 驗證結果 |
| --- | --- |
| Host | macOS 26.5.1，arm64；Python 3.11.14；Apple clang 16.0.0 |
| Isolated environment | `.venv-eye-pose`；torch 2.2.2、torchvision 0.17.2、NumPy 1.26.4 |
| OpenMMLab | MMPose 1.3.2、MMCV 2.1.0（CPU ops 原始碼編譯）、MMEngine 0.10.7、MMDetection 3.3.0 |
| Other key dependencies | OpenCV 4.10.0.84、SciPy 1.13.1、xtcocotools 1.14.3；完整版本見 dependency lock |
| Weight loading | `<All keys matched successfully>`；未載入額外人體或 ImageNet 權重 |
| Metadata | AP-10K 17 點、前 3 名稱 L_Eye/R_Eye/Nose、flip indices 前 3 為 1/0/2 |
| Geometry | 實際官方 affine/inverse 與已知 QA 座標往返；flip roundtrip=0；保存熱圖重解碼最大差 <0.001 px |
| Repeat check | Stage A 與 B 重複動物的 17 點座標與分數完全相同；只是一個重複案例檢查 |
| Dependencies | `pip check` 通過；只驗證本案載入／推論路徑，不代表所有 MMPose 功能均相容 |
| Original environment | 套件版本清單未變；852 個既有受保護檔案中僅預期的 `.gitignore` 改動 |

原提案的 MMCV 2.2.0 是起始候選。檢視固定版本原始碼發現 MMPose 匯入 `RTMOHead` 時載入 `mmdet.utils`，
而 MMDetection 3.3.0 明定 MMCV <2.2.0，因此在使用者已核准的依賴調整範圍內改採 2.1.0。
未更換 HRNet、權重、動物偵測器或上游核心程式。
來源：[MMPose RTMO 匯入](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/mmpose/models/heads/hybrid_heads/rtmo_head.py)、
[MMDetection 版本限制](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/__init__.py)。

安裝器的 `git+file` 複製停滯，終止後改以已核對 commit 的官方 checkout 建 wheel；758 個已打包程式／配置檔逐檔比對。
官方另有 9 個 `mmpose/configs` 範例不打包進 wheel，已明列於來源 manifest；正式配置使用官方 `.mim/configs`。
兩次相容性啟動失敗分別來自本案 wrapper 的 Python symlink 解析與未打包範例的路徑對照，皆保留 log、修正後才執行成功的 attempt 03。
上游提示 AP-10K metadata 改用 `.mim/configs` 路徑；實際 dataset_meta 檢查通過，沒有回退成人體 COCO。

## Localization and Segmentation

定位偏移平均 2.248 px、最大 3.917 px；
SAM Mask 重心誤差平均 1.576 px、最大 6.850 px。
分母固定 12 眼；本輪沒有缺失或空 Mask。所有 GT 像素都在自動 ROI 內，因此沒有因裁切漏掉 GT 的 FN；
IoU 仍以完整原圖格點計算。

| Image | Animal ID | Side | HRNet score | Point in GT | Point-to-centroid px | IoU | Dice | Mask centroid error px |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 20247 | 587181 | left | 0.958 | Yes | 2.797 | 0.538 | 0.700 | 0.954 |
| 20247 | 587181 | right | 0.935 | Yes | 3.067 | 0.635 | 0.777 | 0.488 |
| 20247 | 588042 | left | 0.950 | Yes | 2.576 | 0.900 | 0.947 | 0.088 |
| 20247 | 588042 | right | 0.964 | Yes | 2.113 | 0.830 | 0.907 | 0.601 |
| 125405 | 13692 | left | 0.917 | Yes | 1.289 | 0.692 | 0.818 | 0.689 |
| 125405 | 13692 | right | 0.927 | No | 3.917 | 0.000 | 0.000 | 6.850 |
| 125405 | 17207 | left | 0.930 | Yes | 0.803 | 0.109 | 0.197 | 0.506 |
| 125405 | 17207 | right | 0.914 | Yes | 1.246 | 0.143 | 0.250 | 1.368 |
| 333069 | 599488 | left | 0.983 | No | 2.970 | 0.586 | 0.739 | 0.856 |
| 333069 | 599488 | right | 0.978 | Yes | 1.350 | 0.814 | 0.897 | 0.474 |
| 333069 | 599888 | left | 0.982 | Yes | 1.236 | 0.471 | 0.640 | 0.243 |
| 333069 | 599888 | right | 0.941 | No | 3.617 | 0.014 | 0.028 | 5.798 |

| 觀察到的案例 | 已觀察事實 | 限制或可能原因 |
| --- | --- | --- |
| dog 13692 right | 點在參考外；IoU=0，Mask 重心誤差 6.850 px；疊圖顯示上方毛髮被選中 | 高 heatmap score 未避免小眼睛提示偏移 |
| giraffe 599888 right | IoU=0.014；預測面積 1260 px、參考 18 px；大片頭部被選中 | 單點與較大 ROI 下的語義歧義；不能只靠非空／模型品質分數接受 |
| dog 17207 both eyes | 兩個點都在眼睛參考內，IoU 僅 0.109／0.143；遮罩涵蓋周圍區域 | 點命中不保證眼裂分割範圍正確 |
| giraffe 599488 left | 點在參考外，仍取得 IoU=0.586 | 點命中率與分割品質是不同指標 |

上述歧義成因是結合疊圖的診斷假說，未經消融實驗證明；本輪未依 GT 重新選候選或調參。

## Pixel Distance Measurements

| Image | Endpoints | Reference px | Automatic px | Absolute error px |
| --- | --- | ---: | ---: | ---: |
| 20247 | 587181 L–R | 28.368 | 28.088 | 0.280 |
| 20247 | 588042 L–R | 25.061 | 24.646 | 0.414 |
| 20247 | 587181 R – 588042 R | 385.235 | 385.109 | 0.125 |
| 125405 | 13692 L–R | 34.043 | 29.845 | 4.197 |
| 125405 | 17207 L–R | 25.166 | 23.706 | 1.460 |
| 125405 | 13692 R – 17207 R | 275.428 | 269.819 | 5.609 |
| 333069 | 599488 L–R | 32.844 | 33.004 | 0.160 |
| 333069 | 599888 L–R | 23.325 | 17.379 | 5.946 |
| 333069 | 599488 R – 599888 R | 145.726 | 151.288 | 5.562 |

同隻雙眼最大誤差 5.946 px；跨動物右眼最大誤差 5.609 px。
這些誤差是對既有人工參考的差異，未將有限樣本的誤差當作已校準不確定度或信賴區間。
所有候選仍標記 `visibility_status=unverified`；本輪三圖原本人工選為雙眼可見，不能據此驗證自動可見性篩選。

## Comparison with Human Prompts

| SAM 2.1 Tiny pipeline | Mean IoU | Mean Dice | Centroid error px | Binocular MAE px | Right-eye pair MAE px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 既有人工點＋框／原 ROI | 0.596 | 0.723 | 0.713 | 0.610 | 0.492 |
| 本輪 HRNet 單點／自動 ROI | 0.478 | 0.575 | 1.576 | 2.076 | 3.766 |

此比較同時改變提示來源、提示種類及 ROI，只能描述整體流程差異，不能把全部差距歸因於 HRNet。
未重跑 SAM ViT-B 或 MobileSAM，未據三圖結果選定最終模型。

## Runtime Observations

| CPU FP32 工作 | 本次單次執行 |
| --- | ---: |
| HRNet model load | 0.240 s |
| HRNet 6 動物推論（含 flip test） | 0.585 s；平均 0.098 s/animal |
| HRNet worker peak RSS | 734.9 MiB |
| ROI 幾何計算合計 | 0.000526 s |
| SAM model load | 1.111 s |
| SAM 12 眼推論 | 4.484 s；平均 0.374 s/eye |
| Mask／logits 寫檔合計 | 0.012 s |

以上為一次執行觀察，沒有 warm-up／多次分布，並非端到端 latency benchmark。
YOLO 使用重播；安裝、匯入、模型載入、評分、圖表與全部 I/O 成本不可由平均推論時間代替。

## Reproduction

配置：[auto_eye_localization.toml](../configs/auto_eye_localization.toml)。
完整依賴：[eye_pose_lock_macos_arm64.txt](../configs/eye_pose_lock_macos_arm64.txt)。
來源雜湊：[eye_pose_source_manifest.json](../configs/eye_pose_source_manifest.json)。
既有 SAM 使用原 `.venv`；HRNet subprocess 使用隔離環境。所有研究輸入與輸出都留有 SHA-256。

以下安裝順序供新的 macOS arm64 checkout／尚未存在的 `.venv-eye-pose` 使用；不要覆蓋本次已完成環境：

```bash
.venv/bin/python -m venv .venv-eye-pose
.venv-eye-pose/bin/python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.48.0 numpy==1.26.4 torch==2.2.2
.venv-eye-pose/bin/python -m pip install --no-build-isolation -r configs/eye_pose_lock_macos_arm64.txt
env PATH="$PWD/.venv-eye-pose/bin:$PATH" MMCV_WITH_OPS=1 MAX_JOBS=4 .venv-eye-pose/bin/python -m pip install --no-deps --no-build-isolation 'https://files.pythonhosted.org/packages/a7/19/f97f0fd4d432113a232e75f965335be6b7bb6557ae9faf8051e9a012cadd/mmcv-2.1.0.tar.gz#sha256=d387bcab66b467479b6660310e23746cfc79c6e57acf04094680adb499a5cd3f'
POSE_SOURCE="$(mktemp -d /tmp/animal-eye-pose-source.XXXXXX)"
git init "$POSE_SOURCE"
git -C "$POSE_SOURCE" remote add origin https://github.com/open-mmlab/mmpose.git
git -C "$POSE_SOURCE" fetch --depth=1 origin 5408bc76f5b848cf925a0d1857899011d8c5b497
git -C "$POSE_SOURCE" checkout --detach FETCH_HEAD
.venv-eye-pose/bin/python -m pip install --no-deps --no-build-isolation "$POSE_SOURCE"
.venv-eye-pose/bin/python -m pip check
```

指定 HRNet checkpoint 的[官方下載來源](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.pth)
與 digest 見前表；本機已存在 `weights/eye_pose/`，不需重下載。
完整原始套件清單、來源及安裝失敗紀錄在 `../outputs/phase3/auto_eye_localization_poc_v1/compatibility/`。

本次實際依序執行的研究命令如下。要重跑推論，先複製 TOML 並將 output_directory 改成新目錄，
建立其 `compatibility/` 子目錄，所有階段加上 `--config 新配置`；程式拒絕覆寫已完成輸出。

```bash
.venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage prepare
.venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage compatibility
# Only after Stage A passed:
.venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage pose
.venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage sam
.venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage evaluate
.venv/bin/python -m animal_eye_metrology.auto_eye_visualization
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
.venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage finalize
.venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage verify
```

驗證現有結果只需最後一行，不重新執行模型。驗證器重建最高分候選遮罩、檢查所有檔案 SHA-256，
再從凍結參考重算 12 眼與 9 筆距離。

## Validation and Deliverables

| 產物 | 用途 |
| --- | --- |
| [summary.json](../outputs/phase3/auto_eye_localization_poc_v1/summary.json) | 固定分母與缺失／空遮罩計數 |
| [eye_metrics.csv](../outputs/phase3/auto_eye_localization_poc_v1/eye_metrics.csv) | 每眼點位、分數、命中、IoU／Dice、Mask 重心誤差 |
| [measurement_comparison.csv](../outputs/phase3/auto_eye_localization_poc_v1/measurement_comparison.csv) | 全部 9 筆距離及誤差 |
| [pose_results.json](../outputs/phase3/auto_eye_localization_poc_v1/pose/pose_results.json) | 全部 17 點、原生熱圖、座標矩陣、官方來源與環境 |
| [segmentation.json](../outputs/phase3/auto_eye_localization_poc_v1/segmentation.json) | 自動點、ROI、三候選品質、選取結果、Mask／logits 路徑 |
| [12-eye contact sheet](../outputs/phase3/auto_eye_localization_poc_v1/visualizations/all_12_eyes.png) | 已視覺檢查的逐眼診斷圖 |
| [bear](../outputs/phase3/auto_eye_localization_poc_v1/visualizations/20247_measurements.png)／[dog](../outputs/phase3/auto_eye_localization_poc_v1/visualizations/125405_measurements.png)／[giraffe](../outputs/phase3/auto_eye_localization_poc_v1/visualizations/333069_measurements.png) | 原圖量測疊圖 |
| [compatibility passed](../outputs/phase3/auto_eye_localization_poc_v1/compatibility/passed.json) | Stage A 成功證據與固定 worker digest |
| [runtime_summary.json](../outputs/phase3/auto_eye_localization_poc_v1/runtime_summary.json) | 實際耗時、peak RSS 與 A/B 重複案例一致性 |

新增程式：`auto_eye_localization.py`（管線／評分／重算）、`eye_pose_worker.py`（官方模型隔離程序）、
`auto_eye_visualization.py`（保存結果的圖表）；新增 `test_auto_eye_localization.py`。
新增 4 個配置／來源記錄檔；`.gitignore` 加入 `.venv-eye-pose/`；沒有修改原有模型程式、人工 GT 或舊輸出。

新增測試使用合成 QA 資料，涵蓋虛擬環境 executable、bbox 格式、解剖側別保留、ROI 邊界／無效點、
推論輸入不依賴眼部 GT、漏檢／空 Mask 分母、左右不自動修復、logits 原圖還原及候選選取。
最終結果：**107 項 pytest 通過、Ruff 通過；165 個產物雜湊通過，12 眼與 9 筆距離重算一致**。
pytest 有 53 個來自既有 pycocotools／NumPy 介面的 DeprecationWarning，無失敗。
詳見 [pytest log](../outputs/phase3/auto_eye_localization_poc_v1/compatibility/pytest.log)、
[Ruff log](../outputs/phase3/auto_eye_localization_poc_v1/compatibility/ruff.log)、
[validation JSON](../outputs/phase3/auto_eye_localization_poc_v1_validation.json)。

## Limits and Recommended Next Decision

| 限制 | 下一步建議，尚未執行 |
| --- | --- |
| 單點語義歧義與 ROI 同時影響分割 | 先提出固定資料／ROI 的提示消融，再比較可自動產生的眼框或局部 refinement，避免直接把問題歸因於 SAM 型號 |
| 可見性尚未自動驗證 | 後續核准後加入遮擋、閉眼、側臉與拒絕輸出規則；heatmap score 不作可見性機率 |
| 僅三張目的性開發圖 | 新選樣與按原 image_id 分組的獨立留出集需要另行核准；不報泛化結論 |
| 參考本身有人工變異 | 沿用既有 blind-repeat 結果作背景，不把本輪偏差直接當完整量測不確定度 |
| 預訓練圖像重複未稽核 | 不宣稱嚴格 zero-shot 或零資料洩漏 |
| 平台範圍 | 只驗證本機 CPU；MPS、CUDA、Linux 容器與產品 UI 整合未在本輪驗證 |

建議先核准針對已觀察失敗模式的提示／ROI 對照方案，再決定是否更換定位或分割架構。
