# Fixed-Prompt Eye Segmentation POC Results

## Outcome and Review Scope

2026-11-08 完成核准的三模型固定人工提示比較：3 張圖片、6 隻動物、12 個不同眼睛，
每模型 12 次，共 36 次正式推論；主參考共 36 筆逐眼結果、27 筆距離結果。
所有程序完成、36 個 Mask 均非空，沒有刪除目標、換提示、補點、調 threshold 或以 GT 挑選候選。
主流程記錄約 95.694 秒，包含版本凍結、每眼分程序載入模型、推論、評分與圖片輸出；結尾唯讀重算另計。

以第一輪固定人工參考的平均 IoU、Dice、重心偏差與兩種距離 MAE 衡量，SAM 2.1 Hiera-Tiny 在這批資料領先。
但 SAM ViT-B 的重心偏差中位數較小；三者也都有低 IoU、預測範圍大於人工參考的案例。
本報告不據此自動選定最終模型或宣布分割合格，沒有新設數值通過門檻。

這是提供人工框與點的半自動分割開發 POC，動物身份與解剖左右也由人提供。
誤差指對人工可見眼部 Mask 的偏差，不是已校正的物理誤差或瞳孔／眼球中心準確度。

## Primary Mask and Centroid Results

每模型分母 12 眼；IoU、Dice 是重疊指標，不是正確像素比例或模型信心值。
重心偏差為預測與參考 Mask 重心之間的原圖歐氏距離。

| 模型 | 平均 IoU | 平均 Dice | 重心偏差平均 px | 中位數 px | 最大 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM 2.1 Hiera-Tiny | 0.596 | 0.723 | 0.713 | 0.586 | 2.068 |
| SAM ViT-B | 0.564 | 0.695 | 0.755 | 0.519 | 2.273 |
| MobileSAM ViT-T | 0.552 | 0.686 | 0.806 | 0.684 | 2.986 |

三者均為 12／12 個重心可計算、6／6 隻動物雙眼距離可計算、3／3 組右眼距離可計算，
每張四眼皆可算的圖片為 3／3。這些是幾何完整性，沒有把非空 Mask 自動當成正確或 eligible。
各眼面積、交集／聯集、原圖座標、Δx／Δy、版本及來源在 eye_metrics.csv。

## Distance Errors

雙眼與右眼配對各自統計，不把 6 筆短距離與 3 筆長距離混成一個數字。
MAE 是距離絕對差的平均；RMSE 保留較大偏差的影響。

| 模型 | 量測 | 筆數 | MAE px | RMSE px | 最大絕對差 px | 平均絕對百分比差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SAM 2.1 Hiera-Tiny | 同隻雙眼 | 6 | 0.610 | 0.655 | 0.919 | 2.181% |
| SAM ViT-B | 同隻雙眼 | 6 | 0.731 | 0.953 | 1.943 | 2.730% |
| MobileSAM ViT-T | 同隻雙眼 | 6 | 0.853 | 1.025 | 1.588 | 3.124% |
| SAM 2.1 Hiera-Tiny | 兩動物右眼 | 3 | 0.492 | 0.549 | 0.807 | 0.221% |
| SAM ViT-B | 兩動物右眼 | 3 | 0.558 | 0.578 | 0.678 | 0.264% |
| MobileSAM ViT-T | 兩動物右眼 | 3 | 0.540 | 0.594 | 0.861 | 0.241% |

以下列出九個固定配對，三模型欄為相對同一參考距離的絕對差：

| Image ID | 配對 | 人工參考 px | SAM 2.1 Tiny 絕對差 px | SAM ViT-B 絕對差 px | MobileSAM 絕對差 px |
| --- | --- | ---: | ---: | ---: | ---: |
| 20247 | 587181 雙眼 | 28.368 | 0.657 | 0.171 | 0.443 |
| 20247 | 588042 雙眼 | 25.061 | 0.919 | 1.011 | 1.586 |
| 20247 | 587181／588042 右眼 | 385.235 | 0.212 | 0.344 | 0.262 |
| 125405 | 13692 雙眼 | 34.043 | 0.430 | 0.483 | 0.670 |
| 125405 | 17207 雙眼 | 25.166 | 0.490 | 1.943 | 1.588 |
| 125405 | 13692／17207 右眼 | 275.428 | 0.807 | 0.651 | 0.861 |
| 333069 | 599488 雙眼 | 32.844 | 0.891 | 0.597 | 0.784 |
| 333069 | 599888 雙眼 | 23.325 | 0.275 | 0.182 | 0.044 |
| 333069 | 599488／599888 右眼 | 145.726 | 0.457 | 0.678 | 0.497 |

完整預測距離、有號差、絕對差、百分比差與兩端點座標在 measurement_comparison.csv。
右眼長距離的百分比小，並不代表端點更準；也可能有位移方向抵消的影響。
表格只顯示三位小數，CSV 保留未四捨五入的計算值，不表示 0.001 px 的實際量測精度。

## Per-Image Diagnostics

每列平均四眼；僅是三張特定開發圖，不當作 dog／bear／giraffe 的類別泛化成績。

| Image ID／動物 | 模型 | 平均 IoU | 平均重心偏差 px | 雙眼 MAE px | 右眼配對絕對差 px |
| --- | --- | ---: | ---: | ---: | ---: |
| 20247 bear | SAM 2.1 Tiny | 0.689 | 0.859 | 0.788 | 0.212 |
| 20247 bear | SAM ViT-B | 0.658 | 0.832 | 0.591 | 0.344 |
| 20247 bear | MobileSAM | 0.626 | 1.108 | 1.015 | 0.262 |
| 125405 dog | SAM 2.1 Tiny | 0.444 | 0.667 | 0.460 | 0.807 |
| 125405 dog | SAM ViT-B | 0.443 | 0.720 | 1.213 | 0.651 |
| 125405 dog | MobileSAM | 0.422 | 0.670 | 1.129 | 0.861 |
| 333069 giraffe | SAM 2.1 Tiny | 0.656 | 0.612 | 0.583 | 0.457 |
| 333069 giraffe | SAM ViT-B | 0.591 | 0.712 | 0.390 | 0.678 |
| 333069 giraffe | MobileSAM | 0.607 | 0.641 | 0.414 | 0.497 |

優先 review 這三個觀察，不直接修改人工參考或模型輸出：

| 案例 | 觀察 | 對研究的意義 |
| --- | --- | --- |
| 125405／17207／right | 三模型面積 78／86／94 px，人工參考 14 px；IoU 0.179／0.163／0.149 | 模型區域明顯較大；需依既有眼裂規範檢查是否納入眼周，不能單憑分數決定人工 GT 錯誤 |
| 125405／17207／left | MobileSAM 面積 71 px、參考 15 px，IoU 0.211，但重心只差 0.076 px | 低重疊的 Mask 仍可有接近的重心；量測看似接近不等於輪廓可靠 |
| 20247／588042／left | 三模型重心偏差 2.068／2.273／2.986 px，預測面積 70／64／78 px，參考 30 px | 是本批最大重心偏差來源；局部放大可見邊界差異，需檢查模糊小眼與模型分割範圍 |

所有主參考前景都位於各自人工提示衍生 ROI 內，沒有因裁切掉 GT 像素而漏算分母。
這不證明人工提示已最優，僅排除本輪的 GT 裁切遺失。

## Dog Reference Sensitivity

同一份 dog 四眼預測另對第二輪固定人工 Mask 評分，沒有重跑模型、換提示或挑每模型較有利的一輪。
下表兩欄都只使用同一張 dog 圖的四眼，避免把 12 眼平均與 4 眼平均直接比較：

| 模型 | 第一輪 dog 平均 IoU | 第二輪 dog 平均 IoU | 第一輪 dog 平均重心偏差 px | 第二輪 dog 平均重心偏差 px |
| --- | ---: | ---: | ---: | ---: |
| SAM 2.1 Tiny | 0.444 | 0.422 | 0.667 | 0.541 |
| SAM ViT-B | 0.443 | 0.363 | 0.720 | 0.462 |
| MobileSAM | 0.422 | 0.368 | 0.670 | 0.610 |

第二輪參考下 SAM ViT-B 的 dog 平均重心偏差最小，顯示局部排序會隨人工邊界改變。
同圖兩隻狗的雙眼距離 MAE 分別變為 0.270／0.564／0.480 px；右眼配對絕對差為 0.475／0.319／0.529 px。
這是參考標註敏感度，不是獨立測試或另一批樣本；主摘要仍固定用第一輪 12 眼。

## Frozen Inputs and Execution Policy

| 項目 | 設定 |
| --- | --- |
| 原圖與身份 | COCO 2017 validation；125405、20247、333069；沿用核准的 6 個 animal IDs |
| Prompt | 12 眼皆保存，全部採最新 v1；每眼一個人工框＋一個正點、相同 row 供三模型使用 |
| Primary GT snapshot | 3a3b8b98fba72ce79fd8416138ecd62c26051e528844f9d0ff70fdcacc22af13 |
| Repeat GT snapshot | a154caecdb43a9b471c2164bfceda3b71bf54950293e2dc72c418411eed42b98 |
| Model versions | 沿用相容性試跑的三個官方 commit、checkpoint 及其 SHA-256，沒有重新下載／更換模型 |
| Runtime | macOS arm64、Python 3.11.14、torch 2.9.1、CPU FP32、4 threads、batch=1、seed=42 |
| Sampling | 三張既有目的性開發圖，未重新抽樣；seed 42 用於推論初始化，不是抽樣 |
| ROI | 以人工框為中心，邊長 max(32,4×max(框寬,框高))，取原圖整數裁切範圍；不讀 GT 選框 |
| Candidate selection | multimask=True；取模型預測品質最高候選，平手取較小索引；不用真實 IoU 挑候選 |
| Mask | 原生 predictor 還原至 ROI 像素格的 logits，threshold=0，再放回原圖；無額外平滑、補洞或手修 |

先在提示檔鎖內選定版本，將 12 筆提示、原圖像素、各眼 request 與兩輪完整參考快照複製到新目錄。
模型子程序只讀該眼固定原圖、人工點／框與模型設定，不讀 Eye GT；獨立評分階段才使用參考 Mask。
來源、權重與原圖／提示 digest 逐次核對，計算期間程式來源不得改變。
每眼／每模型使用獨立子程序，沿用既有相容性 worker，避免新增另一套 predictor 介面。
人工 prompts.csv、兩輪人工標註、舊相容性產物與量測基準均未覆寫。

## Evaluation Formulas and Missing Values

```text
IoU = |P intersection G| / |P union G|
Dice = 2 |P intersection G| / (|P| + |G|)
C(M) = (mean(column+0.5), mean(row+0.5)) over nonzero foreground
center_error_px = Euclidean(C(P), C(G))
signed_distance_error = predicted_distance - reference_distance
absolute_distance_error = abs(signed_distance_error)
relative_absolute_error_percent = 100 * absolute_distance_error / reference_distance
MAE = mean(absolute_distance_error)
RMSE = sqrt(mean(signed_distance_error ** 2))
```

三值來源分開：正常非空預測、空預測、程序失敗。空預測對本輪非空 GT 的 IoU／Dice 為 0，
中心與相關距離未定義；程序失敗的原始 IoU／中心／距離保留空值與原因。
model_summary 的 all_targets_failure_as_zero 欄明確將程序失敗以零分計入 Mask 全目標平均，並同列失敗數；
不假裝有一張實際全黑預測。中心與距離只在可定義時計算，必須同時報告可計算筆數。
參考距離為零時百分比未定義，不能把缺值填成零誤差。
本次沒有程序失敗或空預測，因此摘要的各分母均完整，沒有這類懲罰分數影響排名。

## Runtime Observations

| 模型 | 12 眼平均推論 s／眼 | 推論總和 s | 每眼模型載入時間總和 s | 程序最大峰值 RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| SAM 2.1 Tiny | 0.506 | 6.071 | 9.470 | 1.350 |
| SAM ViT-B | 1.849 | 22.184 | 7.515 | 3.634 |
| MobileSAM | 0.341 | 4.095 | 8.260 | 0.861 |

推論時間為前處理＋encoder＋decoder＋原生 ROI 還原，不含模型載入、原圖／權重雜湊、程序啟動與檔案輸出。
run_manifest 的 wall_seconds=95.694，記錄截至寫入結案 manifest 前的凍結、推論、評分與圖表；結尾唯讀驗證另計。
不能把上表推論總和當作完整執行時間。
每眼皆重新載入模型，沒有生產服務的 encoder cache 或長駐模型；沒有暖機／多次重複的正式效能評估。
RSS 是 macOS 單一程序含 Python／模型／activation 的峰值，不是純模型大小或 GPU 記憶體。

## Deliverables and Visual Review

產物集中於 outputs/phase3/eye_segmentation_poc_v1/comparison_v1/：

| 檔案／目錄 | 數量與用途 |
| --- | --- |
| inputs/prompts.csv、proposed_instances.csv、runtime.toml | 固定 12 筆提示、6 筆身份與實際推論設定 |
| inputs/reference_primary/、reference_repeat/ | 兩輪固定人工參考副本及原有 manifest，原始來源不動 |
| input_manifest.json | COCO digest、提示來源／選定順序、參考 digest 與每個固定輸入 SHA-256 |
| cases/{image_animal_side}/input.png、request.json | 每眼相同原圖快照及點／框／ROI，供三模型共用 |
| cases/{eye}/{model}/ | 每模型每眼的原圖尺寸 mask.png、roi_logits.npy、overlay.png、result.json；共 36 組 |
| prediction_outcomes.json、各眼 model.log | 完整 36 次程序狀態、時間與原生警告／錯誤 |
| eye_metrics.csv | 36 筆主參考 Mask、原圖重心、版本與路徑 |
| measurement_comparison.csv | 27 筆主參考距離、誤差、端點座標與版本 |
| model_summary.csv、image_summary.csv | 三模型總覽與 9 筆逐圖摘要 |
| sensitivity_eye_metrics.csv、sensitivity_measurements.csv、sensitivity_summary.csv | 同一預測對第二輪 dog 參考的 12／9／3 筆資料 |
| overlays/ | 3 張逐眼比較圖與 9 張各模型原圖重心／距離對照圖 |
| provenance/、run_manifest.json | 計算與繪圖來源、設定、環境／執行命令及產物 digest |

建議先依以下順序 review；圖中的 L／R 都是動物解剖方向：

1. [Dog 125405 四眼比較](../outputs/phase3/eye_segmentation_poc_v1/comparison_v1/overlays/000000125405_eye_comparison.png)：看後方動物 17207 的左右眼區域。
2. [Bear 20247 四眼比較](../outputs/phase3/eye_segmentation_poc_v1/comparison_v1/overlays/000000020247_eye_comparison.png)：看 588042 左眼的重心偏移。
3. [Giraffe 333069 四眼比較](../outputs/phase3/eye_segmentation_poc_v1/comparison_v1/overlays/000000333069_eye_comparison.png)：比較輪廓與左右眼端點。
4. [SAM 2.1 Tiny dog 原圖距離](../outputs/phase3/eye_segmentation_poc_v1/comparison_v1/overlays/000000125405_sam2_1_tiny_measurements.png)：對照 annotation ID、雙眼與右眼連線。

逐眼圖金色為僅 GT、藍色為僅預測、白色為交集；+ 與 x 分別標參考／預測重心。
各欄使用相同的人框衍生 ROI；nearest-neighbor 放大不增加原圖資訊。
量測圖明確標示 predicted，v 為人工提示版本，不把預測當成新的人工 GT。
三張逐眼圖及一張原圖距離圖已實際開啟檢視身份、圖例、數字與版面。

## Reproduction and Validation

本次執行：

```sh
.venv/bin/python -m animal_eye_metrology.eye_comparison \
  --config configs/eye_segmentation_comparison.toml
```

既有輸出目錄拒絕覆寫。核准新實驗時另建設定與新 output_directory，保留本次資料。
既有數值的唯讀重算：

```sh
.venv/bin/python -m animal_eye_metrology.eye_comparison \
  --verify outputs/phase3/eye_segmentation_poc_v1/comparison_v1
```

重算核對全部產物 digest，從保存的 36 份 logits 還原原圖 Mask，並由固定預測及兩輪參考重新產生所有七份評分／摘要 CSV。
不重新推論、不讀現在的人工提示／GT 工作檔，也不需要即時 COCO 來源或 checkpoint。
數值重現已在同環境驗證；跨平台／版本的逐位元一致性未驗證。

run_manifest.json SHA-256：

```text
eac6d4e2d7f1f359ad6c25998e4f73a9850c49f25af0a5feb12127b59739a281
```

| 新增／修改檔案 | 變更 |
| --- | --- |
| src/animal_eye_metrology/eye_comparison.py | 固定輸入、重用模型 worker、完整評分與唯讀重算 |
| configs/eye_segmentation_comparison.toml | 核准範圍、固定參考 digest 與獨立輸出位置 |
| src/animal_eye_metrology/visualization.py | 新增逐眼比較；原有量測圖新增可選預測標籤，保留原始預設行為 |
| tests/test_metrology.py | 新增已知 3–4–5 中心偏差、距離差與空值，以及失敗／空預測分母測試 |
| README.md、本文件 | 更新結果、操作與限制 |

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests app
.venv/bin/ruff format --check src/animal_eye_metrology/eye_comparison.py \
  src/animal_eye_metrology/visualization.py tests/test_metrology.py
```

實際結果：95 passed，53 個既有 pycocotools／NumPy 棄用警告；Ruff check／format 通過。
36 個原圖 Mask 的 logits 還原、36 筆主眼部結果與 27 筆距離結果重算通過，包含第二輪敏感度資料。
開始前記錄的 806 個既有資料／規範／設定檔 SHA-256 保持不變；保留人工提示與標註、舊基準與相容性結果。
未新增相依套件、下載資料／模型、訓練、自動標註、改寫 CSV Schema、Commit 或 Push。

## Limitations and Suggested Decision

本批只有三張目的性選圖，12 眼及九筆距離共享動物／原圖，不能當成 12 或 9 個獨立影像樣本。
人工標註與提示由同一人提供，資料已參與規範開發及單眼相容性檢視，不是留出驗收；記憶／提示偏差無法排除。
也未完成預訓練資料與這些 COCO 原圖的完整重複影像稽核，不能宣稱無資料洩漏或獨立 zero-shot benchmark。
部分模型框出了更大的區域，但重心接近；反之，較高 IoU 不保證每一筆距離更準。
沒有信賴區間、母體顯著性或物理公分結論；Mask 重心仍不是瞳孔／眼球的解剖中心。

建議把 SAM 2.1 Hiera-Tiny 列為下一輪優先候選，保留 MobileSAM 的成本對照與 SAM ViT-B 的逐例結果。
先審查 17207 左右眼與 588042 左眼是否符合既有眼裂邊界規範，再由負責人決定下一步。
若要改提示預算、邊界定義或後處理，另立新實驗，不在這一輪結果上直接修圖提升成績。
本次停止等待 review，不自動選定最終模型、動物類別或啟動自動眼睛定位。
