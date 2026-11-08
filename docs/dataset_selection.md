# Phase 1 — Dataset Exploration and Selection

## Evidence Status

更新日期：2026-11-07。專案負責人已審查驗收結果並核准 **Phase 1 結案**。
保留 15 張 Selected Images，其中 13 張具至少兩隻完整雙眼候選，作為 Demonstration Candidates。
目前不限制單一動物類別；人工篩選結果不代表整體模型效能或已完成量測 Ground Truth。

| 項目 | 狀態與證據 |
| --- | --- |
| 資料範圍 | Phase 01 已指定 COCO 2017 validation 與 instances_val2017.json |
| 真實資料準備 | 使用者提供兩個壓縮檔，已依設定解壓驗證集註記與影像；程式沒有下載功能 |
| 資料完整性 | 5,000 張影像全部完整解碼、檔名及尺寸符合註記；詳見本機 preparation_manifest.json |
| 資料統計與候選數量 | 已產生 10 類統計，471 張任意多動物、450 張同類多動物、70 張不同類共存影像 |
| Contact sheets | 首次統計依設定關閉；後續已完成四類各 15 張的審核資料包，共 20 頁 |
| 眼睛可見性與解析度 | 已選圖片 47／47 個實例完成五項審核；eye_resolution 與 reviewer 仍留白 |
| 篩選結案 | 15 張 Selected、3 張 Rejected、453 張 Pending；不要求完成剩餘 453 張 |
| 眼睛 Ground Truth | 尚未建立；區分 Landmark 與 Segmentation，標註 UI 未獲核准 |
| 動物類別與模型 | 不限制單一動物類別；負責人其後更新 Phase 2 文件，指定 YOLO11n-seg 的 15 張 POC，現已完成待驗收 |
| 暫定測量方法 | Eye Segmentation Mask → Mask Centroid → Euclidean Distance；Phase 3 才評估實作 |
| 自動測試 | 使用明確標示的人工幾何資料驗證邏輯，不能視為 COCO 統計或模型效能 |

## Phase 1 Closure Decision

### Owner Approved Scope

專案負責人於本次對話核准結案；以下是當時的來源快照，不重新修改選圖或人工審核。
驗收資料位於 outputs/phase1/phase1_final_validation.json，候選清單為
outputs/phase1/measurement_feasibility.csv；原始人工資料分別保存在
image_selection.csv 與 eye_visibility_review_working.csv，兩者 Schema 維持不變。
驗收 JSON 的 phase1_owner_acceptance=pending 是核准前的歷史狀態；本節記錄其後的核准決策，
不回寫歷史驗收報告。Phase 01.md 與 AGENTS.md 亦保持原樣。

| 最終核對 | 結果 |
| --- | ---: |
| COCO validation 影像／動物類別 | 5,000／10 |
| 任意多動物候選影像 | 471 |
| Selected／Rejected／Pending 影像 | 15／3／453 |
| Selected 中所有非 crowd 實例／完成審核 | 47／47 |
| 全眼部工作檔實例／完成審核 | 2,121／212 |
| 具至少兩隻完整雙眼候選的影像 | 13 |
| 上述 13 張中的非 crowd 實例／完整雙眼候選實例 | 38／32 |
| 重複 Annotation ID／重複複合鍵／非法選項／部分必要欄位 | 0／0／0／0 |

### Progressive Filtering Rules

1. 由官方 category metadata 的 supercategory=animal 辨識動物，排除 iscrowd=1。
2. 同圖至少兩個非 crowd 動物才進入 471 張候選，保留 COCO 原始 image_id 與 annotation_id。
3. 人工以 Image Selection 決定 Selected／Rejected／Pending；未處理維持 Pending。
4. Eye Review 僅呈現 Selected 圖片的全部非 crowd 動物，不限制四類或同一物種。
5. 五項必要欄位均為既有合法選項才計作「完成審核」，Uncertain 仍是完成一次判讀，並非合格。
6. 此次結案沿用歷史條件：完成審核 AND left_eye_visibility=Yes AND
   right_eye_visibility=Yes AND measurement_eligibility=Yes。
   一張 Selected 圖片至少有兩隻符合，才列入 13 張 Demonstration Candidates。

Seed 42 的 60 張是先前四類等額試審；後續 15 張 Selected 為人工 Progressive Filtering，
沒有隨機種子，也不是前述抽樣的代表性估計。不得把 13／15 當成 COCO 適用率。

### Demonstration Candidates

| Image ID | 類別 | 完整雙眼候選數 | 合格 Annotation IDs |
| --- | --- | ---: | --- |
| 20247 | bear | 2 | 587181, 588042 |
| 49269 | dog, horse | 2 | 2447, 54326 |
| 52462 | zebra | 2 | 589616, 1822509 |
| 66561 | sheep | 2 | 61524, 62397 |
| 71756 | bear | 2 | 587488, 588072 |
| 81766 | dog | 3 | 2188, 14835, 17520 |
| 125405 | dog | 2 | 13692, 17207 |
| 170739 | elephant | 2 | 582889, 583338 |
| 180011 | giraffe | 3 | 595833, 1414514, 1414822 |
| 198960 | sheep | 2 | 66848, 67945 |
| 205282 | giraffe | 3 | 596049, 599517, 1414076 |
| 269196 | sheep | 5 | 61850, 64938, 65773, 66542, 1817822 |
| 333069 | giraffe | 2 | 599488, 599888 |

另外 5992（sheep）與 41635（cow）各一隻符合完整雙眼條件；保留 Selected，
不將它們改成 Rejected，也不列入此次 13 張清單。
32 隻及同圖兩兩配對最多對應 32 組雙眼距離與 28 組右眼配對；這是組數，尚未產生量測值。

### Eligibility Definitions Proposed for Later Phases

以下僅定義未來衍生欄位，不新增 CSV 欄位、不改寫 measurement_eligibility，
也不替現有 UI 的 image_eye_condition 更換規則。
建議採 Yes／No／Unknown 三值邏輯；Unknown 保留來源原因（尚未審核、Uncertain、欠缺品質確認）。
AND 的規則為任一 No 則 No，全部 Yes 才 Yes，其餘 Unknown。

| 欄位／邏輯值 | 定義 | 衍生規則提案 |
| --- | --- | --- |
| eye_observable(eye_side) | 指定解剖學左／右眼是否能在原圖辨識；單一 instance 需分別表示 left、right | 由對應 left_eye_visibility／right_eye_visibility 取得：Yes→Yes、No→No、Uncertain 或空白→Unknown。不是「任一眼可見」的單一布林值 |
| binocular_measurable | 同一實例是否有可依核准方法重現的雙眼量測 | E_left AND E_right AND A AND Q_left AND Q_right |
| right_eye_measurable | 該實例的解剖學右眼是否可依核准方法重現量測，不要求左眼可見 | E_right AND A AND Q_right |

E_side=eye_observable(side)；A=動物 ID 與解剖學左右已確認；
Q_side=該眼非空、原圖對齊且依未來核准邊界規範完成品質確認的 Eye Segmentation Mask。
目前缺少 A／Q 的獨立確認，不能把雙眼可見直接衍生成 measurable=Yes；
已知必要眼不可見可推為 No，其餘尚未確證時為 Unknown。
同圖任意兩隻的右眼距離另要求不同 annotation_id、相同 image_id，且兩者 right_eye_measurable=Yes。
這與雙眼距離的必要條件不同。

原有 measurement_eligibility 僅保留為 legacy 人工判讀；不把它直接映射為上述任何新欄位。
尤其不能把 legacy No 當成 right_eye_measurable=No，或 legacy Yes 當成雙眼已可量測。
13 張清單是既有篩選快照，不因這份定義提案而重新命名、重算或擴大。

### Known Limitations and Preserved Review Ambiguities

- 有 5 筆 legacy Eligibility=Yes 且一眼為 No：170739/583953、180011/1414641、
  180011/1823152、198960/63253、577149/1412523；最後一筆不在 Selected。
  它們不被納入完整雙眼候選，原始值保持不變。
- 49269/2447、180011/595833 為 Side 且雙眼 Yes；屬待視覺確認的姿態差異，不自動判錯或排除。
- Selected 的 47 筆 eye_resolution、reviewer 皆空白；可見性不保證邊界可準確標註。
- 13 張皆可完整解碼、尺寸與 COCO metadata 一致，EXIF orientation=1；不代表眼睛品質已量化。
- 示範候選偏向人工認為可行的案例，樣本小且類別不均；不得作獨立模型測試集或推估泛化效能。
- COCO 沒有動物眼睛 Ground Truth；動物實例分割準確不等於眼部分割或量測準確。
- 資料完整性與歷史雜湊吻合，但未提供獨立官方參考雜湊，保留 authenticity_verified=false。

### Ground Truth and Measurement Decision Reserved for Phase 3

| 決策 | 核准狀態與用途 |
| --- | --- |
| Eye Landmark Ground Truth | 與 Mask GT 分開定義及保存；人工點、瞳孔／虹膜中心、Mask Centroid 不視為相同標的 |
| Eye Segmentation Ground Truth | 需另訂眼部邊界、遮擋、模糊及品質規範；不是 COCO animal mask |
| 主要測量方法 | 暫定 Eye Segmentation Mask → Mask Centroid → Euclidean Distance |
| Landmark 的地位 | 可於 Phase 3 討論為獨立定位參考；不能悄悄替代 Mask Centroid 的量測方法 |
| 實作時程 | Ground Truth Annotation UI 未獲核准；Phase 2 不做眼睛定位、分割、標記或量測 |

後續若採二值 Eye Mask M，暫定面積 N=sum M，
cx=sum((x+0.5)M)/N、cy=sum((y+0.5)M)/N（原圖左上外框為原點，像素中心在半整數）；
N=0 時中心與距離未定義，不填 0。距離 d=sqrt((cx1-cx2)^2+(cy1-cy2)^2)，單位為 px。
雙眼用同一 instance 的 L／R Mask，右眼配對用同圖不同 instances 的 R Mask。
此為技術決策紀錄，像素慣例、邊界定義與可接受誤差仍須 Phase 3 核准；沒有實作上述計算。
遮擋會使可見區域重心偏移；Mask Centroid 不必等於瞳孔中心、解剖學眼球中心，甚至可能落在非凸 Mask 外。
沒有相機校正及深度資訊，不能將 px 距離解釋為實際長度或三維距離。

### Next Approved Scope

結案後專案負責人更新 [Phase 2 Specification](phase2_instance_segmentation.md)，
並指示依文件實作 YOLO11n-seg、CPU 參考路徑及全部 15 張 Selected Images 的 POC。
該範圍現已完成，實測與重現方式見 [Phase 2 POC Results](phase2_poc_results.md)。
Phase 1 的選圖、人工標記、候選清單與歷史驗收報告保持不變；未訓練或進入眼睛功能。

## Phase 1 Analysis Results Historical Record

下列分析、抽樣及初始規範保留當時狀態；最新結案與核准範圍以上節為準。

執行日期：2026-11-07（Asia/Taipei）。使用本機 COCO 2017 validation 註記，
其 info 宣告 year=2017、version=1.0。輸入含 5,000 個影像紀錄與 36,781 個全部類別註記。
動物類別由 supercategory=animal 取得；以下「非 crowd 實例」只代表註記可計數，
不代表眼睛可見或量測合格。沒有套用大小、姿態或物種偏好門檻。

### Measured Category Counts

| COCO ID | 類別 | 全部動物註記 | Crowd 註記 | 非 crowd 實例 | 含非 crowd 實例影像 | 同類至少兩隻影像 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | bird | 440 | 13 | 427 | 125 | 57 |
| 17 | cat | 202 | 0 | 202 | 184 | 17 |
| 18 | dog | 218 | 0 | 218 | 177 | 29 |
| 19 | horse | 273 | 1 | 272 | 128 | 60 |
| 20 | sheep | 361 | 7 | 354 | 65 | 47 |
| 21 | cow | 380 | 8 | 372 | 87 | 60 |
| 22 | elephant | 255 | 3 | 252 | 89 | 57 |
| 23 | bear | 71 | 0 | 71 | 49 | 17 |
| 24 | zebra | 268 | 2 | 266 | 85 | 64 |
| 25 | giraffe | 232 | 0 | 232 | 101 | 61 |

動物註記合計 2,700，排除 34 筆 crowd 後為 2,666 個實例，分布於 1,016 張不同影像。
各類別影像數不能相加作為唯一影像總數，因為不同類別可能出現在同一影像。
此資料中每個類別的 images_with_annotations 與 images_with_eligible_instances 剛好相同；
CSV 仍分開保留兩種定義。

### Measured Area Distributions

母體為各類別所有非 crowd 實例，**包含單動物影像，並非只限候選影像**。
像素平方與影像面積比例分開計算；比例欄是各實例比例的分位數，不能直接用兩個面積中位數相除。
下表僅為閱讀而四捨五入，CSV 保留原計算精度。

| 類別 | Bbox 面積中位數（px²） | Segmentation 面積中位數（px²） | Bbox 佔影像比例 %：P25／中位數／P75 | Segmentation 佔影像比例 %：P25／中位數／P75 |
| --- | ---: | ---: | --- | --- |
| bird | 750.45 | 293.42 | 0.03／0.27／1.60 | 0.01／0.10／0.59 |
| cat | 64,828.07 | 35,517.10 | 9.86／26.53／48.00 | 4.93／13.92／25.41 |
| dog | 33,426.70 | 15,923.90 | 3.33／12.34／35.61 | 1.57／6.20／18.34 |
| horse | 11,862.87 | 5,350.10 | 1.23／5.42／16.47 | 0.52／2.32／7.15 |
| sheep | 6,393.29 | 3,125.58 | 0.44／2.24／7.32 | 0.25／1.05／3.06 |
| cow | 3,204.17 | 1,578.68 | 0.24／1.20／6.57 | 0.13／0.59／3.29 |
| elephant | 22,392.05 | 10,419.49 | 2.25／7.79／26.81 | 1.20／3.75／15.76 |
| bear | 54,828.08 | 35,139.28 | 5.62／18.61／39.39 | 3.11／12.01／23.07 |
| zebra | 16,619.22 | 8,177.58 | 1.36／6.87／19.04 | 0.67／3.02／9.96 |
| giraffe | 24,594.54 | 7,656.59 | 3.63／9.31／24.68 | 1.18／3.12／8.88 |

area_summary.csv 包含 40 列（10 類 × 4 個指標），各有 count、min、P25、median、P75、max、mean。
Segmentation 面積取原始 area，不用 bbox 代替；這些數值都不是眼部尺寸或量測精度。

### Candidate Image Exports

| 檔案 | 影像數／列數 | 定義 |
| --- | ---: | --- |
| outputs/phase1/multi_any.csv | 471 張 | 任意動物類別合計至少兩個非 crowd 實例 |
| outputs/phase1/multi_same_category.csv | 450 張 | 至少一種動物類別有兩個以上非 crowd 實例 |
| outputs/phase1/multi_different_categories.csv | 70 張 | 至少兩種動物類別各有非 crowd 實例 |
| outputs/phase1/eye_visibility_review.csv | 2,121 列 | 471 張候選影像中每個非 crowd 動物一列；不是影像數 |

同類組與不同類組交集為 49 張，僅同類組 401 張，僅不同類組 21 張；
因此 450 + 70 - 49 = 471。三組檔案保留原始 image ID、annotation ID 及類別計數，沒有重新編號。
其他產物為 category_statistics.csv（10 列）、animal_instances.csv（2,700 列）、
area_summary.csv（40 列）與 run_manifest.json。所有列數均不含 CSV 標頭。

### Observations and Open Questions

| 已有證據支持的觀察 | 尚不能由此推論 |
| --- | --- |
| 同類至少兩隻的影像數：zebra 64、giraffe 61、horse／cow 各 60 | 不代表這些類別的眼睛更清楚或適合最終量測 |
| cat 出現在 184 張影像，但只有 17 張符合同類至少兩隻；dog 為 177 張／29 張 | 單類出現頻率不能取代同類多動物候選供給 |
| bird 的 bbox 面積比例中位數為 0.27%，cat 為 26.53% | 母體包含單動物影像，不能據此推定多動物候選的眼睛解析度 |
| 471 張候選影像可供下一步視覺檢視 | 2,121 個動物尚未經眼部可見性或量測資格審查 |

建議下一步產生候選 contact sheets，搭配原始解析度影像校正人工審查規範。
審查樣本量、抽樣方式、左右眼定義與量測資格仍需專案負責人核准。
不建立綜合排名或自動選類別，也不進入 Phase 2。

### Execution and Validation Record

```sh
.venv/bin/python -m animal_eye_metrology analyze --config configs/phase1.toml
.venv/bin/python -m pytest -q
```

上述分析成功結束（exit code 0）；49 項人工資料測試全數通過，無跳過或警告。
輸出目錄固定為設定中的 outputs/phase1，重跑時須選擇新的輸出目錄，避免覆寫原結果。
本次沒有使用模型、沒有隨機抽樣（seed=null）、沒有更動程式或設定。
contact_sheets.enabled=false，因此此分析執行本身沒有再次讀取影像；先前全量影像驗證屬資料準備紀錄。

輸入註記 SHA-256：

```text
e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f
```

與解壓紀錄的註記雜湊一致。未提供獨立可信參考值，保留 authenticity_verified=false。
run_manifest.json 記錄來源、設定、原始碼雜湊、環境與 7 個 CSV 的 SHA-256。
另外以直接讀取原始 JSON 的獨立核對流程驗證類別計數、候選條件／ID、全部 40 列面積分布，
並以標準函式庫 statistics.quantiles(method="inclusive") 核對分位數。
浮點核對容許相對誤差 1e-12、絕對誤差 1e-9；原始輸出未作四捨五入或改寫。
也確認所有人工欄位為空白、候選檔案存在、7 個產物雜湊一致。
核對通過紀錄另存 outputs/phase1_validation.json，不修改原始分析輸出組。

## Visual Review Batch Historical Preparation

本節記錄 2026-11-07 依使用者指定四類、每類最多 15 張、seed=42 所準備的審核資料。
先重新載入實際註記，確認 manifest 摘要、10 類 metadata、5,000 張影像、471／450／70 張候選、
2,121 列原始審核表；也確認全部來源 CSV 雜湊、同類候選清單與重新計算結果一致。

| 類別 | 符合同類至少兩隻的母體影像 | 抽樣影像 | 該類抽樣目標實例 | 圖內全部非 crowd 實例 | Contact sheets |
| --- | ---: | ---: | ---: | ---: | ---: |
| zebra | 64 | 15 | 49 | 62 | 5 頁 |
| giraffe | 61 | 15 | 45 | 58 | 5 頁 |
| horse | 60 | 15 | 42 | 51 | 5 頁 |
| dog | 29 | 15 | 32 | 43 | 5 頁 |

共 60 張不同影像、168 個各抽樣類別的目標實例；另有 46 個伴隨動物，共 214 列人工審查資料。
例如 giraffe 樣本內也有 zebra；不能直接依人工 CSV 的 animal_category 分組就當成各類的抽樣母體。
必須使用抽樣清單的 sample_category 與 target_annotation_ids 確認比較對象。

抽樣是先將各類合格 image ID 排序，再用一個 random.Random(42)，依 zebra → giraffe → horse → dog
順序不放回抽 15 張；不足 15 張則全取。抽出後僅排序供繪圖，不依眼睛、姿態或面積篩選，也沒有替換難圖。
這是等額的可行性試審，不是全 COCO 或所有動物的代表性估計。完整母體與抽出順序在 review_manifest.json。

產物全部位於 outputs/phase1/visual_review_seed42/：

- sampled_images.csv：60 列，含原始 ID、目標 annotation IDs、原圖路徑、對應 contact sheet 及 seed。
- eye_visibility_review.csv：214 列，直接篩選既有審核表，欄位、順序與資料列格式不變；人工欄位全部留白。
- zebra／giraffe／horse／dog/contact_sheets/page_0001.png 至 page_0005.png：每頁 3 張原圖／bbox／mask 對照。
- REVIEW_GUIDE.md：以動物解剖學左右判讀、原始像素檢視、欄位對應、目標分母及人工操作說明。
- review_manifest.json：來源雜湊、母體／抽樣 ID、抽出順序、Python 版本、執行指令、繪圖設定及產物雜湊。

繪圖沿用 visualization.py，只新增選用 instance_legend 模式，預設行為不變。
審核模式使用每實例配色、短 ID 與引線，另在圖下列出類別及 annotation ID；17 個實例的圖也不重複循環六色。
已檢視各類首頁與各類最密集實例頁，共 8 頁，確認原圖、ID、圖例與疊圖可讀；
這只是圖像產物 QA，沒有替人判斷眼睛或填寫標籤。

本次增加 review_sample.py 作為單一準備入口，未建立新視覺化系統或 Notebook。
執行指令：

```sh
.venv/bin/python -m animal_eye_metrology.review_sample --config configs/phase1.toml
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check --quiet .
```

原精簡 49 項測試加上 2 項抽樣正確性案例，合計 51 項通過。
解剖學左右與此次四類 seed 42 抽樣已由使用者指定；可見性分級、可標註邊界、眼部解析度量法、
量測中心／ground truth 與模糊個案仍需人工確認。審核指南提供操作建議，沒有強制新 schema 或自動評分。
本階段沒有模型推論、訓練、眼部 Ground Truth、距離量測或最終類別選擇。
完成準備後停止，等待使用者人工審核。

## Implemented Analysis Method

### Input and Validation

- 路徑由 TOML 設定；相對路徑以設定檔所在目錄解析。
- 從 categories 的 supercategory == animal 取得動物類別，預期數量為 10。
  不把類別名稱或數字 ID 寫死在分析程式；數量不符會報錯，不自行補類別。
- 檢查唯一 ID、外鍵參照、iscrowd、影像尺寸、有限數值與 polygon／RLE 結構。
  壓縮與未壓縮 RLE 都檢查 run lengths 是否完整覆蓋影像，再交給原生解碼器。
- 保留官方 ID、bbox、segmentation 與 area，不重寫、不修正、不裁切原始註記。
- 註記讀入時記錄 SHA-256；可選擇與使用者提供的可信參考值比對。
  未提供參考時只記錄雜湊，明確標示未驗證官方真實性。
- 啟用 contact sheets 時才檢查所選影像：可解碼性、與 metadata 一致的尺寸及內容 SHA-256。
  未選取影像不宣稱已驗證；只跑註記時完全不讀影像。
- verify-file 可檢查本機 ZIP CRC 與必要成員，不解壓縮、不下載。

### Counting Population

「合格實例」僅表示動物註記且 iscrowd == 0，**不表示眼部量測合格**。
目前沒有大小、姿態、眼睛可見性或物種偏好的額外篩選。

| 欄位 | 定義 |
| --- | --- |
| total_annotations | 該動物類別的所有註記，包含 crowd |
| crowd_annotations | 該類別 iscrowd == 1 的註記數 |
| eligible_instances | 該類別 iscrowd == 0 的註記數 |
| images_with_annotations | 包含該類別任意註記的不同 image ID 數 |
| images_with_eligible_instances | 包含該類別至少一個合格實例的不同 image ID 數 |
| images_with_at_least_two_eligible_instances | 包含該類別至少兩個合格實例的不同 image ID 數 |

三組候選影像獨立套用條件：

1. multi_any：合格動物實例總數 >= 2。
2. multi_same_category：至少一個動物類別的合格實例數 >= 2。
3. multi_different_categories：至少兩個不同動物類別各有合格實例。

第二與第三組可能重疊。混合影像的 CSV 保留全部合格動物 ID，並以
repeated_category_ids 指出哪些類別達到同類兩隻的條件。
動物數只代表 COCO 已註記且非 crowd 的動物，不保證等於影像中的真實動物總數。

### Area Distribution

| 指標 | 計算及限制 |
| --- | --- |
| bbox_area_px2 | 原始 bbox 的 width × height；不裁切至影像範圍 |
| segmentation_area_px2 | 原始註記的 area；不以 bbox 代替，也不重新計算 mask 面積 |
| bbox_area_fraction | bbox_area_px2 / (image width × image height) |
| segmentation_area_fraction | segmentation_area_px2 / (image width × image height) |

每個類別以所有合格動物實例計算分布，包含單動物影像。
輸出 count、min、p25、median、p75、max、mean，分位數以排序後 (n-1)q 線性插值。
零實例類別的 count 為 0，其他分布值留白，不能當作零面積。
animal_instances.csv 同時保留 crowd 註記供稽核，但 area_summary.csv 排除 crowd。
上述面積不是眼睛尺寸、視覺品質或量測誤差的替代指標。

### Visualizations and Manual Template

- 每個 contact sheet tile 並排原圖與動物 bbox／mask，標出 image ID、類別、合格數量及 annotation ID。
- polygon、多部件 polygon、未壓縮 RLE 與壓縮 RLE 交由 pycocotools mask API 轉換。
- 圖上僅疊合非 crowd 動物；原圖保持完整。此處使用的是**動物實例註記**，不是眼部 mask。
- 預設產生所有候選影像。設定 max_images > 0 時只按 image ID 取前 N 張供工程預覽，
  明確記錄為非研究抽樣，不能用來估計眼睛可見率。
- 原始解析度影像才是眼部判讀來源；縮圖會改變眼部可辨識程度。
- 審查模板以 (image_id, annotation_id) 識別單一動物，預填類別與非 crowd 動物數。
  左右眼可見性、姿態、遮擋、眼部解析度、量測資格、排除原因、審查者、規範版本與備註一律空白。
  留白代表尚未審查，不能解讀成不可見或不合格。
- templates/eye_visibility_review.csv 僅提供欄名；真實候選資料列需本機註記分析後產生。

### Reproducibility and Output Protection

統計不使用隨機數，依原始 ID 排序；random_seed 記錄為 null。
每次執行輸出 run_manifest.json，包含資料宣告名稱與 info、來源 SHA-256、完整設定及其雜湊、
命令列及工作目錄、Python／套件版本、分析原始碼雜湊、篩選規則、摘要與各輸出檔雜湊。
若啟用視覺化，另記錄實際選取及驗證的影像。
設定中的資料名稱由使用者宣告，不能單獨證明資料版本；來源與參考雜湊仍需記錄核實。

輸出先寫入暫存目錄，全部成功才發布；非空輸出目錄拒絕覆寫。
人工審查後的檔案應另存並追蹤版本，原始 manifest 雜湊對應的是最初空白模板，
不應用來宣稱後續人工標籤已通過驗證。

## Data Acquisition — Resolved

初始化時僅閱讀官方網頁與 API 文件；其後使用者自行提供兩個壓縮檔，並核准依設定解壓。
目前採用下表選項 A，無須另行下載。其餘選項保留為先前規劃紀錄，不是待執行下載。
官方下載目錄列出 2017 Train/Val annotations 約 241 MB、2017 Val images 約 1 GB；
這些是**官方公布的約略下載標示，並非本次實測檔案大小或資料統計**。
來源：[COCO Download](https://cocodataset.org/#download) 及
[官方下載頁原始碼](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)。

| 選項 | 需核准的項目 | 優點與限制 | 建議 |
| --- | --- | --- | --- |
| A. 使用既有本機資料 | 不下載；你提供本機註記與影像路徑 | 無額外傳輸；仍須驗證來源與完整性 | 若已持有資料，優先使用 |
| B. 分段取得官方資料 | 先 annotations_trainval2017.zip，核對並只使用 instances_val2017.json；之後再單獨核准 val2017.zip | 可先完成所有類別統計，再準備視覺審查；官方註記壓縮包仍含其他檔案 | **沒有本機資料時推薦** |
| C. 一次取得兩個官方壓縮包 | annotations_trainval2017.zip 與 val2017.zip | 一次準備完成，較早占用影像傳輸與儲存空間 | 確定會完整審查時可用 |
| D. 註記加候選影像子集 | 官方註記包；統計後再核准固定 image ID 清單所對應的官方影像 | 可減少影像傳輸；需額外追蹤下載清單與逐檔驗證，不能假裝影像全集已準備好 | 頻寬受限時再考慮 |

官方目錄列出的檔案位置（程式沒有內建下載命令；使用者實際下載網址與日期未提供）：

- [annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
- [val2017.zip](http://images.cocodataset.org/zips/val2017.zip)

實際取得前需核准檔案範圍與本機目的地。取得後記錄來源網址、取得日期、位元組大小與 SHA-256，
檢查 archive CRC、必要成員、JSON 結構，以及實際使用影像的尺寸／可解碼性。
若沒有獨立可信的官方參考雜湊，明確保留「來源真實性未獨立驗證」，不把自算雜湊稱為官方驗證。
不需要下載 train2017 影像、test split 或模型權重。

## Manual Review Options Historical Proposal

解剖學左右及四類 seed 42 的 15 張抽樣已由本次任務指定並採用；
以下其他項目仍是**提案**，尚未套用為 ground truth 定義、篩選器或審查標籤。
此為 Explorer 實作前的規劃；其後 UI 已採 Yes／No／Uncertain 等選項。
最新衍生欄位與 Ground Truth 決策見結案節，不把此處舊提案當成現行標準。

| 決策 | 選項 | 建議及原因 |
| --- | --- | --- |
| 左右眼定義 | 動物解剖學的左／右；或觀察者畫面左／右 | **解剖學左右**，避免姿態改變造成「右眼」意義改變；無法辨別時記 uncertain，不能猜測 |
| 可見性標記 | 二元 visible/not_visible；或 visible/partial/not_visible/uncertain 四級 | **四級**，保留部分遮擋與無法判斷的差異；不可由缺值推定不可見 |
| 頭部姿態 | 人工類別 frontal/oblique/profile/away/uncertain；或要求角度數值 | **先用人工類別**，不在缺乏校正依據時製造精確角度 |
| 遮擋紀錄 | 單一有／無；或 none/partial/severe/uncertain 並附部位或原因 | **分級加備註**，區分眼部、頭部與其他部位；分級邊界需先用試審案例校正 |
| 眼部解析度 | 主觀 high/medium/low；或原始像素下人工圈定審查框並記每眼寬×高 | **記原始像素寬高**，保留可稽核依據；審查框不是眼部 segmentation ground truth，不先設定最低像素門檻 |
| 量測資格 | 單一 eligible/not_eligible；或依單動物雙眼／跨動物右眼分開，並保留 uncertain | **依任務分開**；跨動物還需檢查配對，不能只憑單一動物一列判定配對合格。核准後再調整欄位 |
| 審查範圍 | 全量候選；或先作固定種子的各類別影像抽樣試審 | **先試審校正規範，再依候選量決定全量或抽樣**；樣本量、種子、分層與影像清單需另行核准／記錄 |
| 審查一致性 | 單人審查；或兩人獨立審查共同子集，再處理分歧 | **有資源時採雙人共同子集**；比例與分歧處理方式待核准，不能預先宣稱一致率 |

若核准以解剖學定義左右眼，還需制定遇到側臉、鏡像或左右無法識別時的操作範例。
visible／partial 的邊界、姿態／遮擋分級及眼框的可見範圍也需以具體影像校正。
上述審查不定義眼球、虹膜、瞳孔或眼瞼的正式 segmentation ground truth；這些仍需獨立核准。

## Category Comparison Plan

| 比較面向 | 證據來源 | 目前能否下結論 |
| --- | --- | --- |
| 候選數量 | 真實 COCO 註記的程式計數 | 可描述各類供給，結果見上方；不等於眼部量測可用量 |
| 左右眼可見性與眼部解析度 | 核准規範後的人工審查，須同報審查影像數與動物數 | 否，不能從動物 mask 或 bbox 面積推定 |
| 標註可行性 | 試標時間、可辨識邊界、分歧與排除原因 | 否，需實際試審／試標，不捏造工時 |
| 眼睛定位模型可用性 | 後續調查官方論文／儲存庫，記錄物種、版本、授權及輸出定義 | 否，尚未調查，也不選模型架構 |

不建立尚未核准的綜合加權分數，也不自動挑選勝出類別。
分割準確度、眼部定位誤差、幾何量測誤差與不確定度是不同問題；本階段統計不提供這些效能結果。

## Validation and Execution

從 repository 根目錄執行：

```sh
uv sync --frozen --python 3.11 --extra visualization
uv run --frozen --extra visualization pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
# 本機資料準備且路徑設定完成後才執行：
uv run --frozen animal-eye-metrology analyze --config configs/phase1.toml
uv run --frozen --extra visualization animal-eye-metrology analyze \
  --config configs/phase1.toml --contact-sheets --output outputs/phase1-with-images
```

測試使用非 COCO 的 ID、類別與幾何資料，檢查 metadata 驅動、crowd 排除、候選交集、
零計數類別、分位數、空白審查欄位、輸入不變、polygon／RLE、影像與 ZIP 錯誤，
並以命令列整合測試驗證可重現輸出。單元測試禁止 socket 連線。

### Initial Validation Results — 2026-11-07

以下保留初始化時的歷史紀錄；當時真實資料尚未就位，測試尚未精簡。現況見下一節。

環境：Python 3.11.14、pytest 9.2.0、Ruff 0.17.4、Pillow 12.3.0、pycocotools 2.0.11、NumPy 2.4.6。

| 已執行命令／檢查 | 觀察結果 |
| --- | --- |
| uv sync --python python3.11 --extra visualization | 建立本機開發環境與 uv.lock；取得的是軟體套件，沒有 COCO 或模型資料 |
| uv run --frozen --extra visualization pytest -q | **63 passed，0 skipped，18 warnings**；全部使用人工測試資料 |
| uv run --frozen ruff check . | 通過 |
| uv run --frozen ruff format --check --quiet . | 通過 |
| 預設設定執行 analyze | 因真實註記尚不存在，以預期的 exit code 2 及結構化缺檔訊息結束；未建立研究輸出 |
| python -S 的分析整合測試 | 停用第三方 site-packages 後，註記統計仍可成功執行 |
| 人工檢視合成 contact sheet | 原圖／疊圖、類別、數量及 ID 可讀；未使用真實動物影像 |
| Git ignore 檢查 | 資料、輸出、權重、虛擬環境與 macOS metadata 均被排除 |

18 個警告都來自 pycocotools.mask.decode 在目前 NumPy 環境的 __array__ copy 參數棄用提醒。
解碼與逐像素比對測試通過；警告未隱藏，仍屬後續更新相依套件時需要追蹤的相容性事項。
以上是工程驗證結果，不是 COCO 資料統計、眼部標註品質或模型效能證據。

### Current Preparation and Reduced Tests — 2026-11-07

此節記錄資料準備與測試精簡完成時的狀態，當時尚未進行分析；後續分析結果見文件上方。
使用者核准依 configs/phase1.toml 解壓所需資料，並確認測試精簡方案。
正式程式、CLI、設定與輸出格式均未修改；解壓前後的正式程式／設定 SHA-256 比對一致。

| 本機資料檢查 | 實際觀察 |
| --- | --- |
| 解壓範圍 | annotations/instances_val2017.json 及 val2017 的全部影像；未解壓其他註記 |
| 註記結構 | 現有 load_coco 驗證通過，5,000 個影像紀錄、36,781 個註記紀錄 |
| 影像對應 | 壓縮檔內影像檔名集合與註記完全一致 |
| 影像內容 | 5,000 張全部完整解碼，尺寸與註記一致 |
| 完整性紀錄 | data/coco/preparation_manifest.json，含來源壓縮檔與逐檔 SHA-256、CRC 讀取結果 |
| 研究輸出 | 尚未執行 analyze，未產生真實候選 CSV、contact sheets 或類別統計 |

上列數字僅是本機資料準備檢查，不能當作多動物候選數或模型效能。
使用者實際下載網址、日期與獨立可信的參考雜湊尚未提供；不宣稱已獨立驗證官方來源。
來源壓縮檔保留不變，資料與準備紀錄均不提交 Git。
等價的解壓指令如下；本次以 Python zipfile 逐成員檢查 CRC、解壓並核對內容，
若目的地已存在不同內容即停止，不覆寫既有資料：

```sh
unzip -n annotations_trainval2017.zip annotations/instances_val2017.json -d data/coco
unzip -n val2017.zip -d data/coco
```

| 測試範圍調整 | 理由 |
| --- | --- |
| 保留原有 COCO 解析與分析核心案例 | 驗證 metadata、crowd、ID、三組候選交集、面積、空資料與人工欄位留白 |
| 合併合法 polygon／RLE 至 test_coco.py | 保護自行實作的解析驗證，使用固定編碼，不再測第三方逐像素解碼 |
| test_cli.py 保留最小輸出整合、缺檔、不覆寫、失敗不發布、來源目錄保護 | 這些案例保護研究資料，而非追求 CLI 覆蓋率 |
| 搬移設定路徑與兩個壞影像案例至 test_cli.py | 保留換工作目錄、影像損壞及尺寸不符的獨有情境 |
| 移除 test_config.py、test_visualization.py | 刪除設定參數矩陣、分頁／尺寸／像素、重複 rasterization 檢查 |
| 移除 subprocess 啟動、停用 site-packages、空輸出目錄、模板標頭同步及細部日誌斷言 | 降低環境與實作細節造成的維護負擔 |
| 保留兩個本機檔案驗證案例，移除其中純 CLI 呼叫 | 保護雜湊、ZIP CRC、必要成員與來源真實性標示 |

| 本次執行命令 | 結果 |
| --- | --- |
| .venv/bin/python -m pytest -q | **49 passed，0 skipped，0 warnings** |
| .venv/bin/ruff check . | 通過 |
| .venv/bin/ruff format --check --quiet . | 通過 |

pytest 與既有相依套件保留；沒有新增測試框架、抽象層或覆蓋率要求。
本次沒有觸發舊有解碼警告，是因為移除了逐像素解碼測試，**不是修復或隱藏第三方警告**。
版面、像素對齊與第三方 mask 解碼不再自動驗證；未來若修改視覺化功能，再針對實際風險驗證。

## Files Created or Modified

| 路徑 | 變更與目的 |
| --- | --- |
| README.md | 擴充專案現況、安裝、設定、命令與輸出說明 |
| .gitignore | 增補本機資料、輸出、權重、設定覆寫與 OS 檔案的忽略規則 |
| pyproject.toml、uv.lock | Python 套件、CLI、選用視覺化依賴、Pytest／Ruff 與鎖定版本 |
| configs/phase1.toml | 可攜的資料路徑與視覺化設定，預設只讀註記 |
| src/animal_eye_metrology/__init__.py、__main__.py、cli.py、errors.py | 套件、命令列入口、結構化紀錄與明確錯誤處理 |
| src/animal_eye_metrology/config.py、coco.py | 設定與本機 COCO 結構驗證 |
| src/animal_eye_metrology/analysis.py、outputs.py | 可重現的統計、候選 CSV、空白審查表與 manifest |
| src/animal_eye_metrology/visualization.py、verification.py | 本機影像／遮罩 contact sheets 與本機檔案完整性檢查 |
| templates/eye_visibility_review.csv | 只有欄名的人工審查模板 |
| tests/conftest.py、test_analysis.py、test_coco.py、test_cli.py、test_verification.py | 精簡後的人工幾何 fixtures 與核心／資料保護測試；已移除獨立設定及視覺化測試檔 |
| docs/dataset_selection.md | 方法、證據狀態、下載／人工審查選項、限制及本次交付紀錄 |

## Limitations and Approval Gate

- Phase 1 已核准結案，15 張 Selected 與 13 張 Demonstration Candidates 保留；不限制單一動物類別。
- 選圖和可見性是人工篩選證據，尚未建立 Eye Landmark／Eye Segmentation Ground Truth。
- 原始 legacy Eligibility 保留；新語意僅提案，不修改 CSV 或現有程式判定。
- Phase 2 已依負責人更新的文件完成 YOLO11n-seg 15 張 CPU POC，等待驗收；不擴大為其他模型或全量評估。
- 主要測量方法暫定 Mask Centroid；眼睛標註 UI 與量測實作留待 Phase 3，現在不執行。

## Primary References

- [COCO official download page](https://cocodataset.org/#download)
- [COCO official download page source](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm)
- [COCO API: annotation-to-mask conversion](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)
- [COCO mask API: polygon, RLE and bbox conventions](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)
- [COCO mask implementation: compressed RLE encoding](https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c)

以上來源於 2026-11-07 閱讀，用於查核資料入口與格式；未將其當成已完成的本專案實驗證據。
