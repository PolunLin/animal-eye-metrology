# Phase 3 Eye Segmentation POC Proposal

## Status and Recommendation

日期：2026-11-07。狀態：Proposed，等待負責人核准；本文件不授權模型下載、安裝、推論或訓練。
本次僅查閱既有資料與官方來源，沒有修改人工標註、量測快照、程式、套件或既有研究規範。

建議先以 SAM 2.1 Hiera-Tiny、SAM ViT-B、MobileSAM ViT-T 做三模型、固定提示的半自動眼睛分割 POC。
目的是回答「已知眼睛大致位置時，預訓練模型能否分割可見眼裂內區域，並支援重心距離量測？」。
本階段不宣稱已完成全自動眼睛定位、解剖左右辨識或完整端到端任務。

## Verified Project Baseline

| 已驗證事實 | 對 POC 的意義 |
| --- | --- |
| 第一輪固定 3 圖、6 隻動物、12 眼；Mask 面積 13–42 原圖像素 | 可直接作流程開發與分割診斷，不作獨立測試集 |
| 圖片 125405 dog、20247 bear、333069 giraffe | 三種物種只是三張目的性樣本，不做類別排名 |
| 第一輪可衍生 6 筆雙眼距離及 3 筆同圖兩動物右眼距離 | 每模型最多評估 9 筆距離，需保留失敗分母 |
| dog 四眼第二輪也已凍結；兩輪平均 IoU 0.652、重心位移 0.447 px | 可做參考標註敏感度分析，不能直接當成功門檻或模型精度上限 |
| animal instance POC 已有 YOLO26s-seg 輸出與 COCO matching | 先隔離眼睛分割，後續才能另接實際動物預測管線 |
| 現有 Python 3.11.14、PyTorch 2.9.1、torchvision 0.24.1、macOS arm64 | 先規劃 CPU FP32 相容性試跑；尚無眼睛模型實測耗時 |

本次程序查到 MPS built=True、available=False；這是該執行程序的觀測，不代表整台機器沒有 Apple GPU。
晶片型號與記憶體查詢受限，未據此捏造機型、記憶體或秒數。若之後測 MPS，須另記實際裝置與一致性結果。
參考：[人工量測基準](manual_measurement_baseline.md)、[四眼再標結果](blind_repeat_results.md)。

## Candidate Models and Weight Sources

下列優先順序是工程建議，不是已實測的眼睛準確度排名。

| 候選 | 固定權重候選／官方來源 | 本案比較角色 | 授權與注意事項 |
| --- | --- | --- | --- |
| SAM 2.1 Hiera-Tiny | sam2.1_hiera_tiny.pt；[Meta SAM 2 官方倉庫](https://github.com/facebookresearch/sam2) | 優先試跑；具點／框提示的影像分割 API，小型 SAM 2.1 作主要候選 | 官方明列模型 checkpoints 為 Apache-2.0；不使用影片功能或 demo 字型 |
| SAM ViT-B | sam_vit_b_01ec64.pth；[Meta SAM 官方倉庫](https://github.com/facebookresearch/segment-anything) | 原始 SAM 對照；檢查換成新架構是否真的改善本案結果 | [官方模型卡](https://huggingface.co/facebook/sam-vit-base) 標示 Apache-2.0；CPU 時間／記憶體待量測 |
| MobileSAM ViT-T | weights/mobile_sam.pt；[作者官方倉庫](https://github.com/ChaoningZhang/MobileSAM) | 輕量候選；比較資源成本與小眼睛分割差異 | 倉庫標示 Apache-2.0；Tiny-ViT 替換 encoder，沿用 SAM 提示／解碼流程；不能當完全獨立的訓練來源 |

三者均是通用提示式分割模型，未驗證能精確分割本案只有數十像素的動物眼裂。
SAM 2.1 Tiny 是主要候選的原因是功能與規模適合先做相容性試跑，不保證勝過另兩者。
MobileSAM 的輕量化設計與 CPU 可執行性見[作者論文](https://arxiv.org/abs/2306.14289)；不把論文 GPU 速度當成本機速度。

核准後只從上述官方鏈結擷取指定權重；固定 repository commit／模型 revision、設定檔與 SHA-256，
並保存對應授權文件。此刻未下載權重，故沒有宣稱已核對實際檔案 digest 或下載大小。
優先直接用上游 predictor，不因既有 Ultralytics 已安裝就混用不同包裝、預設後處理或授權。
相依套件須核對與現有 torch／torchvision 的相容性，避免無意升級既有動物分割環境；必要時提出獨立環境方案再核准。

## Alternatives and Scope Choices

| 路線 | 可以回答什麼 | 成本與限制 | 建議 |
| --- | --- | --- | --- |
| 人工固定眼部提示 → SAM 系列 | 已大致知道眼睛在哪時能否切出符合規範的 Mask | 仍有人力提示與側別指定，不能宣稱全自動定位 | 本輪推薦 |
| 文字／視覺定位 → 眼部框 → 分割 | 全自動找眼睛與分割的管線可行性 | 多了一個定位失敗來源；仍須處理動物身份與解剖左右 | 後續獨立比較 |
| YOLO-seg／Mask R-CNN／U-Net 專用眼睛模型 | 在眼睛訓練資料下學習偵測或分割 | 現有 3 圖／12 眼不支持可靠訓練與泛化結論；需新資料與 image-level split | 本輪不訓練 |

SAM 3 系列支援文字／視覺概念提示，值得列入後續自動定位路線；
但查閱時[官方主線安裝要求](https://github.com/facebookresearch/sam3#installation)包含 Python≥3.12、CUDA GPU，
且權重須申請存取、採獨立 SAM License；不作本機 Python 3.11／CPU 第一輪比較的預設。
這是部署成本判斷，不是宣稱 SAM 3 不可能在任何 Mac 或替代執行器上運作，也不預設它能辨識本案眼裂。

既有 COCO 動物 YOLO／Mask R-CNN 權重提供的是動物類別 Mask，不會因換一張標籤表就變成眼睛分割模型。
保留 YOLO26s 作上游動物分割候選，不把其動物 AP 當成眼睛成績。

## Prompt and Crop Protocol Proposed for Approval

| 項目 | 第一輪固定規則提案 |
| --- | --- |
| 標的 | 沿用 eye-visible-aperture-v1，不改為瞳孔、虹膜或整個眼周 |
| 提示來源 | 人只看原圖，為每眼畫一個粗框並點一個眼內正點；不顯示 GT Mask、重心或模型輸出 |
| 提示獨立性 | 禁止由人工 GT Mask 的 bbox／centroid 自動產生提示；若未來這樣做，必須另列 oracle 實驗 |
| 側別 | 操作者提供 animal ID＋解剖眼側；模型不因此獲得「自動分辨左右」的成績 |
| 固定預算 | 每眼一框＋一正點，不在看到結果後逐模型補點、改框或選最好的一次 |
| 眼部上下文 ROI | 以粗框中心取正方形，邊長 max(32, 4×max(框寬,框高)) 原圖 px；超出原圖則裁切，實際範圍保存 |
| 公平比較 | 三模型讀同一 ROI、同一原圖提示、同一眼身份；ROI 只由提示框決定，不讀 GT |
| 原生前處理 | 各模型沿用其官方 resize／padding；保存實際變換，不把一個模型的內部張量硬套另一個 |
| 多 Mask 候選 | 使用預先固定的 multimask 設定；若有多候選，以該模型預測品質分數最高者選一個，平手取較小索引 |
| GT 隔離 | 不能用真實 IoU／重心誤差選候選；模型內部分數不當跨模型準確度分數 |
| 輸出 | 還原 logits 到原圖對應像素格後二值化，存原圖尺寸 0／255 PNG；固定並記錄 threshold／插值方式 |
| 後處理 | 第一輪不補洞、平滑、取最大區塊、手修或超解析；不裁成 GT 形狀 |

粗框／正點本身會提供定位資訊，因此本輪結論限於提示式分割。提示品質影響結果也必須列為限制。
上述 ROI 尺寸是待核准的開發設定，不是已驗證最佳參數；不在 12 眼上反覆搜尋後當獨立測試成績。
如需調參，另記為開發實驗，保留原始執行結果。

提示另存 prompts.csv／manifest，不改 eye_mask_annotations.csv Schema。
最少記錄 image_id、annotation_id、eye_side、prompt_version、point_x/y、box_x0/y0/x1/y1、
prompt_source=manual_original_image、reviewer、時間、image_sha256；實際 ROI／resize／padding 另存每次推論 metadata。
本次不建立檔案、UI 或預填提示。既有輪廓工具能否最低限度提供點／框輸出，實作核准後才決定。

座標原點與像素中心慣例沿用既有方案。若裁切起點為 (x0,y0)、縮放 sx/sy、padding px/py：

```text
x_original = x0 + (u - px) / sx
y_original = y0 + (v - py) / sy
```

須核對各 API 的點座標慣例，避免把模型內部像素中心修正與本專案 +0.5 重複套用。
先用已知幾何驗證裁切／縮放／padding／還原；放大不增加原圖資訊，評估一律在原圖像素格。

## Comparison Design

第一輪使用已凍結的 3 張開發圖片，共 12 個不同眼部目標；同一模型產生 12 個正式預測。
三模型共 36 個正式眼睛 Mask，最多各 9 筆、合計 27 筆距離。相容性診斷執行另記，不重複算獨立樣本。

| 步驟 | 範圍與產出 |
| --- | --- |
| 1. 固定輸入 | 核對三圖與第一輪 GT snapshot digest，人工提示保存後凍結；不隨模型改變 |
| 2. 相容性檢查 | 各模型先一眼，核對載入、點／框、原圖座標與 Mask 輸出；記錄失敗與 CPU 時間，不默默換模型 |
| 3. 固定比較 | 完整 12 眼，每眼固定提示，無訓練、無 GT 導向補點或候選挑選 |
| 4. 分層評估 | Mask → 重心 → 兩種距離；呈現所有逐眼、逐動物與逐圖結果與失敗 |
| 5. 參考敏感度 | 同一份 dog 預測另對第二輪四眼參考評估，不重跑模型、不挑對該模型較有利的 GT |
| 6. Review | 比較資源成本、Mask 與量測差異，決定是否值得進入新的圖片或自動定位實驗 |

第一輪人工 Mask 為主參考，第二輪僅做敏感度分析，不合併平均、不刪除難眼、不以雙輪較高分當主成績。
兩輪既有再標差異不是模型通過門檻，不把「低於 1 px」或「高於同人 IoU」自行定義為成功。
若後續要訂成功門檻，應在正式驗收前由負責人核准，並選不同 image_id 的眼睛資料驗收。

## Metrics and Failure Accounting

令 G 為人工參考 Mask、P 為預測；C 是各自前景像素中心的均值，d 是既有歐氏距離。

| 層次 | 指標／公式 | 報告方式 |
| --- | --- | --- |
| 分割 | IoU=\|P∩G\|/\|P∪G\|；Dice=2\|P∩G\|/(\|P\|+\|G\|) | 12 眼逐筆、平均／中位數、最差案例；分母固定 |
| 重心 | e_c=\|C_P−C_G\|₂，單位原圖 px | 每眼位移，含 Δx／Δy；只在中心有定義時計算 |
| 雙眼距離 | Δd=d_pred−d_ref；絕對差=\|Δd\| | 每動物一筆，每模型分母 6 |
| 跨動物右眼距離 | 同圖、不同 animal ID 的 right/right，公式同上 | 每圖兩動物一筆，每模型分母 3 |
| 相對距離差 | 100×\|Δd\|/d_ref，d_ref=0 時未定義 | 與 px 差一起報，不能用長距離的低百分比掩蓋端點偏差 |
| 幾何可計算率 | 可提供中心的眼數／12；完整雙眼動物數／6；右眼配對數／3；兩動物四眼皆可算的圖數／3 | 非空即可計算不代表分割正確或量測合格；分開標示 |
| 工程 | 載入、前處理、encoder、decoder、還原、端到端時間與程序記憶體 | 同 CPU／threads／精度；人工提示時間另報；模型專屬 cache 策略透明 |

空預測對本次非空參考的 IoU／Dice 為 0；重心與距離保持 Undefined，不填零。
程式錯誤／輸出缺失與空預測分開記錄；主摘要列失敗數，不只平均成功例。
四眼皆非空仍可能是錯誤區域或多隻眼，不能據此宣稱整張圖成功。
不在 12 眼上計算有誤導性的泛化顯著性、類別排行榜或人口層級信賴區間。

## Runtime and Apple Silicon Feasibility

建議 CPU FP32、batch=1、4 threads 作共同基準，seed=42、eval＋inference_mode、無 augment。
seed 是執行設定，不是新增抽樣；不保證不同平台或版本逐位元相同。

SAM 2 上游主要安裝路徑以 Linux／CUDA 為主，官方允許以 SAM2_BUILD_CUDA=0 略過 CUDA extension，
但會少掉小孔洞／小區塊後處理；見[官方安裝說明](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)。
本案提案使用靜態影像 predictor 並關閉這類後處理，不使用影片追蹤、compile、CUDA autocast 或 CUDA 專用範例設定。
這是 CPU 執行方案，不是已完成 macOS 相容性驗證；若出現不支援操作應回報，不隱性換設定。
SAM／MobileSAM 也需在現有 Python／torch 版本做載入試驗，不能以權重較小直接承諾速度。

目前未跑任何眼睛模型，故不提供捏造的分鐘數或記憶體估算。
核准後先各一眼記錄實測，再區分一次性模型載入與每眼成本，估算完整 12 眼耗時；所有模型使用相同硬體比較。
MPS 可作之後的獨立加速實驗，不把 CPU 與 MPS 速度混成架構排名，也不未記錄就自動 fallback。

## Training Data and Leakage Review

| 模型 | 已有官方資訊 | 本案尚不能保證的事項 |
| --- | --- | --- |
| SAM ViT-B | [SAM 論文](https://arxiv.org/abs/2304.02643)說明以 SA-1B 訓練提示式分割 | 未做完整預訓練來源與本機三圖的近重複影像稽核；不宣稱完全無洩漏 |
| SAM 2.1 | [SAM 2 論文](https://arxiv.org/abs/2408.00714)描述 SA-1B 預訓練，以及 SA-V、內部與公開影片資料混合；需對照實際 2.1 checkpoint release | 論文資料概述不是本次指定權重的逐樣本清單；內部資料不可完整稽核 |
| MobileSAM | [作者論文](https://arxiv.org/abs/2306.14289)描述由 SAM teacher 進行 encoder 蒸餾，沿用其 mask decoder | 小型蒸餾訓練不消除 teacher 預訓練來源的重疊風險；不能當與 SAM 無關的獨立預訓練證據 |

未宣稱這些權重以本專案 12 個人工眼睛 Mask 訓練；我們的 POC 也不以它們微調。
但「沒有使用本專案人工 Mask」不等於「從未看過相同原圖」。COCO validation 的名稱也不自動保證對所有模型都是未見資料。
因此目前只稱開發 POC，不稱嚴格零洩漏、獨立泛化或 zero-shot benchmark。

另有兩類可控制的洩漏：不從 GT 產生點／框提示、不用 GT 挑候選 Mask／調 threshold。
之後如訓練專用模型，所有動物、雙眼與裁切必須依原 image_id 分在同一 split，不能拆成不同眼睛隨機分訓練／測試。
新留出圖片只能減少本專案調參洩漏，不能單憑 image-level split 排除未知的預訓練重疊。

## Minimal Deliverables After Approval

擬集中於 outputs/phase3/eye_segmentation_poc_v1/，保持現有 src-layout／CSV／Streamlit，無新資料庫、服務或前端框架。

| 擬產物 | 內容 |
| --- | --- |
| prompts.csv、input_manifest.json | 固定人工點／框、來源、原圖與 GT 版本 digest、crop 策略 |
| 各模型 masks/、predictions.csv | 原圖尺寸預測 Mask、動物與解剖側別、提示／checkpoint 版本、模型分數與失敗原因 |
| eye_metrics.csv | 逐眼 IoU／Dice／重心差、面積與失敗狀態；主參考與再標敏感度分開 |
| measurement_comparison.csv | 每模型 9 筆距離、參考差異及不可算原因 |
| overlays/ | 原圖／人工參考／模型輸出／差異與重心連線 |
| run_manifest.json、summary.json | 資料／權重／提示／程式 digest、環境、設定、實測時間與分母 |

沿用 coco.py、visualization.py、metrology.py 與既有版本讀取／凍結功能；僅增加必要的模型呼叫與座標還原。
Pytest 聚焦已知 ROI／padding 還原、半像素偏移、空預測與正確身份配對，重用既有幾何測試，不追求逐模組覆蓋率。
本次沒有提供不存在的可執行 CLI；實作核准後才新增明確設定與實際執行指令並記錄於 manifest。

## Decisions Requested

| 待核准項目 | 建議 |
| --- | --- |
| 第一輪任務 | 固定人工一框＋一正點的半自動眼睛分割；定位與左右由人提供 |
| 三模型 | SAM 2.1 Hiera-Tiny、SAM ViT-B、MobileSAM ViT-T，使用官方權重 |
| 資料與參考 | 既有三張／12 眼作開發資料；第一輪 GT 為主，dog 第二輪只做敏感度分析 |
| 公平性 | 同一提示／ROI、固定候選選取、無 GT 導向調參、CPU FP32 共同基準 |
| 執行邊界 | 先相容性試跑，再完整比較；不訓練、不擴成 5,000 張、不宣稱自動定位或泛化 |

若負責人希望下一步優先回答「全自動找眼睛」，應改核准文字／視覺定位＋分割管線，
另列定位漏檢、動物配對及解剖左右錯誤；不能直接沿用此半自動 POC 的結果宣稱全自動完成。
COCO 的 5,000 張原圖沒有本案 Eye Mask GT，增加未標註圖片不會自動產生眼睛準確度結論。

## Work Completed This Turn

僅新增本提案文件；核對既有規範、固定 GT／再標報告與動物 POC，查閱模型官方倉庫、模型卡及作者論文。
執行本機 Python／torch／MPS 狀態讀取，沒有模型載入、下載、推論、安裝、訓練、人工資料變更、Commit 或 Push。
本輪無功能變更，未新增測試或重跑 pytest；已有的 88 tests 是前次比較工作結果，不當成本次模型測試。
