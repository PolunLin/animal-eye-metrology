# Eye Segmentation Compatibility Trial

## Outcome

2026-11-08 完成三模型 CPU 相容性試跑：SAM 2.1 Hiera-Tiny、SAM ViT-B、MobileSAM ViT-T。
共同合成圖與同一個真實眼睛提示均通過來源／權重核對、模型載入、點＋框 API、CPU forward、
原圖尺寸二值 Mask 輸出及保存 logits 的還原驗證。

這是相容性結果，沒有計算眼睛 IoU、Dice、重心誤差或距離誤差，也沒有選出準確度最好的模型。
沒有開始完整 12 眼比較、訓練、自動定位、左右辨識或 5,000 張推論。

## Fixed Input and Human Prompts

使用者已保存兩眼提示；本輪只使用已指定的第一眼：

| 項目 | 本輪值 |
| --- | --- |
| Dataset | COCO 2017 validation，本機固定來源 |
| Image／Animal／Side | 125405／13692／anatomical left |
| 原圖 | 000000125405.jpg，640×428 |
| 人工提示版本／來源 | v1，manual_original_image，Reviewer=Polun |
| 人工保存時間 | 2026-11-07T12:43:13.432155+00:00 |
| 原圖正點 x, y | 284.79499865681487, 209.4267565863955 |
| 原圖粗框 x0, y0, x1, y1 | 280.89972807905076, 208.9108873523526, 289.57196359672974, 210.1571272323209 |
| 固定原圖 ROI | [267, 192, 303, 227]，寬 36、高 35 px |
| 提示／原圖快照 | compatibility_manual_v2/request.json、input.png；三模型共用相同 request digest |

點與框由使用者只看原圖提供，沒有從人工 GT bbox／Mask centroid 產生。
原圖與 COCO 身份、人工提示及原圖 digest 均核對後才建立固定 request。
本輪不讀 Eye Mask GT。左右及動物身份由人提供，不能因此宣稱模型自動辨識了眼側。
第一次下載與提示保存發生於 11 月 7 日，安裝復原與本次正式試跑於 11 月 8 日完成。

## Results and Timing

macOS arm64，Python 3.11.14、PyTorch 2.9.1、torchvision 0.24.1，CPU FP32、batch=1、
四個 intra-op threads、一個 inter-op thread、seed=42；eval／inference_mode，無 augment。

| 模型 | 合成圖／真實眼睛 | 真實眼睛模型載入 s | 真實眼睛推論 s | 程序峰值 RSS GiB | 預測面積 px |
| --- | --- | ---: | ---: | ---: | ---: |
| SAM 2.1 Hiera-Tiny | Passed／Passed | 0.717 | 0.407 | 1.358 | 8 |
| SAM ViT-B | Passed／Passed | 0.638 | 2.761 | 3.474 | 9 |
| MobileSAM ViT-T | Passed／Passed | 0.725 | 0.256 | 0.850 | 8 |

推論時間包含 set_image 的前處理／encoder，加 predict 的 decoder／原生原圖尺寸還原；
不含模型載入、權重雜湊、程序啟動、檔案輸出或人工提示時間。
RSS 是每個獨立程序的峰值，包含 Python、模型與推論配置，不是 GPU 記憶體或純模型大小。
每模型依序獨立執行，單一真實眼睛只跑一次，沒有正式暖機或重複計時，受檔案快取及系統負載影響。
以上不足以作完整速度排名或不同物種效果結論；沒有將單眼結果直接外推為可靠的全量耗時。

三者在共同合成圖均輸出 2009 個前景像素；該圖只檢查介面與輸出，不是動物研究資料。
模型內部預測品質分數只用於同模型候選選擇，並非已校準準確率，可能超過 1；沒有裁切或當成跨模型成績。

## Prediction Policy and Coordinates

沿用核准提案：同一人工一框＋一正點、同一 ROI，各模型使用原生前處理。
座標先減 ROI 起點，連續 x/y 直接傳給官方 predictor，保留其內部點編碼；
不自行再加半像素偏移，不以人工 Mask 重心取代人點的位置。
若日後計算輸出 Mask 的重心，才沿用原圖像素中心 (column+0.5,row+0.5)。

三者均使用 multimask_output=True、return_logits=True；以模型自己的最高品質分數取一個候選，平手取最低索引。
使用官方 predictor 還原至 ROI 原像素格的 logits，threshold=0 後放回原圖對應範圍，ROI 外為零。
沒有額外填洞、平滑、取最大區塊、手修或超解析；SAM 2 CUDA extension／其後處理沒有啟用。

每模型保存 ROI logits、原圖尺寸 640×428 的 L-mode 0／255 PNG、原圖 overlay 與結果 JSON。
唯讀驗證核對檔案 digest，從保存 logits 重新產生原圖 Mask，逐像素與 PNG 相符；不宣稱已重跑模型。

[放大查看三模型輸出](../outputs/phase3/eye_segmentation_poc_v1/compatibility_review_v1/single_eye_predictions.png)。
四欄是原圖＋人工提示及三模型輸出；藍色是預測 Mask，未加入人工 GT 比較，沒有標為正確答案。
放大使用 nearest-neighbor，只供檢查既有原圖像素。

## Versions, Downloads and Installation Recovery

| 模型 | 固定官方 commit | checkpoint SHA-256 |
| --- | --- | --- |
| SAM 2.1 Hiera-Tiny | 2b90b9f5ceec907a1c18123530e92e794ad901a4 | 7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69 |
| SAM ViT-B | dca509fe793f601edb92606367a655c15ac00fdf | ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912 |
| MobileSAM ViT-T | f706ad9c4eb7f219c00d9050e46328518ffb65d2 | 6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f |

權重來自提案已核准的官方 URL，保存於 weights/eye_poc/，下載 manifest 記錄來源、大小、時間與實得 SHA-256；
未以獨立上游參考 digest 認證檔案，不將自行算出的 SHA 當成外部來源保證。
官方模型原始碼／設定檔與相應 LICENSE 已保存；來源仍為先前核准的同一組 commit，沒有換成其他模型或封裝。

完整 Git 下載先後遇到 HTTP/2 中斷與 HTTP/1.1 逾時；改採同 commit 的 shallow、blob-filtered sparse checkout，
只取得模型套件與根目錄建置／授權資料，省略歷史、範例資產與重複權重。
對 checkout 及安裝後的 Python／YAML／JSON runtime 檔逐檔比對 digest；CUDA .cu 建置來源不會安裝，未誤當 runtime 缺檔。

SAM 2 的隔離建置原本會額外擷取 torch；本輪改用 SAM2_BUILD_CUDA=0、no-build-isolation，
沿用現有 torch，以 setuptools=80.9.0、wheel=0.45.1 建置。
新增 18 個必要／建置套件，原有 61 個套件的版本全部保持不變；pyproject.toml、uv.lock 不變。
完整實裝版本在 outputs/phase3/eye_segmentation_poc_v1/environment_constraints.txt 與每次 run manifest。

compatibility_synthetic_v1 保留初始診斷：SAM 2 在 CUDA 建置來源檢查階段被拒，並非 forward 失敗；
另兩模型可執行。修正來源檢查後，compatibility_synthetic_v2 三者通過；正式真實眼睛使用 compatibility_manual_v2。
未覆寫初始失敗記錄或將其計入正式眼睛準確度。

## Commands and Reproduction

目前環境已安裝，從 repository 根目錄執行以下唯讀檢查即可：

```sh
.venv/bin/python -m animal_eye_metrology.eye_poc \
  --verify outputs/phase3/eye_segmentation_poc_v1/compatibility_synthetic_v2
.venv/bin/python -m animal_eye_metrology.eye_poc \
  --verify outputs/phase3/eye_segmentation_poc_v1/compatibility_manual_v2
```

本次實際推論命令如下；既有目錄拒絕覆寫。若核准重跑，先調整設定中的 run_revision，再明確建立新版本。
這只執行單眼相容性，沒有把完整 POC 偷塞進命令：

```sh
.venv/bin/python -m animal_eye_metrology.eye_poc \
  --config configs/eye_segmentation_poc.toml --kind synthetic
.venv/bin/python -m animal_eye_metrology.eye_poc \
  --config configs/eye_segmentation_poc.toml --kind manual
```

固定提示頁可從 http://127.0.0.1:8516 使用。服務停止時：

```sh
AEM_WORKSPACE='Eye Segmentation Prompts' \
.venv/bin/streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8516 --server.headless true \
  --browser.gatherUsageStats false
```

只看原圖、畫粗框、點框內正點、填 Reviewer、勾選來源確認後保存；提示另存 prompts/prompts.csv，
每次追加版本，以 image_id＋annotation_id＋eye_side＋prompt_version 識別。舊視窗不能覆蓋較新版本。
保存提示不會自動啟動模型；推論先將選定提示與影像固定在 request.json／input.png。
未保存內容不會跨新連線還原；提示可見性與左右由使用者判讀，不從舊人工結果預填。

新環境可用 configs/eye_poc_requirements.txt 的官方固定 VCS 來源。
若採本次 sparse 來源安裝，vendor/prepare_sources.py 保留實際來源擷取程序及 commit；
它拒絕覆寫既有 vendor 來源，僅適用全新工作副本。下載程序 vendor/download_weights.py 同樣拒絕覆寫。
本次安裝時先以當時 61 個實裝版本產生 constraint，命令為：

```sh
uv pip install --python .venv/bin/python \
  --constraint /tmp/eye_poc_existing_constraints.txt setuptools==80.9.0 wheel==0.45.1
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python --no-build-isolation \
  --constraint /tmp/eye_poc_existing_constraints.txt \
  outputs/phase3/eye_segmentation_poc_v1/vendor/SAM-2 \
  outputs/phase3/eye_segmentation_poc_v1/vendor/segment-anything \
  outputs/phase3/eye_segmentation_poc_v1/vendor/mobile-sam timm==1.0.22
```

這是本次執行紀錄，/tmp constraint 不是可攜交付檔；復原環境請依保存的 environment_constraints.txt，
及官方 source_manifest 的 commit 重新建立來源。模型額外安裝不在主 uv.lock；重新 uv sync 後可能需重新安裝此 optional POC。

## Files

| 新增／修改檔案 | 用途 |
| --- | --- |
| src/animal_eye_metrology/eye_poc.py | 三模型固定提示相容性、分程序執行、來源／權重核對與唯讀輸出驗證 |
| src/animal_eye_metrology/eye_prompts.py | 獨立版本化提示 CSV、有效性檢查、原圖 ROI 與過期視窗保護 |
| app/streamlit_app.py、app/eye_prompt_canvas.js | 沿用單頁工作區，新增原圖點／框操作，不讀 Eye GT |
| configs/eye_segmentation_poc.toml、eye_poc_requirements.txt | 固定模型／CPU／seed／路徑及 optional 套件來源 |
| tests/test_eye_prompts.py | 五個原圖幾何／保存／損壞／重新載入測試，僅用合成資料 |
| README.md、本文件 | 操作、執行、相容性結果與限制 |
| outputs/phase3/eye_segmentation_poc_v1/ | 提示、官方來源紀錄、分開的合成／真實相容性產物及放大圖 |
| weights/eye_poc/ | 三份官方 checkpoint 與下載 manifest；Git 已忽略 |

## Validation and UI QA

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests app
.venv/bin/ruff format --check src/animal_eye_metrology/eye_poc.py \
  src/animal_eye_metrology/eye_prompts.py tests/test_eye_prompts.py app/streamlit_app.py
```

實際結果：93 passed，53 個既有 pycocotools／NumPy 棄用警告；Ruff check／format 通過。
五個新測試涵蓋獨立左右眼與版本保存、過期視窗、無效來源／座標、依框計算 ROI、
已知 logits 原圖還原、損壞偵測，以及 Streamlit 用合成資料保存後重開。
兩份 v2 的三模型輸出均通過 digest 與 logits 還原驗證。
啟動前固定的 724 個原有資料／規範／設定檔 digest 全部不變，包括兩輪人工標註與既有量測基準。
沒有修改 AGENTS.md、Phase 01.md、人工 Mask 或既有人工 CSV，沒有 Commit 或 Push。

Browser plugin not available，使用已快取的 Playwright CLI 驗證本機頁面；無新增瀏覽器相依套件。
流程：開啟 8516 → 原圖提示頁 → 畫框 → 點正點 → 清除未保存測試內容。
瀏覽器測試未按正式資料保存；保存／重開用 pytest 的隔離合成資料驗證。

| 檢查 | 結果與證據 |
| --- | --- |
| 頁面身份／非空 | URL 8516，標題 Animal Eye · Dataset Explorer，原圖畫布與表單存在 |
| 操作 | 框／點更新狀態，清除後恢復未設定；正式測試提示未落盤 |
| Runtime overlay／Console | 沒有 Streamlit exception；Errors=0、Warnings=0 |
| 桌面 | 1440×1100，畫布與原生保存表單可操作；截圖 /tmp/eye_prompt_desktop.png |
| 窄螢幕 | 390×844，原生側欄收合後無頁面水平捲動；截圖 /tmp/eye_prompt_mobile.png；未驗證實際觸控 |
| 保存與載入 | 合成 AppTest 保存 v1、重開恢復原圖框／點；人工兩筆資料由使用者自行保存 |

## Next Step

相容性通過後，仍須先完成其餘 10 眼提示，才能做既有三圖／12 眼的正式固定提示比較。
沿用第一輪固定 Eye GT 作主參考，dog 第二輪作敏感度分析；正式比較才評估 Mask、重心與距離。
提示過窄／過粗或非眼內點的影響須記錄，不看預測後逐模型改提示以提高分數。
這次僅一眼，不代表三物種泛化、全自動定位成功，也沒有排除預訓練原圖重疊風險。
