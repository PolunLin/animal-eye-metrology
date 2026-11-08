# New Image Inference

## Scope and Decisions

本輪依核准路線圖實作第 1、2 項：單張上傳完整推論，以及結果／失敗狀態。
沿用既有模型與量測方式，沒有訓練新模型、改變 GT 定義或晉級 ROI 方法。
依後續核准範圍，眼部推論已從三類擴充至 COCO 十類：鳥、貓、狗、馬、羊、牛、象、熊、斑馬、長頸鹿。
Results Explorer 的 v1／v2 與凍結 A 基線維持原狀。

| 階段 | 實作與理由 | 適用限制 |
| --- | --- | --- |
| 動物偵測／實例分割 | YOLO26s-seg，Ultralytics 8.4.0 隔離套件目錄，CPU | 列出 COCO 動物預測，分數 ≥ 0.5 且框有效才進入後續；保留低分／無效候選 |
| 眼點定位 | 官方 HRNet-W32／AP-10K，獨立 `.venv-eye-pose` 子程序 | COCO 十類皆執行；鳥類為訓練範圍外試驗。L_Eye／R_Eye 是模型解剖通道，沒有人工確認 |
| 眼部分割 | SAM 2.1 Tiny；一個正點、無框、三候選中選模型品質分數最高者 | 不以 GT 重排，不作填洞、平滑或手動修補；模型分數不是已校準的正確率 |
| ROI | A_r100：`max(32, 2*d)`；A_r050：`0.5*max(32, 2*d)` | `d` 是 HRNet 兩眼距離；以眼點為中心、floor／ceil 邊界、限制於原圖。A_r100 必跑 |
| 重心 | 原圖二值遮罩前景像素中心平均，座標為 `(column+0.5, row+0.5)` | 空遮罩沒有重心；不是瞳孔中心 |
| 距離 | 同一動物左右眼重心，以及同圖任意兩動物右眼重心的歐氏距離 | 單位 px；全部標示 candidate／unverified，沒有實體尺度、校準誤差或信賴區間 |

定位沿用既有成對有效性規則：任一眼點不有限、熱圖分數非正、超出圖像／原生動物裁切範圍，
或兩點重合，該動物兩眼都不進入 SAM。這是既有 A 方法限制，並非可見性分類器。
SAM 某眼失敗不阻擋其他眼；動物實例遮罩失敗但框有效時仍可做眼點定位，兩種狀態分開保存。

## COCO Ten-class Execution Policy

`inference_capabilities.py` 是十類啟用政策的共同來源，設定、模型 worker 與介面使用同一清單。
`supported` 表示本次程式是否執行眼部流程，不代表視覺準確度或量測資格通過。

| 動物 | 眼部執行 | 模型範圍欄位 | 驗證界線 |
| --- | --- | --- | --- |
| 貓、狗、馬、羊、牛、象、熊、斑馬、長頸鹿 | 啟用現有 HRNet → SAM | `pose_domain=ap10k_mammal_mapping` | AP-10K 有資料集類別對應，不等於本專案眼部／量測準確度驗收 |
| 鳥 | 同樣啟用現有 HRNet → SAM | `pose_domain=out_of_training_domain`、`experimental=true` | 明確是域外試驗；非空遮罩、正分數或有限距離都不能證明鳥眼定位正確 |

[AP-10K 官方論文](https://arxiv.org/abs/2108.12617)定義其為哺乳動物姿態資料集。
官方註記亦允許推估部分遮擋關鍵點，因此模型輸出不能取代本專案的可見性判定。
來源：[官方標註說明](https://github.com/AlexTheBad/AP-10K/blob/181b1a04755e4dc6fe5616ef7a88496f47bfe228/README.md#3-how-were-the-keypoints-instructed-to-be-labeled)。

`experimental` 及兩個端點的 `pose_domain_a/b` 會傳至距離 CSV／JSON；任一端點是鳥類域外輸出，
該距離也標為試驗性。所有類別的 `qualification`、可見性與左右眼資格仍為 `unverified`。
每個新 run 保存版本 `coco10_ap10k_bird_experimental_v1` 的能力政策；舊三類設定與結果保留原值，
必須重新推論才產生十類版本，不能把舊跳過結果回填成成功。
若要宣稱鳥類可以可靠量測，需要另行選擇／驗證鳥類眼點方法與獨立眼部參考，這次沒有更換模型。

## Inputs and Artifacts

介面將動物與眼部分開呈現。「動物輪廓（實例分割）」直接讀取已保存的 YOLO 動物遮罩，
使用每隻不同色的填色與遮罩邊界；矩形偵測框是另外的可選圖層，不能替代動物輪廓。
「實例遮罩」顯示各隻動物的彩色前景，黑色是背景；逐隻原始二值 PNG 可單獨檢視及下載。
這些展示圖在記憶體中產生，不改寫既有 run／manifest、不修改遮罩，也不重新推論或量測。
輪廓保留原遮罩的孔洞與分離區域，沒有用凸包、平滑或人工補畫更改模型輸出。
若動物遮罩失敗或缺檔，介面保留原因，不將偵測矩形當作分割成果。

輪廓顯示修正後驗證：完整 Pytest 313 passed（同樣 53 個既有警告），Ruff 通過。
瀏覽器實際切換輪廓／實例遮罩、檢視逐隻遮罩並下載；下載的原始 PNG bytes 與保存遮罩相同。
桌面與 390×844 手機版沒有頁面例外／整頁水平溢位；修改前清單的 2,533 個檔案雜湊未變。
修改集中於 `inference_artifacts.py`、`inference_view.py`、其對應測試與操作文件。

輸入只接受真正 JPEG／PNG 的單張靜態影像，限制 20 MiB／12,000,000 像素。
先套用 EXIF 方向，再轉 RGB；所有含透明通道的影像先合成白色背景，不調整解析度。
所有座標與距離都對應這張正規化原圖；保存原始 bytes 與正規化 PNG 的各自 SHA-256。
輸入檔名僅作顯示，不用來推導 COCO／標註 ID 或寫入路徑。

| 檔案 | 內容 |
| --- | --- |
| `input/original.*`、`input/image.png` | 原上傳資料與正規化原圖 |
| `config.toml`、`*_request.json` | 設定副本與實際模型請求，包含路徑與雜湊 |
| `detection.json`、`pose/pose_results.json`、`segmentation.json` | 模型輸出、環境版本、來源核對、候選分數與各自執行時間 |
| `animal_masks/`、`eye_masks/`、`pose/*_heatmaps.npy` | 原圖尺寸二值遮罩及 HRNet 熱圖 |
| `overlay_A_r100.png`、可選 `overlay_A_r050.png` | 原圖尺寸疊圖；不回寫或改變原始遮罩 |
| `animals.csv`、`eyes.csv`、`measurements.csv` | 保留失敗的長表；空值不以 0 代替，無動物時仍有欄位標頭 |
| `result.json` | 完整階段狀態、結果、候選資格、座標、設定、限制、執行命令與日誌路徑 |
| `*.log`、`*_error.json` | 子程序日誌、模型載入或執行失敗原因 |
| `run_manifest.json` | 設定副本與輸出 SHA-256；執行前後來源雜湊及是否變動 |

每次明確按下執行都建立新的 UTC 時間／UUID 目錄。切換檢視、下載及一般 rerun 不重新推論。
圖片、ROI 選項或設定變更後，介面隱藏上一筆結果並要求重新執行，避免對錯圖片。
新圖不讀取 COCO 註記、v1／v2 GT、既有預測配對或人工點框；pose worker 內部 `image_id=0`
只是本次單圖請求鍵，對外動物使用本次偵測 `animal_####`，不冒充人工 annotation_id。

## Status Semantics

| 狀態 | 意義 |
| --- | --- |
| `completed` | 程式與數值檢查完成，不代表分割正確或量測合格 |
| `completed_with_issues` | 至少有無效偵測、動物遮罩問題、未支援類別或不可用眼槽；逐筆列原因 |
| `no_animals` | 沒有通過既有偵測門檻的動物；不等於圖片確定沒有動物 |
| `unsupported_category` | 可保留動物偵測，但未在本輪核准眼部類別範圍 |
| `empty_mask` | 推論完成但沒有前景，不產生重心或假距離 |
| `failed` | 模型載入、執行、來源核對、檔案或格式失敗；不當作正常零偵測 |
| `skipped` | 因上游失敗或沒有有效輸入而未執行 |
| 距離 `candidate` | 兩個遮罩重心可計算，生物學與量測資格仍 `unverified` |
| 距離 `unavailable` | 任一重心不可用；保留配對與端點失敗原因 |

全局模型失敗會寫出已完成的階段、下游未執行狀態與可下載紀錄。每個子程序有 900 秒上限。
介面先檢查本機檔案與權重雜湊，實際執行再核對套件／來源；不在 UI 進程混用 Torch／MMCV 環境。
若來源檔在執行期間變動，manifest 的 `source_unchanged_during_run` 為 false，該執行不應當作可重現研究證據。

## Reproduction

使用目前已通過相容性驗證的本機環境及既有權重，不執行新安裝或下載。

```sh
.venv/bin/python -m animal_eye_metrology.inference_pipeline --preflight
.venv/bin/python -m animal_eye_metrology.inference_pipeline \
  --config configs/inference.toml \
  --image data/coco/val2017/000000125405.jpg --compare-r050
.venv/bin/python -m animal_eye_metrology.inference_pipeline \
  --image data/coco/val2017/000000020247.jpg --compare-r050
.venv/bin/python -m animal_eye_metrology.inference_pipeline \
  --image data/coco/val2017/000000333069.jpg --compare-r050
AEM_WORKSPACE='New Image Inference' .venv/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8517 --server.headless true \
  --browser.gatherUsageStats false
.venv/bin/pytest -q
.venv/bin/ruff check src tests app
.venv/bin/ruff format --check src tests app
```

固定 CPU、seed 42、4 threads；YOLO 設定沿用 image_size 640、confidence_floor 0.001、
候選門檻 0.5、IoU 0.7、max_detections 100、retina_masks、rect=false、float32。
上傳流程不做基準測試預熱，時間包含本次模型載入及子程序開銷，不能與先前穩態效能表直接比較。
設定檔包含三個權重 SHA-256；SAM／HRNet 另核對已釘住的官方來源 commit 與本機來源。

## Validation and Next Steps

### COCO Ten-class Engineering Run

十類政策更新後，使用本機 COCO val2017 做每類一張的流程試跑，固定 CPU／seed 42／4 threads、
A_r100 與 A_r050。模型、閾值、ROI、選罩方式及重心／距離公式均未調整。
這一輪沒有眼部 GT，因此只報流程輸出數，不計算或宣稱眼部 IoU、距離誤差與準確率。

| 目標類別 | image_id | 通過門檻的動物候選數 | 兩條件非空逐眼輸出 | 候選距離列數 |
| --- | --- | --- | --- | --- |
| 鳥 | 41888 | 2 | 8 | 6，全部域外試驗 |
| 貓 | 39769 | 2 | 8 | 6 |
| 狗 | 125405 | 2 | 8 | 6 |
| 馬 | 242724 | 2 | 8 | 6 |
| 羊 | 547383 | 2 | 8 | 6 |
| 牛 | 41635 | 2（預測 1 牛、1 馬） | 8 | 6 |
| 象 | 170739 | 4 | 16 | 20 |
| 熊 | 20247 | 2 | 8 | 6 |
| 斑馬 | 52462 | 2 | 8 | 6 |
| 長頸鹿 | 333069 | 2 | 8 | 6 |

選圖規則在推論前固定：狗／熊／長頸鹿沿用原三張圖；馬／羊沿用本次使用者上傳；
牛／象／斑馬取原 selected15 中該類至少兩隻 noncrowd 動物的最小 image_id；
貓／鳥取本機資料中至少兩個該類 noncrowd bbox 面積各 ≥10,000 px² 的最小 image_id。
COCO 類別／bbox 僅用於選圖，不放入模型請求，也不改寫原 Selected Images 名單。
貓與鳥圖片未經本專案人工眼部標註；鳥圖有頭部朝下／眼睛不清的個體，貓圖有躺睡閉眼個體。
因此這些非空輸出尤其不能當作開眼或合格雙眼量測的證據。

此批共 22 個動物**預測候選**、88 筆逐眼輸出、74 筆候選距離，全部仍 `unverified`。
鳥圖原有三隻鳥但本次只有兩個鳥預測通過門檻；牛圖有跨類預測，顯示動物漏偵測／類別錯誤
仍會影響後續流程。馬圖亦有較大的眼周遮罩，啟用該類不代表眼部邊界已正確。
本輪沒有用事後調閾值、換圖或換模型來消除這些案例。

執行紀錄與每次 run ID 見 `outputs/inference/qa_coco10_20261108/runs.json`、`runs.csv`。
原三張開發圖的 24 張 A_r100／A_r050 遮罩與既有輸出逐像素一致，核對檔為同目錄的
`baseline_compatibility.json`。更新前保護清單的 2,650 個檔案未變；唯一預期修改的舊設定為
`configs/inference.toml`，其中各模型與 ROI 的設定未變。完整測試 **334 passed**、Ruff lint／format 通過。

瀏覽器另實際上傳馬圖與鳥圖，使用預設 A_r100 完成推論。馬圖輸出 4 筆逐眼結果與 3 筆
候選距離；鳥圖畫面顯示域外試驗提示，下載 JSON 的 4 筆逐眼資料與距離 CSV 的 3 筆資料
均保留試驗標記，量測資格仍為 `unverified`。1440×1000 桌面與 390×844 行動版檢查通過，
行動版沒有整頁水平溢位，瀏覽器 console 為 0 errors／0 warnings，沒有 Streamlit 頁面例外。

可由既有本機圖片重跑，所有輸出使用新目錄：

```sh
for image_id in 041888 039769 125405 242724 547383 041635 170739 020247 052462 333069; do
  .venv/bin/python -m animal_eye_metrology.inference_pipeline \
    --image "data/coco/val2017/000000${image_id}.jpg" --compare-r050
done
```

馬、羊的上述 CLI 首次驗證使用原上傳 run 中的 `input/original.jpg` bytes，
因此保存檔名為 `original.jpg`；其 SHA-256 與對應上傳一致，`runs.json` 記錄對應 COCO image_id。

### Original Three-image Integration Run

真實模型驗證沿用三張開發圖（COCO val2017：125405 狗、20247 熊、333069 長頸鹿），
每張重新偵測兩隻動物並執行兩個 ROI 條件。這些圖片曾參與開發，驗證目的為工程連通與既有方法相容性，
不是新增資料或獨立效能評估；不以這次成功執行推論眼部邊界已正確。
單元測試以明確合成輸入涵蓋失敗狀態、空值與座標規則，合成資料不充當研究 GT。

2026-11-08 本機驗證紀錄：

| 開發圖 | 通過門檻動物 | 兩條件逐眼／距離列數 | 單次總耗時（含載入） |
| --- | --- | --- | --- |
| 125405，狗 | 2 | 8／6 | 11.26 秒（Streamlit 操作） |
| 20247，熊 | 2 | 8／6 | 12.13 秒（CLI） |
| 333069，長頸鹿 | 2 | 8／6 | 11.08 秒（CLI） |

上述三次 run ID 依序為 `20261108T075732Z_92c42f74bb73`、
`20261108T075518Z_c0c047244953`、`20261108T075530Z_f5837f39031f`。
這是單次工程觀察，沒有建立效能統計。24 張輸出遮罩與既有相同圖、相同條件的 A_r100／A_r050
逐像素一致，動物框及 HRNet 眼點差異均為 0；核對紀錄位於
`outputs/inference/qa_20261108/baseline_compatibility.json`。
這個核對是在推論完成後獨立執行，不把歷史預測輸入新流程，也不重新評估 GT。

| 驗證 | 結果 |
| --- | --- |
| Pytest | 302 passed；53 個既有 pycocotools／NumPy deprecation warnings |
| Ruff | `check src tests app` 與 `format --check src tests app` 通過 |
| 凍結產物 | 執行前核對清單中的 2,251 個既有標註、設定與成果檔案 SHA-256 全部未變 |
| 真實無動物輸入 | 合成白圖由 YOLO 重新推論；沒有眼部距離，HRNet／SAM 跳過，CSV 保留標頭 |
| 瀏覽器 | 既有 Playwright；Browser plugin 未提供。1440×1000 桌面與 390×844 行動版，無頁面例外／整頁水平溢位 |
| 下載與互動 | 真實上傳→兩條件推論→逐眼切換→PNG／CSV／JSON 下載；JSON 與 8 筆逐眼、6 筆距離 CSV 與保存檔案一致 |
| 錯誤及回歸 | 無效圖片內容有錯誤提示；無動物結果可下載；換輸入隱藏舊結果；Results Explorer 的 v1／v2 並列仍可操作；兩個測試瀏覽器皆 0 console errors／warnings |

主要新增檔案為 `inference_pipeline.py`（流程）、`inference_worker.py`（模型）、
`inference_artifacts.py`（量測／匯出）、`inference_view.py`（介面）、
`inference_capabilities.py`（類別能力政策），均位於 `src/animal_eye_metrology/`；
另有 `configs/inference.toml`、對應的 `tests/test_inference_*.py`。
`app/streamlit_app.py` 增加獨立入口，README 與本指南補上操作及研究界線。

下一步應先提案第 3 項獨立驗證：新圖的選取／物種／留出規則，以及少量 v2 重標一致性抽查，
經專案負責人確認後執行。第 4 項 Docker、Linux CPU 與乾淨環境尚未完成；目前是本機單張 POC，
沒有非同步排程、持久化佇列或取消按鈕。依實際新圖失敗情形再決定邊界精修或可靠性判斷。
