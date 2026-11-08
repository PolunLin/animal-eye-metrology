# Local Setup

## Execution Modes

| 模式 | 必要資料 |
| --- | --- |
| 新圖推論／API | 本機 JPEG／PNG、三個固定權重、兩套執行環境；不需要 COCO 註記或眼睛 GT |
| Results Explorer | 凍結三圖研究產物、原圖與 v1／v2 人工參考 |
| COCO 審核與標註 | 本機 COCO 2017 val2017、官方 instances 註記與指定工作設定 |

Git 不附完整 COCO、權重或歷史 outputs。新圖模式可獨立執行；研究模式缺少對應產物時不會自動補造。

## Fresh Runtime

需要 Python 3.11、可建立虛擬環境的系統與套件下載環境。
HRNet 依賴 MMCV CPU operators，沒有對應 wheel 時需要 C/C++ 編譯工具。
既有真實模型證據來自 macOS arm64 CPU；Linux 範圍以 [Docker 驗證紀錄](docker.md#validation-status) 為準。

從儲存庫根目錄執行：

```sh
cp .env.example .env
python3.11 scripts/bootstrap.py --all
python3.11 scripts/prepare_models.py --download
python3.11 scripts/check_runtime.py --config configs/inference.runtime.toml
```

`bootstrap.py` 建立 `.runtime/app` 與 `.runtime/pose`，準備固定模型程式來源，不更新歷史 `.venv`。
它拒絕覆寫已有目標環境或其環境紀錄；部分失敗也保留紀錄，不能直接重跑來掩蓋原狀態。
可用 `--app`／`--pose` 分別建立環境。模型權重是獨立準備步驟，只有明確指定 `--download` 才允許下載。
已有 `weights/` 時先執行 `python3.11 scripts/prepare_models.py`，只檢查本機檔案。
需要實際載入三個模型時，可在 runtime 檢查加入 `--load-models`；此步驟不是圖片準確度評估。

應用／SAM2 使用 Torch 2.9.1；既有 HRNet 使用 Torch 2.2.2、NumPy 1.26.4、MMCV 2.1.0。
兩套環境保留既有模型行為。主 `uv.lock` 對應原研究專案，單獨 `uv sync` 不能重建完整新圖環境。

## Dependency Sources and Records

| 位置 | 角色 |
| --- | --- |
| `requirements/app.in`、`pose.in` | 應用與 pose 的直接相依來源；Linux 另有 `app-linux.in`／`pose-linux.in` 指定 CPU Torch |
| `requirements/app-macos-arm64.txt` | Python 3.11、macOS arm64 的應用套件版本鎖定 |
| `requirements/pose-macos-arm64.txt` | Python 3.11、macOS arm64 的 pose 套件版本鎖定 |
| `requirements/app-linux-amd64.txt` | Python 3.11、Linux amd64 的應用套件版本鎖定 |
| `requirements/pose-linux-amd64.txt` | Python 3.11、Linux amd64 的 pose 套件版本鎖定 |
| `scripts/bootstrap.py` | 依平台選 lock，另外安裝固定 SAM2／MMPose commit 與 MMCV 2.1.0 CPU operators |
| `.runtime/environment_app.json`、`environment_pose.json` | 命令、狀態、lock SHA-256、實際 Python／平台、套件版本與安裝來源 |

版本解析與成功安裝不等於模型載入／圖片推論驗證。bootstrap 成功後的紀錄狀態為
`installed_not_runtime_validated`，再執行 runtime 檢查及實際圖片流程。
bootstrap 對 Linux amd64／macOS arm64 選擇對應 lock，其他平台會明確拒絕，不猜測套用。

可先查看命令而不建立環境或下載；預覽根目錄也必須沒有既有環境：

```sh
python3.11 scripts/bootstrap.py --all --root .runtime-preview --dry-run
```

要保留已有的 `.runtime` 並建立另一組，可指定全新的 `--root`；配套 inference 設定中的
`python`、`pose_python` 與 `detector_source` 也須對應新位置。不要讓新設定誤用舊環境。

## Interrupted Builds

下載或編譯中斷後，先讀取對應的 `.runtime/environment_*.json`。只有狀態為
`failed` 或已停止的 `building`，且平台、目錄、lock 路徑與 SHA-256 相同，才可續跑：

```sh
python3.11 scripts/bootstrap.py --pose --resume
```

`--resume` 保留 `previous_attempts`，跳過建立 venv，重新核對並安裝既定套件；
不接受成功環境、仍活躍的建置程序或已變動的 lock。修正 lock 後需使用新的環境目錄，保留失敗紀錄。
可用 `--installer pip`／`--installer uv` 明確選擇工具；預設優先使用已安裝的 uv。
套件快取可用 `--cache-dir` 或 `UV_CACHE_DIR`／`PIP_CACHE_DIR` 指定，不得指向既有研究環境、
資料或權重目錄。模型原始碼始終從既定官方 HTTPS commit 安裝；執行時不重新下載。

## Start UI and API

```sh
AEM_INFERENCE_CONFIG=configs/inference.runtime.toml AEM_WORKSPACE='New Image Inference' \
  .runtime/app/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8517 --browser.gatherUsageStats false
```

開啟 [本機介面](http://127.0.0.1:8517)，上傳後按下推論。輸入上限 20 MiB／1,200 萬像素；
套用 EXIF 方向、轉 RGB、透明處合成白色，不縮放原圖。預設 A_r100，可選 A_r050 對照。

另一個終端機啟動 API：

```sh
AEM_INFERENCE_CONFIG=configs/inference.runtime.toml \
  .runtime/app/bin/python -m uvicorn animal_eye_metrology.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

請求與回應見 [API Guide](api.md)。沒有登入，測試帳號不適用；無需雲端 AI 金鑰。
`.env` 是 Compose 的設定入口，本機指令明確指定設定檔。
API、CLI 與 UI 若使用相同輸出目錄，透過 `.inference.lock` 跨程序互斥；已有推論時立即拒絕，
沒有工作佇列。不同輸出目錄不共用此鎖，不能視為跨主機排程器。

## Configuration and Assets

| 位置 | 用途 |
| --- | --- |
| `configs/inference.runtime.toml` | 全新本機環境路徑與固定推論設定 |
| `configs/inference.docker.toml` | 容器內環境與輸出位置 |
| `configs/sam2_runtime.toml` | 應用專用 SAM2 執行／來源設定 |
| `configs/inference.toml` | 保留的歷史本機 POC 設定 |
| `weights/` | YOLO26s-seg、HRNet-W32、SAM 2.1 Tiny；核對 SHA-256 |
| `outputs/inference/` | 獨立 run：輸入、遮罩、CSV／JSON、設定、日誌、manifest |

缺檔、來源或雜湊不符時，先查看檢查結果及日誌；不要以任意升版、改名或回填零值排除錯誤。
歷史設定路徑以設定檔所在目錄解析，工程入口使用配套 runtime 設定。

## Checks and Troubleshooting

```sh
.runtime/app/bin/python -m pytest -q
.runtime/app/bin/ruff check src tests app scripts
.runtime/app/bin/ruff format --check src tests app scripts
```

| 現象 | 檢查 |
| --- | --- |
| 權重／來源缺失 | 模型與 runtime 檢查、配套設定及執行根目錄 |
| MMCV 匯入／operators 失敗 | pose 建置日誌、固定 Torch／NumPy／MMCV 及編譯工具 |
| YOLO 版本不符 | 新應用載入 Ultralytics 8.4.0，歷史研究可能使用 8.3.230 |
| 開啟舊工作區 | 設定 `AEM_WORKSPACE='New Image Inference'` 或手動切換 |
| 有距離但眼睛分錯 | 候選值不代表合格；檢視逐眼遮罩、原因與資格欄位 |
| Results Explorer 缺檔 | 備妥凍結研究產物；先用新圖模式或已提交的 CSV 範例 |

單元測試使用隔離合成資料，不執行完整 COCO 實驗。模型實跑另外保存設定、版本與輸入／輸出雜湊。
歷史研究入口見 [文件索引](index.md)。
