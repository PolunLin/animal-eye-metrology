# Animal Eye Metrology

以 COCO 2017 多動物影像示範：**動物實例分割 → 眼部定位與分割 → 雙眼及跨動物右眼距離**。
提供 Streamlit 介面、FastAPI API、Docker 部署與 CSV／PNG／JSON 下載。
交付固定使用 YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny，ROI 採 A_r100。

距離是原圖上的**眼部遮罩重心距離（px）**，不是瞳孔距離或公分。
新圖結果標記 `candidate / unverified`；沒有人工參考時不產生準確率或誤差。

## Assignment Deliverables

| 要求 | 交付位置 |
| --- | --- |
| 完整程式碼 | `src/`、`app/`、`scripts/`、`tests/` |
| 介紹、技術棧、本機與 Docker 步驟 | 本文件 |
| API 文件 | [API Guide](docs/api.md)；啟動後的 [Swagger UI](http://127.0.0.1:8000/docs) |
| 測試帳號 | 不適用；本機展示沒有登入流程，也不需 AI API key |
| 環境變數範例 | [.env.example](.env.example) |
| 測試資料 CSV | [measurements.csv](examples/measurements.csv)、[來源與欄位](examples/README.md) |
| 系統架構圖 | [Architecture](docs/architecture.md) |
| 模型理由、量測公式與驗證 | [Methodology](docs/methodology.md) |

## Technology Stack

| 部分 | 採用技術與用途 |
| --- | --- |
| 應用 | Python 3.11、Streamlit、FastAPI／Uvicorn；本機檔案保存 |
| 動物輪廓 | YOLO26s-seg／Ultralytics 8.4.0，輸出各動物框與實例遮罩 |
| 眼點 | HRNet-W32／AP-10K，提供解剖左右眼通道與位置 |
| 眼部遮罩 | SAM 2.1 Tiny，以預測眼點及局部 ROI 分割 |
| 重現與品質 | 固定來源 commit／權重 SHA-256、CPU FP32、seed 42、4 threads、Pytest、Ruff |

三個模型分別負責不同子任務；本交付沒有模型競賽、微調訓練或研究工作區。
應用使用 Torch 2.9.1，HRNet 使用 Torch 2.2.2／MMCV 2.1.0，以兩個 Python 環境隔離相依。

## Local Setup

支援 Python 3.11、macOS arm64 或 Linux amd64。MMCV CPU operators 必要時由原始碼編譯，
需要 C/C++ 編譯工具；安裝套件與模型時需要網路。從專案根目錄執行：

```sh
cp .env.example .env
python3.11 scripts/bootstrap.py --all
python3.11 scripts/prepare_models.py --download
python3.11 scripts/check_runtime.py --config configs/inference.runtime.toml --load-models
```

`bootstrap.py` 依平台鎖定檔建立 `.runtime/app`、`.runtime/pose`，拒絕覆寫成功建立的環境。
已有權重時以 `python3.11 scripts/prepare_models.py` 檢查即可；一般推論不自動下載。
安裝中斷時先檢查 `.runtime/environment_*.json`；只有失敗／已停止且設定一致的建置可用
`--app --resume` 或 `--pose --resume` 接續。套件版本見 `requirements/` 的四份平台鎖定檔。

啟動介面：

```sh
AEM_INFERENCE_CONFIG=configs/inference.runtime.toml \
  .runtime/app/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8517 --browser.gatherUsageStats false
```

開啟 [本機介面](http://127.0.0.1:8517)，上傳圖片、執行推論，檢視動物／眼部遮罩與兩種距離。
另一個終端機可啟動 API：

```sh
AEM_INFERENCE_CONFIG=configs/inference.runtime.toml \
  .runtime/app/bin/python -m uvicorn animal_eye_metrology.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

`.env` 供 Compose 讀取；本機 Python 指令明確設定 `AEM_INFERENCE_CONFIG`。
CLI 使用同一流程：

```sh
.runtime/app/bin/python -m animal_eye_metrology --image data/000000125405.jpg
```

## COCO Example

範例為 COCO 2017 validation 的 `125405`：兩隻狗，官方非 crowd annotation IDs
為 `13692`、`17207`。這是既有開發示範圖，不是獨立測試集。
[輸入清單](examples/input_manifest.csv) 記錄原圖網址與 SHA-256；不需下載整份 COCO。

```sh
mkdir -p data
curl --fail --location \
  https://s3.amazonaws.com/images.cocodataset.org/val2017/000000125405.jpg \
  --output data/000000125405.jpg
python3.11 - <<'PY'
import hashlib
from pathlib import Path
assert hashlib.sha256(Path('data/000000125405.jpg').read_bytes()).hexdigest() == (
    'aeb172b088be9b334a97fcd4df1fe89b5730bc5836765a23c82f2b8b4ff3d03b'
)
PY
```

## Docker

目標為 **Linux amd64、CPU**；UI 與 API 共用同一映像、各自執行程序。

```sh
cp .env.example .env
python3.11 scripts/prepare_models.py --download
docker compose build
docker compose up
```

UI：[localhost:8517](http://127.0.0.1:8517)；API：[localhost:8000/docs](http://127.0.0.1:8000/docs)。
預設只綁定主機 `127.0.0.1`。`weights/`、`data/` 唯讀掛載，`outputs/` 保存結果，
停止容器後仍保留。埠號與掛載路徑可在 `.env` 調整。

```sh
docker compose run --rm api python scripts/check_runtime.py \
  --config configs/inference.docker.toml --load-models
docker compose run --rm api python -m pytest -q
docker compose logs --tail=100
docker compose down
```

建置會安裝固定套件與模型程式碼；推論不下載資料。Linux arm64／GPU 不在目前驗證範圍。
CI 提供輕量測試與選擇性 CPU 容器檢查；這些工程檢查不代替模型準確度驗證。

## Validation

```sh
.runtime/app/bin/python -m pytest -q
.runtime/app/bin/ruff check src tests app scripts
.runtime/app/bin/ruff format --check src tests app scripts
.runtime/app/bin/python scripts/check_delivery.py
```

測試使用隔離合成資料，驗證幾何、座標、API、失敗狀態與保存；不下載資料或模型。
啟動 API 後，可用既有示範圖檢查真實流程及下載：

```sh
.runtime/app/bin/python scripts/smoke_api.py \
  --image data/000000125405.jpg --report outputs/api_smoke.json
```

模型準確度需另用人工眼部參考，分開評估 IoU／Dice、重心誤差與兩類距離 MAE／RMSE。
現有小樣本觀察、公式及驗證界線見 [Methodology](docs/methodology.md)。

## Repository Layout

```text
app/                      Streamlit entry point
src/animal_eye_metrology/  API, CLI, inference, models, measurement, UI and core helpers
configs/                  Local / Docker inference and pinned model configuration
requirements/             Application / pose dependencies and platform locks
scripts/                  Environment, model, runtime and delivery checks
tests/                    Final pipeline contract and geometry tests
examples/                 Recorded baseline CSV and input provenance
docs/                     API, architecture and methodology
```

研究紀錄、替代模型、訓練工具、歷史程式與答辯筆記存於 `local_only/`，由 Git 與 Docker 排除。
資料集、權重、環境及執行產物也不提交。公開版本不依賴本機研究備份。

## License

自行撰寫的程式採 [MIT](LICENSE)；上游模型、套件與資料遵循各自授權：
[Ultralytics](https://github.com/ultralytics/ultralytics/blob/v8.4.0/LICENSE)、
[MMPose](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/LICENSE)、
[SAM 2](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE)。
