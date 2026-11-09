# Animal Eye Metrology

以 COCO 2017 多動物影像示範：**動物實例分割 → 眼部定位與分割 → 雙眼及跨動物右眼距離**。
提供 Streamlit 介面、FastAPI API、Docker 部署與 CSV／PNG／JSON 下載。
交付固定使用 YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny，ROI 採 A_r100。

距離是原圖上的**眼部遮罩重心距離（px）**，不是瞳孔距離或公分。
新圖結果標記 `candidate / unverified`；沒有人工參考時不產生準確率或誤差。

## Reviewed Validation Results

2026-11-10 固定目前已保存的 v2 人工審核，對 **26 張 COCO 多動物照片**重新執行原模型與 A_r100。
其中 **16 張、42 眼**符合量測參考條件；其餘眼睛的模糊、遮擋、未審核等狀態完整保留。

![本次人工審核開發驗證：覆蓋率、分割重疊與距離誤差](examples/reviewed_validation/overview.png)

| 評估項目 | 本次結果 | 分母／可計算數 |
| --- | ---: | --- |
| 眼睛 Mean IoU／Dice | **0.540／0.584** | 42 眼，含 2 眼無法配對而記 0 |
| 平均遮罩重心誤差 | **3.683 px** | 40／42 眼可計算 |
| 同隻動物雙眼 MAE／RMSE | **4.665／9.769 px** | 13／13 組，來自 8 張圖 |
| 不同動物右眼 MAE／RMSE | **3.576／4.783 px** | 16／16 組，來自 7 張圖 |

**單隻貓：眼部輪廓與雙眼距離**

![單隻橘貓：原圖、審核參考與模型眼部遮罩](examples/reviewed_validation/eye_examples.png)

展示選用 COCO `555705` 左側橘貓（動物 `49839`）：雙眼距離參考與預測皆為 **45.142 px**，
與本次審核參考的差為 **0 px**。本例審核後的輪廓與既有預測相同，作為流程展示；整體表現以上表完整驗證為準。

**兩隻貓：完整動物輪廓、眼部遮罩與跨動物距離**

![兩隻貓完整畫面：動物與眼部分割、各自雙眼及跨右眼距離](examples/reviewed_validation/two_cats.png)

同張 COCO `555705` 圖片展示橘貓與虎斑貓，量測均使用原圖眼部遮罩重心：

| 量測 | 模型距離 |
| --- | ---: |
| 橘貓雙眼 | 45.142 px |
| 虎斑貓雙眼 | 52.430 px |
| 兩隻貓的右眼 | 111.884 px |

這是**人工審核的小樣本開發驗證**，部分參考已有模型預測曝光，不能當作獨立泛化準確度。
雙眼距離最大誤差達 **30.754 px**；距離算得出來不代表量測合格。跨右眼 16 組中有 10 組來自同一張圖，配對並非獨立樣本。
[完整結果、逐眼／逐距離 CSV 與重現步驟](examples/reviewed_validation/README.md) 保留失敗與排除分母；
原本三張示範圖的歷史結果另列於 [Methodology](docs/methodology.md#recorded-observations)，不混合比較。

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

三個模型分別負責不同子任務。專案包含動物分割微調、學習率診斷與再評估；
目前推論仍採原始預訓練權重。方法、歷史結果及重現命令見 [Training](docs/training.md)。
介面的「訓練與實驗」頁籤可查看保存結果、曲線與下載比較 CSV。
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

## Eye Annotation

「眼睛標註與審核」頁籤提供 v2 人工輪廓、草稿與審核、版本保存與遮罩下載。
準備工作區與新增批次見 [Eye Annotation](docs/eye_annotation.md)。頁面不會自動訓練或更換模型。

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
需要已啟動的 Docker Engine／Desktop、Compose v2 與 Python 3.11（只用於下載／核對權重）。
建置需連網並編譯 MMCV，首次耗時較長。Apple Silicon 會模擬 amd64，建置與推論速度可能較慢。
請先依上方 COCO Example 取得示範圖，或準備自己的多動物 JPEG／PNG。

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
docker compose exec -T api python scripts/smoke_api.py \
  --image data/000000125405.jpg --report outputs/api_smoke.json
docker compose logs --tail=100
docker compose down
```

建置會安裝固定套件與模型程式碼；推論不下載資料。Linux arm64／GPU 不在目前驗證範圍。
CI 提供輕量測試與選擇性 CPU 容器檢查；這些工程檢查不代替模型準確度驗證。
容器 smoke 檢查包含真實圖片、兩種距離、CSV／PNG 下載、無效圖片及無動物圖片。
報告保存在主機 `outputs/api_smoke.json`；Docker 模式不需要先建立本機 `.runtime/`。

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

本次已用固定人工審核參考分開計算 IoU／Dice、重心誤差與兩類距離 MAE／RMSE。
可直接[重播保存的驗證結果](examples/reviewed_validation/README.md#replay-scores)，不需模型或原圖。
公式及驗證界線見 [Methodology](docs/methodology.md)。

## Repository Layout

```text
app/                      Streamlit entry point
src/animal_eye_metrology/  API, CLI, inference, models, measurement, UI and core helpers
configs/                  Local / Docker inference and pinned model configuration
requirements/             Application / pose dependencies and platform locks
scripts/                  Environment, model, runtime and delivery checks
tests/                    Final pipeline contract and geometry tests
examples/                 Baseline and reviewed validation evidence, CSV and provenance
docs/                     API, architecture and methodology
```

訓練程式、設定、測試與精簡實驗證據納入公開交付；其他歷史研究與答辯筆記仍保存在 `local_only/`，由 Git 與 Docker 排除。
完整資料集、權重、環境及原始執行產物不提交；`examples/` 保留精簡結果與可重播評分的參考／預測遮罩。公開版本不依賴本機研究備份。

## License

自行撰寫的程式採 [MIT](LICENSE)；上游模型、套件與資料遵循各自授權：
[Ultralytics](https://github.com/ultralytics/ultralytics/blob/v8.4.0/LICENSE)、
[MMPose](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/LICENSE)、
[SAM 2](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE)。
