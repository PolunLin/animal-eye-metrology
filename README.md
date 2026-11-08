# Animal Eye Metrology

AI-Based Segmentation and Distance Measurement.

以 COCO 多動物影像為研究起點，串接動物實例分割、眼部定位／分割與像素距離量測。
提供本機 Streamlit 工作區與 HTTP API，保留人工參考、失敗狀態、設定與來源紀錄。

## Capabilities and Limits

| 功能 | 目前行為 |
| --- | --- |
| 單張圖片推論 | 上傳 JPEG／PNG，執行 YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny |
| 動物與眼睛 | 各動物的實例遮罩、眼點、眼部遮罩與重心；啟用 COCO 十類，鳥類明確標為域外試驗 |
| 距離 | 同隻動物左右眼、同圖任意兩隻動物右眼的遮罩重心距離，單位 px |
| 輸出 | PNG、逐動物／逐眼／候選距離 CSV、JSON、日誌與來源雜湊 |
| 研究工作區 | COCO 篩選、人工眼部標註、固定提示與 v1／v2 成果比較；需另備研究資料與凍結產物 |

**新圖輸出都是候選量測，資格仍為 `unverified`。** 遮罩重心不是瞳孔中心，像素距離不是實體長度。
左右眼、可見性與可靠量測資格尚未完成獨立驗證，不能外推全物種準確率。
新圖沒有人工參考時不產生 IoU、Dice 或距離誤差。

## Technology Stack

| 部分 | 技術與理由 |
| --- | --- |
| 應用 | Python 3.11、Streamlit、FastAPI／Uvicorn、本機檔案；無資料庫與登入系統 |
| 動物分割 | YOLO26s-seg／Ultralytics 8.4.0；同時提供每隻動物的框與實例遮罩 |
| 眼部定位 | HRNet-W32／AP-10K、固定 MMPose commit；使用動物眼點通道，隔離相依環境 |
| 眼部分割 | SAM 2.1 Tiny、固定官方 commit；以眼點與局部 ROI 產生眼部遮罩 |
| 環境 | 應用 Torch 2.9.1；HRNet Torch 2.2.2／MMCV 2.1.0，維持兩套環境 |
| 品質與紀錄 | CPU FP32、seed 42、4 threads、Pytest、Ruff、來源／權重 SHA-256 |

模型理由、公式、人工參考與限制見 [Methodology](docs/methodology.md)。

## Local Setup

從儲存庫根目錄執行；需要 Python 3.11。重建工具使用全新的 `.runtime/`，保留歷史 `.venv`，
拒絕覆寫已存在的目標環境。版本來源與四份平台鎖定檔見 [Setup](docs/setup.md#dependency-sources-and-records)。

```sh
cp .env.example .env
python3.11 scripts/bootstrap.py --all
python3.11 scripts/prepare_models.py --download
python3.11 scripts/check_runtime.py --config configs/inference.runtime.toml
AEM_INFERENCE_CONFIG=configs/inference.runtime.toml AEM_WORKSPACE='New Image Inference' \
  .runtime/app/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8517 --browser.gatherUsageStats false
```

開啟 [本機介面](http://127.0.0.1:8517)。只有明確使用 `--download` 才取得模型權重；
一般推論不自動下載模型或 COCO。已有權重時可先執行模型檢查。
完整環境檢查、本機 API 與研究模式見 [Setup](docs/setup.md)。

## Docker

首個目標為 **Linux amd64、CPU**。同一映像保留應用與 HRNet 的隔離 Python 環境，
Compose 提供 UI 與 API 兩個服務。權重唯讀掛載，輸出獨立保存。

```sh
cp .env.example .env
python3.11 scripts/prepare_models.py --download
docker compose build
docker compose up
```

介面預設 [localhost:8517](http://127.0.0.1:8517)，API 預設 [localhost:8000](http://127.0.0.1:8000)。
Linux amd64 的建置、容器測試、三模型載入與 HTTP 推論已有成功驗證；
建置、掛載及 CI 證據見 [Docker](docs/docker.md)。

## API and Access

[API 操作與範例](docs/api.md)；API 啟動後提供 [互動文件](http://127.0.0.1:8000/docs)。

**測試帳號：不適用。** 本機研究原型沒有帳號、密碼或登入流程，無需外部 AI API key。
`.env.example` 提供非機密執行設定；此交付未宣稱具備公開網路身分驗證或多使用者服務能力。

## Notebook Demo

開啟 [Demo Notebook](notebooks/demo.ipynb)，逐步展示原圖、動物／眼部分割、眼部放大、
兩種候選距離與幾何驗算，直接呼叫專案 Python 程式。
預設重播指定的歷史結果；缺少本機 run 時退回已提交的 CSV。切換 `MODE = "live"` 才執行模型。

```sh
uv venv --python 3.11 .runtime/notebook
uv pip install --python .runtime/notebook/bin/python -r requirements/notebook.txt
.runtime/notebook/bin/python -m jupyterlab notebooks/demo.ipynb
```

Notebook 環境獨立於模型環境；完整推論仍需本機權重與配套設定。
模式切換、重現命令與研究限制見 [Notebook Demo Guide](docs/notebook_demo.md)。

## Examples and Validation

| 交付項目 | 入口 |
| --- | --- |
| 系統架構圖 | [Architecture](docs/architecture.md) |
| 互動 Demo | [Notebook](notebooks/demo.ipynb)、[使用說明](docs/notebook_demo.md) |
| 測試 CSV | [measurements.csv](examples/measurements.csv)、[來源與欄位](examples/README.md) |
| 公式與驗證 | [Methodology](docs/methodology.md#measurement-and-formula-verification) |
| 操作文件索引 | [Documentation Index](docs/index.md) |

```sh
.runtime/app/bin/python -m pytest -q
.runtime/app/bin/ruff check src tests app scripts
.runtime/app/bin/ruff format --check src tests app scripts
```

單元測試以合成幾何與隔離資料驗證公式、座標、失敗狀態與保存，不把合成資料當研究 GT。
真實模型與 Docker 的驗證方式見 [Setup](docs/setup.md#validation) 與 [Docker](docs/docker.md#validation-status)。

## Repository Layout

```text
app/                         Streamlit entry point and drawing components
src/animal_eye_metrology/    Pipeline, model workers, measurement, review and API
configs/                     Application and frozen research configurations
requirements/                Runtime and build dependency definitions
notebooks/                   Step-by-step Python demo; recorded or live inference
scripts/                     Fresh environment, model and readiness tools
tests/                       Automated tests using isolated fixtures
examples/                    Recorded CSV and provenance notes
docs/                        Seven project guides: index, setup, Docker, API, architecture, methods, notebook
Dockerfile, compose.yaml     CPU container and services
.env.example                 Non-secret configuration template
data/, weights/, outputs/    Local assets and generated artifacts; ignored by Git
```

Git 只保留程式、執行設定、依賴定義、自動測試、空輸出的 Demo Notebook、小型 CSV 範例，
以及說明此專案的上述七份文件。Phase 研究紀錄、提案、Agent 筆記、執行報告與歷史文件快照
在本機保留並由 `.gitignore` 排除；資料集、權重、虛擬環境、快取及推論產物也不提交。
研究模式的完整原圖、人工參考與凍結產物需另外備妥，不是乾淨 checkout 的附帶資料。

## License

本專案自行撰寫的原始程式採 [MIT](LICENSE)，不將上游模型、權重、套件或資料重新授權為 MIT。

| 上游 | 官方授權／來源 |
| --- | --- |
| YOLO／Ultralytics | [8.4.0 原始碼 LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.4.0/LICENSE)、[程式與模型授權說明](https://www.ultralytics.com/license) |
| HRNet／MMPose | [固定 commit 的 Apache-2.0 LICENSE](https://github.com/open-mmlab/mmpose/blob/5408bc76f5b848cf925a0d1857899011d8c5b497/LICENSE)、[AP-10K 模型來源](https://github.com/open-mmlab/mmpose/tree/5408bc76f5b848cf925a0d1857899011d8c5b497/configs/animal_2d_keypoint/topdown_heatmap/ap10k) |
| SAM 2.1 | [固定 commit 的 Apache-2.0 LICENSE](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE)、[模型與 checkpoint 來源](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md) |

資料與模型的取得方式見 [Setup](docs/setup.md)；使用者仍須遵守各上游授權。
