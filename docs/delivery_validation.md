# Delivery Validation

日期：2026-11-08（Asia/Taipei）。這份紀錄分開列出既有基線、本次程式回歸、環境重建與容器驗證。
歷史成功不替代本次新環境或 Linux 執行證據；沒有執行的項目保持待驗證。

## Scope

本輪整理應用啟動、環境定義、Docker、最小 HTTP API、介面模組與交付文件。
沿用 YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny、A_r100／A_r050 與遮罩重心像素距離。
不更新人工 GT、研究樣本、模型權重、預測候選選取或歷史研究成績。

## Verification Matrix

| 範圍 | 已確認結果／目前狀態 | 證據與解讀 |
| --- | --- | --- |
| 整理前全套基線 | 334 passed，53 warnings | 本輪變更前由既有本機環境執行；警告數與測試成功分開記錄 |
| 介面拆分針對性回歸 | 101 passed，50 warnings | 模組整理後的針對性測試；不當成最終全套結果 |
| 原始資料／權重／研究產物保護 | 8,023 檔 SHA-256 全部相同，無缺檔 | 本次工作機基線 `/tmp/aem_delivery_audit/protected.json`；不將新增文件當作受保護歷史產物 |
| CSV 範例 | 原 bytes／SHA-256、6 筆距離公式與 candidate／unverified 核對通過 | 見 [Example](../examples/README.md)，沒有執行模型或新增準確度評估 |
| 文件 | 新文件本地連結、Markdown fences 檢查通過 | 歷史 README 快照保留根目錄連結語境，不改寫歷史文件 |
| 最終全套 Pytest／Ruff | 400 passed，54 warnings；Ruff 92 檔通過 | 含下載重試／備援測試；見下方命令 |
| 四份平台 lock／重建命令 | 四份鎖定檔、交付檢查及 uv.lock 一致性通過 | 應用與 pose 均實際重建；Linux 容器結果另列 |
| 全新 macOS app／pose 環境 | 兩套新環境安裝與三模型實際載入通過 | Python 3.11.14；758 個 MMPose 來源檔核對通過 |
| Linux amd64 Docker 建置 | 最終 Linux amd64 建置、兩套匯入與 CPU ops、400 項容器測試通過 | 91fbeeb 的完整 CI 成功；先前失敗紀錄保留於下方 |
| Docker runtime 檢查／模型載入 | 三模型實際載入、來源／權重核對、HTTP smoke 全部通過 | 758 個 MMPose 來源檔；使用相同原圖與原權重 |
| UI／API 真實圖片與下載 | API／UI 真實圖片、下載及失敗狀態通過 | 8 筆逐眼、6 筆候選距離；結果仍為 unverified |
| Linux arm64／GPU／獨立泛化 | 未執行 | 不屬本輪完成聲明 |

## Commands to Record

以下為驗證入口，最終結果須依實際執行環境與日誌補齊：

```sh
.runtime/app/bin/python -m pytest -q
.runtime/app/bin/ruff check src tests app scripts
.runtime/app/bin/ruff format --check src tests app scripts
python3.11 scripts/bootstrap.py --all --root .runtime-preview --dry-run
python3.11 scripts/check_runtime.py --config configs/inference.runtime.toml
python3.11 scripts/check_runtime.py --config configs/inference.runtime.toml --load-models
docker compose config
docker compose build
docker compose up -d
docker compose exec ui python scripts/check_runtime.py --config configs/inference.docker.toml --load-models
```

環境重建保留每套 `environment_*.json`，包含採用 lock SHA-256、命令、平台與實際套件清單。
失敗／部分建立的環境與建置日誌應保留；bootstrap 拒絕覆寫已有環境。
受保護清單位於本次工作機的暫存位置，並非 Git 交付資料；最終報告應另外記錄核對檔數與是否有變動。

## Local Runtime Evidence

`outputs/inference/runtime_validation_20261108/` 保存本機驗收，沒有覆寫歷史結果：

| 紀錄 | 結果 |
| --- | --- |
| `runtime_check.json` | app／pose 版本、CPU ops、固定來源／權重及三模型實際載入均 ready |
| `protected_files_check.json` | 8,023 檔未變，排除原執行快取及 vendored 套件目錄 |
| `api_smoke.json` | 新圖、兩類距離、JSON／CSV／PNG、無效輸入與合成無動物圖通過 |
| `baseline_comparison.json` | 同一張開發圖的 8 個遮罩逐像素相同；眼點、ROI、重心、候選選擇與 6 筆距離完全相同 |

真實 API run：`20261108T095341Z_c519a172ecbc`；合成空白圖 run：`20261108T095358Z_9f11dc8b84e4`。
比對來源是原開發 run `20261108T075732Z_92c42f74bb73`；輸入 SHA-256 相同。
這是環境／程式一致性驗證，不是新增 GT、精度成績或獨立泛化驗收。

本機新環境最終全套為 400 passed、54 warnings、14.14 秒；Ruff lint／format 92 檔通過。
原生 Linux CI 在初次交付 commit `a5823de` 完成 386 passed、54 warnings；之後移除未採用的
本機 Git sourcecache 功能及其兩個測試，保留官方 HTTPS 來源及套件快取。

較早兩輪本機全套曾出現 Streamlit AppTest 30 秒逾時。單獨測試、相同前段順序、全套模組
收集後的單一測試及後續完整重跑皆通過；未找到確定根因，沒有提高 timeout 或放寬條件。
CI 保留 `faulthandler_timeout=15`，原失敗日誌保存在本機驗收目錄。

## UI Evidence

新環境介面 `http://127.0.0.1:18517`：既有圖片上傳 → A_r050 對照 → 推論完成 → CSV／JSON／PNG 下載。
UI run `20261108T095547Z_183662a70f51`：8 筆眼部輸出、6 筆候選距離，全部維持 unverified。
三類下載與保存檔逐 bytes 相同；1440×1000、390×844 均無水平溢位或 Streamlit 例外，
console 0 errors／0 warnings；手機切換動物與眼側不重新推論。手機僅 viewport，未驗證實機觸控。

Browser plugin not available；使用既有離線 Playwright CLI，未安裝瀏覽器相依。
截圖、DOM／console 及下載核對在本機 `/tmp/aem_delivery_audit/`。

## Container Findings Retained

第一輪 Linux 映像可建置，主環境／pose 匯入及 CPU NMS 通過。容器全套測試發現
`AEM_WORKSPACE=New Image Inference` 被測試繼承，選圖測試因此未進入其預期工作區；
已由 autouse fixture 隔離 AEM 環境變數，再以容器預設環境重播本機全套，384 passed／54 warnings。
沒有修改應用的預設工作區或放寬斷言。

原 COCO 自訂 HTTPS 網域憑證不符合主機名；工程 fixture 改由同一官方 S3 儲存位置的有效 HTTPS
路徑取得。實取 148,266 bytes，SHA-256 `aeb172b088be9b334a97fcd4df1fe89b5730bc5836765a23c82f2b8b4ff3d03b`，
與原開發圖片完全相同。沒有停用憑證驗證或換圖。

69d3706 的 Linux 容器內完整測試為 384 passed／54 warnings（33.01 秒）。其後下載權重時
發生 HTTP 504，實際模型載入與 HTTP smoke 被跳過。已加入有限次重試、逐檔診斷與官方
固定 asset API 備援，並將圖片／權重準備移到昂貴建置前。原 canonical 與 SHA 保留；
404 或雜湊不符仍直接失敗，不以其他模型取代。

## Final Linux CPU Acceptance

[GitHub Actions run 42472036879](https://github.com/PolunLin/animal-eye-metrology/actions/runs/42472036879)
對 commit `91fbeeb0ae029af6e664d6dfa76d891070e7ad16` 完整成功：

| 項目 | 實際結果 |
| --- | --- |
| Linux host unit tests | 400 passed／54 warnings；Ruff 通過 |
| Docker image | amd64／Linux，ID `sha256:f768ff3d88c5e2aac2b479f341c15b5811d7484a056600719e8d8afa9f794190` |
| 容器測試 | 400 passed／54 warnings，25.33 秒 |
| CPU runtime | app Torch 2.9.1+cpu／vision 0.24.1+cpu；pose Torch 2.2.2+cpu／vision 0.17.2+cpu |
| 真實模型載入 | YOLO26s-seg、SAM 2.1 Tiny、HRNet-W32／AP-10K，全部 ready |
| 來源／權重 | 三個 SHA、固定 commit、758 個 MMPose 檔案通過 |
| HTTP smoke | health／OpenAPI／無效上傳／兩動物／兩類距離／候選語意／保存／CSV／PNG／無動物全部通過 |

Linux run：`20261108T105012Z_bef5f82eb15f`；合成空白圖：`20261108T105107Z_3cee677c0a84`。
HTTP 結果含 8 筆逐眼輸出（4 實際眼 × 2 ROI 條件）與 6 筆候選距離，全部 unverified。
實際推論容器無網路，沒有臨時下載模型或 GT。

同一原圖／正規化像素的 Linux／macOS 比較：8 個遮罩逐像素相同，重心與 6 筆距離相同；
HRNet 眼點的最大跨平台歐氏位移為 `0.00006103515625 px`。這是單一既有開發圖的工程一致性，
不是與 GT 的誤差，也不保證其他圖片或硬體逐像素一致。
詳見本機新產物 `linux_macos_comparison.json`、`linux_runtime_check.json` 與 `linux_image-inspect.json`。

CI artifact `cpu-runtime-evidence`：ID `13613998475`，2,216,392 bytes，
ZIP SHA-256 `1aab2f094c9386a7d793b08484be34da574622c3a1073002e5e2591dd626b902`，保留 14 天。
本機已下載核對 ZIP hash，原始紀錄在 `/tmp/aem_delivery_audit/linux-final/`；
驗證摘要及跨平台比較在 `outputs/inference/runtime_validation_20261108/`。

## Scientific Limits

測試成功、模型載入、非空遮罩與 HTTP 200 都不能證明量測正確。
新圖距離資格仍是 `unverified`；缺失端點留空而非補零。工程驗證不新增 GT、IoU／Dice 或泛化成績。
跨作業系統模型差異應另行保存，不預設逐像素一致。
