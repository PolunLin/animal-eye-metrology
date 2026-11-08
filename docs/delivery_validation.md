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
| 原始資料／權重／研究產物保護 | 已建立 8,023 檔的 SHA-256 基線；最終核對待完成 | 本次工作機基線 `/tmp/aem_delivery_audit/protected.json`；不將新增文件當作受保護歷史產物 |
| CSV 範例 | 原 bytes／SHA-256、6 筆距離公式與 candidate／unverified 核對通過 | 見 [Example](../examples/README.md)，沒有執行模型或新增準確度評估 |
| 文件 | 新文件本地連結、Markdown fences 檢查通過 | 歷史 README 快照保留根目錄連結語境，不改寫歷史文件 |
| 最終全套 Pytest／Ruff | 待完成 | 另記完整命令、測試／警告數與結果 |
| 四份平台 lock／重建命令 | 待最終一致性核對 | lock 解析或 dry run 通過不等於套件安裝與模型 forward 通過 |
| 全新 macOS app／pose 環境 | 待實際結果 | 不使用原 `.venv` 的成功代替新環境 |
| Linux amd64 Docker 建置 | 待實際結果 | 需成功完成映像，不以檔案／Compose 靜態檢查代替 |
| Docker runtime 檢查／模型載入 | 待實際結果 | 檢查來源、版本、權重與實際載入，不等於分割準確度 |
| UI／API 真實圖片與下載 | 待實際結果 | 需保存 run、階段狀態、CSV／JSON／PNG 與失敗路徑 |
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

## Scientific Limits

測試成功、模型載入、非空遮罩與 HTTP 200 都不能證明量測正確。
新圖距離資格仍是 `unverified`；缺失端點留空而非補零。工程驗證不新增 GT、IoU／Dice 或泛化成績。
跨作業系統模型差異應另行保存，不預設逐像素一致。
