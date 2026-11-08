# Docker

## Target

首個目標為 **Linux amd64、CPU、Python 3.11**。同一映像包含 `/opt/venvs/app`（介面、API、YOLO、SAM2）
與 `/opt/venvs/pose`（HRNet／MMPose）；Compose 以同一映像啟動 UI 與 API 兩個服務。

## Validation Status

| 範圍 | 狀態 |
| --- | --- |
| 原 macOS arm64 CPU 推論 | 已有歷史實跑，見 [New Image Inference](new_image_inference.md) |
| 這次乾淨本機環境 | 待本次實際驗證紀錄；見 [Delivery Validation](delivery_validation.md) |
| Docker 建置／Linux 真實推論 | 待本次實際驗證；見 [Delivery Validation](delivery_validation.md) |
| Linux arm64、GPU、跨平台逐像素一致 | 未宣稱完成驗證 |

## Build and Run

從儲存庫根目錄執行：

```sh
cp .env.example .env
python3.11 scripts/prepare_models.py --download
docker compose build
docker compose up
```

已有權重時可先以 `python3.11 scripts/prepare_models.py` 核對，不重複下載。
建置會取得固定套件與模型程式來源；權重由明確準備步驟取得。啟動與推論不下載模型或 COCO。

| 服務 | 預設入口 |
| --- | --- |
| UI | [http://127.0.0.1:8517](http://127.0.0.1:8517) |
| API | [http://127.0.0.1:8000](http://127.0.0.1:8000)；[操作文件](api.md) |

`.env.example` 列出可調整的埠號與掛載設定；複製為 `.env` 後再修改。
沒有登入流程，測試帳號不適用，不需要外部 AI API key。
Compose 預設只將服務埠綁定主機 `127.0.0.1`。`.env` 的 `AEM_INFERENCE_CONFIG` 是本機啟動提示；
Compose 明確使用 `/app/configs/inference.docker.toml`，避免容器載入主機 `.runtime` 路徑。

## Storage

| 資料 | 掛載 | 說明 |
| --- | --- | --- |
| `weights/` | 唯讀 | 固定權重與 SHA-256 |
| `outputs/` | 可寫 | 新 run、圖片、遮罩、CSV／JSON、日誌與 manifest |
| `data/` | 研究模式選用、唯讀 | COCO／人工研究原圖；新圖推論不依賴完整 COCO |

主機虛擬環境、權重、COCO 與歷史 outputs 不作為原始碼打包。
停止或重建容器後，主機輸出目錄仍保留新 run。研究模式仍需對應的完整凍結產物與設定。
UI 與 API 共用輸出根目錄時，由 `.inference.lock` 跨程序互斥；忙碌時 API 回傳 409，
UI／CLI 保留忙碌錯誤。這是本機檔案系統協作鎖，沒有排隊、取消或跨主機調度。

映像建置由 `bootstrap.py --all --root /opt/venvs` 使用 Linux 的兩份平台 lock，
保留 `/opt/venvs/environment_app.json` 與 `environment_pose.json` 的實際安裝清單。
四份 lock 與直接相依來源的關係見 [Setup](setup.md#dependency-sources-and-records)。

```sh
docker compose logs --tail=100
docker compose down
```

## Acceptance

| 檢查 | 通過條件 |
| --- | --- |
| 建置／環境 | 乾淨建置、兩套相依檢查、來源及權重核對通過 |
| 正常圖片 | UI／API 完成動物→眼點→眼遮罩→距離，PNG／CSV／JSON 可下載 |
| 無動物／無效圖 | 保留空值及原因，沒有假距離或正常成功假象 |
| 保存 | 停止容器後仍能從主機讀取 run 及 manifest |
| 科學界線 | 保留 candidate／unverified，無 GT 不輸出誤差或準確度 |
| 跨平台比較 | 固定圖片與設定，保存實際差異；不預設跨系統遮罩逐像素一致 |

MMCV 編譯成功仍需模型載入與 forward 驗證。本機測試與 Linux 容器測試是分開的證據；
若無法建置或執行，應列出待驗證項目，不能將靜態檢查寫成 Docker 通過。
