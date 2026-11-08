# HTTP API

## Scope

這是本機單張 CPU 展示 API，沿用既有 YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny
與遮罩重心距離計算。沒有登入功能，測試帳號為「不適用」。所有距離仍為像素候選量測；
可見性、解剖左右眼與量測資格維持 `unverified`。新圖片沒有人工參考，不產生 IoU／Dice 或距離誤差。

## Start

從專案根目錄、已完成環境與模型準備後啟動；`AEM_INFERENCE_CONFIG` 選擇推論設定檔。

```sh
AEM_INFERENCE_CONFIG=configs/inference.runtime.toml .runtime/app/bin/python -m uvicorn \
  animal_eye_metrology.api:app --host 127.0.0.1 --port 8000 --workers 1
```

啟動後可開啟 [Swagger UI](http://127.0.0.1:8000/docs)、[ReDoc](http://127.0.0.1:8000/redoc)，
或取得 [OpenAPI JSON](http://127.0.0.1:8000/openapi.json)。文件頁不會載入模型。
這些是本機服務連結，啟動前不代表已有公開服務。

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/health` | 檢查設定、本機直譯器／來源／權重檔與權重雜湊；不啟動模型推論 |
| POST | `/predict` | multipart `file`，可選 `compare_r050=true`；同步完成既有推論並回傳完整結果 JSON |
| GET | `/runs/{run_id}` | 讀取已保存結果，不重新推論；允許在權重缺失時查看既有結果 |
| GET | `/runs/{run_id}/artifacts/{artifact_key}` | 下載結果 `artifacts` 登記的 CSV／疊圖檔；不接受任意檔案路徑 |

```sh
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body -F file=@data/coco/val2017/000000125405.jpg \
  -F compare_r050=true http://127.0.0.1:8000/predict
```

將回應的 `run_id` 代入下列請求；範例字串是位置說明，不是假造的執行紀錄。

```sh
curl --fail-with-body "http://127.0.0.1:8000/runs/<run_id>"
curl --fail-with-body "http://127.0.0.1:8000/runs/<run_id>/artifacts/measurements_csv" \
  --output measurements.csv
```

`/predict` 的回應沿用 CLI／介面的 `result.json`，包括 `status`、`stages`、`animals`、`eyes`、
`measurements`、`artifacts`、`limitations`、設定與來源資訊。`artifacts` 內的值是本次 run
的相對檔案路徑；使用其「鍵」呼叫下載端點，例如 `measurements_csv`、`A_r100_overlay`。
新請求建立獨立目錄，不從圖片檔名推導 COCO 身份，也不讀 GT／人工提示。

## Status Semantics

| HTTP status | Meaning |
| --- | --- |
| 200 | 請求完成或成功讀取結果；須另看 JSON 的 `status`，不代表量測正確 |
| 400 | 圖片內容無效、空檔、格式不符、像素數超限或多幀圖片 |
| 404 | run 不存在／識別碼無效、未登記的產物或產物路徑超出 run 目錄 |
| 409 | 此 API 或共用輸出目錄的其他入口已有推論執行中；沒有排隊，完成後再重試 |
| 413 | 圖片 bytes 超過設定限制（預設 20 MiB） |
| 422 | 缺少 `file`、表單欄位型別不合法等請求格式錯誤 |
| 500 | 產物無法讀取或流程尚未回傳可讀取結果即失敗；檢查服務日誌 |
| 503 | 設定無效、本機必要檔案缺失或權重雜湊不符；health 的 `detail` 提供狀態 |

如果 pipeline 成功保存「模型執行失敗」的結果，HTTP 仍為 200、JSON `status="failed"`，
並保留階段失敗與下游缺失。不可將 HTTP 200 直接解讀為模型成功或準確度通過。
`candidate` 只表示兩個重心可計算；距離為空時保留 `unavailable` 與原因，不補零。

## Execution Limits

使用單一 Uvicorn worker。共用推論入口會鎖定輸出目錄的 `.inference.lock`，
API／CLI／Streamlit 使用同一輸出目錄時跨程序互斥；API 同時另有程序內的立即拒絕鎖。
這是 Linux／macOS 本機檔案系統的協作鎖；不同輸出目錄不共用，不能視為跨主機排程器。
同步請求沒有工作佇列、取消按鈕或恢復執行功能；
每個模型子程序沿用設定中的 timeout，客戶端斷線不等於取消推論。

API 以內容核對 JPEG／PNG，不單靠副檔名或 MIME；沿用 1,200 萬像素限制與不縮放的原圖座標。
multipart 解析後最多從檔案讀取 `max_upload_bytes + 1` bytes。這是本機展示服務，
沒有增加公網部署用的驗證、反向代理請求大小限制或使用者隔離。

## Contract Tests

```sh
.runtime/app/bin/pytest -q tests/test_api.py tests/test_execution_lock.py
```

測試只使用合成圖片、假推論結果與隔離暫存目錄，涵蓋內容／大小檢查、API 文件、缺少環境、
忙碌狀態、模型失敗結果、唯讀下載，以及路徑／符號連結越界。不視為模型準確度驗證。

## Engineering Smoke Check

API 與已核對權重準備完成後，可用既有開發圖檢查完整 HTTP 流程：

```sh
.runtime/app/bin/python scripts/smoke_api.py \
  --image data/coco/val2017/000000125405.jpg --report outputs/api_smoke.json
```

此指令不下載圖片或模型。它執行既有 A_r100／A_r050 方法，檢查兩類距離、保存結果、
CSV／PNG 下載、錯誤上傳及合成空白圖；JSON 報告與退出碼同時保留成功或失敗。
使用的仍是既有開發圖片，這是工程連通驗證，不新增人工參考或獨立準確度成績。
