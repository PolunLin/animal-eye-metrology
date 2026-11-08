# HTTP API

依 [README](../README.md#local-setup) 準備環境與權重後啟動：

```sh
AEM_INFERENCE_CONFIG=configs/inference.runtime.toml \
  .runtime/app/bin/python -m uvicorn animal_eye_metrology.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

互動文件：[Swagger UI](http://127.0.0.1:8000/docs)、[ReDoc](http://127.0.0.1:8000/redoc)、
[OpenAPI JSON](http://127.0.0.1:8000/openapi.json)。測試帳號不適用；本機展示沒有登入功能。

## Endpoints

| Method | Path | 說明 |
| --- | --- | --- |
| GET | `/health` | 檢查設定、直譯器、來源與權重雜湊；不執行推論 |
| POST | `/predict` | multipart `file`；固定 A_r100，同步回傳結果 JSON |
| GET | `/runs/{run_id}` | 讀取已保存結果，不重新推論 |
| GET | `/runs/{run_id}/artifacts/{artifact_key}` | 下載結果登記的 CSV／PNG；不接受任意路徑 |

```sh
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body -F file=@data/000000125405.jpg http://127.0.0.1:8000/predict
```

將回應的 `run_id` 代入：

```sh
curl --fail-with-body "http://127.0.0.1:8000/runs/<run_id>"
curl --fail-with-body "http://127.0.0.1:8000/runs/<run_id>/artifacts/measurements_csv" \
  --output measurements.csv
```

## Response and Errors

結果包含 `status`、`stages`、`animals`、`eyes`、`measurements`、`artifacts`、設定與來源。
距離單位為 px；缺失端點用 JSON `null`／CSV 空值，保留原因。
`candidate` 不代表眼睛分對或量測合格；沒有 GT 不計算準確度。

| HTTP | 意義 |
| --- | --- |
| 200 | 請求完成；仍須檢查 JSON `status`，模型失敗也可能保存為 `failed` 結果 |
| 400／413／422 | 圖片內容無效／超過 bytes 上限／缺少必要欄位 |
| 404 | run 或產物不存在，或檔案路徑越界 |
| 409 | 共用輸出目錄已有推論，稍後重試 |
| 500／503 | 產物或執行錯誤／環境與權重未就緒 |

只接受靜態 JPEG／PNG，預設 20 MiB／1,200 萬像素；套用 EXIF、RGB 正規化，不縮放原圖。
每個模型子程序有 timeout；同步 API 沒有工作佇列，客戶端斷線不等於取消執行。
部署與真實 HTTP smoke 指令見 [README](../README.md#validation)。
