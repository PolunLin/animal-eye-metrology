# Recorded Baseline CSV

`measurements.csv` 保留既有 run 的 **A_r100 全部三列**：兩筆同隻雙眼與一筆跨動物右眼距離。
只移除 A_r050 對照列；保留原 CSV 標頭與每列 bytes，沒有重算、四捨五入或挑成功列。

| 項目 | 來源 |
| --- | --- |
| Image | COCO 2017 validation，`125405`，640×428，兩隻狗 |
| Run | `20261108T075732Z_92c42f74bb73` |
| Pipeline | YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny；CPU、seed 42、4 threads |
| Original CSV SHA-256 | `4a8ea8409a1d63bc5c3efd35b446c5ce17bcbf5f860ef686f5d1132a8b41bf11` |
| CSV SHA-256 | `cc07eefa1d9c7d34ffbea272a9dc449ff0e5b02c676d08e8ee455d833abae8a0` |
| Source result.json SHA-256 | `269dade32d6dc3ace244507155b349294635a85caee3b79f5aaf59161ac833ae` |
| Original JPEG SHA-256 | `aeb172b088be9b334a97fcd4df1fe89b5730bc5836765a23c82f2b8b4ff3d03b` |

原圖取得方式與雜湊見 [input_manifest.csv](input_manifest.csv) 及 [README](../README.md#coco-example)。
完整歷史 run 不附於 Git；這份 CSV 是開發圖輸出，可驗算幾何，不是獨立準確度成績。

## Fields

| 欄位 | 意義 |
| --- | --- |
| `condition_id` | 固定 A_r100 |
| `measurement_type` | binocular＝同隻雙眼；right_to_right＝不同動物右眼 |
| `prediction_id_a/b` | run 內動物 ID，不是 COCO annotation ID |
| `eye_side_a/b` | 解剖左右通道，不依畫面 x 排序 |
| `cx_a_px/cy_a_px/cx_b_px/cy_b_px` | 原圖眼遮罩重心 |
| `distance_px`、`units` | 重心歐氏距離，px |
| `status`、`qualification` | 本例為 candidate／unverified；不是合格判定 |
| `reason` | 缺失原因；本例可計算，因此留空 |

UTF-8 BOM，Python 請以 `utf-8-sig` 讀取。既有 CSV 早於域外標記欄位，保持原 schema。

## Read and Verify

```sh
.runtime/app/bin/python scripts/check_delivery.py
```

此檢查核對 CSV 雜湊、有限端點、兩種配對、候選狀態與歐氏距離公式，不評估分割正確性。

## Recorded Evaluation

[evaluation_summary.csv](evaluation_summary.csv) 保留既有 A_r100 基線對兩版人工參考的單列摘要，原標頭與列 bytes 不變。
來源為 `outputs/phase3/definition_comparison_v2/condition_summary.csv`，不是本次重跑的成績；三張開發圖共 12 眼、6 筆雙眼與 3 筆跨右眼距離。

- 原始摘要 SHA-256：`478e5f2520bc588707b88a1c3d7f6da5f7fc277f807f5f9431f7aeb642cb90b2`
- 提交摘要 SHA-256：`13223852535f0be8b5fa2117360af97c6050e21c38fbb95e15e29ddd16d54044`
- 分母、失敗／空遮罩數、兩版參考分開保存；數值解讀見 [Methodology](../docs/methodology.md#recorded-observations)。完整人工遮罩與原圖需另備，不能僅從摘要重建準確度評估。
