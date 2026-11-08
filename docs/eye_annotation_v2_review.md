# v2 Annotation Qualification Review

## Follow-up Closure

2026-11-08 同日後續：負責人完成狗13692兩眼重新描繪與審核，最新左右眼皆revision 4，
面積123／52 px；人工歷史增至31筆。最新12眼資料完整性、資格欄位及獨立輪廓重建全部通過，
原先「僅圈極薄眼裂」的具體疑慮已解除。已凍結v2並完成18組固定輸出比較。

最終核對另存 `outputs/phase3/eye_annotation_v2_review_final_20261108/`，
來源69檔SHA-256與前次稽核分開保存；不以舊報告背書新修訂。
最新結果見 [Frozen v1 / v2 Comparison](eye_definition_v2_comparison_results.md)。
低解析度限制與人審責任仍保留，不宣稱自動解剖認證。

**以下保留首次核對的歷史紀錄，待處理狀態已由上述後續結果解除。**

## Decision

2026-11-08 核對結果：**12／12 眼通過資料完整性及已登錄資格檢查；
狗 13692 的兩眼仍需人工複核 v2 邊界／可見性，尚未凍結 v2 或執行正式比較。**
負責人已回覆：「我會重新檢查／修訂這兩眼，再通知你」。
這是既定「核對通過後才凍結」的待處理項目，沒有修改標註規範、刪除樣本或變更模型。

本紀錄對應複核當時的 28 筆歷史修訂。後續人工保存可能增加修訂；
不得把本表當成持續更新的最新狀態，也不能直接使用較舊 Accepted 修訂繞過新草稿。

## Recorded Qualification

資料範圍維持 COCO 2017 validation 的 20247、125405、333069，6 隻動物、12 眼。
所有下表修訂皆由 `Polun` 登錄為 `Visible / side_confirmed=Yes / Accepted / Pass`。
revision 編號是同一 v2 protocol 內的保存版本，不是 v1／v2 定義編號。

| 圖片 | 動物 | 眼側（動物自身） | 最新修訂 | Mask 面積（px） | 本次複核 |
| --- | --- | --- | ---: | ---: | --- |
| 20247 | 熊 587181 | left | 3 | 67 | 完整性通過；未見同類明顯邊界疑慮 |
| 20247 | 熊 587181 | right | 2 | 64 | 同上 |
| 20247 | 熊 588042 | left | 2 | 122 | 同上 |
| 20247 | 熊 588042 | right | 2 | 37 | 同上 |
| 125405 | 狗 13692 | left | 3 | 7 | **待人工複核 v2 邊界／可見性** |
| 125405 | 狗 13692 | right | 2 | 6 | **待人工複核 v2 邊界／可見性** |
| 125405 | 狗 17207 | left | 2 | 135 | 完整性通過；未見同類明顯邊界疑慮 |
| 125405 | 狗 17207 | right | 2 | 98 | 同上 |
| 333069 | 長頸鹿 599488 | left | 3 | 30 | 同上 |
| 333069 | 長頸鹿 599488 | right | 3 | 50 | 同上 |
| 333069 | 長頸鹿 599888 | left | 2 | 18 | 同上 |
| 333069 | 長頸鹿 599888 | right | 2 | 28 | 同上 |

程式檢查包含：身份與 scope、準備清單／規範／原圖雜湊、CSV 與 revision 一致性、
原圖尺寸的單通道 0／255 非空 Mask、Mask 雜湊、有效座標、封閉輪廓及無未完成頂點。
另以獨立 polygon rasterization 核對原圖像素中心，12 張 Mask 逐像素一致。
結構上可組成 6 筆雙眼及 3 筆跨動物右眼距離；**此處沒有宣稱已取得通過語義複核的凍結量測參考**。
解剖左右、可見性與眼瞼邊界仍需人工判讀，程式旗標檢查不等同自動解剖認證。

## Pending Boundary Review

狗 13692 的原圖呈現窄眼裂外觀，現有多邊形約 1–2 原圖像素高；
放大檢視尚無法確認輪廓已納入可辨識眼瞼外緣，或是否應記為 Partial／Uncertain。
這是待釐清的觀察，並非已判定標註錯誤。面積小、比 v1 小，均不是拒絕理由；
本次也未用與模型輪廓的相似程度決定 Ground Truth。

依[已核准規範](eye_annotation_protocol_v2.md)：
「模糊、極小或眼瞼外界不清」須記 Uncertain；不能只畫眼裂後當作已確認 v2。
可辨識時依原圖修訂，不能為通過資格而固定擴張或補出不可見範圍。
如有不合格眼，保留其狀態，不刪除分母、不替換圖片，也不自動把它改成 Pass。

複核產物位於 `outputs/phase3/eye_annotation_v2_review_20261108/`，
包含 12 眼對照、狗 13692 細線／原圖像素複核圖、獨立核對程式、JSON 與來源雜湊。
這個目錄是稽核證據，**不是凍結 GT 或模型成績目錄**。

## Validation and Continuation

| 項目 | 結果 |
| --- | --- |
| `annotation_session --status` | 複核時 saved=12、recorded eligible=12；此工具不判讀語義邊界 |
| 完整 Pytest | 206 passed；53 個既有 pycocotools／NumPy 棄用警告 |
| Ruff lint／format | 通過；81 個 Python 檔案格式符合 |
| 舊來源保護 | 2,081 個既有產物／設定／Phase 1 CSV 雜湊未變 |
| v2 凍結／比較 | 尚未執行，等待兩眼人工複核完成 |

本次已執行：

```sh
.venv/bin/python -m animal_eye_metrology.annotation_session --config configs/eye_annotation_v2.toml --status
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

收到人工複核完成通知後，重新檢查全部最新版本及來源，通過才依已核准設定凍結、驗證及比較：

```sh
.venv/bin/python -m animal_eye_metrology.measurement_baseline --config configs/measurement_baseline_v2.toml
.venv/bin/python -m animal_eye_metrology.measurement_baseline --verify outputs/phase3/manual_measurement_baseline_v2
.venv/bin/python -m animal_eye_metrology.definition_comparison --config configs/definition_comparison_v2.toml --check
.venv/bin/python -m animal_eye_metrology.definition_comparison --config configs/definition_comparison_v2.toml
```

後續仍沿用 18 條件的固定輸出，核對 216 筆逐眼、162 筆距離及 18 組彙總；
不重跑模型、不重選候選、不更動 v1、A_r100／A_r050 或原 γ＝0.5 未晉級結論。
定義差異不能當成模型進步，三張開發圖不能當成獨立泛化驗收。
