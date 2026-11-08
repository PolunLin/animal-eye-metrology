# Phase 3B Manual Eye Mask Annotation

## Scope

依負責人「好的 3B」指示，沿用 [Phase 3A 建議規範](phase3a_eye_annotation_protocol.md)
及試標清單，在現有 Streamlit 單頁加入 Eye Mask Annotation。
採可見眼裂內眼部區域、解剖左右、原圖二值 PNG＋獨立 CSV；本輪不收 Landmark。
Phase 3A 的 Draft／Proposed 檔案保留當時快照，本次實作範圍記錄於此。

| 項目 | 範圍 |
| --- | --- |
| 圖片 | 125405 dog、20247 bear、333069 giraffe |
| 動物 | 13692、17207；587181、588042；599488、599888，共 6 隻 |
| 眼睛槽位 | 每隻 left／right，合計 12 個，不預設都可標 |
| 已實作 | 動物／眼側選擇、框選放大、輪廓修正、審核、草稿／版本保存與重開 |
| 本次不含 | 眼睛模型、AI 提示、Landmark、重心／距離計算、訓練與 5,000 張推論 |

工具不從動物 Mask 自動猜眼睛，也不將舊 Phase 1 Yes 複製成新 Eye GT 結果。

## Start and Annotate

從 repository 根目錄執行；本機已有 Streamlit 1.67，可直接啟動：

```sh
.venv/bin/streamlit run app/streamlit_app.py
```

開啟啟動訊息中的本機網址，選 **Eye Mask Annotation**。既有頁面可重新整理後切換工作區。
使用 Streamlit 原生 v2 元件與 Canvas，沒有額外前端框架或 npm build。
[Streamlit v2 官方 API](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.component)
支援直接提供 JavaScript 與雙向狀態，不需另建服務。
explorer 最低 Streamlit 版本調整為 1.67、uv.lock 同步；其他套件版本不變。
新環境沿用 README 的 uv 安裝方式與需要的 optional extras。

1. 選 Image ID、Animal Instance 及動物自身的 left／right。
   展開「查看完整原圖與動物身份」核對，不能按畫面左右自動賦值。
2. 「操作」選「框選放大」，拖出頭部／眼部範圍；放開後回到「描繪輪廓」。
3. 沿可見眼部邊界逐點點擊，至少三點後按「完成輪廓」。可拖曳頂點修正。
   「復原一步」移除最後一個未完成頂點；若沒有未完成頂點，移除最後一條完成輪廓。
4. 若確有外物／毛髮遮擋，可選「扣除遮擋」描繪扣除區域；紅色表示扣除。
   多條輪廓依建立順序加入或扣除，不自動補洞、平滑或取最大區塊。
5. 先「儲存草稿」後，可展開「已保存的原圖尺寸 Mask」檢視或下載 PNG。
6. 填寫可見性、側別確認、Mask 審核、量測品質及實際 Reviewer。
   接受 Mask 前勾選「我已確認眼側及輪廓符合規範」，再按「儲存審核結果」。
7. 切換前建議保存。輪廓在同一連線切換左右眼／動物時保留；
   關閉或重新整理前必須保存草稿，未保存內容不會跨新連線還原。

「回到動物」只恢復 COCO 動物 bbox 附近的視野，不清除輪廓。放大和像素格不會增加原圖資訊。

## Review Fields

| 選項／操作 | 定義 |
| --- | --- |
| Visible | 能辨認目標可見眼部區域，仍需審核邊界 |
| Partial | 遮擋／明顯半閉眼；可保存分割參考，不能指定量測 Pass |
| NotVisible | 沒有可辨認眼部；清除輪廓後保存原因，不製造全黑 GT PNG |
| Uncertain | 目前無法確定，保留未知與原因 |
| side_confirmed=Yes | 審核者已核對動物自身的解剖左右；初始化為 Unknown |
| Accepted | 已完成非空輪廓，側別確認、可見性與 Reviewer 均已填寫 |
| Rejected／Fail | 原因必填，不修改 Phase 1 Eligibility |
| measurement_quality=Pass | 使用者主動選擇；需 Accepted＋Visible＋側別 Yes＋非空 Mask |
| 儲存草稿 | Pending、量測品質 Unknown；可保存部分頂點，不代表已審核或可測量 |

夜視只見亮點而沒有眼裂邊界時用 Uncertain，不把亮點直接當眼睛。
Accepted／Pass 是人工判斷，程式只檢查必要條件，不自動證明標註正確。

## Persistence and Coordinates

預設輸出為 outputs/phase3/ground_truth/：

```text
source_manifest.json
eye_mask_annotations.csv
masks/{image_id}/{annotation_id}_{eye_side}_v{version}.png
revisions/{image_id}_{annotation_id}_{eye_side}_v{version}.json
.annotations.lock
```

CSV 使用 3A 提議的 13 欄：

```text
image_id,annotation_id,eye_side,mask_path,visibility,side_confirmed,
mask_review_status,measurement_quality,exclusion_reason,reviewer,
protocol_version,annotation_version,annotated_at_utc
```

每次保存追加新版本，唯一鍵為 image_id＋annotation_id＋eye_side＋annotation_version。
重新開啟以該眼最高已提交版本為目前版本，完整歷史仍保留；後續評估必須明確凍結版本。
初始化只有來源 manifest 與 CSV 欄名，不預填 12 筆人工結果。

PNG 為原圖尺寸、單通道 L、值為 0／255。原點是原圖左上外框，像素中心在
(column+0.5,row+0.5)。多邊形採 even-odd fill，恰在邊界上的像素中心納入。
輪廓點在前端已轉為原圖座標；版本 JSON 另存未完成頂點、放大區域／顯示轉換、
原圖 digest、Mask digest 及工具資訊。無非空完成區域時 mask_path 留白，不輸出假的 Mask。

AEM_EYE_PROPOSAL 可指定試標清單，預設 outputs/phase3/phase3a/proposed_instances.csv。
AEM_EYE_OUTPUT 可指定獨立 Phase 3 子目錄，不能覆寫提案目錄或 Phase 1／2。
既有 AEM_CONFIG 與 Phase 1 操作方式仍可使用。

## Integrity and Concurrent Windows

保存前核對 COCO 身份與原圖 digest。載入核對 CSV schema、重複鍵、版本 JSON、PNG 尺寸／編碼及 digest；
損壞時停止並顯示錯誤，不修補人工結果。檔案鎖保護合併更新，CSV 暫存後原子替換，
PNG／JSON 使用新檔名並先保存，CSV 最後提交。另一視窗已更新同眼時拒絕舊版本保存，須人工重新載入整合。
崩潰留下未被 CSV 引用的檔名會被跳過，不覆寫、不自動當作 GT。

## Verification and Limits

| 檔案 | 變更 |
| --- | --- |
| app/streamlit_app.py、app/eye_canvas.js | 新增同頁工作區、原生互動描邊及審核操作 |
| src/animal_eye_metrology/eye_annotations.py | 原圖像素化、來源核對、版本保存與讀取 |
| tests/test_eye_annotations.py | 五個合成資料測試，涵蓋幾何、完整性及保存流程 |
| pyproject.toml、uv.lock | Streamlit 最低版本同步到已安裝的 1.67 |
| README.md、本文件 | 執行、操作、資料格式與限制 |
| outputs/phase3/ground_truth/ | 初始化來源 manifest 與空白 CSV，沒有人工標註列 |

```sh
.venv/bin/pytest -q
.venv/bin/ruff check src tests app
.venv/bin/ruff format --check app/streamlit_app.py \
  src/animal_eye_metrology/eye_annotations.py tests/test_eye_annotations.py
```

測試以獨立合成資料檢查像素中心／扣除區域、空／未知眼睛、左右隔離、重開與歷史版本、
過期視窗衝突、寫入失敗回復、孤立版本檔保護、損壞 Mask 及 Streamlit 草稿保存。
瀏覽器以 Playwright 驗證實際操作；此環境沒有 Browser plugin。
首次發現 v2 重跑造成重複畫布，已清理舊節點與 ResizeObserver，並重新驗證操作。
瀏覽器測試資料與證據位於 /tmp，不是研究 GT；正式資料不包含合成輪廓。

| 實測核對 | 結果 |
| --- | --- |
| pytest | 82 passed，53 個既有上游類型的 pycocotools／NumPy 棄用警告 |
| Ruff | check 通過，三個變更 Python 檔 format 通過 |
| 合成瀏覽器操作 | 四點描邊→完成輪廓→人工選項→保存 v1→左右切換→重開→框選放大→保存草稿 v2 |
| 原圖座標 | 合成 Mask 為 100×80、L、0／255，9 個前景像素；改變放大區域後兩版本 PNG digest 相同 |
| 版本與隔離 | v1 保留，右眼維持空白；重開恢復左眼輪廓，整頁只有一個畫布 |
| 正式資料 | 3 張／6 隻／12 個槽位可載入；驗證時正式 CSV 為 0 筆、Mask 為 0 張 |
| 版面 | Chrome 桌面 1440×1050、手機寬度 390×844 可顯示，手機寬度無水平溢出；未驗證觸控描邊 |

完整核對記錄見 outputs/phase3/phase3b_validation.json。啟動的正式本機驗收頁為
http://127.0.0.1:8514；若服務已停止，重新執行上列啟動命令並使用當次顯示的網址。

這是單機小量多邊形工具，沒有完整多人標註平台、眼睛模型或誤差容許值。
支援滑鼠／指標操作；觸控與其他瀏覽器需另驗證，沒有完整鍵盤描邊。
建議先在 125405 試標四眼，完成重標一致性檢查後，再擴至另外兩張與後續模型評估。
