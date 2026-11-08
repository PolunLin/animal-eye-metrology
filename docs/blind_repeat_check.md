# Four-Eye Blind Repeat Check

四眼重標及兩輪比較已完成，數值與限制見 [Four-Eye Repeatability Results](blind_repeat_results.md)。
以下保留本輪原始操作方式與準備時狀態。

## Purpose and Scope

依負責人核准，僅重標 COCO 圖片 125405 的兩隻狗、四眼。
沿用 eye-visible-aperture-v1、原圖二值 PNG 與既有 13 欄 CSV，
不修改正式標註、已凍結基準、應用程式或相依套件。

這是同一標註者的再標一致性試驗。工作區不載入第一輪輪廓、審核值或量測數字；
使用者先前已看過基準對照圖，無法排除記憶效應，不能宣稱完全盲化或獨立驗收。
四眼仍是原先的四個目標，不增加獨立樣本數，也不代表多人一致性或真實量測誤差。

## Prepared Files

工作根目錄為 outputs/phase3/blind_repeat_v1/：

| 相對路徑 | 內容 |
| --- | --- |
| inputs/proposed_instances.csv | 原試標清單中的兩筆身份資料，沒有眼睛輪廓與審核結果 |
| annotations/source_manifest.json | 相同 COCO 來源 digest、兩隻動物身份與既有座標慣例 |
| annotations/eye_mask_annotations.csv | 原 Schema，初始化只有欄名、零筆標註 |
| annotations/.annotations.lock | 沿用現有保存鎖 |
| trial_manifest.json | 核准範圍、來源／基準 digest、選取方式、盲化限制與後續比較計畫 |
| preparation_validation.json | 四眼空白狀態、載入切換及原檔完整性檢查紀錄 |

本輪 masks/ 與 revisions/ 由你保存標註時才建立；不複製任何舊 Mask、頂點、放大視野或人工判斷。
trial_manifest 的初始筆數只記錄準備時的狀態，不是即時完成數；實際進度以工作 CSV 最新版本為準。
標註者與時間取實際審核列，不預填 Reviewer 或捏造再標時間間隔。

## Start

本次已啟動本機專用頁面：http://127.0.0.1:8515 。
請使用此網址，選 **Eye Mask Annotation**；原本的 8514 頁面屬於第一輪，這次不要在那裡保存。

若服務停止，從 repository 根目錄重新執行以下單一命令；請保留兩個 AEM_EYE 設定：

```sh
AEM_CONFIG=configs/phase1.toml \
AEM_EYE_PROPOSAL=outputs/phase3/blind_repeat_v1/inputs/proposed_instances.csv \
AEM_EYE_OUTPUT=outputs/phase3/blind_repeat_v1/annotations \
.venv/bin/streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8515 --server.headless true \
  --browser.gatherUsageStats false
```

每次重啟仍讀取本輪獨立 CSV；本輪已保存的輪廓可以恢復，不會退回第一輪工作目錄。
若 8515 已有本輪服務，直接開啟網址，不用再啟動第二個程序。

## Annotation Steps

1. 選 Eye Mask Annotation，確認側欄「Eye GT 工作檔」以
   outputs/phase3/blind_repeat_v1/annotations 結尾，圖片只有 125405。
   第一次進入應顯示 1 張圖片、4 個眼睛槽位、已保存 0、Mask 已接受 0；之後會隨本輪保存增加。
2. 按下表選擇動物與解剖眼側；可展開「查看完整原圖與動物身份」確認 COCO 框。
   左右以動物自身為準，不依畫面左右，不以記憶中的第一輪位置代替判斷。
3. 選「框選放大」，放大頭／眼部；自行沿可見眼裂內眼部區域逐點描繪，排除眼瞼及毛髮，
   按「完成輪廓」。放大不增加原圖資訊。必要時拖動頂點修正。
4. 依本次觀察填可見性、解剖左右確認、Mask 審核、量測品質與相同的實際 Reviewer。
   只有你確認符合原規範才選 Accepted／Pass，勾選確認後按「儲存審核結果」。
   無法確認請保留 Uncertain／Unknown 與原因，不為湊滿四筆而猜測。
5. 每眼保存後再切換下一眼；關閉或重新整理前至少保存草稿。
   Pending 草稿不算審核完成；「已保存 4」也不自動代表四眼都符合量測條件。

| 順序 | Image ID | Animal Instance | 解剖眼側 |
| --- | --- | --- | --- |
| 1 | 125405 | 13692 | left |
| 2 | 125405 | 13692 | right |
| 3 | 125405 | 17207 | left |
| 4 | 125405 | 17207 | right |

重標期間請不要開啟第一輪 Mask、輪廓 JSON、重心 CSV 或量測對照圖，也不要照著舊輪廓描。
本工具沒有鎖住其他目錄，這項隔離需要操作配合；原圖與 COCO 動物框可正常使用。
輪廓可在本輪內修正並追加版本，保存歷史保留；後續比較前另行凍結本輪選定版本。

## Completion and Comparison

四個槽位完成本輪人工審核後，告知「四眼重標完成」。若有不確定或無法標註的眼睛，照實記錄並告知。
之後才核對保存內容與來源、凍結本輪版本，與已驗收的第一輪基準依 image_id／annotation_id／eye_side 配對：

| 層次 | 後續比較 |
| --- | --- |
| 輪廓 | 每眼 Mask IoU 與 Dice；原圖像素對齊 |
| 位置 | 每眼兩次 Mask 重心的歐氏位移，px |
| 量測 | 兩筆同隻雙眼距離、一筆兩動物右眼距離的有號差與絕對差，px |

不完整／不可測的眼睛明確列出，不用 0 取代缺值，不只報告成功配對。
目前沒有預設數值通過門檻，也沒有執行比較、AI、訓練或擴增標註範圍。

## Preparation Verification

沿用 load_targets、initialize_store、load_annotations 核對來源與 Schema。
Streamlit AppTest 載入真實原圖並切換兩隻動物及左右眼，四個槽位均為版本 0、
空輪廓、空人工紀錄，審核選項未預填、Reviewer 空白；沒有按保存或寫入合成標註。
本次只準備資料與說明，沒有修改程式，未新增測試案例或重跑全套 pytest。
此檢查驗證頁面狀態，不是本次瀏覽器描邊測試。
