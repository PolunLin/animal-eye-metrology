# Eye Region Annotation Protocol v2

## Approval and Scope

專案負責人於本次對話核准：包含可辨識眼瞼邊緣，排除眼周毛髮與色塊。
規範識別碼為 **eye-visible-eyelid-v2**；這是新標的，不是更改 v1 標註是否正確的判決。
v1 `eye-visible-aperture-v1`、其人工歷史、A_r100 原基線、A_r050 對照及既有晉級結論全部保留。
本輪不訓練、不重跑模型、不修改 ROI、提示或候選選擇。

| 項目 | 本輪範圍 |
| --- | --- |
| 圖片 | 既有 COCO 2017 validation：125405（狗）、20247（熊）、333069（長頸鹿） |
| 動物 | 13692、17207；587181、588042；599488、599888，共 6 隻 |
| 眼睛 | 每隻動物自身的 left／right，共 12 個待人工重標槽位 |
| 標註來源 | 原圖與 COCO 動物身份框；從空白輪廓開始，不載入 v1／模型輪廓或眼點 |
| 用途 | 定義開發、標註一致性與凍結模型輸出比較；不是新增獨立驗收集 |
| 隨機性 | 沿用已核准的目的性選圖與清單順序，沒有重新抽樣或隨機 seed |

## Target and Boundary Rules

**標註可見眼裂內的眼球表面，加上直接相鄰且能辨認的眼瞼邊緣。**
眼瞼邊緣指緊接眼裂的可辨識結構；不等同整片眼周皮膚、毛髮或顏色區塊。
外界沿可辨認眼瞼邊緣的外側描繪，不使用固定擴張像素或固定比例。

| 情況 | 納入／排除與操作 |
| --- | --- |
| 可見眼球表面 | 納入眼裂內可辨識的虹膜、瞳孔與鞏膜；不把它們拆成不同目標 |
| 上下眼瞼與眼角 | 納入直接相鄰且可辨識的眼瞼邊緣，止於能辨認的外緣；不外推眼眶範圍 |
| 黑色眼周／毛色斑塊 | 不能因為深色、連成一片或模型選到而納入；必須能辨認是眼瞼邊緣才算 |
| 毛髮、睫毛與淚痕 | 排除；跨過目標的毛髮視為遮擋，不把毛髮輪廓畫成眼瞼 |
| 眼內反光 | 在可辨識目標內的反光仍納入；不依亮度挖洞 |
| 模糊、極小或眼瞼外界不清 | 記 Uncertain 與原因；可存草稿，不能任意膨脹，也不能只畫眼裂後當作已確認 v2 |
| 外物遮擋／半閉眼 | 只畫能觀察的部分；記 Partial，量測品質 Fail 或 Unknown，不補出不可見範圍 |
| 閉眼／只剩亮點 | 不推測隱藏眼裂；依可辨識情況記 NotVisible／Uncertain，不當完整 v2 量測眼 |
| 孔洞、多區塊 | 依可見情況保存，註明原因；不自動補洞、連接、平滑或套橢圓 |

原圖像素是資訊上限；放大與像素格只供描繪，不能增加細節。
不要求 v2 面積一定大於 v1，也不先顯示 v1 供對齊；獨立重標可能產生其他邊界差異。
無法依本規範一致判讀時保留未知，先討論規範，不在同一 protocol 下自行更改規則。

## Identity, Coordinates, and Review

- 左右按動物自身的解剖方向判讀，先看完整頭部、鼻部與姿態，不以畫面左右排序替代。
- `image_id`／`annotation_id` 沿用 COCO 身份；左右槽位是待確認項目，不預填側別已確認。
- 原圖左上外框為 `(0,0)`，像素中心為 `(column+0.5,row+0.5)`。
- 每次保存建立新修訂；Mask 為完整原圖尺寸、單通道 0／255 PNG。
- 不預填 reviewer、可見性、Accepted 或 Pass；reviewer 由實際標註者輸入。

| 狀態 | 意義 |
| --- | --- |
| Pending | 尚未審核，可保存未完成輪廓 |
| Accepted | 人工確認輪廓符合 v2、側別已確認；不是自動品質判定 |
| Visible＋Accepted＋side_confirmed=Yes＋Pass | 沿用既有完整眼部量測資格，才可進入本輪完整比較的凍結參考 |
| Partial／Uncertain／NotVisible | 保留原因與可用草稿，不強迫填 Pass；不可產生假的完整眼睛 Mask |

**12 個槽位皆審閱，不等於 12 個眼睛都可量測。**
目前凍結比較沿用完整 12 眼／9 距離的範圍；若有無法量測者，先輸出待處理清單，
不自行刪除樣本、改分母、替換圖片或要求標註者改成 Pass。

## Independent Relabeling Procedure

1. 開啟專用 v2 標註頁，核對頁首規範與獨立輸出目錄。
2. 先看原圖與動物身份框；選定動物及其自身眼側，再自行框選放大。
3. 從空白輪廓描繪 v2。標註期間不開 Results Explorer、舊標註或模型結果視窗；
   不複製 v1、不採用 SAM 輪廓、不以自動膨脹產生新 GT。
4. 先存草稿，逐項填寫可見性、側別、審核、量測品質與實際 reviewer。
   輪廓可辨且符合規範才勾人工確認並保存；不確定則留下具體原因。
5. 完成同一隻動物另一眼，依清單完成全部 12 槽位；換圖後重新核對身份與側別。
6. 檢查完成狀態。若有 Pending／Uncertain／Partial，先保留，不執行完整比較。
7. 全部符合既有凍結資格後，建立獨立的 `manual_measurement_baseline_v2`，
   保存原圖、Mask、版本、規範副本、來源與 SHA-256，再進行唯讀比較。

使用者先前已看過模型結果，因此此流程只能稱「不顯示舊輪廓的獨立重標」，
不能稱完全盲測、未見資料或獨立泛化驗收。分開目錄可避免直接複製，不能消除記憶影響。
如後續要評估 v2 再標一致性，應另開第二次空白重標目錄，不把本次 v1→v2 差異當作同規範重標誤差。
第二次重標與新增驗收圖片不在本次自動執行範圍內。

## Frozen-Output Comparison

| 比較 | 要報告的內容與解讀 |
| --- | --- |
| 人工 v1 → v2 | 面積、Mask IoU／Dice、重心位移、6 筆雙眼與 3 筆跨右眼距離變化；是定義差異，不是模型誤差或再標一致性 |
| 同一模型輸出 vs v1／v2 | 逐眼 IoU、Dice、重心偏差；逐筆距離與誤差；分開列出兩版參考，不改寫原成績 |
| A_r100／A_r050 | 原基線與對照優先呈現；預測 Mask、眼點、ROI、候選與距離均沿用已凍結值 |
| 其他既有條件 | 18 組完整保留角色；B／C／D 沿用原實驗人工提示，是診斷，不能稱 v2 人工提示上限 |
| 失敗與資格 | 不隱藏空 Mask／失敗列，距離缺值不填零；候選可見性仍 unverified，不因更換 GT 自動合格 |

v2 量測名稱為「**v2 眼部區域重心距離**」，不是瞳孔中心距離、眼球解剖中心距離或真實長度。
沿用 Mask 像素中心平均與歐氏距離，單位 px；不新增校正、Landmark 或評分門檻。
新定義下 IoU 較高只表示對新標的更符合；不代表模型能力進步，亦不撤銷 γ＝0.5 原未晉級結論。
完整比較由獨立設定與輸出目錄執行；未具備人工 v2 參考前，不產生任何實驗成績。

## Reproduction

```sh
.venv/bin/python -m animal_eye_metrology.annotation_session --config configs/eye_annotation_v2.toml --prepare
.venv/bin/python -m animal_eye_metrology.annotation_session --config configs/eye_annotation_v2.toml --status
AEM_ANNOTATION_SESSION=configs/eye_annotation_v2.toml .venv/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8518 --server.headless true --browser.gatherUsageStats false
```

人工重標完成並核對資格後才執行：

```sh
.venv/bin/python -m animal_eye_metrology.measurement_baseline --config configs/measurement_baseline_v2.toml
.venv/bin/python -m animal_eye_metrology.definition_comparison --config configs/definition_comparison_v2.toml --check
.venv/bin/python -m animal_eye_metrology.definition_comparison --config configs/definition_comparison_v2.toml
```

準備命令拒絕覆寫既有工作目錄；不要刪除工作目錄來重跑。
狀態檢查不創造審核資料；凍結與比較命令拒絕覆寫原產物。
