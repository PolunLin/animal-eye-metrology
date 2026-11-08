# Phase 3A Eye Annotation Protocol Draft

## Scope and Approval Status

本階段準備眼睛標註規範、資料格式與 3 張試標圖片提案，供專案負責人核准。
目前沒有人工 Eye Mask、Eye Landmark、眼睛模型或距離量測結果。
以下所有新定義皆為提案；核准前不套用到既有人工資料，也不啟動 Ground Truth UI。

沿用既有決策：Eye Landmark GT 與 Eye Segmentation GT 分開，主要量測方法暫定為
**Eye Segmentation Mask → Mask Centroid → Euclidean Distance**。
Mask 重心是指定可見區域的幾何重心，不等於瞳孔中心或解剖學眼球中心。

## Verified Starting Point

| 已有資料 | 已確認事實 | 對本階段的意義 |
| --- | --- | --- |
| Phase 1 | 15 張 Selected，其中 13 張含至少兩隻既有完整雙眼候選，共 32 隻 | 是候選篩選結果，尚不是眼睛 GT |
| YOLO26s 15 張 POC | 13 張中有 12 張至少恢復兩隻既有合格動物，總共 27／32 隻 | 可追蹤動物 ID；不保證眼睛邊界可標 |
| eye_visibility_review_working.csv | Selected 的 47 筆核心欄位已填，eye_resolution、reviewer 仍留白 | 不以空白視為不合格，也不自動補值 |
| 269196 羊群圖 | 動物 Mask 合併，只有 1／5 隻合格動物配對成功 | 保留為上游失敗案例，不刪除或偷偷替換 |
| 試標提案 | 3 張原圖可解碼、尺寸符合 COCO、EXIF orientation=1；六個目標均有動物 TP | 僅核對來源與實例，不據此填寫眼睛 GT |

來源：[Phase 1 research record](dataset_selection.md)、
[YOLO26s selected-image results](phase2_yolo26s_selected_results.md)。
原始 eye_visibility_review.csv 是保留的空白模板，既有人工結果在 working 檔；兩者均不修改。

## Eye Region Definition Options

| 選項 | Mask 範圍 | 用途與限制 | 建議 |
| --- | --- | --- | --- |
| A Visible eye aperture | 可見眼裂內、能辨識為眼球表面的投影區域；可包含可見虹膜、瞳孔與鞏膜 | 對應「可見眼部區域重心」，會受姿態、眼瞼開合及遮擋影響 | 第一版試標規範候選 |
| B Iris | 僅可見虹膜，是否包含瞳孔須另定義 | 標的不同，須能清楚區分虹膜邊界 | 暫不混入 A |
| C Pupil | 僅瞳孔 | 標的不同；反光、模糊或小眼睛時可能無法辨認 | 暫不混入 A |

建議先試 A，因為本專案暫定量測的是 Eye Mask 重心距離。
若研究題目實際要求瞳孔中心，必須重新確認標的與方法，不能把 A 的重心改名為瞳孔中心。

### Proposed Boundary Rules for Option A

| 情況 | 標註規則 |
| --- | --- |
| 正常可辨識眼睛 | 依可見眼裂內的眼球表面描邊；排除眼瞼皮膚、眼周毛髮、睫毛與淚痕 |
| 眼內反光 | 明確位於已辨識眼部區域內的反光仍屬該區域，不因亮度而挖洞 |
| 夜視亮點／過曝 | 若只見亮點、無法辨認眼裂範圍，標 Uncertain；不能把亮斑輪廓當作眼睛輪廓 |
| 外物／毛髮遮擋 | 只畫可見部分，遮住的像素不補畫，標 Partial；保留為分割參考，不直接納入嚴格量測集 |
| 閉眼或明顯半閉眼 | 不推測藏在眼瞼後方的範圍；依可辨識情況記 NotVisible／Partial／Uncertain |
| 模糊、小眼睛、邊界不可辨 | 允許保存草稿與原因，但不核准為可測量 GT；不得靠平滑、插值或超解析臆造邊界 |
| 多區塊或孔洞 | 依實際可見範圍保存，不自動補洞、取最大區塊或以橢圓代替；異常形狀須人工複核 |

放大只改善操作與檢視，不增加原始影像資訊。第一輪不預設「眼寬幾個像素就合格」；
應在實際試標後，記錄區域大小與再標一致性，再提出可供新驗收資料使用的品質門檻。

## Anatomical Left and Right

左右以動物自身的解剖學方向定義，必須同時查看完整頭部、鼻部及姿態。
正面動物的左右通常與觀看者相反，但不能用螢幕左右排序自動賦值。
側轉、遮擋或無法判斷時保持 Unknown，不強迫指定為已確認。

每隻動物有 left／right 兩個待確認標註槽位。未確認解剖方向的草稿不得升為可量測資料；
若只辨識到一隻眼但無法確定側別，可以留觀察備註，不把它複製成兩隻眼或當成右眼。
本輪不水平翻轉原圖；動物 ID 使用官方 annotation_id，不使用模型預測順序代替。

## Three Image Trial Proposal

採目的性選圖，從既有 13 張候選中選出不同物種及臉部大小的三張日間圖片。
沒有新的隨機抽樣，因此沒有抽樣 seed；這是規範開發樣本，不是留出測試集或類別排名依據。
六隻動物在既有 working CSV 皆為左右眼 Yes、legacy Eligibility Yes，且均有 YOLO26s 動物 TP。
這些既有標記只作選圖依據，不等於新的眼睛邊界、解剖側別或可量測性已經確認。

| 試標順序 | Image ID | 動物／Annotation IDs | 原圖尺寸 | 預計眼睛槽位上限 | 試標重點 |
| --- | --- | --- | --- | ---: | --- |
| 1 | [125405 原圖](../data/coco/val2017/000000125405.jpg) | dog：13692、17207 | 640×428 | 4 | 前後兩張臉大小不同，先確認毛髮與眼裂邊界是否可重複描繪 |
| 2 | [20247 原圖](../data/coco/val2017/000000020247.jpg) | bear：587181、588042 | 640×457 | 4 | 較小且較模糊的後方臉；原圖身體截斷不能誤算成模型切頭 |
| 3 | [333069 原圖](../data/coco/val2017/000000333069.jpg) | giraffe：599488、599888 | 428×640 | 4 | 小頭部與眼部解析度限制；檢查能否可靠確認左右及邊界 |

總計 3 張、6 隻動物、最多 12 個不同眼部目標；沒有預先保證 12 個都能成為有效 Mask。
若全部條件成立，最多可衍生 6 組同隻雙眼距離、3 組同圖兩隻右眼距離；目前均未產生。
所有提案列為 Proposed，待負責人核准；不代表最終動物類別選擇。

完整清單為 [proposed_instances.csv](../outputs/phase3/phase3a/proposed_instances.csv)，
以每隻動物一列保存原始 IDs、bbox、尺寸、影像 digest、選擇理由與用途。
不在這份提案中填寫新的 left/right visibility、品質結果或 eye_resolution。

71756 夜視反光、81766 長毛與分數邊界、269196 實例合併保留為之後的壓力案例，
不混進第一輪日間試標；完整 13 張候選及所有既有失敗紀錄仍保留。
如果本輪眼睛不可標，先記錄原因並提出替代提案，不能直接挑掉失敗後宣稱成功率。

## Proposed Ground Truth Data Format

以下是未來 Phase 3B 的資料契約提案，目前不建立空白 GT 檔案或 placeholder Mask。
沿用現有 src-layout、Streamlit 與 CSV；不新增資料庫或一般化標註平台。

### Eye Segmentation Ground Truth

每個眼睛 Mask 為原圖尺寸的單通道 8-bit PNG，像素僅為 0（背景）／255（眼部），
無透明度。計算時用 M=PNG/255 得到 0／1，不對提交後的 Mask 自動做形態學修補。
建議相對路徑為 masks/{image_id}/{annotation_id}_{eye_side}_v{annotation_version}.png。
修訂另存新版本，不覆寫已核准版本；無可標眼睛時不偽造全黑 PNG 冒充已完成標註。

未來 eye_mask_annotations.csv 建議每隻眼睛每個版本一列：

| 欄位 | 定義 |
| --- | --- |
| image_id、annotation_id | 原始 COCO 圖片及動物身份 |
| eye_side | left／right，動物自身的方向；草稿可尚未確認，但不得當成已核准側別 |
| mask_path | 相對於輸出目錄的 PNG 路徑；無有效 Mask 時留白 |
| visibility | Visible／Partial／NotVisible／Uncertain；未審核留白 |
| side_confirmed | Yes／No／Unknown；初始化為 Unknown，不由螢幕位置推定 |
| mask_review_status | Pending／Accepted／Rejected／Uncertain；Accepted 只代表分割標註審核通過 |
| measurement_quality | Pass／Fail／Unknown；區分可接受的分割參考與適合嚴格量測的標註 |
| exclusion_reason | 不可標、側別不明、反光、遮擋等原因；Rejected／Fail 應有理由 |
| reviewer | 實際執行審核者，不自動替使用者署名 |
| protocol_version、annotation_version | 邊界規範版本與標註修訂版本 |
| annotated_at_utc | 實際保存標註的 UTC 時間，不預填虛構時間 |

歷史唯一鍵為 image_id＋annotation_id＋eye_side＋annotation_version；
同一版本不得有重複紀錄。量測與評估必須明確指定使用哪個版本，不以最後寫入的檔案悄悄取代。
來源 manifest 記錄 COCO 版本、原圖尺寸與 digest、規範版本、標註工具版本及每個 Mask digest。
若實際使用 ROI，亦記錄裁切框、縮放比及畫布 padding，不能只保留會隨模型版本改變的框。
這些都是 Phase 3 新產物，不變更 Phase 1 CSV Schema。

### Eye Landmark Ground Truth

建議本輪先不收 Landmark，以降低成本並避免把主要方法改成手點中心。
若日後需要，另存獨立 eye_landmarks.csv，欄位至少包含 image_id、annotation_id、eye_side、
landmark_definition、x、y、reviewer、protocol_version、annotation_version。
瞳孔中心、人工視覺中心與 Mask 重心是不同標的；從 Mask 算出的重心是衍生值，不當作獨立人工 Landmark GT。

## Original Image Coordinates

建議沿用既有研究紀錄中的慣例：原圖左上外框為 (0,0)，x 向右、y 向下；
第 row=r、column=c 個像素中心為 (c+0.5,r+0.5)。PNG 尺寸永遠為原圖 W×H，保存後再次核對。

若未來標註工具裁切 ROI 的左上角為 (x0,y0)，縮放比為 sx、sy，
顯示畫布內的影像左上 padding 為 (px,py)，畫布點 (u,v) 對應：

```text
x_original = x0 + (u - px) / sx
y_original = y0 + (v - py) / sy
```

不得把螢幕／瀏覽器頁面的座標直接保存為原圖座標。
sx、sy 必須由實際顯示影像區域計算，不能只用名義放大倍率；UI 實作時另處理畫布偏移與像素比例。
區域裁切與縮放是顯示操作，保存前轉回原圖再產生最終二值 Mask；原圖本身不重採樣或覆寫。

對非空 Mask，令 N=sum M，則：

```text
cx = sum((column + 0.5) * M) / N
cy = sum((row + 0.5) * M) / N
distance_px = sqrt((cx_a - cx_b)^2 + (cy_a - cy_b)^2)
```

沒有 Mask 或 N=0 時，中心與距離為未定義，保留空值及原因，不能填 0。
同隻雙眼使用同一 annotation_id 的 L／R；兩隻右眼必須同一 image_id、不同 annotation_id。
結果為二維影像像素距離，沒有相機校正與深度時不轉換成公分或三維距離。
上述數學與座標慣例本次只寫成規格，沒有實作計算。

## Measurability and Unknown States

| 衍生狀態 | 提案規則 |
| --- | --- |
| eye_observable(side) | 未來該側的新審核 Visible／Partial→Yes、NotVisible→No、Uncertain／空白→Unknown；不回寫 legacy 可見性 |
| A_side | 該眼的動物 ID 對應已核對，且 side_confirmed=Yes 才為 Yes；明確 No→No，其餘 Unknown |
| Q_side | 非空且原圖對齊的 Eye Mask、mask_review_status=Accepted、measurement_quality=Pass 均具備才為 Yes |
| binocular_measurable | E_left AND E_right AND A_left AND A_right AND Q_left AND Q_right |
| right_eye_measurable | E_right AND A_right AND Q_right，不要求左眼可測 |

三值 AND：任何必要條件明確 No 則 No，全部 Yes 才 Yes，其餘 Unknown。
此處將既有定義的 A 細化為每眼的 A_side，避免左眼未確認而誤否決已確認的右眼。
未審核資料保持 Unknown；legacy Eligibility=Yes 不直接變成 measurable=Yes。
Q 的缺件／Pending／Uncertain 保持 Unknown；已明確 Rejected、Fail 或無可見眼部則可判 No，並記錄理由。
Partial 的可見區域可有合格的分割參考 Mask，但因遮擋／半閉眼會移動可見區域重心，
本版建議 measurement_quality=Fail，不列為完整眼部重心量測；此嚴格規則亦待負責人確認。
尺寸不符、破損 Mask 或重複鍵是資料錯誤，須修正，不當成人工 No／Unknown 而略過。

## Future Trial Procedure and Validation

核准規範與資料後，Phase 3B 才製作最小標註操作。建議先在 125405 的四個眼部槽位試標；
若輪廓或解剖方向無法確認，記錄 Uncertain／原因，再回到規範討論，不以猜測補齊四眼。
初步操作可行後才擴至另外兩張，最多 12 個不同眼部目標。

為檢查再標一致性，建議對第一張的四隻眼睛在另一次操作中重標，先不顯示上一版輪廓。
這是相同四眼的重複版本，不增加獨立樣本數；一位審核者重標只能估計同人一致性，
不能宣稱已完成多人標註者一致性驗證。

未來須分開報告：

| 驗證層次 | 要報告的內容 |
| --- | --- |
| 標註完整性 | 可見性、側別、可接受 Mask、可量測眼睛的數量與各種未知／排除原因 |
| Mask 一致性 | 原圖像素面積、可見區域寬高、重標 IoU 與 Dice |
| 幾何一致性 | 同一眼兩次 Mask 的重心差；同隻雙眼與同圖兩隻右眼距離的差異，單位 px |
| 任務完成率 | 每張是否能提供兩隻動物各自的雙眼量測，並提供兩隻右眼距離；分母保留失敗圖 |

目前不捏造 1 px、某個 IoU 或固定眼寬為已驗證的成功門檻。
可用這三張開發樣本提出容差，但在眼睛模型評估前須核准並凍結，再以不同 image_id 驗收。
同張圖的所有動物與眼睛放在同一資料分組，避免相近像素同時用於開發與驗收。
若日後用人工點／框提示模型，須另記提示來源，標為互動式分割，不宣稱全自動定位。

數學程式實作時沿用 pytest，重點測試非空／空 Mask 重心、已知幾何距離、
ROI／原圖轉換與同圖不同動物右眼配對。這次文件與清單變更不新增測試框架或數學程式。

## Decisions Requested

| 待核准事項 | 建議 |
| --- | --- |
| 眼睛邊界標的 | 選 A，可見眼裂內的眼部區域，依上列反光／毛髮／遮擋規則 |
| 部分可見與側別不明 | 可保存參考／草稿，嚴格量測不放行；未知不當成 Rejected |
| 主要 GT | 先做 Eye Segmentation GT，暫不額外收 Landmark |
| 保存與座標 | 原圖二值 PNG＋獨立 CSV，採半整數像素中心與版本化保存 |
| 試標資料 | 提議 125405、20247、333069 的六隻動物，最多 12 隻眼；尚未選定最終類別 |

核准這些規範與樣本後，才能確認 Phase 3B 的標註工具實作範圍；
本文件本身不授權眼睛模型下載／推論、訓練、5,000 張分析或 Ground Truth UI 開發。

## Deliverables and Checks

| 產物 | 內容 |
| --- | --- |
| 本文件 | 邊界選項、解剖左右、例外規則、未來 GT 格式、座標與重標流程、核准事項 |
| outputs/phase3/phase3a/proposed_instances.csv | 6 筆動物提案；只有既有資料與提案用途，沒有新的眼睛標註 |
| outputs/phase3/phase3a/proposal_manifest.json | 來源 digest、選圖方法、3 圖／6 動物／最多 12 眼與 Proposed 狀態 |
| outputs/phase3/phase3a/validation.json | 資料來源、ID、影像尺寸、提案數量與原有資料雜湊核對結果 |

實際 Eye Mask、Landmark、距離與模型結果數量均為 0。
執行只使用現有 CSV／JSON／圖片讀取、影像尺寸及 digest 核對；沒有執行模型命令。
核對結果：3 張圖片、6 個唯一動物 ID、6 個文件連結均通過，234 個受保護檔案 digest 不變。
本次正式程式不變，採上述資料與文件核對，未新增或重跑 pytest。
沒有修改 AGENTS.md、Phase 01.md、既有人工 CSV、模型結果或正式程式，也沒有 Commit／Push。
