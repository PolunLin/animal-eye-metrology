# Methodology and Validation

## Task Decomposition and Model Choice

本專案分開處理動物實例分割、眼點定位、眼部分割、幾何量測、量測不確定度與評估。
動物輪廓正確不代表眼睛分割正確；眼睛 IoU 提升也不保證距離誤差改善。

| 工作 | 方法 | 選擇依據與限制 |
| --- | --- | --- |
| 動物分割 | YOLO26s-seg／Ultralytics 8.4.0 | 同時輸出動物框與實例遮罩，框可供姿態模型裁切；保留既定模型，不宣稱全 COCO 最佳 |
| 眼點定位 | HRNet-W32／AP-10K、固定 MMPose commit | 有動物解剖左右眼通道並通過本機 CPU 相容性；眼點不是本案遮罩重心或可見性判斷 |
| 眼部分割 | SAM 2.1 Hiera-Tiny、固定官方 commit | 接受定位點作提示，將眼部定位與遮罩生成分開；局部 ROI 限定輸入範圍，仍可能分到眼睛周圍外觀 |
| 提示／ROI | HRNet 單正點、無眼框；A_r100，選配 A_r050 | 保留既有基線與對照；不用 GT 調提示、排序候選或修補輪廓 |

預設推論沿用固定預訓練權重，不在推論時訓練或微調。SAM2 固定來源為
`2b90b9f5ceec907a1c18123530e92e794ad901a4`，MMPose 為
`5408bc76f5b848cf925a0d1857899011d8c5b497`；權重 SHA-256 由推論設定與準備工具核對。
模型候選品質分數不是經校準的正確率；此文件不將本機開發紀錄當作公開效能基準。

對 HRNet 兩眼點距離 $d_{\mathrm{HRNet}}$，以各預測眼點為中心的方形 ROI 邊長為：

$$
L=f\max(32,2d_{\mathrm{HRNet}})
$$

A_r100 使用 $f=1$，A_r050 使用 $f=0.5$；左／上邊界取 floor，右／下取 ceil，
再裁至圖片邊界。每眼只用其預測正點，不使用人工提示或 GT 回填。
兩條件是同一圖片的對照，不是獨立樣本。

## Data and Ground Truth

COCO 範例來源為 2017 validation；示範應選含至少兩隻動物的圖片，
並保存資料版本、image_id、來源與檔案雜湊。提交的單張圖片來源清單在
[`examples/input_manifest.csv`](../examples/input_manifest.csv)，距離樣例在
[`examples/measurements.csv`](../examples/measurements.csv)。完整資料集與人工參考需另外準備。
COCO 動物實例遮罩可以評估動物分割，不能當眼睛 GT；開發圖也不能當獨立測試集。

### Ground Truth Definitions

| 定義 | 範圍 | 使用規則 |
| --- | --- | --- |
| v1：`eye-visible-aperture-v1` | 可見眼裂 | 保留原始成績與凍結參考 |
| v2：`eye-visible-eyelid-v2` | 可見眼裂＋緊鄰可辨識眼瞼邊緣，排除毛髮／色塊 | 同一模型輸出另外評分，不能把換定義稱為模型進步 |

解剖左右及可見範圍由人判讀；程式完整性檢查不代替人的語義審核。
定位 Landmark GT 與分割 Mask GT 是不同標的。兩種既有分割定義應分開保存版本，
不混用來比較成績。研究標註還需本機原始規範、資格審核與凍結參考；這裡只說明量測語意，
不以簡介代替完整標註流程。

## Measurement and Formula Verification

對原圖二值眼部遮罩 $M_{r,c}\in\{0,1\}$，前景數 $N=\sum_{r,c}M_{r,c}$。
原點為影像左上外框，x 向右、y 向下，像素中心位於半整數。

$$
c_x=\frac{\sum_{r,c}(c+0.5)M_{r,c}}{N},\qquad
c_y=\frac{\sum_{r,c}(r+0.5)M_{r,c}}{N}
$$

兩個端點 $a=(c_{x,a},c_{y,a})$、$b=(c_{x,b},c_{y,b})$ 的距離：

$$
d(a,b)=\sqrt{(c_{x,a}-c_{x,b})^2+(c_{y,a}-c_{y,b})^2}\quad[\mathrm{px}]
$$

雙眼距離配對同一動物的解剖 left／right；跨右眼距離配對同圖不同動物的 right／right，
每個無序動物組合只計一次。$N=0$ 或缺失端點時未定義，CSV 留空、JSON 使用 null，不補零。
有限的零距離與缺值是不同情況。

實作在 `metrology.py`。`tests/test_metrology.py` 使用已知合成幾何驗證單像素中心 `(0.5, 0.5)`、
平移後重心同量平移、3–4–5 三角形、空遮罩／無效值及雙眼／跨右眼配對。
這是公式與工程測試，不是動物 GT 或模型準確度。

```sh
.runtime/app/bin/python -m pytest -q tests/test_metrology.py
```

提交的 [CSV 範例](../examples/README.md) 另提供不需模型的端點公式核對。
有人工參考時，應核對遮罩與版本，從原圖格點重新計算參考重心與距離，
再與預測比較；不能只檢查程式成功執行或 CSV 中的算式。

## Evaluation Metrics

對預測遮罩 $P$ 與人工參考 $G$：

$$
\mathrm{IoU}=\frac{|P\cap G|}{|P\cup G|},\qquad
\mathrm{Dice}=\frac{2|P\cap G|}{|P|+|G|}
$$

$$
e_{\mathrm{centroid}}=\|c(P)-c(G)\|_2,\qquad
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|d_i^{\mathrm{pred}}-d_i^{\mathrm{ref}}|
$$

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(d_i^{\mathrm{pred}}-d_i^{\mathrm{ref}})^2}
$$

| 層次 | 使用與缺失規則 |
| --- | --- |
| 動物分割 | 官方 COCOeval Mask AP／AR，另記固定操作點配對；不是眼睛準確度 |
| 定位 | 點是否落入對應參考、對參考遮罩重心的偏移；不稱瞳孔 landmark 誤差 |
| 眼部分割 | 原圖格點 IoU／Dice；兩個空遮罩未定義，不記完美分數 |
| 重心／距離 | 分開報重心偏差、雙眼 MAE／RMSE、跨右眼 MAE／RMSE；對可定義端點計數值，同列完整分母與缺失數 |
| 失敗 | 不刪除失敗目標；既有 failure-as-zero 分割摘要與有效距離摘要依欄位定義解讀 |

新圖沒有 GT 時不產生以上誤差。定位、分割、重心及距離是不同指標，不能互相代替。

## Validation and Uncertainty

評估前需固定資料分割、人工參考定義、配對規則、失敗分母與量測容忍值。
將開發資料與獨立測試資料分開，逐眼檢查輪廓與重心，分開報告分割指標與兩類距離誤差。
更換參考定義造成的分數差異屬定義敏感度，不能當作模型進步。
環境、模型載入及 API 成功只證明工程流程可執行，不提供上述準確度證據。

| 限制 | 界線 |
| --- | --- |
| 可見性／側別 | 新圖未驗證；閉眼、遮擋或高分點不自動成為合格量測 |
| 物種 | 十類啟用不代表全類準確度通過；鳥類為哺乳動物姿態模型的域外試驗 |
| 尺度／深度 | 無相機校正、已知尺度或深度，不提供公分或三維距離 |
| 不確定度 | MAE、有限重標差異與 v1／v2 差異不是已校準信賴區間 |
| 泛化 | 尚無獨立留出驗收；新增資料與研究方法仍需負責人核准 |

每次研究記錄資料版本／選樣、模型版本／commit、設定、seed、分母與命令；
實驗報告與完整產物保存在本機，不納入公開專案文件。新圖沒有人工參考時，
所有量測資格保持 `unverified`，不提供偽造的準確率或不確定度。
