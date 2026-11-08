# Methodology and Validation

## Model Choice

| 任務 | 固定方法 | 選擇理由與界線 |
| --- | --- | --- |
| 動物實例分割 | YOLO26s-seg／Ultralytics 8.4.0 | 同時提供每隻動物的框與遮罩；框供後續定位使用，不宣稱全 COCO 最佳 |
| 眼部定位 | HRNet-W32／AP-10K | 提供動物解剖左右眼通道，避免僅憑畫面左／右命名；不等於已驗證可見性 |
| 眼部分割 | SAM 2.1 Tiny | 用預測眼點與局部 ROI 產生二值眼部遮罩；點定位與遮罩生成分開處理 |
| 幾何量測 | 眼部遮罩重心間的歐氏距離 | 在原圖像素上可直接驗算；不宣稱瞳孔中心、實體長度或三維距離 |

使用固定預訓練權重，不在此交付訓練或微調。SAM2 commit：
`2b90b9f5ceec907a1c18123530e92e794ad901a4`；MMPose commit：
`5408bc76f5b848cf925a0d1857899011d8c5b497`。三個權重的 SHA-256 在推論設定中固定。
CPU FP32、seed 42、4 threads；套件鎖定與安裝紀錄用於重現環境。

## Data and Eye Definition

示範取自 COCO 2017 validation，圖片至少包含兩個動物實例；提交範例為 `125405` 的兩隻狗。
原圖網址、雜湊及用途見 [input_manifest.csv](../examples/input_manifest.csv)。這是開發示範，
不是獨立留出測試。動物輪廓可使用官方 COCO 實例遮罩評估；COCO 沒有本案眼部遮罩 GT。

人工眼部參考曾凍結為兩版，不能混用其分數：

| 版本 | 遮罩範圍 |
| --- | --- |
| `eye-visible-aperture-v1` | 可見眼裂 |
| `eye-visible-eyelid-v2` | 可見眼裂與緊鄰可辨識眼瞼邊緣，排除毛髮與色塊 |

左右眼採解剖側別。標註者需確認可見性及側別，再以相同原圖格點描繪與凍結遮罩。
定位 landmark 與遮罩重心是不同標的；不能把對重心的偏移稱為瞳孔定位誤差。

## Fixed Inference Rule

HRNet 雙眼預測距離為 $d_h$，A_r100 的正方形 ROI 邊長為：

$$L=\max(32,2d_h)$$

以各眼點為中心，左／上取 floor，右／下取 ceil，裁至影像邊界。
每眼使用一個預測正點、無人工框；SAM 候選取最高模型品質分數，logits > 0 為前景，
不補洞或平滑。推論不讀 GT，不以 GT 改提示或挑選候選。
模型品質分數未經校準，不能當眼部分割正確率。

## Measurement and Formula Verification

對原圖二值眼部遮罩 $M_{r,c}\in\{0,1\}$，令 $N=\sum_{r,c}M_{r,c}$。
原點在影像左上外框，x 向右、y 向下，像素中心為半整數：

$$
c_x=\frac{\sum_{r,c}(c+0.5)M_{r,c}}{N},\qquad
c_y=\frac{\sum_{r,c}(r+0.5)M_{r,c}}{N}
$$

端點 $a=(c_{x,a},c_{y,a})$、$b=(c_{x,b},c_{y,b})$ 的距離為：

$$d(a,b)=\sqrt{(c_{x,a}-c_{x,b})^2+(c_{y,a}-c_{y,b})^2}\quad[\mathrm{px}]$$

| 量測 | 配對 |
| --- | --- |
| 同隻動物雙眼 | 同一動物的 anatomical left／right |
| 不同動物右眼 | 同一張圖的不同動物 right／right，每個無序組合只算一次 |

$N=0$ 或缺失端點時未定義，CSV 留空、JSON 使用 null；不補零。實作見
`measurement/metrology.py` 與 `inference/inference_artifacts.py`。
測試驗證單像素中心 `(0.5, 0.5)`、平移、3–4–5 三角形、配對及缺失狀態。
這些是工程與公式測試，不是動物 GT。

## Evaluation Metrics

對預測遮罩 $P$ 與人工參考 $G$：

$$\mathrm{IoU}=\frac{|P\cap G|}{|P\cup G|},\qquad
\mathrm{Dice}=\frac{2|P\cap G|}{|P|+|G|}$$

$$e_c=\|c(P)-c(G)\|_2$$

對 $n$ 筆可定義的距離：

$$\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^n|d_i^{pred}-d_i^{ref}|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^n(d_i^{pred}-d_i^{ref})^2}$$

動物分割使用 COCO Mask AP／AR（IoU 0.50:0.05:0.95），眼部分割另計 IoU／Dice。
重心誤差、雙眼 MAE／RMSE、跨右眼 MAE／RMSE 分開報告。
同列完整目標數、有效數、推論失敗及空遮罩數；兩個空遮罩不算完美重疊。
距離缺值不納入數值平均，必須保留缺失率，避免只報成功案例。

## Recorded Observations

下表只保留交付的 **A_r100 固定預測**。來源為既有凍結評分，
COCO 圖片 `20247`、`125405`、`333069`（熊、狗、長頸鹿各兩隻）：
12 眼、6 筆雙眼及 3 筆跨右眼距離，當次皆可計算，沒有推論失敗或空遮罩。

| 人工參考 | Mean IoU | Mean Dice | 重心誤差 px | 雙眼 MAE／RMSE px | 跨右眼 MAE／RMSE px |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 0.478 | 0.575 | 1.576 | 2.076／3.038 | 3.766／4.561 |
| v2 | 0.522 | 0.617 | 1.738 | 2.008／2.792 | 3.383／4.188 |

未四捨五入摘要與來源雜湊見 [Recorded Evaluation](../examples/README.md#recorded-evaluation)。
兩列是同一批預測對不同參考定義的評分；分數差異不是模型進步。
這是目的性選取的三張開發圖，動物框沿用當時凍結 YOLO 預測；不是本次版本重跑、
獨立泛化成績或自動可見性驗收。完整人工遮罩與評分產物留在本機研究備份及 outputs。

## Validation and Limits

| 驗證層次 | 方法與限制 |
| --- | --- |
| 公式與工程 | Pytest、CSV 端點驗算、失敗測試、來源雜湊；不證明模型準確度 |
| 視覺與參考 | 對照原圖逐眼檢查輪廓、側別及重心；從凍結人工遮罩重算參考距離 |
| 獨立驗收 | 事先固定參考定義、資料分割、配對、失敗分母與容忍值；目前尚未完成 |
| 尺度與深度 | 沒有相機校正、已知尺度或深度，只報影像平面 px |
| 誤差與不確定度 | 小樣本 MAE、重標差異或參考定義差異不是已校準信賴區間 |
| 可見性與物種 | 側臉、遮擋、閉眼、小眼睛可能失敗；十類啟用不等於十類準確度通過，鳥類為 AP-10K 域外試驗 |

每次執行保存輸入、設定、模型版本、seed、命令、狀態及雜湊。
新圖沒有人工參考時不產生 IoU／Dice 或距離誤差，所有量測資格保留 `unverified`。
