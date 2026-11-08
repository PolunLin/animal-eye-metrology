# Methodology and Validation

## Task Decomposition and Model Choice

本專案分開處理動物實例分割、眼點定位、眼部分割、幾何量測、量測不確定度與評估。
動物輪廓正確不代表眼睛分割正確；眼睛 IoU 提升也不保證距離誤差改善。

| 工作 | 方法 | 選擇依據與限制 |
| --- | --- | --- |
| 動物分割 | YOLO26s-seg／Ultralytics 8.4.0 | 已保存四模型 CPU 試跑與 15 張 Selected Images 比較，候選恢復結果支持沿用；不代表全 COCO 最佳 |
| 眼點定位 | HRNet-W32／AP-10K、固定 MMPose commit | 有動物解剖左右眼通道並通過本機 CPU 相容性；眼點不是本案遮罩重心或可見性判斷 |
| 眼部分割 | SAM 2.1 Hiera-Tiny、固定官方 commit | 三模型／12 眼固定人工提示比較中，平均 IoU、Dice 與兩種距離 MAE 領先；不當成全自動泛化結論 |
| 提示／ROI | HRNet 單正點、無眼框；A_r100，選配 A_r050 | 保留既有基線與對照；不用 GT 調提示、排序候選或修補輪廓 |

完整紀錄：[動物比較](phase2_yolo26s_selected_results.md)、
[三模型固定提示比較](eye_segmentation_comparison_results.md)、[自動定位](phase3c_auto_eye_localization_poc_results.md)。
沿用預訓練權重，沒有訓練或微調；模型候選品質分數不是經校準的正確率。

## Data and Ground Truth

來源為 COCO 2017 validation。選樣、排除 crowd、人工示範圖與資料完整性紀錄見
[Dataset Selection](dataset_selection.md)。COCO 動物實例遮罩可以評估動物分割，不能當眼睛 GT。

主要眼部開發比較為 20247（熊）、125405（狗）、333069（長頸鹿），共 6 動物、12 眼、
6 筆雙眼與 3 筆跨右眼距離。三圖經目的性選取並參與開發，不是獨立測試集。
十類工程試跑的非空輸出也不是新增的眼部準確度證據。

| 定義 | 範圍 | 使用規則 |
| --- | --- | --- |
| v1：`eye-visible-aperture-v1` | 可見眼裂 | 保留原始成績與凍結參考 |
| v2：`eye-visible-eyelid-v2` | 可見眼裂＋緊鄰可辨識眼瞼邊緣，排除毛髮／色塊 | 同一模型輸出另外評分，不能把換定義稱為模型進步 |

規範與修訂見 [v2 Protocol](eye_annotation_protocol_v2.md)、[Qualification Review](eye_annotation_v2_review.md)、
[Definition Comparison](eye_definition_v2_comparison_results.md)。解剖左右及可見範圍由人判讀；
程式完整性檢查不代替人的語義審核。定位 Landmark GT 與分割 Mask GT 是不同標的。

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

既有人工參考可用對應歷史環境及已備妥的凍結產物唯讀重算：

```sh
.venv/bin/python -m animal_eye_metrology.measurement_baseline \
  --verify outputs/phase3/manual_measurement_baseline_v2
```

該驗證核對遮罩／版本並重算重心與距離，不重新推論或建立 GT。
提交的 [CSV 範例](../examples/README.md) 另提供不需模型的端點公式核對。

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

| 層次 | 使用與缺失規則 |
| --- | --- |
| 動物分割 | 官方 COCOeval Mask AP／AR，另記固定操作點配對；不是眼睛準確度 |
| 定位 | 點是否落入對應參考、對參考遮罩重心的偏移；不稱瞳孔 landmark 誤差 |
| 眼部分割 | 原圖格點 IoU／Dice；兩個空遮罩未定義，不記完美分數 |
| 重心／距離 | 分開報重心偏差、雙眼 MAE／RMSE、跨右眼 MAE／RMSE；對可定義端點計數值，同列完整分母與缺失數 |
| 失敗 | 不刪除失敗目標；既有 failure-as-zero 分割摘要與有效距離摘要依欄位定義解讀 |

新圖沒有 GT 時不產生以上誤差。定位、分割、重心及距離是不同指標，不能互相代替。

## Evidence and Uncertainty

已保存的 A_r100 對 v1 平均 IoU 為 0.478，同一預測改對 v2 評分後為 0.522；
重心誤差則由 1.576 變為 1.738 px。這是參考定義敏感度，不是模型進步，
也顯示重疊與幾何品質須分開報告。完整來源與分母見 [v1／v2 比較](eye_definition_v2_comparison_results.md)。

| 限制 | 界線 |
| --- | --- |
| 可見性／側別 | 新圖未驗證；閉眼、遮擋或高分點不自動成為合格量測 |
| 物種 | 十類啟用不代表全類準確度通過；鳥類為哺乳動物姿態模型的域外試驗 |
| 尺度／深度 | 無相機校正、已知尺度或深度，不提供公分或三維距離 |
| 不確定度 | MAE、有限重標差異與 v1／v2 差異不是已校準信賴區間 |
| 泛化 | 尚無獨立留出驗收；新增資料與研究方法仍需負責人核准 |

每輪記錄資料版本／選樣、模型版本／commit、設定、seed、分母與命令。
本次環境／文件整理不重新選模型、改 GT、更新量測定義或回寫歷史成績。
