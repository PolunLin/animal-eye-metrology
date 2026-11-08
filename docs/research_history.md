# animal-eye-metrology

Animal Eye Metrology: AI-Based Segmentation and Distance Measurement.

本專案旨在研究多動物影像中的實例分割、眼部定位／分割與距離量測。
**Phase 1 已由專案負責人核准結案**，現有工具用於 COCO 2017 validation 資料探索與人工篩選。
已完成 YOLO11n-seg 的 15 張 CPU POC、四模型固定 100 張試跑，
以及 YOLO26s-seg 對原有 15 張 Selected Images 的分割驗收資料；
不限制單一動物類別，已完成三張圖片的 12 個人工 Eye Mask，並建立 9 筆像素距離參考；
另已完成四眼再標差異分析、三個眼睛分割模型對固定人工提示的 12 眼開發 POC，
以及 HRNet-W32/AP-10K 自動定位接 SAM 2.1 Tiny 的三圖 POC、12 組提示／ROI 對照、
6 組幾何自動框實驗；自動框主方法未通過晉級條件，保留既有基線。
尚未進行獨立泛化驗收。
已依負責人核准完成 v2 獨立重標：可見眼裂＋可辨識眼瞼邊緣，排除眼周毛髮與色塊。
狗 13692 兩眼經人工修訂至 revision 4，最新12眼核對通過；v2已凍結，18組固定輸出兩版比較完成。
A_r100 對 v2 的平均 IoU 為0.522，但重心誤差1.738 px，並非全面改善；v1、A基線及原不晉級結論均保留。
詳見 [v1 / v2 Comparison Results](docs/eye_definition_v2_comparison_results.md)。
Streamlit 成果頁已支援 v1、v2 與兩版並列，固定同一模型輸出查看分割、重心、距離與失敗案例。

## Current Status

- 資料來自本機，沒有資料下載功能；人工結果寫入獨立工作 CSV。
- 使用者已提供 COCO 壓縮檔，並依設定解壓驗證集註記與 5,000 張影像。
  檔名、完整解碼與尺寸檢查通過；已完成本機註記分析，共 471 張多動物候選影像。
  結果位於 outputs/phase1/；測試仍僅使用人工幾何資料。
- Streamlit Explorer 採漸進式篩選：保留 15 張 Selected Images，47／47 個動物完成眼部審核。
  其中 13 張包含至少兩隻完整雙眼候選，共 32 隻，作為 Demonstration Candidates；
  不要求完成其餘 453 張 Pending，也不將示範圖當成獨立模型測試集。
- 已用 seed 42 為 zebra、giraffe、horse、dog 各抽 15 張，產生 60 張影像的人工審核資料包；
  位於 outputs/phase1/visual_review_seed42/；既有人工結果保存在獨立眼部審核工作檔。
- 統計動物實例的分割面積，不代表眼部可見性或量測準確度。
- legacy Eligibility 與既有 CSV 保持不變；eye_observable、binocular_measurable、
  right_eye_measurable 僅提出未來定義，尚未套用新 Schema 或判定規則。
- Eye Landmark GT 與 Eye Segmentation GT 分開；主要量測方法暫定
  Eye Segmentation Mask → Mask Centroid → Euclidean Distance 已建立人工 Mask 量測基準。
  Phase 3B 工具保留獨立人工資料；四眼再標、三模型固定提示比較及自動定位接分割均已完成。
- 自動流程重播三圖的六個凍結 YOLO 預測，產生 12 眼分割及 9 筆候選像素距離。
  HRNet 使用獨立 `.venv-eye-pose`；全部候選仍為 `visibility_status=unverified`。
  非空 Mask 不代表已通過眼睛可見性、側別或量測可靠性驗證。

結案結果、篩選與已知限制見 [Phase 1 Research Record](docs/dataset_selection.md)。
目前的實作要求見 [Phase 2 Specification](docs/phase2_instance_segmentation.md)，
實際數字、模型來源、失敗案例及限制見 [Phase 2 POC Results](docs/phase2_poc_results.md)。
原始 15 張 POC 保留；四模型相容性與估時結果見
[Phase 2 100-image Pilot](docs/phase2_pilot100_results.md)。未訓練或微調模型。

## Automatic Eye Localization POC

以下 POC、提示／ROI 與自動框表格保留原 **v1** 成績；v2 的並列結果見上方比較報告。

已完成本機 CPU 相容性驗證，以及 COCO 20247、125405、333069 的固定範圍開發 POC。
流程為凍結動物預測框 → HRNet-W32/AP-10K 眼點 → 預測點衍生 ROI → SAM 2.1 Tiny → Mask 重心與像素距離。

| 指標 | 已保存結果 |
| --- | ---: |
| HRNet 點落在對應人工眼睛 Mask 內 | 9／12 |
| 非空眼睛 Mask | 12／12 |
| 平均 IoU／Dice | 0.478／0.575 |
| 同隻動物雙眼距離 MAE | 2.076 px |
| 跨動物右眼距離 MAE | 3.766 px |

完整逐眼結果、失敗案例、環境版本、來源與重現命令見
[Automatic Eye Localization Results](docs/phase3c_auto_eye_localization_poc_results.md)。
狗與長頸鹿案例包含定位偏移及毛髮／大片頭部誤分；這些數字只描述三張已選開發圖。
目前 CLI 固定此三圖範圍；Streamlit 已整合唯讀成果檢視，
但不提供新圖推論，不能稱為任意圖片的完整自動應用。

```sh
.venv/bin/python -m animal_eye_metrology.auto_eye_localization --stage verify
```

以上只驗證已保存的 165 個產物，重算 12 眼與 9 筆距離，不重新推論。

## Phase 3C Prompt / ROI Ablation

已完成沿用三圖與人工參考的 12 組對照：兩種點來源 × 是否加入人工框 × 三種 ROI。
共 144 個眼部結果、432 個候選及 108 筆距離；原基線 12 眼重用，新增 132 次提示推論。

| 條件 | 平均 IoU | 雙眼距離 MAE | 跨右眼距離 MAE |
| --- | ---: | ---: | ---: |
| A：HRNet 點、無框、1× ROI（原自動基線） | 0.478 | 2.076 px | 3.766 px |
| A：HRNet 點、無框、0.5× ROI | 0.486 | 0.854 px | 0.980 px |
| B：人工點、無框、1× ROI | 0.532 | 0.569 px | 0.714 px |
| C：HRNet 點＋人工框、1× ROI | 0.597 | 0.428 px | 0.245 px |
| D：人工點＋人工框、1× ROI | 0.594 | 0.401 px | 0.375 px |

只有 A 組屬全自動提示；B／C／D 為人工介入診斷。該輪顯示人工框資訊有診斷價值，
但 dog 17207 使用人工點框後仍過度分割；小 ROI 亦有逐眼退步，未替換正式預設。
完整 12 組、定位、逐筆距離、案例圖與重現命令見
[Prompt / ROI Ablation Results](docs/phase3c_prompt_roi_ablation_results.md)。

```sh
.venv/bin/python -m animal_eye_metrology.prompt_roi_ablation --stage verify
```

## Phase 3C Geometric Automatic Eye Boxes

已依核准方案完成 3 種框比例 × 點＋框／只有框，共 6 組、72 眼結果、216 候選及 54 筆距離。
ROI 固定為 A_r100，框完全由 HRNet 預測產生；沒有使用人工點框、訓練或硬截斷 Mask。

| 條件 | 平均 IoU | 重心偏差均值 | 雙眼距離 MAE | 跨右眼距離 MAE |
| --- | ---: | ---: | ---: | ---: |
| A_r100 | 0.478 | 1.576 px | 2.076 px | 3.766 px |
| A_r050 | 0.486 | 1.319 px | 0.854 px | 0.980 px |
| γ=0.5 點＋框（預先指定主方法） | 0.243 | 2.137 px | 1.717 px | 1.432 px |
| γ=0.3 點＋框（探索） | 0.490 | 1.390 px | 0.696 px | 1.102 px |

**主方法未晉級**：10 項研究門檻通過 8 項，分割 IoU 與重心偏差兩項不通過。
γ=0.3 與 A_r050 互有取捨，保留為候選；沒有事後改稱主方法成功。
較大框雖完整包含參考眼部，仍常分到眼周。六組結果、逐眼反例與晉級判定見
[Geometric Automatic Eye Box Results](docs/phase3c_auto_eye_box_results.md)。

```sh
.venv/bin/python -m animal_eye_metrology.auto_eye_box --stage verify
```

## Proposed Next Directions

以下區分已完成工作與後續建議；新增資料、模型與研究方法仍需明確定義後執行。

| 順序 | 方向 | 具體交付物與決策依據 |
| --- | --- | --- |
| 已完成 | v2 定義比較與介面整合 | 12眼已核對凍結，18組兩版比較完成；Streamlit已支援v1／v2切換與並列，保留A基線 |
| 暫緩研究 | 局部眼部範圍與外觀精修 | 幾何框主方法未晉級；熱圖與外觀精修尚未執行，熱圖寬度不當作眼睛尺寸 |
| 暫緩研究 | 眼部邊界精修與候選品質判斷 | dog17207對v2改善；優先研究dog13692、giraffe599888殘餘失敗與bear588042邊界穩定度，保留原v1紀錄 |
| 研究必要 | 難例與獨立驗證 | 制定新增資料與標註方案，按原 image_id 分組，分開開發與留出驗證，包含遮擋、側臉、閉眼及小眼睛 |
| 研究必要 | 可靠性與量測資格 | 評估可見性／側別、錯誤接受與拒絕輸出的取捨；擴充參考重標證據，區分輪廓、重心與距離差異 |
| 已完成 | 成果檢視與新圖推論 | Results Explorer 保留 v1／v2；New Image Inference 提供上傳、完整 CPU 推論、階段狀態及結果下載 |
| 工程後續 | 容器與效能驗證 | 驗證 Linux CPU、MMCV 建置及隔離環境互通，再完成 Docker／乾淨環境測試；GPU 效能需另行實測 |

人工點／框只用於診斷，不當成全自動成績；三圖對照用於選擇下一步方法，不作泛化驗收。
ROI 改變同時影響上下文與有效尺度，對照時固定原 HRNet 裁切中心，避免替換點時連帶移動 ROI。

## New Image Inference

在 Streamlit 選擇 **New Image Inference**，上傳單張 JPG／PNG，按「開始自動推論」。
本輪整合既有 **YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny**，
現在啟用 COCO 全部十類動物：**鳥、貓、狗、馬、羊、牛、象、熊、斑馬、長頸鹿**。
其中鳥類會執行相同的眼部流程，但明確標為 **AP-10K 訓練範圍外試驗**；
其餘九類的資料集物種對應也不代表已通過本專案準確度驗證。
預設 **A_r100**，可勾選同時執行 **A_r050** 對照，不更換或覆寫歷史基線。

```sh
AEM_WORKSPACE='New Image Inference' .venv/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8517 --server.headless true \
  --browser.gatherUsageStats false

.venv/bin/python -m animal_eye_metrology.inference_pipeline --preflight
.venv/bin/python -m animal_eye_metrology.inference_pipeline \
  --image data/coco/val2017/000000125405.jpg --compare-r050
```

| 項目 | 行為 |
| --- | --- |
| 輸入 | 單張 JPEG／PNG；20 MiB、1,200 萬像素上限；套用 EXIF 方向、轉 RGB，透明處合成白色；不縮放 |
| 完整推論 | 重新偵測動物並產生眼點、分割、重心；不讀取人工點框、GT、舊偵測或標註配對 |
| 物種與結果說明 | 十類皆啟用；鳥類的域外試驗標記傳遞至動物、眼睛、距離及 CSV／JSON。結果上方先顯示逐眼完成／缺失原因 |
| 動物輪廓 | 獨立顯示 YOLO 動物遮罩的實際邊界與每隻不同色的實例遮罩；可選顯示矩形偵測框，另可下載逐隻原始二值遮罩 |
| 眼部結果 | 分開顯示定位、眼部分割、重心與量測；A_r100／A_r050 的局部放大使用同一顯示範圍 |
| 失敗 | 保留未支援、無效眼點、空遮罩、子程序失敗與超時；無法計算的距離為空值並附原因 |
| 下載 | 動物輪廓／實例遮罩圖、逐隻原始動物遮罩、每條件眼部疊圖 PNG、動物／逐眼／候選距離 CSV、完整 JSON |
| 保存 | 每次建立 `outputs/inference/<UTC時間_UUID>/`，保存原圖、正規化圖、模型遮罩、請求、日誌、設定與來源雜湊 |
| 科學界線 | 全部距離均為**候選量測**；左右眼、可見性與量測資格未驗證。新圖無 GT，不產生 IoU／Dice／距離誤差 |

設定位於 `configs/inference.toml`，可用 `AEM_INFERENCE_CONFIG` 指定工程路徑。
沿用 `.venv`、獨立 Ultralytics 8.4.0 目錄與 `.venv-eye-pose`，不安裝或自動下載模型。
這是單張本機 CPU POC；Docker、乾淨環境與新資料獨立驗證屬後續工作。
舊的馬／羊結果會保留當次「未啟用」紀錄；重新按下推論才使用十類設定，舊結果不會被補寫。
這個範圍是 COCO 十類，沒有宣稱涵蓋 COCO 以外所有動物，或可以量測被遮擋、閉合的眼睛。
詳細狀態與重現說明見 [New Image Inference Guide](docs/new_image_inference.md)。

## Read-only Results Explorer

使用現有環境啟動，或在既有 Streamlit 選擇 **Results Explorer**：

```sh
AEM_WORKSPACE='Results Explorer' .venv/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8517 --server.headless true \
  --browser.gatherUsageStats false
```

| 分頁 | 可檢視內容 |
| --- | --- |
| 總覽比較 | 切換 v1／v2／兩版並列；A_r100、A_r050優先，另可加入人工診斷；IoU、Dice、重心、距離誤差與固定分母；並列可展開v2−v1變更量 |
| 逐眼檢視 | 選圖、動物及解剖左右眼；固定原基線對照另一條件。並列時每個條件各顯示兩版GT，預測Mask／重心／點框／ROI保持相同；共享包含兩版GT及完整預測的視野 |
| 距離量測 | 圖中預測幾何固定；表格依版本切換參考距離與誤差，保留A_r100、A_r050與所選條件；並列可看兩版誤差差值 |
| 失敗案例 | 狗17207、狗13692、長頸鹿599888右眼、熊588042左眼；明確分開v1歷史與v2解讀，不把狗17207新版仍一概判失敗；舊網格標為v1 |
| 來源與驗證 | v1／v2參考與比較manifest、重現指令；下載所選版本長表或固定兩版差異寬表 |

預設為 `configs/results_definitions.toml`，釘住已凍結的v2參考與正式比較來源，
並引用原 `configs/results_workspace.toml`。原設定不修改，仍可用 `AEM_RESULTS_CONFIG`
指定原設定只看v1；所有路徑相對設定檔解析，不依賴歷史manifest的絕對執行路徑。
首次開啟預設v1，選「v1／v2 並列」即可查看同一輸出對兩版參考的差異。
此入口在載入 COCO 或初始化人工工作檔之前分流，只需要已保存的成果與凍結圖片。

資料層核對已讀取檔案的 manifest 雜湊、人工參考版本，以及原 A_r100 與重用結果一致性。
雙定義adapter另外核對v2 snapshot／protocol、五組來源manifest、比較CSV／JSON一致性、
v1原值重現、相同樣本與分母、預測路徑／重心／距離恆定；只載入已保存評分，不重新計分。
A_r100 逐眼／距離保留 original automatic 來源；摘要來自已核對的 ablation A_r100，
不重複計入同一基線。合併後共 **18 條件、216 筆逐眼結果、162 筆距離**，
仍只有 **3 張開發圖、6 隻動物、12 個參考眼與 9 筆參考距離**。
並列顯示／下載的兩列是同一預測對兩版參考，不是增加獨立樣本。
長表CSV含 `reference_definition`、`reference_protocol`，檔名標示v1／v2／paired；
兩版差異寬表使用v1_／v2_欄位及已保存的delta欄位。
讀取缺檔、來源雜湊不符或樣本不一致時顯示錯誤，不自行修復或覆寫產物。
頁面只核對使用到的檔案；完整實驗驗證仍使用各模組的 `--stage verify`。

圖像疊圖只在記憶體中繪製，沿用已保存 Mask、座標與距離；不重新推論或修改量測定義。
只有框條件不把 HRNet 幾何中心畫成 SAM 正點。
定位命中、候選最佳GT IoU、框涵蓋等只保存原v1診斷，與v2評分分開；
原晉級判定及舊條件網格皆標示v1。B／C／D沿用原v1人工提示，不稱為v2人工提示上限。
人工提示不列為全自動成績；γ＝0.5主方法未晉級，A基線不變。
所有保存候選仍為 `visibility_status=unverified`；Defined 只表示距離可計算。
Mask 重心不等於瞳孔中心，px 距離不等於實際長度；尚未進行獨立泛化驗收。

## Independent v2 Eye Relabeling

新規範：[Eye Region Annotation Protocol v2](docs/eye_annotation_protocol_v2.md)。
識別碼 `eye-visible-eyelid-v2`；範圍是可見眼裂加上緊鄰且可辨識的眼瞼邊緣，
排除毛髮、睫毛、淚痕及毛色斑塊；外緣不清請保留 Uncertain，不任意擴張。
完整標註規則、特殊情況、資格與比較解讀見規範。

已準備 `outputs/phase3/eye_annotation_v2/`：3 張原圖副本、6 個動物身份、
12 眼待標清單、凍結規範與來源 manifest，以及全新空白人工工作檔。
準備時 **0 筆人工標註、0 個人工 Mask**；沒有複製 v1 或採用模型輪廓。
即時進度請用下列 `--status`，不能把本段的準備時數字當成之後的最新狀態。

2026-11-08 首次核對發現狗13692兩眼需邊界複核；負責人已完成新修訂，左右眼皆revision 4。
最新31筆歷史中的12眼全部Accepted／Pass，完整性及獨立輪廓轉Mask核對通過；
已建立獨立v2凍結參考並完成18組比較，無新推論或候選重排。
首次／最終複核均保留；詳見 [Qualification Review](docs/eye_annotation_v2_review.md) 與
[Comparison Results](docs/eye_definition_v2_comparison_results.md)。

```sh
.venv/bin/python -m animal_eye_metrology.annotation_session --config configs/eye_annotation_v2.toml --status
AEM_ANNOTATION_SESSION=configs/eye_annotation_v2.toml .venv/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8518 --server.headless true --browser.gatherUsageStats false
```

`AEM_ANNOTATION_SESSION` 會限制頁面只開啟 v2 標註工作區，忽略舊的 GT／proposal 環境路徑，
載入準備好的原圖與本輪草稿，不提供 Results Explorer 或舊提示入口。
實際使用者須自行從原圖描繪、確認解剖左右、填 Reviewer 與審核；
每次保存只在獨立 v2 目錄追加修訂。凍結資格計數不會替使用者判斷可見性。
已看過模型輸出，因此不是完全盲測；本輪亦未完成 v2 再標一致性或獨立泛化驗收。

| 產物／入口 | 用途 |
| --- | --- |
| `configs/eye_annotation_v2.toml` | 可攜式準備與標註設定；首次準備才使用 `annotation_session --prepare`，已存在則拒絕覆寫 |
| `relabel_scope.csv` | 同三圖、六動物、12眼工作清單；不是 Ground Truth，也不預填審核 |
| `ground_truth/` | 實際人工標註、版本與Mask；與v1完全分開 |
| `configs/measurement_baseline_v2.toml` | 人工完成後凍結12個合格v2眼與9筆像素距離；缺件不回退舊版本 |
| `configs/definition_comparison_v2.toml` | 同一批18條件凍結輸出分別對v1/v2計算；原A結果保持不變 |

本次人工完成、資格核對後已執行以下命令；既有凍結及比較目錄拒絕覆寫：

```sh
.venv/bin/python -m animal_eye_metrology.measurement_baseline --config configs/measurement_baseline_v2.toml
.venv/bin/python -m animal_eye_metrology.definition_comparison --config configs/definition_comparison_v2.toml --check
.venv/bin/python -m animal_eye_metrology.definition_comparison --config configs/definition_comparison_v2.toml
```

`--check` exit 2 表示等待人工v2凍結參考，不建立成績目錄；exit 1 表示輸入錯誤。
比較目錄已存在時會拒絕再次建立；現在請使用凍結參考的 `--verify` 與報告中的獨立稽核命令。
12 槽位已審閱不等於12眼可測；Uncertain／Partial 不自動改 Pass，也不自行刪除分母。
已完成輸出分開列出：人工v1→v2的Mask／重心／距離變化、同一模型的兩版分割與距離誤差。
沿用原始12眼、6雙眼及3跨右眼分母／缺值規則；B/C/D仍是原實驗人工提示診斷，
不稱為v2人工提示上限；不自動晉級模型，也不把定義改變稱為模型能力進步。

## Manual Mask Measurement Baseline

眼睛分割已完成三模型的合成圖／單眼 CPU 相容性試跑，以及完整 12 眼固定人工提示比較。
Mask、重心及兩種距離誤差、敏感度與逐眼對照圖見
[Fixed-Prompt Eye Segmentation Results](docs/eye_segmentation_comparison_results.md)。
固定人工提示可在 http://127.0.0.1:8516 的 Eye Segmentation Prompts 工作區保存；
結果、執行指令與 optional 套件安裝見 [Eye Segmentation Compatibility](docs/eye_segmentation_compatibility.md)。

已凍結三張圖片、六隻動物、12 隻眼睛的最新 Accepted／Pass 版本，
輸出 12 個原圖重心、6 組雙眼距離、3 組同圖兩動物右眼距離與 3 張對照圖。
完整數值、固定版本與限制見 [人工 Mask 量測基準](docs/manual_measurement_baseline.md)。
產物位於 outputs/phase3/manual_measurement_baseline_v1/，原始人工標註保持不變。
這些是可見眼部 Mask 衍生的 px 參考值，尚不是模型成績或實際長度。

可以唯讀驗證已保存的基準，無需重新操作標註工具：

```sh
.venv/bin/python -m animal_eye_metrology.measurement_baseline \
  --verify outputs/phase3/manual_measurement_baseline_v1
```

建立基準的設定為 configs/measurement_baseline.toml，命令詳見報告；既有目錄拒絕覆寫。
已另完成四眼再標比較、三模型相容性試跑與 12 眼開發比較；不當成全自動眼睛定位或獨立驗收成績。

## Four-Eye Blind Repeat Check

已準備圖片 125405 的獨立四眼重標工作目錄 outputs/phase3/blind_repeat_v1/annotations/。
本輪專用頁面為 http://127.0.0.1:8515，請選 Eye Mask Annotation；
重啟命令與完整步驟見 [四眼盲重標操作說明](docs/blind_repeat_check.md)。
原始標註與人工量測基準保留；本輪不載入第一輪輪廓與審核值。
四眼重標及兩輪比較已完成，逐眼 Mask／重心及三筆距離變化見
[Four-Eye Repeatability Results](docs/blind_repeat_results.md)。

## Phase 3B Manual Eye Masks

現有 Streamlit 已加入 **Eye Mask Annotation** 工作區，可選動物／解剖左右、框選放大、
描繪及調整輪廓、保存草稿與人工審核結果。每次保存建立原圖尺寸 PNG 與獨立 CSV 新版本，
不覆寫歷史標註、Phase 1 人工資料或 COCO 官方 Annotation。

```sh
.venv/bin/streamlit run app/streamlit_app.py
```

進入頁面後選 Eye Mask Annotation，先做 125405 的四眼試標。
資料保存於 outputs/phase3/ground_truth/，初始只有 CSV 欄位與來源 manifest，沒有預填眼睛結果。
操作、審核欄位、版本保存與限制見 [Phase 3B 操作說明](docs/phase3b_annotation_tool.md)。
使用現有 Streamlit 1.67 原生互動元件，沒有新增前端框架；最低版本與 uv.lock 已同步。
這次不提供眼睛模型、Landmark 點位或距離計算。

## Phase 3A Annotation Proposal

Phase 3A 的 [眼睛標註規範草案](docs/phase3a_eye_annotation_protocol.md) 已整理，
包含可見眼部／虹膜／瞳孔選項、解剖左右、遮擋與反光、原圖 PNG＋CSV 格式及座標規範。
試標提議為 125405（dog）、20247（bear）、333069（giraffe），共 6 隻動物、最多 12 隻眼。
清單與來源核對位於 outputs/phase3/phase3a/，保留提出時的 Proposed 快照。
其後依「好的 3B」採上述建議規範與試標範圍製作人工工具；Landmark、眼睛模型與距離未實作。

## YOLO26s Selected Images Review

依負責人核准，暫用 YOLO26s-seg 處理既有 15 張 Selected Images；完整結果與
13 張候選的逐圖連結見 [YOLO26s Selected Images Review](docs/phase2_yolo26s_selected_results.md)。
本批 Mask AP=75.20／100、TP／FP／FN=37／6／10；既有合格動物恢復 27／32 隻，
12／13 張至少恢復兩隻。269196 羊群圖出現實例合併，須優先審查；Phase 1 人工結果不變。
上述配對不代表眼睛已能分割或測量，模型仍為暫定 POC，等待本次驗收。

沿用既有本機權重及 100 張試跑的獨立 Ultralytics 8.4.0 目錄：

```sh
PYTHONPATH=outputs/phase2/pilot100/ultralytics_8_4_0 \
  .venv/bin/python -m animal_eye_metrology.phase2 \
  --config configs/phase2_yolo26s_selected.toml
```

輸出位於 outputs/phase2/yolo26s_seg_selected15_cpu/，包括 15 張 overlays、
candidate_recovery.csv、兩模型 candidate_comparison.csv、視覺觀察 visual_review_notes.csv，
以及預測／配對／評估／manifest。既有輸出拒絕覆寫；重跑請加 --output 指定新的
outputs/phase2 子目錄。本次沒有執行 5,000 張、眼睛模型或 Ground Truth UI。

## Phase 2 Four-model Timing Pilot

四模型為 YOLO11n-seg、YOLO26n-seg、Mask R-CNN ResNet-50 FPN v2、YOLO26s-seg。
從完整 val2017 以 seed 42 固定抽 100 張，全部模型使用相同影像及順序，
依序以 CPU、FP32、batch=1、4 threads 執行，完成後推估 5,000 張處理時間。
這個試跑不代表完整資料集的準確度結論，沒有執行 5,000 張推論。

設定為 configs/phase2_pilot.toml；輸出位於 outputs/phase2/pilot100/。
YOLO11 沿用現有 Ultralytics 8.3.230；YOLO26 使用獨立的 8.4.0 套件目錄，
pyproject.toml、uv.lock 與既有環境不需升級。初次建置該目錄可執行：

```sh
uv pip install --python .venv/bin/python --no-deps \
  --target outputs/phase2/pilot100/ultralytics_8_4_0 'ultralytics==8.4.0'
```

權重須事先存放於 weights/，來源與 SHA-256 見設定檔；程式本身不下載。
本次實際執行順序如下。既有 sample、模型結果與比較表均拒絕覆寫，
重跑須複製設定檔到 configs/ 並指定另一個 output，再對各命令加入 --config。

```sh
.venv/bin/python -m animal_eye_metrology.phase2_pilot prepare
.venv/bin/python -m animal_eye_metrology.phase2_pilot run --model yolo11n_seg
PYTHONPATH=outputs/phase2/pilot100/ultralytics_8_4_0 \
  .venv/bin/python -m animal_eye_metrology.phase2_pilot run --model yolo26n_seg
.venv/bin/python -m animal_eye_metrology.phase2_pilot run --model maskrcnn_v2
PYTHONPATH=outputs/phase2/pilot100/ultralytics_8_4_0 \
  .venv/bin/python -m animal_eye_metrology.phase2_pilot run --model yolo26s_seg
.venv/bin/python -m animal_eye_metrology.phase2_pilot summarize
```

sampled_images.csv 與 sample_manifest.json 記錄抽樣及原圖雜湊；
comparison.csv 提供實測與估時，各模型資料夾保留逐張時間、預測 RLE、COCO 評估及版本。
詳細計時範圍、記憶體與限制見上述試跑報告。完整 5,000 張比較仍待核准。

## Phase 2 Segmentation POC

使用既有 uv 環境安裝必要的選用依賴；原 Phase 1／Explorer 的版本未升級：

```sh
uv sync --frozen --extra visualization --extra explorer --extra segmentation
```

先將官方 YOLO11n-seg 權重放在 weights/yolo11n-seg.pt；取得指令如下，不包含 COCO 下載。
此檔案是模型的二進位 checkpoint，不提交 Git。

```sh
mkdir -p weights
curl -fL --output weights/yolo11n-seg.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt
```

configs/phase2.toml 固定 checkpoint SHA-256、CPU、640×640 letterbox、FP32、seed=42、
AP confidence floor=0.001、NMS IoU=0.7、每圖全類 max_det=100，
診斷操作點 confidence=0.5／Mask IoU=0.5。
推論程式只讀本機 checkpoint，缺檔或雜湊不符即停止，不會自動下載權重或資料。

```sh
# 完整執行：全部 15 張 Selected Images
.venv/bin/python -m animal_eye_metrology.phase2 --config configs/phase2.toml

# 輸出已存在時拒絕覆寫；重跑請指定新的 run 目錄
.venv/bin/python -m animal_eye_metrology.phase2 --config configs/phase2.toml \
  --output outputs/phase2/yolo11n_seg_cpu_review

# 可先明確執行單張 smoke test，不把它稱為完整 POC
.venv/bin/python -m animal_eye_metrology.phase2 --config configs/phase2.toml \
  --limit 1 --output outputs/phase2/smoke_review_1
```

既有完整結果位於 outputs/phase2/yolo11n_seg_cpu/：

| 產物 | 內容 |
| --- | --- |
| predictions_coco.json | 原圖尺寸的 COCO RLE、category_id、score、image_id、獨立 prediction_id；AP 用低分數預測 |
| prediction_instances.json | 同一批預測另保留原圖 xywh bbox；不將 prediction_id 當成 annotation_id |
| instance_matches.csv | 固定操作點的 TP／FP／FN、ignored、below_confidence、IoU／Dice，以及 bbox 已配對但 Mask 未配對的診斷 |
| metrics.json | 官方 COCOeval Mask AP／AR、8 個有 GT 類別的結果、逐圖診斷與時間；無 GT 類別為 null |
| run_manifest.json、image_results.json | 來源／權重／程式／產物雜湊、版本、設定、原圖尺寸、逐圖狀態及時間 |
| overlays/*.png | 15 張原圖／官方 GT／預測的三欄對照，預測欄僅畫 confidence>=0.5 |

本次 Mask AP=0.626946、TP／FP／FN=33／7／14；完整數值見 POC Results。
這是 15 張人工選圖的實測，不能作為全 COCO 或獨立泛化成績。
框與 Mask 診斷都沿用官方 COCOeval；沒有另外實作 AP。
這個原始 15 張 POC 僅使用 YOLO11n-seg；後續四模型試跑另外保存。
該動物 POC 未執行全量 5,000 張或 MPS；後續人工眼部標註、量測與單眼分割相容性進度見上方紀錄。

Ultralytics 程式與模型適用其 AGPL-3.0／Enterprise 授權選項，
本專案的 MIT 宣告不覆蓋上游條款；公開或部署時需依使用方式處理。
見 [官方授權](https://www.ultralytics.com/license)。

## Setup

需要 Python 3.11+ 與 uv。下列安裝命令會取得 Python 開發／視覺化套件，**不會下載 COCO**。

```sh
uv sync --frozen --python 3.11
# 若需要 contact sheets 或執行影像損壞／尺寸不符測試：
uv sync --frozen --python 3.11 --extra visualization
```

統計核心僅使用 Python 標準函式庫；Pillow 與 pycocotools 是視覺化選用依賴。
相依套件版本由 uv.lock 固定。

## Local Dataset Configuration

編輯 configs/phase1.toml，或複製為被 Git 忽略的 configs/phase1.local.toml。
所有 TOML 內的路徑都相對於**設定檔所在目錄**解析。

```text
data/coco/
├── annotations/instances_val2017.json
└── val2017/
    └── <original COCO file names>
```

資料尚未取得時不需要建立空假資料。缺檔會以結構化錯誤訊息結束，不會自動下載。
只跑註記統計時不需要影像目錄。

目前本機資料已備妥；完整性與逐檔 SHA-256 紀錄位於被 Git 忽略的
data/coco/preparation_manifest.json。來源壓縮檔保留在專案根目錄；僅解壓所需驗證集，
沒有解壓訓練集註記、captions 或 person keypoints。

## Commands

先在沒有影像的情況下分析本機註記：

```sh
uv run --frozen animal-eye-metrology analyze --config configs/phase1.toml
```

影像可用後，使用新的輸出目錄產生原圖／動物 bbox 與 mask 對照 contact sheets：

```sh
uv run --frozen --extra visualization animal-eye-metrology analyze \
  --config configs/phase1.toml --contact-sheets --output outputs/phase1-with-images
```

--output 的相對路徑以目前工作目錄為基準。輸出目錄必須全新或為空，避免覆寫人工審查。
contact_sheets.max_images = 0 表示全部候選影像；正整數僅產生按 image ID 排序的預覽，
**不構成研究抽樣**。眼部審查必須回到原始解析度，不能只依縮圖判讀。

核准下載且本機檔案已備妥後，可檢查檔案雜湊、ZIP CRC 與必要成員；此命令不解壓縮：

```sh
uv run --frozen animal-eye-metrology verify-file annotations_trainval2017.zip \
  --zip --required-member annotations/instances_val2017.json
```

如有獨立可信的 SHA-256 參考值，可加 --expected-sha256；註記 JSON 的參考值也可寫入設定。
自行計算的雜湊及 ZIP CRC 能追蹤版本／檢查損壞，不能單獨證明官方來源。

## Outputs

| 檔案 | 內容 |
| --- | --- |
| category_statistics.csv | 全部動物類別的原始、crowd 與合格實例／影像計數 |
| animal_instances.csv | 所有動物註記的原始 ID、iscrowd、bbox、官方 area 與面積比例 |
| area_summary.csv | 合格實例的 bbox／segmentation 面積分布，含像素平方與影像面積比例 |
| multi_any.csv | 至少兩個合格動物實例的影像 |
| multi_same_category.csv | 至少同類兩個合格實例的影像 |
| multi_different_categories.csv | 至少兩種動物類別的合格實例共存影像 |
| eye_visibility_review.csv | 候選影像中每個合格動物一列；人工欄位留白 |
| contact_sheets/page_*.png | 選用的原圖／bbox／mask 對照圖 |
| run_manifest.json | 資料與程式雜湊、設定、環境、執行方式、篩選規則及輸出雜湊 |

CSV 的 category_ids、category_names、category_counts、repeated_category_ids、annotation_ids
儲存為 JSON 欄位；三組候選影像可重疊，不能相加當成總數。
程式不修改官方註記，且失敗的執行不會發布部分輸出目錄。

## Manual Review Batch

本批 contact sheets 沿用既有 visualization.py，另啟用實例圖例、不重複配色與較大面板。
抽樣清單保留原始 image／annotation ID；人工 CSV 完整沿用上層表格的 schema。
60 張影像共 214 列審查資料，包含 168 個抽樣目標實例與 46 個伴隨物種實例；
比較各類別時應依 sampled_images.csv 的 sample_category／target_annotation_ids 界定母體。

操作說明：[Visual Review Guide](outputs/phase1/visual_review_seed42/REVIEW_GUIDE.md)。
左右眼採動物解剖學方向；用原始影像判讀，不能把可見直接當成可準確分割。

可重現的準備指令如下；既有資料包拒絕覆寫，重跑時請用新的 Phase 1 子目錄：

```sh
.venv/bin/python -m animal_eye_metrology.review_sample --config configs/phase1.toml
# 重跑範例：另存於同一 Phase 1 目錄下
.venv/bin/python -m animal_eye_metrology.review_sample --config configs/phase1.toml \
  --output outputs/phase1/visual_review_seed42-repeat
```

## Streamlit Dataset Explorer

本機單頁工具，重用 coco.py、analysis.py 與 visualization.py。
使用 macOS／Linux（CSV 寫入鎖使用標準函式庫 fcntl），不需要 API 服務、資料庫、登入或容器。

從專案根目錄安裝選用套件並啟動：

```sh
uv sync --frozen --python 3.11 --extra visualization --extra explorer
.venv/bin/python -m streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

開啟 http://127.0.0.1:8502。若該連接埠已有此工具執行，直接開啟網址即可；
若由其他服務使用，可更換 --server.port。停止時在啟動工具的終端機按 Ctrl+C。

| 工作區 | 操作與定義 |
| --- | --- |
| Image Selection（預設） | 瀏覽圖片、填選填備註，再按 Selected／Rejected／Pending，立即保存；預設儲存後自動下一張，可取消勾選 |
| 資料範圍與篩選 | 預設全部 471 張多動物候選；可切換 Seed 42 的 60 張。保留動物類別、圖片狀態篩選與 Image ID／上一張／下一張 |
| 圖片顯示 | 原圖與 COCO 註記分頁；Bounding Box／Mask 獨立開關。不同實例沿用既有配色及 Annotation ID |
| Eye Review | 只顯示 Selected 圖片，可選擇圖中全部非 crowd 動物實例，不要求逐一完成所有動物 |
| 眼部表單 | 左右眼 Yes／No／Uncertain；姿態 Front／Side／Other；遮擋 None／Partial／Heavy；Eligibility Yes／No／Uncertain；Reviewer Notes。左右依動物的解剖學方向判讀 |
| 選填與裁切 | 保留 eye_resolution、exclusion_reason、reviewer；選取實例 bbox 加粗，可查看原圖裁切。放大不增加原始解析度 |
| 每圖雙眼候選提示 | 依全圖已儲存紀錄，計算左右眼與 Eligibility 均為 Yes 且五個選項已完成的實例。至少兩隻時提示可先停止審核其他動物；精確分割與量測仍待確認 |
| 待確認 | 未審核或關鍵欄位 Uncertain 不當成不合格；提示不會自動改變 Selected／Rejected 狀態 |
| Dataset Overview | 保留全資料類別／候選統計；選圖區顯示全候選三種狀態數，眼部區顯示目前範圍已選圖及實例審核進度 |

Seed 42 的類別篩選依 sampled_images.csv 的 sample_category 選圖；全部候選依實際動物類別選圖。
進入 Eye Review 後，可審核已選圖中的其他類別動物，避免漏掉同圖可供研究的實例。
五個選擇欄位全填（含 Uncertain）代表完成一次人工判讀，不代表量測合格。
目前不提供 Ground Truth 中心標記或距離計算。

### CSV Persistence

兩個工作區使用不同 CSV，既有人工結果不會自動轉成 Selected。

| 工作檔（預設位置） | 唯一鍵與 Schema |
| --- | --- |
| outputs/phase1/image_selection.csv | image_id；欄位固定為 image_id, selection_status, review_notes。新候選預設 Pending |
| outputs/phase1/eye_visibility_review_working.csv | image_id + annotation_id；沿用既有 17 欄。已存在時直接載入，不因選圖狀態變動而刪除審核 |

眼部工作檔首次不存在時，才由原始 eye_visibility_review.csv 載入建立。Schema 保持：

```text
image_id,file_name,annotation_id,category_id,animal_category,animal_count,category_animal_count,left_eye_visibility,right_eye_visibility,head_pose,occlusion,eye_resolution,measurement_eligibility,exclusion_reason,reviewer,review_protocol_version,notes
```

1. 在 Image Selection 按狀態按鈕，保存圖片選擇與備註，再前往 Eye Review。
2. 依動物本身的解剖學左右判讀，填完五個選項後按「儲存審核」；只更新 (image_id, annotation_id) 對應的一筆。
3. 每次儲存先在檔案鎖內重新載入，再用暫存檔原子替換，避免半份 CSV 與跨視窗漏寫。
4. 若同一筆已被另一個視窗改過，拒絕覆蓋，按「重新載入圖片選擇」或「重新載入此實例」後再人工確認。
5. 切換影像或實例前先儲存；尚未送出的備註及表單不自動保存。重新整理會載入已儲存資料。
6. 側欄可分別下載兩份 CSV，或在「CSV 儲存位置」指定其他同 schema 的獨立工作檔。
   路徑必須在 Phase 1 輸出目錄內，不能指向原始分析或抽樣產物。

重複鍵、未知實例、欄位或類別 metadata 不符時，工具會拒絕載入，不會自行刪除或修正原檔。
舊有非空標記若不在新選項中，會顯示提示並等待人工重新選擇，不自動把 partial 當成 Yes。
不要在工具儲存的同時用試算表軟體直接覆寫同一工作檔；外部編輯器不會使用此檔案鎖。
目前沒有 Windows 檔案鎖支援、多人管理、完整修訂歷史或自動備份；可自行另存 CSV 副本。

測試或其他本機資料可設定 AEM_CONFIG（設定檔位置）、AEM_SELECTION_CSV（選圖工作檔）、
AEM_REVIEW_CSV（眼部工作檔）；兩個工作檔仍須位於該設定的 Phase 1 輸出目錄。
測試只使用合成資料，不填寫真實人工判讀。更新程式後，請重啟 Streamlit 以載入新的模組。

## Validation

v1／v2成果介面整合後：**228項測試通過**，另有53個既有pycocotools／NumPy棄用警告；
Ruff lint與格式檢查通過。新增17個資料層案例、5個合成AppTest，涵蓋跨版本固定預測、
來源／protocol／snapshot防混用、失敗缺值、CSV、legacy相容與唯讀保護。
真實瀏覽器使用既有Playwright CLI（Browser plugin未提供），桌面1440×1000及手機390×844
核對v1／v2／並列、換圖換動物、分割疊圖、距離、歷史失敗說明及下載；
v2摘要9列、狗單眼兩版4列、狗距離兩版12列下載均與保存值一致，console無error／warning。
介面改動前後2,250個既有Phase3產物／設定／Phase1 CSV的SHA-256全部未變。
此為已保存成果的檢視驗證，不是新推論或新增模型驗收。

本輪工程檔案：新增 `results_definitions.py`、`configs/results_definitions.toml`，
更新 `results_view.py`，新增 `test_results_definitions.py` 與 `test_results_definitions_view.py`；
原 `results_workspace.toml`、已凍結模型／標註／比較結果均保留。
實際使用現有 `.venv` 執行 `python -m pytest -q`、`ruff check .`、`ruff format --check .`；
未新增依賴或啟動模型推論。

```sh
uv run --frozen --extra visualization --extra explorer pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

v2 獨立重標與比較準備完成後完整驗證為 **206 項測試通過**，
另有 53 個既有 pycocotools／NumPy 棄用警告；Ruff 與格式檢查通過。
本輪新增 25 項合成資料測試：跨 protocol 防污染、獨立準備與規範雜湊、
專用 AppTest、v2 凍結與兩版定義比較、缺資料阻擋、原分母／缺值保留及來源變更拒絕。
準備後核對 2,081 個既有產物／設定／Phase 1 CSV 全部未變；當時比較器回報 awaiting_v2_reference。
2026-11-08 人工完成後再次執行全部測試與 Ruff，仍為206通過／53警告，舊2,081檔雜湊未變。
其後狗13692兩眼完成revision4；最新12眼核對通過並凍結v2，18組／216逐眼／162距離比較完成。
獨立數值稽核36,789項通過，最大差3.55×10⁻¹⁵；1,271個列入manifest檔案雜湊通過，
預測Mask／重心／距離兩版一致，v1成績全部重現；實際結果與限制見兩版比較報告。
測試中的合成 Mask 不是真實 GT。
v2 真實頁面另以既有 Playwright（Browser plugin 未提供）核對桌面1440×1000與手機390×844：
規範、空白輪廓、未預填Reviewer、換圖／換眼均正常；無未處理例外，console為0 errors／0 warnings。
正式頁面僅做檢視操作；保存、重開與防污染以合成資料AppTest驗證，沒有代填真實人工輪廓。

前輪成果檢視整合完成時為181項；當輪新增22項合成資料測試涵蓋來源與基線一致性、只框提示、
圖像座標與尺寸、失敗／缺檔狀態、AppTest 換圖／條件切換與唯讀保護。
Results Explorer 另以既有 Playwright CLI（Browser plugin 未提供）驗證真實成果：
桌面 1440×1000、手機 390×844；換圖後動物選項、只框提示、失敗案例與距離視圖正常。
18 列條件比較與 6 列狗圖片距離 CSV 下載數值與保存結果一致；
重啟後新頁 console 為 0 errors／0 warnings，手機文件寬度未超出 viewport。
2,080 個既有 Phase 3 產物、實驗設定及 Phase 1 CSV 的 SHA-256 全部未變。
先前 159 項驗證紀錄見 [自動眼框驗證](docs/phase3c_auto_eye_box_results.md#reproduction-and-validation)。
覆蓋類別、原圖座標／RLE、IoU／Dice、官方 matching／crowd／空預測、人工標註／提示保存、
重心與距離計算、自動 ROI、失敗分母、logits 重建、提示隔離、配對對照、失敗快取、
自動框像素涵蓋、只有框輸入與晉級規則；
測試使用合成 QA 資料。
涵蓋解析、crowd 排除、多動物篩選、面積統計、ID 保留、
空白人工標籤與必要的輸入／輸出保護；不追求覆蓋率或每模組一份測試。
已移除啟動環境、設定細項、contact sheet 版面／像素及重複檢查。
合法 polygon／RLE 的解析測試不需要視覺化依賴；只有損壞影像與尺寸不符兩項測試
需要 visualization extra，未安裝時會跳過；AppTest 需要 explorer 與 visualization extra。
保留 pytest 供後續 Metrology 數學驗證。

AppTest 驗證選圖儲存後自動換圖、重開載入、Eye Review 限已選圖、取消後再選回保留既有人工結果，
以及眼部儲存 → 切換另一實例不帶入資料 → 重開載入原結果。
CSV 測試驗證 schema 不變、重複鍵拒絕、不同紀錄合併、同筆舊版本衝突拒絕，
以及未審核／Uncertain 保持待確認、找到兩隻後可先停止審核的提示邏輯。
pycocotools／NumPy 有上游解碼棄用警告，未隱藏；不影響目前測試通過。

以下保留 Explorer 改版時的驗證快照；當時尚未完成最終選圖，最新數字以 Current Status 及 Phase 1 結案紀錄為準。
當時驗證：68 passed，46 則上述棄用警告；Ruff 與格式檢查通過。
真實資料以既有 Playwright CLI 驗證（Browser plugin 未提供）：桌面 1440×1000 的預設工作區、
換圖、類別篩選、Mask 開關與 Eye Review 空狀態正常；390×844 完成基本版面檢查。
重啟後的新瀏覽器工作階段無 console error／warning。寫入流程只使用合成資料。
既有 13 份資料／規範檔的雜湊未變，包含 2,121 列眼部工作檔與其中 168 筆已完成人工審核；
新圖片選擇檔的 471 筆候選皆初始化為 Pending。

## Project Layout

```text
configs/                    Portable TOML configuration
app/streamlit_app.py        Single-page local dataset explorer
src/animal_eye_metrology/   Offline analysis and command-line entry point
templates/                  Header-only manual review template
tests/                      Synthetic fixtures and automated tests
docs/dataset_selection.md  Methods, evidence status, and pending approvals
docs/phase2_instance_segmentation.md  Owner-updated Phase 2 implementation specification
docs/phase2_poc_results.md   Verified YOLO11n-seg CPU POC and reproduction commands
data/                       Local data, ignored by Git
outputs/                    Generated results, ignored by Git
```

直接使用安裝後的命令列入口，不另建重複的腳本包裝層。
下載選項、人工審查規範與研究限制見 [Dataset Selection](docs/dataset_selection.md)。
