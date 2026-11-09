# Training and Experiments

專案包含動物實例分割監督式微調、development-only 學習率診斷，以及選模凍結後的驗收。
本頁記載 2026-11-08 的既有實驗；恢復程式與頁面沒有重新訓練，也沒有替換推論權重。
HRNet-W32／AP-10K 與 SAM 2.1 Tiny 在此流程使用預訓練模型，沒有眼部模型訓練。

## Recorded Results

第一輪使用 256 fitting／64 development／64 internal holdout，每模型 3 epochs、384 updates。

| 模型 | 原始 Mask AP | 第一輪微調 Mask AP | Best epoch |
| --- | ---: | ---: | ---: |
| YOLO11n-seg | 51.21 | 30.24 | 2 |
| YOLO26n-seg | 53.07 | 45.62 | 3 |
| YOLO26s-seg | 63.71 | 53.39 | 3 |
| Mask R-CNN v2 | 66.77 | 67.00 | 1 |

YOLO11 fitting AP 也下降，不能只據 holdout 下降稱為過擬合。
其後以相同 YOLO11 初始化、CPU4、seed42、fitting256／development64、1 epoch 做 LR-only 對照：
初始化 development AP 50.3867；LR 1e-3 為 37.1610，1e-4 為 49.2096，1e-5 為 51.8054。
這支持原配方更新幅度過大的解釋，因果範圍限此模型／資料／seed／epoch。
Mask-only 對照只增加約 1.13 AP points，仍低於初始化，未充分解釋退步。

第二輪僅將 LR 0.001 改為 0.00001，三個 YOLO 都從原始權重重新開始。
下表使用另一組 fresh64，同一組影像比較三個條件，不能與上表不同 holdout 混比。

| 模型 | 原始 AP | 第一輪 AP | 第二輪 AP | 第二輪−原始（點） |
| --- | ---: | ---: | ---: | ---: |
| YOLO11n-seg | 51.44 | 37.00 | 53.02 | +1.58 |
| YOLO26n-seg | 57.37 | 50.10 | 57.91 | +0.53 |
| YOLO26s-seg | 61.81 | 52.97 | 62.10 | +0.28 |

第二輪最佳 epoch 分別為 1／2／1；由 development 選定，並在首次 fresh 推論前凍結三模型選擇。
保存紀錄顯示九組評估重算一致、三份原生權重重新載入一致。這是歷史驗證紀錄，並非本次重新評估。
目前正式推論仍使用原始 YOLO26s-seg → HRNet-W32/AP-10K → SAM 2.1 Tiny，固定 A_r100。

## Why the Original Weights Remain

微調實驗用來檢查訓練配方與動物分割表現，是否替換交付權重則需要整條量測流程的證據。
YOLO26s-seg 第二輪在同一組 fresh64 上由 61.81 AP 增至 62.10 AP，依未四捨五入值計算增加 0.28 點；
這是單 seed、小樣本的觀察，尚不足以證明穩定改善，也不能推論眼點、眼部遮罩或距離誤差會改善。

動物框改變會影響後續 HRNet 輸入與 SAM 提示區域。專案尚未完成替換微調權重後，
與原始權重在同一固定眼部參考上的端到端誤差、缺失率及部署回歸比較。
因此正式推論保留已有固定來源、雜湊與工程驗證的原始權重，讓交付結果可追溯；
這是目前證據範圍下的交付決定，不代表微調一定較差。

HRNet-W32／AP-10K 與 SAM 2.1 Tiny 也維持原始預訓練權重，本專案尚未訓練這兩個眼部模型。
後續若要替換模型或權重，需先確認資料切分與端到端驗收方式；本次交件不自動啟動新訓練。

## Dataset and Method

- 來源：COCO 2017 train2017，官方 `annotations_trainval2017.zip` 中的 `annotations/instances_train2017.json`。
- 候選：至少兩個非 crowd 動物；任一類含 crowd 的圖排除；保留同圖所有類別原始標註，不把人物刪成背景。
- seed42 稽核 768 個候選，651 通過、74 crowd 排除、43 缺圖；原始影像與解碼遮罩保存 SHA-256。
- 檔案／RGB 相同或 dHash Hamming distance≤2 作傳遞分組，每組取一圖；seed43 依序分 holdout／development／fitting。
- Fitting：256 圖／1,354 instances；development：64 圖／318 instances；第一輪 holdout：64 圖／377 instances。
- Fresh64：排除第一輪所有已用群組，按 group_id 排序後以 seed44 抽樣；352 instances。原官方 val2017 與眼部參考均未加入。
- SGD momentum0.9、weight decay0.0005、clip norm10、batch2、img640、3 epochs、seed42；無增強／warmup／EMA／梯度累積。
- YOLO 凍結前 10 blocks 與 DFL，BatchNorm running stats 固定、保留 80 類 head；Mask R-CNN 凍結 backbone、保留 91-channel head。
- Bitmap masks 經 nearest resize，可能遺失極小物體；第一輪記錄 10 個小實例在 160 grid 消失，不改寫原始 GT。
- 評估：官方 COCOeval segm AP@[.50:.95]、十動物類、maxDets 1/10/100、confidence floor .001、原圖遮罩；AP 表示為 0–100。
- 第一輪 YOLO11 為 MPS，其他為 CPU4；第二輪與評估都為 CPU4／FP32，不能把不同裝置耗時當架構排名。

只有單 seed 與小樣本，未估計信賴區間；小幅 AP 增加不證明顯著改善。
COCO train 可能已用於上游預訓練，fresh 指本專案尚未使用的群組，不是上游未見資料。
自動去重不保證場景獨立；動物 AP 也不代表眼睛定位／眼遮罩／距離誤差改善。
已檢閱的 holdout 不可在繼續調參後仍稱為新的獨立驗收。

## Environment

Python 3.11；Torch 2.9.1、TorchVision 0.24.1；YOLO11 使用 Ultralytics 8.3.230，YOLO26 使用 8.4.0。
獨立環境避免版本衝突；應用環境不需改動。以下建立新環境，不要覆蓋已有環境：

```sh
python3.11 -m venv .runtime/train11
.runtime/train11/bin/python -m pip install -r requirements/training-yolo11.txt
.runtime/train11/bin/python -m pip install --no-deps -e .
python3.11 -m venv .runtime/train26
.runtime/train26/bin/python -m pip install -r requirements/training-yolo26.txt
.runtime/train26/bin/python -m pip install --no-deps -e .
```

直接相依固定版本；這兩份訓練需求不是跨平台完整 transitive lock，不承諾逐位元重現。
原始訓練驗證平台為 macOS arm64；Linux／Docker 完整訓練尚未驗證。
權重由 `configs/training_models.toml` 記錄原來源與 SHA-256；須先放入 `weights/`。
訓練不自動下載資料或模型；缺檔／版本或 SHA 不符會停止。

## Execution

在專案根目錄執行。準備既有 `data/train2017/` 原圖與根目錄的 `annotations_trainval2017.zip`。
預設設定指向歷史輸出，會拒絕覆寫。重跑前複製兩份設定為 `.local.toml`，
只調整下列路徑，保留研究參數；模型設定仍指向同目錄的 `training_models.toml`：

| 設定 | 路徑欄位範例（相對 configs） |
| --- | --- |
| `segmentation_training_pilot.local.toml` | `output = "../outputs/training/reproduction/pilot"` |
| `segmentation_training_recovery.local.toml` | `output = "../outputs/training/reproduction/recovery/retrain"` |
| 同上 | `data_directory = "../outputs/training/reproduction/recovery/data"` |
| 同上 | `previous_output = "../outputs/training/reproduction/pilot"` |

```sh
.runtime/train11/bin/python -m animal_eye_metrology.research.training prepare --config configs/segmentation_training_pilot.local.toml
.runtime/train11/bin/python -m animal_eye_metrology.research.training run --config configs/segmentation_training_pilot.local.toml --model yolo11n_seg --device mps
.runtime/train26/bin/python -m animal_eye_metrology.research.training run --config configs/segmentation_training_pilot.local.toml --model yolo26n_seg --device cpu
.runtime/train26/bin/python -m animal_eye_metrology.research.training run --config configs/segmentation_training_pilot.local.toml --model yolo26s_seg --device cpu
.runtime/train26/bin/python -m animal_eye_metrology.research.training run --config configs/segmentation_training_pilot.local.toml --model maskrcnn_v2 --device cpu
.runtime/train26/bin/python -m animal_eye_metrology.research.training summarize --config configs/segmentation_training_pilot.local.toml
.runtime/train11/bin/python -m animal_eye_metrology.research.training_diagnosis --config configs/segmentation_training_pilot.local.toml --output outputs/training/reproduction/diagnosis
.runtime/train26/bin/python -m animal_eye_metrology.data.training_recovery_data --source outputs/training/reproduction/pilot/data --output outputs/training/reproduction/recovery/data
```

Linux 的 YOLO11 使用 `--device cpu`，需記錄與原實驗的裝置差異。
資料供給不同時 prepare 的候選與切分也可能不同，應先比對來源 manifest，不把新結果冒充歷史重現。
LR 診斷會凍結當前程式副本；原歷史診斷使用的 adapter 快照仍保留在本機原產物中。

以下先完成所有 `train`，才依序執行各模型 `finalize`：

```sh
.runtime/train11/bin/python -m animal_eye_metrology.research.training_recovery train --config configs/segmentation_training_recovery.local.toml --model yolo11n_seg
.runtime/train26/bin/python -m animal_eye_metrology.research.training_recovery train --config configs/segmentation_training_recovery.local.toml --model yolo26n_seg
.runtime/train26/bin/python -m animal_eye_metrology.research.training_recovery train --config configs/segmentation_training_recovery.local.toml --model yolo26s_seg
.runtime/train11/bin/python -m animal_eye_metrology.research.training_recovery finalize --config configs/segmentation_training_recovery.local.toml --model yolo11n_seg
.runtime/train26/bin/python -m animal_eye_metrology.research.training_recovery finalize --config configs/segmentation_training_recovery.local.toml --model yolo26n_seg
.runtime/train26/bin/python -m animal_eye_metrology.research.training_recovery finalize --config configs/segmentation_training_recovery.local.toml --model yolo26s_seg
.runtime/train11/bin/python scripts/export_training_weights.py --config configs/segmentation_training_recovery.local.toml --model yolo11n_seg
.runtime/train26/bin/python scripts/export_training_weights.py --config configs/segmentation_training_recovery.local.toml --model yolo26n_seg
.runtime/train26/bin/python scripts/export_training_weights.py --config configs/segmentation_training_recovery.local.toml --model yolo26s_seg
```

`best.pt`／`last.pt` 是 state_dict＋optimizer 的研究格式；`native_weights.pt` 是原生推論格式。
匯出會逐張量核對，拒絕覆寫；不會切換應用權重。

## Published Evidence and Validation

[來源與檔案雜湊](../examples/training/manifest.json) 記錄原檔 SHA-256、公開副本 SHA-256 與轉換。
公開 JSON 僅把本機根目錄絕對路徑轉成相對路徑；數值未改。

- [第一輪比較](../examples/training/pilot_comparison.csv)
- [第二輪同圖比較](../examples/training/recovery_comparison.csv)
- [學習率對照](../examples/training/lr_diagnosis.json)
- [歷史評估重算驗證](../examples/training/evaluation_verification.json)

`examples/training/` 另含三個 YOLO 的 epoch loss／development AP 與原生匯出驗證。
完整 GT、權重、逐步日誌、預測與原始 provenance 保留於本機 `outputs/training/`，不納入 Git。
公開紀錄可獨立展示；完整評估重算需取得原預測／標註，公開摘要本身不足以重算 AP。

```sh
.runtime/app/bin/python -m pytest -q
.runtime/app/bin/ruff check src tests app scripts
.runtime/app/bin/ruff format --check src tests app scripts
.runtime/app/bin/python scripts/check_delivery.py
```

測試使用合成資料，覆蓋資料身份、互斥切分、LR-only 約束、選模隔離、目標轉換、權重匯出與紀錄完整性。
輕量 CI 缺少模型相依時會跳過對應測試；完整應用環境執行模型相關合成測試，不下載或訓練真實模型。
