# Notebook Demo

[`notebooks/demo.ipynb`](../notebooks/demo.ipynb) 直接呼叫專案既有 Python 程式，逐步展示動物實例分割、
眼點定位、眼部分割與重心距離。預設重播明確指定的歷史結果；只有選擇 `live` 才執行模型。
Notebook 不需要先啟動 API 或 Streamlit。

## Start

在專案根目錄建立獨立 Python 3.11 Notebook 環境並安裝已鎖定的依賴：

```sh
uv venv --python 3.11 .runtime/notebook
uv pip install --python .runtime/notebook/bin/python -r requirements/notebook.txt
.runtime/notebook/bin/python -m jupyterlab notebooks/demo.ipynb
```

選擇這個環境的 Python kernel，依序執行所有儲存格。`requirements/notebook.in` 是直接依賴來源，
`requirements/notebook.txt` 是版本鎖定；不要為了 Notebook 升級既有模型環境。
Notebook 使用自己的輕量環境；`live` 的三個模型仍由設定檔指定的 app／pose 子程序執行。
模型環境與權重準備方式見 [Local Setup](setup.md)，Notebook 本身不下載圖片、資料集或權重。

## Modes and Configuration

修改 Notebook 開頭的設定儲存格，再重新依序執行：

| 設定 | 預設值／用途 |
| --- | --- |
| `MODE` | `"replay"`；也可選 `"live"` 或 `"csv"` |
| `REPLAY_RUN` | `ROOT / "outputs/inference/20261108T075732Z_92c42f74bb73"`；固定既有 run，可手動指定其他完整 run |
| `IMAGE_PATH` | `ROOT / "data/coco/val2017/000000125405.jpg"`；`live` 的本機 JPEG／PNG，可替換 |
| `CONFIG_PATH` | 預設 `ROOT / "configs/inference.runtime.toml"`，可由 `AEM_INFERENCE_CONFIG` 指定 |

| 模式 | 讀取資料 | 執行模型 | 缺檔行為與證據界線 |
| --- | --- | --- | --- |
| `replay` | 指定 run 的結果、原圖、遮罩與產物 | 否 | 缺少 `result.json` 時清楚告知，退回已提交的 CSV；歷史結果不代表現場推論成功 |
| `live` | 指定本機圖片與推論設定 | 是 | 先檢查本機依賴；缺失或模型失敗會顯示原因，不以歷史結果冒充本次結果 |
| `csv` | `examples/measurements.csv` | 否 | 僅展示有來源紀錄的距離與公式驗算；沒有原圖或遮罩，不能展示分割影像 |

Git 不附原圖、權重與完整歷史 run，因此乾淨 checkout 的預設展示可退回 CSV。
CSV 來源與固定 SHA-256 見 [Recorded Measurement CSV](../examples/README.md)。
若使用歷史 `.venv`／`.venv-eye-pose` 環境，請明確將 `CONFIG_PATH` 改為
`ROOT / "configs/inference.toml"`，或在啟動 JupyterLab 前設定：

```sh
AEM_INFERENCE_CONFIG=configs/inference.toml \
  .runtime/notebook/bin/python -m jupyterlab notebooks/demo.ipynb
```

不會自動切換設定或模型環境。`live` 每次執行推論儲存格都會建立新的 run；
與 API／CLI／Streamlit 共用輸出目錄時沿用既有互斥鎖，忙碌時須等待其他推論完成。

## Demo Flow and Reproduction

展示順序為環境與來源 → 輸入／模式 → 原圖與動物輪廓 → 眼部疊圖及 ROI 放大 →
逐眼狀態 → 全部距離列 → 公式、遮罩重心與來源雜湊檢查。
完整 run 缺少部分產物時，應依 Notebook 顯示的缺失原因解讀，不從缺失影像推定模型成功。

Notebook 直接沿用 `inference_pipeline.run_inference`／`load_result`、既有視覺化及幾何函式，
不複製或更改模型與量測方法。完整推論保存輸入、設定、執行命令、模型版本、日誌與
`run_manifest.json`；CPU、seed 42、4 threads 等設定由選用的原設定檔控制。
重播的版本與設定應以該 run 的紀錄為準，不能用目前環境資訊回填歷史紀錄。

可在沒有互動介面的環境執行整本 Notebook；以下沿用 Notebook 中已儲存的模式，
預設為 `replay`，不強制切換成 `live`：

```sh
mkdir -p outputs/notebook_demo
.runtime/notebook/bin/python -m jupyter nbconvert \
  --to notebook --execute notebooks/demo.ipynb \
  --output demo.executed.ipynb --output-dir outputs/notebook_demo \
  --ExecutePreprocessor.timeout=3600
```

檢視 `outputs/notebook_demo/demo.executed.ipynb` 中的模式、來源與檢查結果。
執行產物保留在已忽略的 `outputs/`，不要提交含執行輸出的 Notebook；
版本控制中的 `notebooks/demo.ipynb` 應保留空輸出。

## Measurement and Validation Limits

| 項目 | 解讀方式 |
| --- | --- |
| 座標與距離 | 原圖二值眼遮罩前景的像素中心平均，座標為 `(column + 0.5, row + 0.5)`；重心間歐氏距離，單位 px |
| 兩種距離 | 同隻動物左右眼，以及同圖不同動物的右眼配對；不跨圖片配對 |
| 左右眼 | 沿用模型解剖通道，不按畫面位置重排；可見性與左右資格仍未驗證 |
| 狀態與缺失 | 顯示所有狀態與失敗列，保留 `unavailable`、空值及原因；不補零或只呈現成功案例 |
| 候選量測 | `candidate` 只代表兩個重心可計算，`qualification` 仍為 `unverified`；不是瞳孔距離或實體長度 |
| 公式／雜湊檢查 | 驗證保存資料的幾何一致性與完整性；不證明眼點、分割邊界或量測準確 |
| GT 與不確定性 | 新圖沒有人工參考，不產生 IoU／Dice、距離誤差、校準誤差或信賴區間；同圖 ROI 條件不是獨立樣本 |
| 鳥類 | 新版結果保留 `experimental`／`pose_domain`；哺乳動物姿態模型的域外輸出仍是試驗性結果 |

舊 CSV 可能沒有新版域外欄位，應保留原 schema，不將欄位缺失解讀為已通過驗證。
流程 `completed`、有限距離或非空遮罩均不能代替準確度評估。
模型與產物說明見 [New Image Inference](new_image_inference.md)；新的資料選擇、
GT、評估或量測方法仍須另行提案並經專案負責人確認。

## Local Validation — 2026-11-08

以下是本次本機觀察，不代表新環境安裝驗收或模型準確度評估。
使用 Python 3.11.14、JupyterLab 4.6.6 與獨立 Notebook 環境，以真正的 Jupyter kernel 執行。

| 檢查 | 結果 |
| --- | --- |
| Notebook 格式／語法 | 通過；19 個儲存格，提交版本沒有執行輸出 |
| `replay` | 指定原 run 完整執行；6 筆既有距離、13 張展示影像；公式、遮罩重心與 manifest 雜湊一致 |
| `csv` | 從 `notebooks/` 執行通過；6 筆既有距離與固定 CSV 雜湊一致 |
| 缺 run 退回 | 明確顯示退回 CSV，執行通過；沒有模型或圖像輸出 |
| `live`：`configs/inference.toml` | 既有本機環境完成完整流程；run `20261108T094616Z_c8d8b2cc0d45`，3 筆候選距離、12 張展示影像，幾何與產物雜湊一致 |
| `live`：`configs/inference.runtime.toml` | 本次動物分割完成，但新 pose 環境缺少 `mmpose`，眼點定位失敗；run `20261108T094302Z_30edae7d8e34` 保留失敗狀態與空距離，未當作成功 |
| 原產物保護 | 重播 run 的檔案雜湊在測試前後相同，Notebook 原檔仍為空輸出 |
| 套件相容性 | 獨立 Notebook 環境的 94 個套件通過 `uv pip check` |
| 相關自動測試 | 59 passed；只驗證幾何與流程行為，不評估模型準確度 |
| Ruff／交付檢查 | Notebook lint、格式與 `check_delivery.py` 通過 |

`preflight()` 檢查檔案及權重雜湊，不保證子程序套件能匯入；新 pose 環境需完成原有安裝流程。
本機現場 Demo 可明確選擇已驗證的歷史設定，啟動命令見上文，Notebook 不會自行切換環境。
執行後的 Notebook 保存在 `outputs/notebook_demo/demo.{replay,csv,fallback,live}.executed.ipynb`；
完整現場結果保存在對應 `outputs/inference/<run_id>/`，均未納入 Git。

相關檢查命令：

```sh
.venv/bin/python -m pytest -q tests/test_metrology.py tests/test_inference_artifacts.py tests/test_inference_pipeline.py
.venv/bin/ruff check notebooks/demo.ipynb
.venv/bin/ruff format --check notebooks/demo.ipynb
.venv/bin/python scripts/check_delivery.py
uv pip check --offline --cache-dir .runtime/cache/uv-notebook --python .runtime/notebook/bin/python
```
