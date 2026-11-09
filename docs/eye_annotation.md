# Eye Annotation and Review

目前頁面僅提供 **eye-visible-eyelid-v2**，不提供 v1 選項。
本次恢復人工標註工具，不啟動訓練、不替換推論權重、不改動距離公式。

## Mask Definition

標註完整「可見眼裂內的眼球表面＋直接相鄰且可辨識的眼瞼邊緣」。
包含可見的虹膜、瞳孔、鞏膜與目標內反光；排除毛髮、睫毛、淚痕與眼周色塊。
不把整片眼周皮膚算作眼睛，也不補出被遮擋或看不見的部分。

| 情況 | 操作 |
| --- | --- |
| 完整可見 | 描繪符合 v2 的完整區域，確認動物自身左右 |
| 部分遮擋／半閉眼 | 只畫可見部分，標 Partial；量測品質 Fail 或 Unknown |
| 看不到 | NotVisible，不畫猜測的遮罩 |
| 外緣不清／極小／模糊 | Uncertain，保存原因；不要為通過審核而擴張輪廓 |
| 孔洞／多區塊 | 可加入輪廓、扣除遮擋；不自動補洞、平滑或連接 |

左右採動物自身解剖側別，不是畫面左右。原圖資訊是上限，放大不增加細節。

## Centers and Measurement

目前距離量測使用遮罩像素重心：所有前景像素中心座標取平均，像素中心為 `(column+.5,row+.5)`。
頁面顯示的是當前完成輪廓的重心預覽，草稿不等於已審核參考。
使用者已確認中心點統一採遮罩重心，不需另外點眼裂或瞳孔中心。
後續 HRNet 的眼點標籤將由經審核的 v2 遮罩衍生，須記錄遮罩版本與來源；
這與 AP-10K 原眼點標的不能直接視為相同，也不能宣稱是瞳孔定位。
本頁目前只保存遮罩及顯示重心，尚未執行訓練資料匯出或微調。

## Workflow

1. 開啟「眼睛標註與審核」，選圖片、動物實例及解剖眼側。
2. 展開完整原圖確認身份，使用「框選放大」查看眼部。
3. 逐點描繪，至少三點後按「完成輪廓」。可拖曳頂點、復原、加入或扣除區域。
4. 保存草稿；逐項填寫可見性、側別確認、審核結果、量測品質與實際 Reviewer。
5. 符合規範才勾選人工確認並儲存審核；Accepted 不等於可量測，Pass 還要求 Visible 與確認左右。
6. 切換眼側／圖片；關閉頁面前保存。CSV、完整尺寸 PNG 與修訂 JSON 保留在工作區。

不預填 Reviewer、Accepted 或 Pass。不確定、拒絕與失敗的樣本應保留，不能靜默刪掉改善成績。
每次保存寫入新 revision；不同視窗同時修改會拒絕過期版本，必須重新載入再人工整合。

## Data Preparation

上傳圖片執行推論只需 README 的模型環境與一張 JPEG／PNG，不需要完整 COCO 或人工標註工作區。
下列準備只適用於「眼睛標註與審核」；請使用已取得並核對來源的 COCO 原圖與官方實例標註。

| 用途 | 必要資料 |
| --- | --- |
| 上傳推論示範 | 自備圖片，或 README 的 COCO `125405` 示範圖 |
| 新增 val2017 標註批次 | `data/coco/val2017/` 原圖及 `data/coco/annotations/instances_val2017.json` |
| 新增 train2017 標註批次 | `data/train2017/` 原圖及 `data/coco/annotations/instances_train2017.json` |
| 預設三圖重標工作區 | `configs/phase1.toml` 指定的 val 原圖／JSON，以及既有 `outputs/phase3/phase3a/proposed_instances.csv` |

新批次會建立自己的 proposal；不需要預設三圖工作區先準備成功。
只有 README 的一張示範圖不足以建立預設 20 張新批次，且三張既有開發圖會從新批次候選排除。
請先確認本機資料足夠，再於頁面指定已確認的來源與張數；此步驟不會下載圖片或分配訓練／驗收用途。

若官方標註仍保存在專案根目錄的 `annotations_trainval2017.zip`，可從專案根目錄執行以下命令，
僅將選定來源的原始 JSON bytes 複製到資料目錄。範例使用 `val2017`；需要 train 時將參數改為 `train2017`。
目標已存在會拒絕覆寫；已有 JSON 時直接保留並核對來源即可，不必重跑。

```sh
python3.11 - val2017 <<'PY'
import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile

split = sys.argv[1]
if split not in {"val2017", "train2017"}:
    raise SystemExit("Choose val2017 or train2017.")
target = Path("data/coco/annotations") / f"instances_{split}.json"
if target.exists():
    raise SystemExit(f"Refusing to overwrite: {target}")
with ZipFile("annotations_trainval2017.zip") as archive:
    payload = archive.read(f"annotations/instances_{split}.json")
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("xb") as handle:
    handle.write(payload)
print(f"{target}: SHA-256 {hashlib.sha256(payload).hexdigest()}")
PY
```

Docker Compose 只將主機 `data/` 掛載至容器 `/app/data`，不掛載根目錄 ZIP；因此使用 Docker 標註前，
先在主機完成上述 JSON 準備，並依表格放置原圖。若 `.env` 自訂 `DATA_DIRECTORY`，
請在該目錄保留相同相對結構。`outputs/` 掛載保存批次與審核修訂。
這些步驟補齊來源位置；主推論的 Docker smoke 檢查不代表所有人工標註互動均已完成容器驗收。

## Workspace and New Batches

預設設定 `configs/annotation_workspace.toml` 使用三張既有開發圖（狗／熊／長頸鹿，6 動物／12 眼槽位），
輸出在獨立的 `outputs/phase3/annotation_workspace_v2_centroid/`，不覆寫原 `eye_annotation_v2` 的凍結標註。
新工作區從空白人工標註開始，不複製模型輪廓或既有 GT。這批示範圖不自動納入新訓練或獨立驗收。

**乾淨 clone 不包含上述 proposal CSV 或歷史工作區。** 以下 `--prepare` 命令僅適用於已備妥三圖來源與
既有 proposal 的環境，不能單靠命令重建缺少的來源清單。首次使用可依上節備妥本機資料後，
使用頁面的「新增標註批次」建立獨立工作區；預設工作區的缺檔提示不會阻止此操作。

```sh
.runtime/app/bin/python -m animal_eye_metrology.annotation.annotation_session --config configs/annotation_workspace.toml --prepare
.runtime/app/bin/python -m animal_eye_metrology.annotation.annotation_session --config configs/annotation_workspace.toml --status
```

`--prepare` 拒絕覆寫；已準備的工作區直接開啟或使用 `--status`。
設定與規範一旦凍結不能原地更改。原圖、來源標註及準備 manifest 皆核對 SHA-256。
使用預設三圖工作區時，需自行提供已核准的原圖、COCO 實例標註及 proposal CSV；此工具不下載資料。

新增批次可直接在頁面操作：展開「新增標註批次：選擇 train／val」，選來源、照片數（1–100）與 Seed，按「建立待標註批次」。
預設來源 val2017、20 張、seed42；僅按下建立後才抽取。建立後可從「目前標註批次」重新開啟。

系統讀取本機 JSON（不存在時讀既有 `annotations_trainval2017.zip`），以至少兩個非 crowd 動物為候選，
按 image ID 排序再以指定 seed 打亂，核對影像解碼、尺寸、SHA-256。
排除已建立批次的同來源 image ID、完全相同圖片檔案及三張既有 val 示範圖；近重複／同場景仍待人工審查。
保留選中圖片的所有原始 COCO annotations，逐隻非 crowd 動物建立左右眼待標註槽位。

產物存於 `outputs/phase3/eye_annotation_batches/`，包含獨立來源、圖片副本、設定、排除原因、程式 SHA、seed 與選取順序。
所有新眼睛保持未標註，`dataset_role=unassigned`，沒有自動訓練／驗收切分。
COCO val 可用於微調，但同圖不能再作該次模型的獨立驗收。選來源本身不代表已確認新訓練或評估策略。

也可建立自訂資料設定、proposal CSV 與 session TOML，透過 `AEM_ANNOTATION_SESSION` 指定：

```sh
AEM_ANNOTATION_SESSION=configs/annotation_new.local.toml \
  .runtime/app/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8517
```

proposal 欄位：`image_id,annotation_id,animal_category,file_name,image_width,image_height,coco_bbox_xywh,proposed_eye_sides,image_sha256`。
動物框與身份須符合來源 COCO JSON；`proposed_eye_sides` 為 JSON `["left","right"]`，只代表待標註槽位。
目前不支援任意圖片直接上傳後自動建立動物身份，也不會將預測框冒充原始 GT。
批次圖片、物種、訓練／開發／驗收切分需另行確認，避免已看過的開發圖流入獨立驗收。

## Training Boundary

使用者已選擇分階段訓練 HRNet 眼點定位與 SAM 眼部分割。
v2 遮罩與遮罩重心標的已確認。實際微調仍需新資料批次、切分與訓練設定。
標註保存不會自動啟動訓練，也不自動更換模型。
