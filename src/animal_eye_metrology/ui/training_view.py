"""Read-only presentation of recorded supervised fine-tuning evidence."""

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st


def load_records(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Verify published evidence before presenting it as a historical result."""
    directory = root / "examples/training"
    manifest = json.loads((directory / "manifest.json").read_text())
    for record in manifest["records"]:
        path = directory / record["file"]
        if (
            path.parent != directory
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError("Published training evidence failed its integrity check.")
    tables = []
    for name in ("pilot_comparison.csv", "recovery_comparison.csv"):
        with (directory / name).open(newline="", encoding="utf-8") as handle:
            tables.append(list(csv.DictReader(handle)))
    return tables[0], tables[1]


def expanded_training_view(job: dict[str, Any], selection: dict[str, Any]) -> bytes:
    """Present the frozen expanded experiment and export the displayed AP comparison."""
    result = job["result"]
    counts = job["split_counts"]
    st.subheader("最新大型訓練：YOLO26s-seg")
    st.caption("2026-11-09 完成 · COCO train2017 · 保存結果")
    a, b, c = st.columns(3)
    a.metric("Fitting", f"{counts['train']:,} 張")
    b.metric("Development", f"{counts['development']:,} 張")
    c.metric("Holdout", f"{counts['final_holdout']:,} 張")
    st.caption(
        f"{job['epochs']} epochs、{result['optimizer_steps']:,} 次更新；"
        "seed 42、batch 2、LR 0.00001；MPS 訓練、CPU 評估。"
    )
    baseline_dev = result["baseline_development_mask_ap"] * 100
    trained_dev = result["best_development_mask_ap"] * 100
    baseline_holdout = result["baseline_holdout_mask_ap"] * 100
    trained_holdout = result["finetuned_holdout_mask_ap"] * 100
    rows = [
        {
            "權重": "原始預訓練",
            "Development AP": baseline_dev,
            "Holdout AP": baseline_holdout,
        },
        {
            "權重": f"微調最佳 epoch {result['best_epoch']}",
            "Development AP": trained_dev,
            "Holdout AP": trained_holdout,
        },
        {
            "權重": "微調 − 原始（AP 點）",
            "Development AP": trained_dev - baseline_dev,
            "Holdout AP": trained_holdout - baseline_holdout,
        },
    ]
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            name: st.column_config.NumberColumn(name, format="%.2f")
            for name in ("Development AP", "Holdout AP")
        },
    )
    st.caption(
        "十動物類 COCO Mask AP@[.50:.95]（0–100）。各欄使用同一組圖片比較兩份權重；"
        "development 供選模，holdout 在選模凍結後評估。差值為 AP 點。"
    )
    st.subheader("為什麼最後保留原始權重？")
    if (
        selection["selected_candidate"] != "pretrained"
        or selection["holdout_used_for_selection"] is not False
        or selection["status"] != "frozen_before_final_holdout_prediction"
    ):
        raise ValueError("Expanded experiment does not match the documented model decision.")
    st.info(
        f"微調最佳 development AP 為 {trained_dev:.2f}，低於原始模型的 {baseline_dev:.2f}。"
        "因此在 holdout 推論前就凍結選擇，保留原始權重；沒有用 holdout 回頭選模型。"
    )
    st.write(
        f"凍結後，同組 holdout 的原始模型 AP 為 {baseline_holdout:.2f}、"
        f"微調為 {trained_holdout:.2f}（差 {trained_holdout - baseline_holdout:+.2f} AP 點），"
        "也未顯示改善。微調候選已匯出保存，但沒有替換正式推論權重。"
    )
    st.write(
        "正式推論維持原始 YOLO26s-seg → HRNet-W32／AP-10K → SAM 2.1 Tiny、A_r100。"
        "替換微調權重後的端到端眼距比較尚未完成。"
    )
    st.warning(
        "本輪只有單一 seed、3 epochs，未估計信賴區間。與早期實驗的資料切分、更新次數及"
        "訓練裝置不同，不能直接混比 AP，也不能將差異單獨歸因於資料量。"
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def training_workspace(root: Path) -> None:
    """Show archived results without loading models, private outputs or training workers."""
    st.header("訓練與實驗")
    st.caption("2026-11-08 小樣本實驗／2026-11-09 大型訓練 · 動物實例分割微調")
    st.info("目前推論維持原始預訓練權重。HRNet／SAM 未在這段實驗中訓練。")
    try:
        pilot, recovery = load_records(root)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"無法讀取已驗證的實驗紀錄：{error}")
        return
    phase = st.radio(
        "實驗階段",
        ["最新大型訓練", "第二輪微調", "第一輪微調", "學習率診斷"],
        horizontal=True,
    )
    directory = root / "examples/training"
    if phase != "最新大型訓練":
        st.markdown(
            "**早期小樣本：第一輪微調 → AP 下降 → development 學習率對照 → 第二輪微調 → 驗收**"
        )
        a, b, c = st.columns(3)
        a.metric("Fitting", "256 張")
        b.metric("Development", "64 張")
        c.metric("第二輪 Fresh holdout", "64 張")
        st.caption("COCO train2017，多動物影像；3 epochs、batch 2、每模型 384 次更新、seed 42。")
    if phase == "最新大型訓練":
        try:
            job = json.loads((directory / "expanded_yolo26s_job_status.json").read_text())
            selection = json.loads((directory / "expanded_yolo26s_selection.json").read_text())
            download_data = expanded_training_view(job, selection)
        except (OSError, ValueError, KeyError) as error:
            st.error(f"無法讀取已驗證的實驗紀錄：{error}")
            return
        filename = "expanded_yolo26s_comparison.csv"
    elif phase == "第二輪微調":
        st.subheader("同一組 fresh64：訓練前後比較")
        rows: list[dict[str, Any]] = [
            {
                "模型": row["label"],
                "預訓練 AP": float(row["fresh_pretrained_ap"]) * 100,
                "第一輪 AP": float(row["fresh_previous_v1_ap"]) * 100,
                "第二輪 AP": float(row["fresh_retrained_v2_ap"]) * 100,
                "相對預訓練差值（點）": float(row["change_vs_pretrained_ap_points"]),
            }
            for row in recovery
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        selected = st.selectbox("查看訓練曲線", [row["model"] for row in recovery], index=2)
        epochs = json.loads((directory / f"{selected}_epochs.json").read_text())
        left, right = st.columns(2)
        with left:
            st.caption("訓練 mean loss（僅在同模型內解讀）")
            st.line_chart(
                [{"Epoch": e["epoch"], "Mean loss": e["mean_loss"]} for e in epochs],
                x="Epoch",
                y="Mean loss",
            )
        with right:
            st.caption("Development Mask AP（0–100）")
            st.line_chart(
                [{"Epoch": e["epoch"], "Mask AP": e["development_mask_ap"] * 100} for e in epochs],
                x="Epoch",
                y="Mask AP",
            )
        filename = "recovery_comparison.csv"
    elif phase == "第一輪微調":
        st.subheader("第一輪 internal holdout：64 張")
        st.dataframe(
            [
                {
                    "模型": r["model"],
                    "預訓練 AP": float(r["baseline_holdout_mask_ap"]) * 100,
                    "微調 AP": float(r["finetuned_holdout_mask_ap"]) * 100,
                    "最佳 Epoch": int(r["best_epoch"]),
                }
                for r in pilot
            ],
            hide_index=True,
            width="stretch",
        )
        st.warning("此處與第二輪使用不同的 64 張影像，兩張表的分數不能直接混比。")
        filename = "pilot_comparison.csv"
    else:
        st.subheader("只改學習率的 development 對照")
        diagnosis = json.loads((directory / "lr_diagnosis.json").read_text())
        st.caption("YOLO11n-seg、CPU、1 epoch；只用 fitting／development，未用 holdout 調參。")
        st.dataframe(
            [
                {
                    "Learning rate": r["learning_rate"],
                    "Development AP": r["development_mask_ap"] * 100,
                }
                for r in diagnosis["candidate_results"].values()
            ],
            hide_index=True,
            width="stretch",
        )
        st.write("後續固定採用 LR 0.00001，其他數值訓練超參數保持不變。")
        filename = "lr_diagnosis.json"
    if phase != "最新大型訓練":
        download_data = (directory / filename).read_bytes()
    st.download_button(
        "下載本階段紀錄",
        download_data,
        file_name=filename,
        mime="application/json" if filename.endswith("json") else "text/csv",
    )
    st.warning(
        "Mask AP 是動物實例分割指標，不是眼距準確率；"
        "COCO train 可能與上游預訓練資料重疊，不能稱為完全未見資料的泛化測試。"
    )
    with st.expander("方法、重現命令與紀錄來源"):
        guide = (root / "docs/training.md").read_text()
        st.markdown(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", guide))
        st.download_button("下載訓練說明", guide, file_name="training.md", mime="text/markdown")
