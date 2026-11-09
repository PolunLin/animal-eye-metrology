"""Read-only presentation of recorded supervised fine-tuning evidence."""

import csv
import hashlib
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


def training_workspace(root: Path) -> None:
    """Show archived results without loading models, private outputs or training workers."""
    st.header("訓練與實驗")
    st.caption("2026-11-08 保存的實驗紀錄 · 動物實例分割微調")
    st.info("目前推論維持原始預訓練權重。HRNet／SAM 未在這段實驗中訓練。")
    try:
        pilot, recovery = load_records(root)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"無法讀取已驗證的實驗紀錄：{error}")
        return
    st.markdown(
        "**第一輪微調 → 發現 AP 下降 → development 學習率對照 → 第二輪微調 → 固定選模後驗收**"
    )
    a, b, c = st.columns(3)
    a.metric("Fitting", "256 張")
    b.metric("Development", "64 張")
    c.metric("第二輪 Fresh holdout", "64 張")
    st.caption("COCO train2017，多動物影像；3 epochs、batch 2、每模型 384 次更新、seed 42。")
    phase = st.radio("實驗階段", ["第二輪微調", "第一輪微調", "學習率診斷"], horizontal=True)
    directory = root / "examples/training"
    if phase == "第二輪微調":
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
    st.download_button(
        "下載本階段紀錄",
        (directory / filename).read_bytes(),
        file_name=filename,
        mime="application/json" if filename.endswith("json") else "text/csv",
    )
    st.warning(
        "Mask AP 是動物實例分割指標，不是眼距準確率。單 seed、小樣本，未估計信賴區間；"
        "COCO train 可能與上游預訓練資料重疊，不能稱為完全未見資料的泛化測試。"
    )
    with st.expander("方法、重現命令與紀錄來源"):
        guide = (root / "docs/training.md").read_text()
        st.markdown(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", guide))
        st.download_button("下載訓練說明", guide, file_name="training.md", mime="text/markdown")
