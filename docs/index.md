# Documentation Index

## Review and Run

| 文件 | 用途 |
| --- | --- |
| [README](../README.md) | 能力、技術棧、交付入口與限制 |
| [Setup](setup.md) | 新環境、模型準備、本機啟動與測試 |
| [Docker](docker.md) | 容器目標、掛載、命令與驗證狀態 |
| [API](api.md) | HTTP 介面、請求／回應與操作範例 |
| [Notebook Demo](notebook_demo.md) | Jupyter 展示、重播／現場推論、幾何驗算與重現命令 |
| [Architecture](architecture.md) | 架構圖、執行隔離與模組責任 |
| [Methodology](methodology.md) | 模型理由、GT、公式、評估與不確定度 |
| [CSV Example](../examples/README.md) | 既有小型輸出、欄位與來源核對 |
| [Delivery Validation](delivery_validation.md) | 本次基線、回歸、產物保護、新環境與容器驗證紀錄 |

## Research Records

研究文件保留各階段的條件與當時狀態；早期「尚未實作」不覆蓋後來完成的功能。
目前啟動以 Setup／Docker 為準；歷史命令仍對應原環境與凍結產物。

| 階段 | 規範與結果 |
| --- | --- |
| 資料選圖 | [Dataset Selection](dataset_selection.md) |
| 動物分割 | [Specification](phase2_instance_segmentation.md)、[原始 POC](phase2_poc_results.md)、[100-image Pilot](phase2_pilot100_results.md)、[YOLO26s Selected Images](phase2_yolo26s_selected_results.md) |
| 人工眼部標註 | [v1 Proposal](phase3a_eye_annotation_protocol.md)、[Annotation Tool](phase3b_annotation_tool.md)、[v2 Protocol](eye_annotation_protocol_v2.md)、[Qualification Review](eye_annotation_v2_review.md) |
| 人工量測／重標 | [Manual Baseline](manual_measurement_baseline.md)、[Repeat Check](blind_repeat_check.md)、[Repeat Results](blind_repeat_results.md) |
| 眼睛分割 | [CPU Compatibility](eye_segmentation_compatibility.md)、[Fixed-Prompt Comparison](eye_segmentation_comparison_results.md) |
| 自動提示／ROI | [Automatic Localization](phase3c_auto_eye_localization_poc_results.md)、[Prompt/ROI Ablation](phase3c_prompt_roi_ablation_results.md)、[Automatic Eye Boxes](phase3c_auto_eye_box_results.md) |
| 定義敏感度 | [Frozen v1/v2 Comparison](eye_definition_v2_comparison_results.md) |
| 新圖應用歷史 | [New Image Inference](new_image_inference.md) |
| 整理前 README | [Research History](research_history.md) |

`research_history.md` 是整理前 README 的完整原始快照；相對連結保留原先以儲存庫根目錄為起點的寫法。
操作入口請使用本索引，不將快照中的當時狀態視為最新能力。
完整 COCO、權重與研究 outputs 不提交 Git，乾淨 checkout 的歷史產物連結可能不存在。
可直接檢視的數值例子在 `examples/`，不冒充完整實驗資料包。
