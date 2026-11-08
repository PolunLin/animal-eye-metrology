# Documentation Index

本目錄只保留理解、執行與驗證專案所需的七份文件。
Phase 紀錄、研究提案、Agent 筆記、執行報告與歷史快照屬本機工作資料，不納入 Git。

## Project Guides

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

新使用者先閱讀 README，再依 Setup 或 Docker 啟動。模型選擇、人工參考定義、距離公式及
評估界線統一放在 Methodology，避免同一規則散落在多份階段報告。

## Validation

環境與自動測試命令見 [Setup](setup.md#validation)，容器與實際模型流程見
[Docker](docker.md#validation-status)。既有 [CSV 範例](../examples/README.md) 可在沒有模型時
驗算端點距離，但不是獨立準確率評估。

完整 COCO、權重、人工參考與研究 outputs 不提交 Git。缺少本機研究產物時，
使用新圖推論或 CSV 模式；程式不會補造原圖、標註或實驗結果。
