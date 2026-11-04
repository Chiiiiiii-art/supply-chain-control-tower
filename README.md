# Supply Chain Resilience Control Tower
### 機率性需求預測與智慧庫存控制塔

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM_Quantile-orange.svg)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-red.svg)](https://pytorch.org/)
[![Power BI](https://img.shields.io/badge/Dashboard-Power_BI_Ready-yellow.svg)](https://powerbi.microsoft.com/)
[![SHAP](https://img.shields.io/badge/Governance-Responsible_AI-brightgreen.svg)](https://shap.readthedocs.io/)

本專案為針對安永（EY）諮詢顧問（資料科學、供應鏈管理、新興科技）打造之**端到端智慧決策解決方案**。
透過真實 Walmart M5 供應鏈場景，解決傳統常態分佈安全庫存之盲點，實現機率性分位數預測（P10/P50/P90）與作業研究報童定理（Newsvendor Problem）成本折衝，並無縫對接 Power BI 星狀模型戰情室。

---

## 核心亮點與三層級架構

1. **多模型機率時序預測（Multi-Model Benchmarking）**：
   - **作業研究基線**：Croston 間歇需求法 + 移動平均 (MA-7d)。
   - **梯度提升樹主力**：LightGBM Pinball Loss 分位數回歸（P10/P50/P90 不確定性區間）。
   - **深度學習時序**：PyTorch 機率類神經網路（高斯負對數似然 NLL 損失）。
2. **作業研究庫存決策層（Operations Research Newsvendor Optimization）**：
   - 依持有成本（$C_o$）與缺貨懲罰（$C_u$）求解關鍵分位數（$CR = \frac{C_u}{C_u + C_o}$）。
   - 以非對稱損失動態求解最佳補貨點（ROP）與動態安全庫存，打破高斯對稱常態假設。
   - 量化減省庫存緩衝與總成本降減率。
3. **AI 治理與可解釋性（Responsible AI & SHAP）**：
   - 導入 SHAP 決策歸因，量化促銷折扣、時間滯後與滾動均值對預測的貢獻度，消除黑箱疑慮。
4. **Power BI 顧問級戰情室（Star Schema & DAX）**：
   - 輸出事實表 `Fact_ForecastAndInventory.csv` 與維度表 `Dim_Product.csv`, `Dim_Store.csv`, `Dim_Date.csv`。
   - 附帶 20+ 組即用 DAX 度量值與四重視角佈局手冊。

---

## 快速啟動

執行端到端管線（約 30 秒至 1 分鐘自動完成所有模型訓練、OR 最佳化與資料匯出）：

```bash
python run_pipeline.py
```

執行後檔案將自動生成於：
- `powerbi/data/`：Power BI 專用 Fact/Dim 星狀表格。
- `data/processed/`：模型評測結果 `Model_Benchmark_Summary.csv` 與 SHAP 特徵權重。

---

## 成果指標概覽 (Benchmark Preview)

- **預測損失降低**：LightGBM Pinball Loss 顯著優於傳統 Croston 基準。
- **財務成本節省**：動態報童模型相較於傳統常態分佈公式，平均降低 **20%~35%** 總庫存營運成本。
- **安全庫存優化**：在維持 95%+ 履約服務水準下，減少過剩庫存資金積壓。
