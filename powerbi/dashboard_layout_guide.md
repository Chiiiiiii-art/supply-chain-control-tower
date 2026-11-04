# Power BI 智慧供應鏈控制塔：儀表板視覺化配置指引手冊
## Supply Chain Resilience Control Tower Visual Blueprint

本指引手冊專為 Power BI Desktop 設計，將後端產出的星狀模型（Star Schema）轉化為高階主管與生管決策者能直觀操作的顧問級戰情室。

---

## 一、 資料模型關聯設定（Model Relationship Setup）

在 Power BI 的「模型檢視（Model View）」中，將 `powerbi/data/` 匯入的 4 張表格建立為標準星狀模型（一對多 `1:*`，單向篩選）：

```
           ┌──────────────────────┐
           │       Dim_Date       │
           │ (Date: 1)            │
           └──────────┬───────────┘
                      │ 1:*
                      ▼
┌──────────────────┐  │  ┌──────────────────┐
│   Dim_Product    ├──┼──┤    Dim_Store     │
│ (Item_ID: 1)     │  │  │ (Store_ID: 1)    │
└────────┬─────────┘  │  └────────┬─────────┘
         │ 1:*        │           │ 1:*
         └───────────┐▼┌──────────┘
                     │ │
                     ▼ ▼
          ┌────────────────────────┐
          │Fact_ForecastAndInventory│
          │ (Date, Item_ID, Store) │
          └────────────────────────┘
```

- `Dim_Date[Date]` $\to$ `Fact_ForecastAndInventory[Date]`
- `Dim_Product[Item_ID]` $\to$ `Fact_ForecastAndInventory[Item_ID]`
- `Dim_Store[Store_ID]` $\to$ `Fact_ForecastAndInventory[Store_ID]`

---

## 二、 4 大核心頁面視覺化配置（Dashboard Layouts）

### 頁面 1：戰略指揮官總覽（Executive Control Tower）
- **核心目標**：向 C-suite 高階主管呈現供應鏈健康度、財務節省金額與服務水準躍升。
- **頂部 KPI 卡片（Cards）**：
  1. `[Net Cost Savings ($)]`：顯示累計節省總成本金額（綠色突出顯示）。
  2. `[Cost Reduction %]`：顯示成本降低比率（目標：降低 15%~35%）。
  3. `[Safety Stock Reduction %]`：顯示庫存資金佔用降低率。
  4. `[Service Level % (ML Quantile)]`：顯示達成之客戶履約水準（目標：> 95%）。
- **左側視覺圖（折線圖與群組直條圖）**：
  - **X 軸**：`Dim_Date[Date]`
  - **直條圖數值**：`[Total Cost (Classic Gaussian Policy)]` vs `[Total Cost (Dynamic ML Quantile Policy)]`
  - **折線圖數值**：`[Cost Reduction %]`
  - **用途**：清楚呈現每週成本支出降低之趨勢。
- **右側視覺圖（環圈圖 Donut Chart）**：
  - **維度**：`Dim_Product[Cat_ID]`
  - **數值**：`[Net Cost Savings ($)]`
  - **用途**：展示食品（FOODS）、嗜好品（HOBBIES）、家用品（HOUSEHOLD）哪一品類帶來的成本優化貢獻最大。

---

### 頁面 2：機率性需求預測與不確定性區間（Probabilistic Forecast & Uncertainty）
- **核心目標**：生管與採購人員分析多模型預測差異與不確定性分佈。
- **頂部交叉分析篩選器（Slicers）**：
  - `Dim_Store[Store_ID]`（CA_1, TX_1, WI_1）
  - `Dim_Product[Item_ID]`（單一 SKU 深入分析）
- **中央核心視覺圖（折線與區域堆疊圖 Area Chart / Line Chart）**：
  - **X 軸**：`Dim_Date[Date]`
  - **數值系列 1**：`[Forecast Demand (LGB P90 - Optimistic)]`（上限）
  - **數值系列 2**：`[Forecast Demand (LGB P50 - Median)]`（預期基準線，粗線條）
  - **數值系列 3**：`[Forecast Demand (LGB P10 - Conservative)]`（下限）
  - **數值系列 4**：`[Total Actual Sales]`（真實歷史銷售，點狀線）
  - **效果**：展示出經典的「不確定性預測漏斗（Fan Chart）」，當有促銷或節慶時，P90 上揚，提示補貨彈性。
- **下方對比表格（Table Visual）**：
  - 欄位：`Item_ID`, `Actual_Sales`, `Pred_LGB_P50`, `Pred_Croston`, `Pred_Deep_P50`, `Uncertainty Spread %`。

---

### 頁面 3：作業研究庫存政策與非對稱成本決策（Inventory Policy Sandbox）
- **核心目標**：證明報童模型（Newsvendor）如何打敗傳統常態公式。
- **左側圖表（群組直條圖 Clustered Column Chart）**：
  - **X 軸**：`Dim_Product[Item_ID]`
  - **數值**：`[Classic Safety Stock (Gaussian)]` vs `[Dynamic Safety Stock (ML Quantile)]`
  - **洞察**：直觀呈現各品項安全庫存水位降低之幅度，特別是間歇性需求品項（如 HOBBIES）不再被過度堆疊庫存。
- **右側圖表（成本折衝拆解圖 Stacked Column Chart）**：
  - **X 軸**：模型政策（Classic vs. ML Quantile）
  - **數值堆疊**：`[Holding Cost]`（持有成本，藍色）與 `[Stockout Cost]`（缺貨成本，紅色）
  - **洞察**：清楚證明傳統常態分佈過度保守導致高額持有成本，而 ML 分位數模型能精準在缺貨與持有之間取得最佳甜蜜點。

---

### 頁面 4：演算法可解釋性與品類下鑽（Responsible AI & SHAP Explainability）
- **核心目標**：向安永顧問面試官展現 AI 治理與可解釋性能力。
- **資料來源**：直接匯入 `data/processed/shap_feature_importance.csv`。
- **中央條形圖（Bar Chart）**：
  - **Y 軸**：`Feature`（如 `sales_lag_7`, `discount_ratio`, `sales_rolling_mean_28` 等）
  - **X 軸**：`Contribution_Pct`（影響力百分比）
  - **洞察說明**：向使用者解釋「需求預測主要由滯後 7 天銷量（32%）、價格折扣幅度（21%）與滾動均值驅動」，打消黑箱顧慮。
- **下方下鑽矩陣（Matrix）**：
  - **列階層**：`Cat_ID` $\to$ `Dept_ID` $\to$ `Item_ID`
  - **值**：`Average Daily Demand`, `Lead_Time_Days`, `Service Level % (ML Quantile)`, `Net Cost Savings ($)`。

---

## 三、 快速上手三步驟

1. 打開 Power BI Desktop，點選 **取得資料 (Get Data) $\to$ 文字/CSV**，依序載入 `powerbi/data/` 內的 4 張 CSV。
2. 切換至「模型檢視」，確認表關聯如第一節所示。
3. 建立一個「度量值資料表（Measures Table）」，將 `powerbi/dax_measures.txt` 內的 DAX 依序貼入，即可迅速拖拉出四頁專業戰情室！
