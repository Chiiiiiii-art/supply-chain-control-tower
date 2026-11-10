# Power BI 智慧供應鏈控制塔：儀表板視覺化配置與優化指引手冊
## 顧問級戰情室版（繁體中文指引 + 專業外商英文標題）

本手冊專為 Power BI Desktop 設計，將星狀模型轉化為符合外商與管理顧問（EY / McKinsey / BCG）標準的儀表板。全文件採中文步驟說明，圖表標題、度量值標籤與欄位名稱採用精簡俐落的商務英文。

---

## 一、 資料模型星狀關聯設定

在 Power BI 的「模型檢視 (Model View)」中確認 4 張表格關聯（一對多 `1:*`，單向篩選）：

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

---

## 二、 4 大頁面精簡英文標題與視覺配置

### 頁面 1：戰略指揮官總覽 (Executive Control Tower)
- **頂部頁面橫幅 (Page Banner)**：
  - **大標題**：`Executive Supply Chain Control Tower`
  - **副標題**：`Enterprise inventory capital efficiency, cost savings & fulfillment SLA`
- **頂部 4 張 KPI 卡片（建議 1x4 單列橫排，消除留白）**：
  > ⚠️ **必改設定**：在卡片格式窗格中，將 **「類別標籤 (Category Label)」關閉 (Off)**，只保留上方自訂的視覺效果標題，避免文字重複堆疊！
  1. **Card 1: Net Cost Savings**
     - 標題：`Net Cost Savings`
     - 數值：`[Net Cost Savings ($)]`（格式設為 `$#,##0`）
     - 副標題 (`fx`)：`Dyn_Sub_CostSavings`
  2. **Card 2: Total Cost Reduction**
     - 標題：`Cost Reduction %`
     - 數值：`[Cost Reduction %]`（格式設為 `0.0%`）
     - 副標題 (`fx`)：`Dyn_Sub_CostReduction`
  3. **Card 3: Dynamic Buffer Shift**
     - 標題：`Dynamic Buffer Shift`
     - 數值：`[Safety Stock Reduction %]`
     - 副標題 (`fx`)：`Dyn_Sub_SafetyStock`（解釋係為高風險品項增補庫存）
  4. **Card 4: Fulfillment SLA (OTIF)**
     - 標題：`Fulfillment SLA`
     - 數值：`[Service Level % (ML Quantile)]`
     - 副標題 (`fx`)：`Dyn_Sub_ServiceLevel`
- **主要趨勢圖（折線與群組直條圖）**：
  - **精簡標題**：`Weekly Cost & Savings Trend`
  - **X 軸**：`Dim_Date[Date]`（或週階層）
  - **直條數值**：`[Total Cost (Classic Gaussian Policy)]` vs `[Total Cost (Dynamic ML Quantile Policy)]`
  - **折線數值**：`[Cost Reduction %]`（次 Y 軸）
- **品類佔比圖（環圈圖 Donut Chart）**：
  - **精簡標題**：`Savings by Category`
  - **圖例**：`Dim_Product[Cat_ID]`
  - **數值**：`[Net Cost Savings ($)]`

---

### 頁面 2：機率性需求預測與不確定性區間 (Probabilistic Forecast)
- **頂部頁面橫幅 (Page Banner)**：
  - **大標題**：`Probabilistic Demand Forecasting & Tail Risk Engine`
  - **副標題**：`Multi-model pinball loss benchmarking & uncertainty funnel`
- **頂部切片器（Slicers，橫向緊湊排列，避免巨大留白）**：
  1. `Store Location`：改為橫向藥丸/磚塊按鈕，縮減高度。
  2. `SKU Selector`：**新增 `Dim_Product[Item_ID]` 下拉式切片器**（開啟搜尋功能）。
- **中央預測漏斗圖（折線圖 Line Chart）**：
  - **精簡標題**：`Demand Uncertainty Funnel (P10 / P50 / P90)`
  - **副標題 (`fx`)**：`Dyn_Sub_UncertaintyBand`
  - **X 軸**：`Dim_Date[Date]`
  - **線條系列**：
    - `[Forecast Demand (LGB P90 - Optimistic)]`（淺藍/淡紫細虛線）
    - `[Forecast Demand (LGB P50 - Median)]`（深藍/深紅粗實線）
    - `[Forecast Demand (LGB P10 - Conservative)]`（淺藍/淡紫細虛線）
    - `[Total Actual Sales]`（深灰點狀實線）
- **評測對比表（Table Visual）**：
  - **精簡標題**：`Model Benchmark & Spread Summary`
  - **欄位重新命名（雙擊欄位修改）**：
    - `SKU ID`
    - `Actual Sales`
    - `LightGBM P50`
    - `Croston`
    - `PyTorch Deep`
    - `Spread %`（開啟資料橫條 Data Bar）

---

### 頁面 3：作業研究庫存政策沙盒 (Inventory Policy Sandbox)
- **頂部頁面橫幅 (Page Banner)**：
  - **大標題**：`Operations Research Inventory Policy Sandbox`
  - **副標題**：`Newsvendor asymmetric cost trade-off vs. Gaussian baseline`
- **頂部切片器**：縮小切片器高度至頂部工具列大小，釋放垂直畫布空間。
- **左側安全庫存圖（群組直條圖）**：
  - **精簡標題**：`Safety Stock: Gaussian vs. ML Quantile`
  - **X 軸**：`Dim_Product[Item_ID]`
  - **數值**：`[Classic Safety Stock (Gaussian)]` vs `[Dynamic Safety Stock (ML Quantile)]`
- **右側成本折衝圖（堆疊直條圖）**：
  - **精簡標題 1**：`ML Policy: Holding vs. Stockout Cost`
  - **精簡標題 2**：`Classic Policy: Holding vs. Stockout Cost`
  - **改善建議**：統一兩圖的 Y 軸最大刻度（皆鎖定為 100 千），避免視覺誤導。

---

### 頁面 4：AI 治理與品類下鑽 (Responsible AI & Drill-down)
- **頂部頁面橫幅 (Page Banner) ⚠️【必改錯誤】**：
  - 原截圖誤貼成 Page 1 標題！
  - **正確大標題**：`Responsible AI & Decision Attribution Matrix`
  - **正確副標題**：`SHAP feature attribution explaining demand volatility & multi-echelon drill-down`
- **左側特徵貢獻圖（橫向條形圖）**：
  - **精簡標題**：`SHAP Feature Importance (% Contribution)`
  - **Y 軸**：`Feature`
  - **X 軸**：`Contribution_Pct`
- **右側品類下鑽矩陣（Matrix Visual）⚠️【必改階層錯誤】**：
  - **精簡標題**：`Category Drill-down & Savings Waterfall`
  - **列 (Rows) 順序必改**：
    $$\text{Cat\_ID (品類)} \longrightarrow \text{Dept\_ID (部門)} \longrightarrow \text{Item\_ID (商品 SKU)}$$
    *(目前截圖中 Item_ID 跑到了 Dept_ID 上方，展開順序嚴重錯誤)*
  - **值 (Values) 欄位顯示名稱**：
    - `Avg Daily Demand`
    - `Avg Lead Time (Days)` *(確認聚合為平均值)*
    - `Fulfillment SLA (ML)`
    - `Net Cost Savings ($)`

---

## 三、 動態解說 DAX 度量值庫 (英文業務動態文字)

在度量值資料表加入以下度量值，並於圖表「格式」$\to$「一般」$\to$「標題」$\to$「副標題」點擊 **`fx`** 綁定：

```dax
// =====================================================================================
// 1. 財務節省動態說明 (P1 Subtitle)
// =====================================================================================
Dyn_Sub_CostSavings = 
VAR Savings = [Net Cost Savings ($)]
VAR Pct = [Cost Reduction %]
RETURN
"Realized " & FORMAT(Savings, "$#,##0") & " savings (" & FORMAT(Pct, "0.0%") & " reduction), driven by asymmetric stockout mitigation."

// =====================================================================================
// 2. 安全庫存調整動態說明 (P1 Subtitle - 解釋負值為策略性增庫)
// =====================================================================================
Dyn_Sub_SafetyStock = 
VAR SS_Red = [Safety Stock Reduction %]
RETURN
IF(
    SS_Red < 0,
    "Asymmetric buffer expansion (+" & FORMAT(ABS(SS_Red), "0.0%") & ") on high-margin SKUs prevents costly stockouts.",
    "Released excess buffer (-" & FORMAT(SS_Red, "0.0%") & ") on stable SKUs, unlocking capital."
)

// =====================================================================================
// 3. 成本降減基準動態說明 (P1 Subtitle)
// =====================================================================================
Dyn_Sub_CostReduction = 
VAR Pct = [Cost Reduction %]
RETURN
"Newsvendor policy outperforms baseline (Benchmark: 15%–35%)."

// =====================================================================================
// 4. 履約率服務水準動態說明 (P1 Subtitle)
// =====================================================================================
Dyn_Sub_ServiceLevel = 
VAR SL_ML = [Service Level % (ML Quantile)]
VAR SL_Classic = [Service Level % (Classic)]
VAR Diff = SL_ML - SL_Classic
RETURN
"Current SLA: " & FORMAT(SL_ML, "0.0%") & " (" & IF(Diff >= 0, "+" & FORMAT(Diff, "0.0%"), FORMAT(Diff, "0.0%")) & " vs. Gaussian baseline)."

// =====================================================================================
// 5. 預測漏斗不確定性區間說明 (P2 Subtitle)
// =====================================================================================
Dyn_Sub_UncertaintyBand = 
VAR AvgSpread = [Uncertainty Spread %]
RETURN
"P10–P90 spread is " & FORMAT(AvgSpread, "0.0%") & " of median demand; shaded band guides procurement agility."

// =====================================================================================
// 6. 前置時間平均值 (修復矩陣欄位計數問題)
// =====================================================================================
Average Lead Time Days = 
AVERAGE(Dim_Product[Lead_Time_Days])
```

---

## 四、 Power BI Desktop 操作位置速查指引

1. **關閉卡片重複標籤（解決 P1 重複顯示兩個標題）**：
   - 點選卡片圖 $\to$ 右側 **「格式」** 窗格 $\to$ **「視覺效果 (Visual)」** 標籤 $\to$ 將 **「類別標籤 (Category label)」切換為關閉 (Off)**。
2. **修改圖表自訂英文標題**：
   - 點選圖表 $\to$ 右側 **「格式」** $\to$ **「一般 (General)」** 標籤 $\to$ **「標題 (Title)」** $\to$ 在「文字」輸入手冊中的精簡英文標題。
3. **綁定動態副標題 (`fx`)**：
   - 在「標題」設定下方展開 **「副標題 (Subtitle)」** 並開啟 $\to$ 點擊文字框旁的 **`fx` 按鈕** $\to$ 格式樣式選「欄位值」，選取對應的 `Dyn_Sub_...` 度量值。
4. **修正 P4 矩陣階層倒置**：
   - 點選 P4 矩陣 $\to$ 在右側「建置視覺效果」窗格的「列 (Rows)」區域中，用滑鼠將 `Dept_ID` 拖曳到 `Item_ID` 上方（確保順序為 `Cat_ID` $\to$ `Dept_ID` $\to$ `Item_ID`）。
5. **消除切片器巨大留白**：
   - 點選 Store Location 切片器 $\to$ 拖拉縮減高度（壓扁至頂部一排）$\to$ 在其右方空出的區域加入 `Item_ID` 切片器（樣式設為下拉式 Dropdown，點擊切片器右上角三點開啟「搜尋」功能）。
