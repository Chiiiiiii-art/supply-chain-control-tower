"""
Star Schema Data Exporter for Power BI Desktop
Generates production-grade Fact and Dimension tables adhering to Dimensional Modeling best practices.
"""

import os
import pandas as pd

def export_powerbi_tables(df_optimized: pd.DataFrame, df_calendar: pd.DataFrame, output_dir: str):
    """
    Exports clean Star Schema tables for Power BI import:
    - Fact_ForecastAndInventory
    - Dim_Product
    - Dim_Store
    - Dim_Date
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fact Table
    fact_cols = {
        "date": "Date",
        "item_id": "Item_ID",
        "store_id": "Store_ID",
        "sales": "Actual_Sales",
        "sell_price": "Sell_Price",
        "is_promo": "Is_Promo",
        "pred_croston": "Pred_Croston",
        "pred_ma7": "Pred_MA7",
        "pred_lgb_p10": "Pred_LGB_P10",
        "pred_lgb_p50": "Pred_LGB_P50",
        "pred_lgb_p90": "Pred_LGB_P90",
        "pred_deep_p10": "Pred_Deep_P10",
        "pred_deep_p50": "Pred_Deep_P50",
        "pred_deep_p90": "Pred_Deep_P90",
        "safety_stock_classic": "Safety_Stock_Classic",
        "safety_stock_quantile": "Safety_Stock_Quantile",
        "rop_classic": "ROP_Classic",
        "rop_quantile": "ROP_Quantile",
        "holding_cost_classic": "Holding_Cost_Classic",
        "stockout_cost_classic": "Stockout_Cost_Classic",
        "total_cost_classic": "Total_Cost_Classic",
        "holding_cost_quantile": "Holding_Cost_Quantile",
        "stockout_cost_quantile": "Stockout_Cost_Quantile",
        "total_cost_quantile": "Total_Cost_Quantile",
        "cost_savings_dollar": "Cost_Savings_Dollar",
        "is_stockout_classic": "Is_Stockout_Classic",
        "is_stockout_quantile": "Is_Stockout_Quantile"
    }
    
    available_fact_cols = {k: v for k, v in fact_cols.items() if k in df_optimized.columns}
    df_fact = df_optimized[list(available_fact_cols.keys())].rename(columns=available_fact_cols)
    fact_path = os.path.join(output_dir, "Fact_ForecastAndInventory.csv")
    df_fact.to_csv(fact_path, index=False)
    
    # 2. Dim_Product (strictly unique by Item_ID)
    dim_product = df_optimized.groupby("item_id").first().reset_index()[
        ["item_id", "dept_id", "cat_id", "base_price", "lead_time_days", "unit_stockout_cost", "unit_holding_cost"]
    ]
    dim_product.columns = ["Item_ID", "Dept_ID", "Cat_ID", "Base_Price", "Lead_Time_Days", "Unit_Stockout_Cost", "Unit_Holding_Cost"]
    product_path = os.path.join(output_dir, "Dim_Product.csv")
    dim_product.to_csv(product_path, index=False)
    
    # 3. Dim_Store (strictly unique by Store_ID)
    dim_store = df_optimized.groupby("store_id").first().reset_index()[["store_id", "state_id"]]
    dim_store["Store_Type"] = "Supercenter"
    dim_store.columns = ["Store_ID", "State_ID", "Store_Type"]
    store_path = os.path.join(output_dir, "Dim_Store.csv")
    dim_store.to_csv(store_path, index=False)
    
    # 4. Dim_Date
    df_cal = df_calendar.copy()
    df_cal["month_name"] = pd.to_datetime(df_cal["date"]).dt.strftime("%B")
    df_cal["event_name"] = df_cal["event_name"].fillna("None")
    cal_cols = {
        "date": "Date",
        "year": "Year",
        "month": "Month",
        "month_name": "Month_Name",
        "day": "Day",
        "weekday": "Weekday",
        "is_weekend": "Is_Weekend",
        "event_name": "Event_Name",
        "has_event": "Has_Event",
        "snap_CA": "Snap_CA",
        "snap_TX": "Snap_TX",
        "snap_WI": "Snap_WI"
    }
    dim_date = df_cal[list(cal_cols.keys())].rename(columns=cal_cols)
    date_path = os.path.join(output_dir, "Dim_Date.csv")
    dim_date.to_csv(date_path, index=False)
    
    print("\nExported Star Schema tables to Power BI folder:")
    print(f"  • {fact_path} ({len(df_fact):,} rows)")
    print(f"  • {product_path} ({len(dim_product):,} SKUs)")
    print(f"  • {store_path} ({len(dim_store):,} Stores)")
    print(f"  • {date_path} ({len(dim_date):,} Dates)")
    return fact_path, product_path, store_path, date_path
