"""
Operations Research Inventory Optimization Engine
Implements Newsvendor Critical Fractile Model under asymmetric holding and stockout penalties.
Compares dynamic machine learning quantiles against traditional Gaussian safety stock.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

def compute_newsvendor_policies(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes classic Gaussian inventory policy vs. dynamic ML quantile policy.
    Simulates total inventory cost (holding + stockout) and financial savings.
    """
    df = df_results.copy()
    
    # Critical Fractile CR = Cu / (Cu + Co)
    df["critical_fractile"] = df["unit_stockout_cost"] / (df["unit_stockout_cost"] + df["unit_holding_cost"])
    df["critical_fractile"] = np.clip(df["critical_fractile"], 0.50, 0.98)
    
    # -------------------------------------------------------------
    # 1. Traditional Policy (Gaussian assumption with historical std)
    # -------------------------------------------------------------
    # Z-score corresponding to service level CR
    df["z_score_classic"] = norm.ppf(df["critical_fractile"])
    
    # Historical rolling mean & std over 28 days
    hist_mean = df["sales_rolling_mean_28"].fillna(df["sales"].mean())
    hist_std = df["sales_rolling_std_28"].fillna(df["sales"].std())
    lead_time_sqrt = np.sqrt(df["lead_time_days"])
    
    df["lead_time_demand_classic"] = hist_mean * df["lead_time_days"]
    df["safety_stock_classic"] = np.round(df["z_score_classic"] * hist_std * lead_time_sqrt, 1)
    df["safety_stock_classic"] = np.maximum(df["safety_stock_classic"], 1.0)
    df["rop_classic"] = np.round(df["lead_time_demand_classic"] + df["safety_stock_classic"], 1)
    
    # -------------------------------------------------------------
    # 2. Dynamic ML Quantile Policy (Non-parametric / LightGBM)
    # -------------------------------------------------------------
    # Interpolate target quantile from P50 and P90
    p50 = df["pred_lgb_p50"]
    p90 = df["pred_lgb_p90"]
    p10 = df["pred_lgb_p10"]
    cr = df["critical_fractile"]
    
    # Piecewise interpolation for quantile Q(CR)
    q_cr = np.where(
        cr >= 0.50,
        p50 + ((cr - 0.50) / 0.40) * (p90 - p50),
        p10 + ((cr - 0.10) / 0.40) * (p50 - p10)
    )
    df["pred_quantile_cr"] = np.round(q_cr, 2)
    
    df["lead_time_demand_ml"] = p50 * df["lead_time_days"]
    # Dynamic safety stock directly scaled from tail spread
    df["safety_stock_quantile"] = np.round(np.maximum(0, (df["pred_quantile_cr"] - p50) * lead_time_sqrt * 1.5), 1)
    df["rop_quantile"] = np.round(df["lead_time_demand_ml"] + df["safety_stock_quantile"], 1)
    
    # -------------------------------------------------------------
    # 3. Financial Cost Simulation (Holding vs Stockout Penalty)
    # -------------------------------------------------------------
    # Lead time actual demand (proxied by rolling forward lead time sales)
    actual_lt_demand = df["sales"] * df["lead_time_days"]
    
    # Classic Cost Simulation
    stockout_qty_classic = np.maximum(0, actual_lt_demand - df["rop_classic"])
    overstock_qty_classic = np.maximum(0, df["rop_classic"] - actual_lt_demand)
    df["stockout_cost_classic"] = np.round(stockout_qty_classic * df["unit_stockout_cost"], 2)
    df["holding_cost_classic"] = np.round(overstock_qty_classic * df["unit_holding_cost"], 2)
    df["total_cost_classic"] = df["stockout_cost_classic"] + df["holding_cost_classic"]
    
    # ML Quantile Cost Simulation
    stockout_qty_quantile = np.maximum(0, actual_lt_demand - df["rop_quantile"])
    overstock_qty_quantile = np.maximum(0, df["rop_quantile"] - actual_lt_demand)
    df["stockout_cost_quantile"] = np.round(stockout_qty_quantile * df["unit_stockout_cost"], 2)
    df["holding_cost_quantile"] = np.round(overstock_qty_quantile * df["unit_holding_cost"], 2)
    df["total_cost_quantile"] = df["stockout_cost_quantile"] + df["holding_cost_quantile"]
    
    # Cost savings metrics
    df["cost_savings_dollar"] = np.round(df["total_cost_classic"] - df["total_cost_quantile"], 2)
    df["inventory_buffer_reduction"] = np.round(df["safety_stock_classic"] - df["safety_stock_quantile"], 1)
    
    # Service Level Status flags
    df["is_stockout_classic"] = (stockout_qty_classic > 0).astype(int)
    df["is_stockout_quantile"] = (stockout_qty_quantile > 0).astype(int)
    
    # Summary reporting
    total_classic_cost = df["total_cost_classic"].sum()
    total_ml_cost = df["total_cost_quantile"].sum()
    cost_reduction_pct = ((total_classic_cost - total_ml_cost) / total_classic_cost) * 100
    
    stockout_rate_classic = df["is_stockout_classic"].mean() * 100
    stockout_rate_quantile = df["is_stockout_quantile"].mean() * 100
    
    avg_ss_classic = df["safety_stock_classic"].mean()
    avg_ss_quantile = df["safety_stock_quantile"].mean()
    ss_reduction_pct = ((avg_ss_classic - avg_ss_quantile) / avg_ss_classic) * 100
    
    print("\n" + "="*70)
    print("          OPERATIONS RESEARCH INVENTORY OPTIMIZATION IMPACT")
    print("="*70)
    print(f"  Total Cost (Classic Gaussian):       ${total_classic_cost:,.2f}")
    print(f"  Total Cost (Dynamic ML Quantile):    ${total_ml_cost:,.2f}")
    print(f"  Financial Savings Generated:         ${(total_classic_cost - total_ml_cost):,.2f} ({cost_reduction_pct:.2f}% Reduction)")
    print(f"  Stockout Incident Rate:              {stockout_rate_classic:.1f}% -> {stockout_rate_quantile:.1f}%")
    print(f"  Average Safety Stock Required:       {avg_ss_classic:.1f} units -> {avg_ss_quantile:.1f} units ({ss_reduction_pct:.1f}% Buffer Reduced)")
    print("="*70 + "\n")
    
    return df
