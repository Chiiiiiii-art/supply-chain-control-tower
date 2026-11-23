"""
Operations Research Inventory Optimization Engine
Implements Newsvendor Critical Fractile Model under asymmetric holding and stockout penalties.
Uses 1/L daily cycle normalization to eliminate overlapping-window double-counting, and reports
full ablation across Classic Gaussian, Croston-NB, ML-Gaussian, and Direct CQR ML Quantile policies.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

LT_QUANTILE_GRID = [0.50, 0.75, 0.85, 0.90, 0.95]


def _interp_lt_quantile(df: pd.DataFrame, cr: np.ndarray) -> np.ndarray:
    """Piecewise linearly interpolate direct L-day quantile Q_{D_{t,L}}(CR) across grid."""
    cols = [f"pred_lt_p{int(round(q * 100))}" for q in LT_QUANTILE_GRID]
    q_matrix = df[cols].values
    return np.array([np.interp(c, LT_QUANTILE_GRID, row) for c, row in zip(cr, q_matrix)])


def _eval_cycle_costs(
    actual_lt: np.ndarray,
    rop: np.ndarray,
    cu: np.ndarray,
    co: np.ndarray,
    lt_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 1/L cycle-normalized holding, stockout, total costs and stockout flag."""
    stockout_qty = np.maximum(0.0, actual_lt - rop)
    overstock_qty = np.maximum(0.0, rop - actual_lt)
    stockout_cost = np.round((stockout_qty * cu) / lt_days, 2)
    holding_cost = np.round((overstock_qty * co) / lt_days, 2)
    total_cost = np.round(stockout_cost + holding_cost, 2)
    is_stockout = (stockout_qty > 0).astype(int)
    return holding_cost, stockout_cost, total_cost, is_stockout


def compute_newsvendor_policies(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Classic Gaussian, Croston-NB, ML-Gaussian ablation, and Direct ML Quantile
    policies over exact forward L-day demand with 1/L cycle cost normalization.
    """
    if "lt_sales" not in df_results.columns:
        raise ValueError("df_results must include exact forward L-day 'lt_sales'.")

    df = df_results.copy()
    df["_orig_idx"] = np.arange(len(df))
    df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    # Critical Fractile CR = Cu / (Cu + Co)
    cr_raw = df["unit_stockout_cost"] / (df["unit_stockout_cost"] + df["unit_holding_cost"])
    df["critical_fractile"] = np.clip(cr_raw, 0.50, 0.95)

    # 1. Classic Gaussian Policy (28-day rolling mean + Gaussian safety stock)
    df["z_score_classic"] = norm.ppf(df["critical_fractile"])
    hist_mean = df["sales_rolling_mean_28"].fillna(df["sales"].mean())
    hist_std = df["sales_rolling_std_28"].fillna(df["sales"].std())
    lead_time_sqrt = np.sqrt(df["lead_time_days"])

    df["lead_time_demand_classic"] = hist_mean * df["lead_time_days"]
    df["safety_stock_classic"] = np.round(df["z_score_classic"] * hist_std * lead_time_sqrt, 1)
    df["rop_classic"] = np.round(df["lead_time_demand_classic"] + df["safety_stock_classic"], 1)

    # 2. Direct ML Lead-Time Quantile Policy (exact Newsvendor optimum S* = Q_{D_{t,L}}(CR))
    q_lt_cr = _interp_lt_quantile(df, df["critical_fractile"].values)
    df["pred_quantile_cr"] = np.round(q_lt_cr, 2)
    df["lead_time_demand_ml"] = np.round(df["pred_lt_mu"], 1)
    df["rop_quantile"] = np.round(q_lt_cr, 1)
    df["safety_stock_quantile"] = np.round(np.maximum(0.0, df["rop_quantile"] - df["lead_time_demand_ml"]), 1)

    # 3. Ablation Policies: ML Mean + Gaussian SS, and Croston-SBA Negative Binomial
    df["rop_ml_gaussian"] = np.round(
        df["pred_lt_mu"] + df["z_score_classic"] * hist_std * lead_time_sqrt, 1
    )
    if "pred_croston_lt_cr" in df.columns:
        df["rop_croston_nb"] = np.round(df["pred_croston_lt_cr"], 1)

    # 4. 1/L Cycle-Normalized Cost Simulation over exact L-day demand
    actual_lt_demand = df["lt_sales"].values.astype(float)
    lt_days = np.maximum(1.0, df["lead_time_days"].values.astype(float))
    cu = df["unit_stockout_cost"].values.astype(float)
    co = df["unit_holding_cost"].values.astype(float)

    (
        df["holding_cost_classic"],
        df["stockout_cost_classic"],
        df["total_cost_classic"],
        df["is_stockout_classic"],
    ) = _eval_cycle_costs(actual_lt_demand, df["rop_classic"].values, cu, co, lt_days)

    (
        df["holding_cost_quantile"],
        df["stockout_cost_quantile"],
        df["total_cost_quantile"],
        df["is_stockout_quantile"],
    ) = _eval_cycle_costs(actual_lt_demand, df["rop_quantile"].values, cu, co, lt_days)

    (
        _,
        _,
        df["total_cost_ml_gaussian"],
        df["is_stockout_ml_gaussian"],
    ) = _eval_cycle_costs(actual_lt_demand, df["rop_ml_gaussian"].values, cu, co, lt_days)

    if "rop_croston_nb" in df.columns:
        (
            _,
            _,
            df["total_cost_croston_nb"],
            df["is_stockout_croston_nb"],
        ) = _eval_cycle_costs(actual_lt_demand, df["rop_croston_nb"].values, cu, co, lt_days)

    df["cost_savings_dollar"] = np.round(df["total_cost_classic"] - df["total_cost_quantile"], 2)
    df["inventory_buffer_reduction"] = np.round(df["safety_stock_classic"] - df["safety_stock_quantile"], 1)

    df = df.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)

    # Summary & Ablation Reporting
    total_classic_cost = df["total_cost_classic"].sum()
    total_ml_gauss_cost = df["total_cost_ml_gaussian"].sum()
    total_ml_cost = df["total_cost_quantile"].sum()
    cost_reduction_pct = ((total_classic_cost - total_ml_cost) / total_classic_cost) * 100

    stockout_rate_classic = df["is_stockout_classic"].mean() * 100
    stockout_rate_ml_gauss = df["is_stockout_ml_gaussian"].mean() * 100
    stockout_rate_quantile = df["is_stockout_quantile"].mean() * 100

    avg_ss_classic = df["safety_stock_classic"].mean()
    avg_ss_quantile = df["safety_stock_quantile"].mean()
    ss_reduction_pct = ((avg_ss_classic - avg_ss_quantile) / avg_ss_classic) * 100

    from src.models.evaluate_benchmark import diebold_mariano_test

    dates_vec = df["date"].values
    dm_c_stat, dm_c_p = diebold_mariano_test(
        df["total_cost_classic"].values, df["total_cost_quantile"].values, dates=dates_vec
    )
    dm_g_stat, dm_g_p = diebold_mariano_test(
        df["total_cost_ml_gaussian"].values, df["total_cost_quantile"].values, dates=dates_vec
    )

    print("\n" + "="*76)
    print("      OPERATIONS RESEARCH INVENTORY OPTIMIZATION & ABLATION (1/L NORMALIZED)")
    print("="*76)
    print(f"  1. Classic Gaussian (MA28):          ${total_classic_cost:,.2f}  | Cycle SLA: {100-stockout_rate_classic:.1f}%")
    if "total_cost_croston_nb" in df.columns:
        total_croston_cost = df["total_cost_croston_nb"].sum()
        sla_croston = (1.0 - df["is_stockout_croston_nb"].mean()) * 100
        print(f"  2. Croston-SBA (NB Quantile):        ${total_croston_cost:,.2f}  | Cycle SLA: {sla_croston:.1f}%")
    print(f"  3. Ablation (ML Mean + Gaussian SS): ${total_ml_gauss_cost:,.2f}  | Cycle SLA: {100-stockout_rate_ml_gauss:.1f}%")
    print(f"  4. Direct ML CQR Quantile Policy:    ${total_ml_cost:,.2f}  | Cycle SLA: {100-stockout_rate_quantile:.1f}%")
    print("-" * 76)
    print(f"  Total Savings vs. Classic:           ${(total_classic_cost - total_ml_cost):,.2f} ({cost_reduction_pct:.2f}% | Date-DM t={dm_c_stat:.2f}, p={dm_c_p:.2e})")
    print(f"  Quantile Tail Gain vs. ML-Gaussian:  ${(total_ml_gauss_cost - total_ml_cost):,.2f} ({((total_ml_gauss_cost - total_ml_cost)/total_ml_gauss_cost)*100:.2f}% | Date-DM t={dm_g_stat:.2f}, p={dm_g_p:.3f})")
    print(f"  Average Safety Stock Required:       {avg_ss_classic:.1f} units -> {avg_ss_quantile:.1f} units ({ss_reduction_pct:.1f}% Buffer Reduced)")
    print("="*76 + "\n")

    return df
