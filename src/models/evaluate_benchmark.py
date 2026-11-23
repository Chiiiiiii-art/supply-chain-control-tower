"""
Model Benchmark & Comprehensive Evaluation Engine
Computes Pinball Loss (Quantile Loss), L1-consistent MAE (P50), L2-consistent RMSE (Mean),
and Diebold-Mariano (DM) statistical significance tests across candidate architectures.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from typing import Tuple


def _pinball_per_obs(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> np.ndarray:
    """Element-wise Pinball Loss vector for quantile q."""
    err = y_true - y_pred
    return np.maximum(q * err, (q - 1.0) * err)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Calculates mean Pinball Loss for a specific quantile q."""
    return float(np.mean(_pinball_per_obs(y_true, y_pred, q)))


def diebold_mariano_test(
    loss_baseline: np.ndarray,
    loss_candidate: np.ndarray,
    h: int = None,
    dates: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Two-sided Diebold-Mariano test comparing average loss differential
    d = L_base - L_cand using a Newey-West Bartlett long-run variance estimator.
    When `dates` is supplied for stacked panels, loss differentials are averaged cross-sectionally
    by calendar date first to prevent cross-series lag leakage and pseudo-replication.
    Positive DM_Stat indicates candidate outperforms baseline (lower loss).
    """
    d = np.asarray(loss_baseline, dtype=float) - np.asarray(loss_candidate, dtype=float)
    if dates is not None:
        d = pd.Series(d).groupby(np.asarray(dates), sort=True).mean().values
    n = len(d)
    if n < 2:
        return 0.0, 1.0
    if h is None:
        h = max(1, int(np.floor(n ** (1.0 / 3.0))))

    d_mean = float(np.mean(d))
    d_cent = d - d_mean
    lrv = float(np.dot(d_cent, d_cent) / n)
    for k in range(1, h + 1):
        if k >= n:
            break
        weight = 1.0 - k / (h + 1.0)
        gamma_k = float(np.dot(d_cent[k:], d_cent[:-k]) / n)
        lrv += 2.0 * weight * gamma_k

    lrv = max(lrv, 1e-12)
    dm_stat = d_mean / np.sqrt(lrv / n)
    p_val = float(2.0 * student_t.sf(np.abs(dm_stat), df=n - 1))
    return float(dm_stat), p_val


def _avg_pinball_per_obs(
    y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray
) -> np.ndarray:
    """Average P10/P50/P90 Pinball Loss per observation for DM testing."""
    return (
        _pinball_per_obs(y_true, p10, 0.1)
        + _pinball_per_obs(y_true, p50, 0.5)
        + _pinball_per_obs(y_true, p90, 0.9)
    ) / 3.0


def _score_model(
    model_name: str,
    category: str,
    y_true: np.ndarray,
    mu: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    train_sec: float,
    baseline_loss_vec: np.ndarray = None,
    dates: np.ndarray = None,
) -> dict:
    """Compute L1-optimal MAE (P50), L2-optimal RMSE (Mean), Pinball losses, and DM test."""
    mae = float(np.mean(np.abs(y_true - p50)))
    rmse = float(np.sqrt(np.mean((y_true - mu) ** 2)))
    l10 = pinball_loss(y_true, p10, 0.1)
    l50 = pinball_loss(y_true, p50, 0.5)
    l90 = pinball_loss(y_true, p90, 0.9)
    loss_vec = _avg_pinball_per_obs(y_true, p10, p50, p90)

    if baseline_loss_vec is None:
        dm_stat, dm_p = 0.0, 1.0
    else:
        dm_stat, dm_p = diebold_mariano_test(baseline_loss_vec, loss_vec, dates=dates)

    return {
        "Model": model_name,
        "Category": category,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Pinball_P10": round(l10, 3),
        "Pinball_P50": round(l50, 3),
        "Pinball_P90": round(l90, 3),
        "Avg_Pinball_Loss": round((l10 + l50 + l90) / 3.0, 3),
        "DM_Stat": round(dm_stat, 2),
        "DM_p_value": float(f"{dm_p:.3e}"),
        "Training_Time_Sec": round(float(train_sec), 2),
    }


def evaluate_all_models(df_eval: pd.DataFrame, time_stats: dict) -> pd.DataFrame:
    """
    Evaluates baseline, LightGBM, and Deep ZINB-NLL models on holdout test set.
    """
    y_true = df_eval["sales"].values.astype(float)
    croston_loss_vec = _avg_pinball_per_obs(
        y_true,
        df_eval["pred_croston_p10"].values,
        df_eval["pred_croston_p50"].values,
        df_eval["pred_croston_p90"].values,
    )

    eval_dates = df_eval["date"].values if "date" in df_eval.columns else None
    records = [
        _score_model(
            "OR Baseline (Croston-SBA)",
            "Operations Research",
            y_true,
            df_eval["pred_croston"].values,
            df_eval["pred_croston_p10"].values,
            df_eval["pred_croston_p50"].values,
            df_eval["pred_croston_p90"].values,
            time_stats.get("croston_time", 0.05),
            baseline_loss_vec=None,
            dates=eval_dates,
        ),
        _score_model(
            "OR Baseline (MA-7d)",
            "Operations Research",
            y_true,
            df_eval["pred_ma7"].values,
            df_eval["pred_ma7_p10"].values,
            df_eval["pred_ma7_p50"].values,
            df_eval["pred_ma7_p90"].values,
            time_stats.get("ma7_time", 0.02),
            baseline_loss_vec=croston_loss_vec,
            dates=eval_dates,
        ),
        _score_model(
            "LightGBM Quantile (Pinball)",
            "Tree-based GBDT",
            y_true,
            df_eval["pred_lgb_mu"].values,
            df_eval["pred_lgb_p10"].values,
            df_eval["pred_lgb_p50"].values,
            df_eval["pred_lgb_p90"].values,
            time_stats.get("total_lgb", 1.5),
            baseline_loss_vec=croston_loss_vec,
            dates=eval_dates,
        ),
        _score_model(
            "Deep Probabilistic (ZINB-NLL)",
            "Deep Neural Network",
            y_true,
            df_eval["pred_deep_mu"].values,
            df_eval["pred_deep_p10"].values,
            df_eval["pred_deep_p50"].values,
            df_eval["pred_deep_p90"].values,
            time_stats.get("train_time_deep", 3.2),
            baseline_loss_vec=croston_loss_vec,
            dates=eval_dates,
        ),
    ]
    
    df_benchmark = pd.DataFrame(records)
    print("\n" + "="*82)
    print("           SUPPLY CHAIN MULTI-MODEL BENCHMARK RESULTS")
    print("="*82)
    print(df_benchmark.to_string(index=False))
    if "pred_lt_mu" in df_eval.columns and "lt_sales" in df_eval.columns:
        y_lt = df_eval["lt_sales"].values.astype(float)
        lt_rmse = float(np.sqrt(np.mean((y_lt - df_eval["pred_lt_mu"].values) ** 2)))
        lt_mae = float(np.mean(np.abs(y_lt - df_eval["pred_lt_p50"].values)))
        lt_cov = {
            f"P{q}": round(float((y_lt <= df_eval[f"pred_lt_p{q}"].values).mean()), 3)
            for q in (50, 75, 85, 90, 95)
            if f"pred_lt_p{q}" in df_eval.columns
        }
        print("-" * 82)
        print(f"  Direct L-Day LT LightGBM -> RMSE(Mean): {lt_rmse:.3f}, MAE(P50): {lt_mae:.3f}, CQR Coverage: {lt_cov}")
    print("="*82 + "\n")
    return df_benchmark
