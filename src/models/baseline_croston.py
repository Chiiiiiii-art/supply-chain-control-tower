"""
Baseline Operations Research Forecasting Models
Implements Croston's Intermittent Demand Method, Moving Average, and Simple Exponential Smoothing.
"""

import numpy as np
import pandas as pd

def croston_forecast(series: np.ndarray, alpha: float = 0.1) -> float:
    """
    Croston's Method for intermittent demand.
    Separates demand magnitude and demand intervals.
    Returns the expected demand rate per period.
    """
    non_zeros = np.nonzero(series)[0]
    if len(non_zeros) == 0:
        return 0.0
    if len(non_zeros) == 1:
        return float(series[non_zeros[0]] / max(1, len(series)))
    
    # Calculate intervals between positive demands
    intervals = np.diff(non_zeros)
    magnitudes = series[non_zeros]
    
    z = magnitudes[0]  # smoothed magnitude
    p = intervals[0] if len(intervals) > 0 else 1.0  # smoothed period interval
    
    for i in range(1, len(intervals)):
        z = alpha * magnitudes[i] + (1 - alpha) * z
        p = alpha * intervals[i] + (1 - alpha) * p
        
    p = max(1.0, p)
    return float(z / p)

def evaluate_baseline_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fit baseline models per (item_id, store_id) series and evaluate on test_df.
    """
    predictions = []
    
    for (item, store), sub_train in train_df.groupby(["item_id", "store_id"]):
        sales_hist = sub_train["sales"].values
        
        # 1. Croston Rate
        croston_rate = croston_forecast(sales_hist, alpha=0.15)
        
        # 2. 7-Day Moving Average
        ma_7 = float(np.mean(sales_hist[-7:])) if len(sales_hist) >= 7 else float(np.mean(sales_hist))
        
        # 3. 28-Day Moving Average
        ma_28 = float(np.mean(sales_hist[-28:])) if len(sales_hist) >= 28 else float(np.mean(sales_hist))
        
        # Store for test matching
        sub_test = test_df[(test_df["item_id"] == item) & (test_df["store_id"] == store)].copy()
        sub_test["pred_croston"] = croston_rate
        sub_test["pred_ma7"] = ma_7
        sub_test["pred_ma28"] = ma_28
        predictions.append(sub_test)
        
    df_eval = pd.concat(predictions, ignore_index=True)
    return df_eval
