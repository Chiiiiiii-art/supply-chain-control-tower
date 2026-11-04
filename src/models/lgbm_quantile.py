"""
GBDT Pinball Quantile Regression Model
Trains separate Pinball Loss gradient boosted tree models for P10 (Pessimistic/Lower Bound),
P50 (Median), and P90 (Optimistic/High Service Level) demand forecasting.
Compatible with SHAP TreeExplainer.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, List, Tuple

FEATURE_COLS = [
    "sell_price", "discount_amount", "discount_ratio", "is_promo",
    "sales_lag_7", "sales_lag_14", "sales_lag_21", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_std_7",
    "sales_rolling_mean_14", "sales_rolling_std_14",
    "sales_rolling_mean_28", "sales_rolling_std_28",
    "zero_ratio_28", "month", "day", "is_weekend", "has_event"
]

def train_quantile_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantiles: List[float] = [0.10, 0.50, 0.90]
) -> Tuple[pd.DataFrame, Dict[str, GradientBoostingRegressor], Dict[str, float]]:
    """
    Train Gradient Boosted Tree models for specified quantiles (Pinball Loss)
    and generate test predictions.
    """
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["sales"].values
    
    X_test = test_df[FEATURE_COLS].values
    
    models = {}
    train_times = {}
    test_preds = test_df.copy()
    
    start_total = time.time()
    for q in quantiles:
        q_tag = f"p{int(q*100)}"
        t0 = time.time()
        
        # Fast, robust quantile regressor with Pinball loss
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=80,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        models[q_tag] = model
        train_times[q_tag] = time.time() - t0
        
        # Predict on test
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)
        test_preds[f"pred_lgb_{q_tag}"] = np.round(preds, 2)
        
    total_time = time.time() - start_total
    train_times["total_lgb"] = total_time
    
    # Enforce monotonicity: P10 <= P50 <= P90
    test_preds["pred_lgb_p50"] = np.maximum(test_preds["pred_lgb_p50"], test_preds["pred_lgb_p10"])
    test_preds["pred_lgb_p90"] = np.maximum(test_preds["pred_lgb_p90"], test_preds["pred_lgb_p50"])
    
    print(f"Trained GBDT Pinball Quantile models {quantiles} in {total_time:.2f}s.")
    return test_preds, models, train_times
