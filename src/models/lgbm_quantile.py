"""
LightGBM Pinball Quantile Regression Model
Trains daily Pinball Loss models (P10, P50, P90) for forecasting benchmarks and
direct lead-time demand models (mean + P50..P95 quantiles) for Newsvendor optimization.
Compatible with SHAP TreeExplainer.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import time
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

FEATURE_COLS = [
    "item_code", "dept_code", "cat_code", "store_code",
    "sell_price", "is_promo", "promo_rate_7", "is_snap",
    "sales_lag_1", "sales_lag_2", "sales_lag_3",
    "sales_lag_7", "sales_lag_14", "sales_lag_21", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_std_7",
    "sales_rolling_mean_14", "sales_rolling_std_14",
    "sales_rolling_mean_28", "sales_rolling_std_28",
    "zero_ratio_28", "month", "day", "is_weekend", "has_event"
]

LT_FEATURE_COLS = FEATURE_COLS + ["lead_time_days", "lt_wknd_days", "lt_snap_days", "lt_event_days"]
LT_QUANTILES = [0.50, 0.75, 0.85, 0.90, 0.95]
CATEGORICAL_COLS = ["item_code", "dept_code", "cat_code", "store_code"]

_LGBM_BASE_PARAMS = dict(
    n_estimators=120,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.06,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    min_child_samples=20,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)


def _enforce_monotonicity(df: pd.DataFrame, prefix: str, quantiles: List[float]) -> None:
    """Enforce non-crossing quantile predictions in ascending order."""
    tags = [f"p{int(round(q * 100))}" for q in sorted(quantiles)]
    for i in range(1, len(tags)):
        prev_col = f"{prefix}{tags[i - 1]}"
        curr_col = f"{prefix}{tags[i]}"
        df[curr_col] = np.maximum(df[curr_col], df[prev_col])


def train_quantile_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantiles: List[float] = [0.10, 0.50, 0.90]
) -> Tuple[pd.DataFrame, Dict[str, LGBMRegressor], Dict[str, float]]:
    """
    Train daily LightGBM Pinball Quantile models and direct L-day lead-time
    expectation/quantile models, returning monotonic test predictions.
    """
    sorted_quantiles = sorted(quantiles)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["sales"].values
    X_test = test_df[FEATURE_COLS]

    models = {}
    train_times = {}
    test_preds = test_df.copy()

    start_total = time.time()
    for q in sorted_quantiles:
        q_tag = f"p{int(round(q * 100))}"
        t0 = time.time()
        model = LGBMRegressor(objective="quantile", alpha=q, **_LGBM_BASE_PARAMS)
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)
        models[q_tag] = model
        train_times[q_tag] = time.time() - t0
        test_preds[f"pred_lgb_{q_tag}"] = np.round(np.clip(model.predict(X_test), 0, None), 2)

    _enforce_monotonicity(test_preds, "pred_lgb_", sorted_quantiles)

    # Daily conditional expectation model for L2-consistent RMSE evaluation
    m_daily_mu = LGBMRegressor(objective="regression", **_LGBM_BASE_PARAMS)
    m_daily_mu.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)
    models["mu"] = m_daily_mu
    test_preds["pred_lgb_mu"] = np.round(np.clip(m_daily_mu.predict(X_test), 0, None), 2)

    daily_time = time.time() - start_total
    train_times["total_lgb"] = daily_time

    # Direct causal L-day lead-time demand models (eliminates non-additivity of daily medians)
    train_lt = train_df.dropna(subset=["lt_sales"]).copy()
    lt_dates = pd.to_datetime(train_lt["date"])
    cal_cut = lt_dates.max() - pd.Timedelta(days=60)
    # Exclude fit-fold windows whose L-day horizon overlaps the calibration fold
    lt_end = lt_dates + pd.to_timedelta(train_lt["lead_time_days"] - 1, unit="D")
    tr_fit = train_lt[lt_end <= cal_cut]
    tr_cal = train_lt[lt_dates > cal_cut]

    X_train_lt = train_lt[LT_FEATURE_COLS]
    y_train_lt = train_lt["lt_sales"].values
    X_test_lt = test_df[LT_FEATURE_COLS]

    m_lt_mu = LGBMRegressor(objective="regression", **_LGBM_BASE_PARAMS)
    m_lt_mu.fit(X_train_lt, y_train_lt, categorical_feature=CATEGORICAL_COLS)
    models["lt_mu"] = m_lt_mu
    test_preds["pred_lt_mu"] = np.clip(m_lt_mu.predict(X_test_lt), 0, None)

    for q in sorted(LT_QUANTILES):
        q_tag = f"p{int(round(q * 100))}"
        # Conformal Quantile Regression (CQR) residual calibration on chronological holdout
        cqr_offset = 0.0
        if len(tr_fit) > 0 and len(tr_cal) > 0:
            m_cal = LGBMRegressor(objective="quantile", alpha=q, **_LGBM_BASE_PARAMS)
            m_cal.fit(tr_fit[LT_FEATURE_COLS], tr_fit["lt_sales"].values, categorical_feature=CATEGORICAL_COLS)
            res_cal = tr_cal["lt_sales"].values - np.clip(m_cal.predict(tr_cal[LT_FEATURE_COLS]), 0, None)
            cqr_offset = max(0.0, float(np.quantile(res_cal, q)))

        m_lt = LGBMRegressor(objective="quantile", alpha=q, **_LGBM_BASE_PARAMS)
        m_lt.fit(X_train_lt, y_train_lt, categorical_feature=CATEGORICAL_COLS)
        models[f"lt_{q_tag}"] = m_lt
        test_preds[f"pred_lt_{q_tag}"] = np.clip(m_lt.predict(X_test_lt) + cqr_offset, 0, None)

    _enforce_monotonicity(test_preds, "pred_lt_", LT_QUANTILES)
    train_times["total_lgb_with_lt"] = time.time() - start_total

    print(
        f"Trained LightGBM daily quantiles {sorted_quantiles} in {daily_time:.2f}s "
        f"(+ CQR direct LT grid in {train_times['total_lgb_with_lt']:.2f}s total)."
    )
    return test_preds, models, train_times
