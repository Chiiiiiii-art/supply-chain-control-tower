"""
Baseline Operations Research Forecasting Models
Implements causal 1-step-ahead Syntetos-Boylan Approximation (Croston-SBA),
Moving Average (MA-7d / MA-28d), and parametric count distribution quantiles
(Negative Binomial / Poisson) for principled Pinball Loss evaluation.
"""

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


def croston_sba_rolling(series: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    Causal 1-step-ahead Syntetos-Boylan Approximation (Croston-SBA) for intermittent demand.
    Separates demand magnitude (z) and inter-demand interval (p), applying the
    (1 - alpha / 2) bias-correction factor before updating with each realized observation.
    """
    n = len(series)
    preds = np.zeros(n, dtype=float)
    z = None
    p = 1.0
    q = 0
    sba_factor = 1.0 - (alpha / 2.0)

    for t in range(n):
        if z is not None:
            preds[t] = sba_factor * (z / max(1.0, p))
        else:
            preds[t] = 0.0

        y_t = float(series[t])
        q += 1
        if y_t > 0.0:
            if z is None:
                z = y_t
                p = float(q)
            else:
                z = alpha * y_t + (1.0 - alpha) * z
                p = alpha * float(q) + (1.0 - alpha) * p
            q = 0

    return preds


def _count_quantiles(hat_mu: np.ndarray, disp_ratio: float, q: float) -> np.ndarray:
    """
    Derive parametric discrete count quantiles (Negative Binomial when overdispersed,
    otherwise Poisson) using variance-to-mean dispersion estimated on training history.
    """
    mu_safe = np.maximum(1e-3, hat_mu)
    if disp_ratio > 1.05:
        p_nb = 1.0 / disp_ratio
        r_nb = np.maximum(1e-3, mu_safe / (disp_ratio - 1.0))
        return nbinom.ppf(q, n=r_nb, p=p_nb).astype(float)
    return poisson.ppf(q, mu=mu_safe).astype(float)


def evaluate_baseline_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate causal 1-step-ahead rolling forecasts and count-distribution quantiles
    for Croston-SBA and Moving Average baselines across each (item_id, store_id) series.
    """
    predictions = []

    for (item, store), sub_test in test_df.groupby(["item_id", "store_id"]):
        sub_train = train_df[
            (train_df["item_id"] == item) & (train_df["store_id"] == store)
        ].sort_values("date")
        sub_test_sorted = sub_test.sort_values("date").copy()

        train_sales = sub_train["sales"].values.astype(float)
        test_sales = sub_test_sorted["sales"].values.astype(float)
        full_sales = np.concatenate([train_sales, test_sales])
        n_train = len(train_sales)

        # Causal 1-step-ahead rolling point forecasts
        croston_full = croston_sba_rolling(full_sales, alpha=0.15)
        full_s = pd.Series(full_sales)
        ma7_full = full_s.shift(1).rolling(7, min_periods=1).mean().fillna(0.0).values
        ma28_full = full_s.shift(1).rolling(28, min_periods=1).mean().fillna(0.0).values

        croston_test = croston_full[n_train:]
        ma7_test = ma7_full[n_train:]
        ma28_test = ma28_full[n_train:]

        # Training-only dispersion ratio (sigma^2 / mu) to parameterize P10/P50/P90
        mu_train = float(np.mean(train_sales)) if n_train > 0 else 0.0
        var_train = float(np.var(train_sales, ddof=1)) if n_train > 1 else 0.0
        disp_ratio = (var_train / mu_train) if mu_train > 0.0 else 1.0

        sub_test_sorted["pred_croston"] = np.round(croston_test, 4)
        sub_test_sorted["pred_croston_p10"] = _count_quantiles(croston_test, disp_ratio, 0.10)
        sub_test_sorted["pred_croston_p50"] = _count_quantiles(croston_test, disp_ratio, 0.50)
        sub_test_sorted["pred_croston_p90"] = _count_quantiles(croston_test, disp_ratio, 0.90)

        sub_test_sorted["pred_ma7"] = np.round(ma7_test, 4)
        sub_test_sorted["pred_ma7_p10"] = _count_quantiles(ma7_test, disp_ratio, 0.10)
        sub_test_sorted["pred_ma7_p50"] = _count_quantiles(ma7_test, disp_ratio, 0.50)
        sub_test_sorted["pred_ma7_p90"] = _count_quantiles(ma7_test, disp_ratio, 0.90)

        sub_test_sorted["pred_ma28"] = np.round(ma28_test, 4)

        # Lead-time Negative Binomial quantile baseline for Croston-SBA ablation
        lt = max(1, int(sub_test_sorted["lead_time_days"].iloc[0]))
        cr = np.clip(
            sub_test_sorted["unit_stockout_cost"].values
            / (sub_test_sorted["unit_stockout_cost"].values + sub_test_sorted["unit_holding_cost"].values),
            0.50,
            0.95,
        )
        if "lt_sales" in sub_train.columns:
            tr_lt = sub_train["lt_sales"].dropna().values.astype(float)
            mu_lt = float(np.mean(tr_lt)) if len(tr_lt) > 0 else 0.0
            var_lt = float(np.var(tr_lt, ddof=1)) if len(tr_lt) > 1 else 0.0
            disp_lt = (var_lt / mu_lt) if mu_lt > 0.0 else disp_ratio
        else:
            disp_lt = disp_ratio
        sub_test_sorted["pred_croston_lt_cr"] = _count_quantiles(croston_test * lt, disp_lt, cr)
        predictions.append(sub_test_sorted)

    df_eval = pd.concat(predictions, ignore_index=True)
    return df_eval
