"""
Explainable AI (XAI) Module for Supply Chain Demand Governance
Computes SHAP (SHapley Additive exPlanations) values across the lead-time
inventory decision models (lt_p50 demand level and lt_p90 safety-stock tail risk).
"""

import os
import shap
import numpy as np
import pandas as pd
from src.models.lgbm_quantile import FEATURE_COLS, LT_FEATURE_COLS


def _mean_abs_shap(model, X_sample: pd.DataFrame) -> np.ndarray:
    """Compute per-feature mean absolute SHAP values for a single tree model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap_arr = shap_values[0] if isinstance(shap_values, list) else shap_values
    return np.mean(np.abs(shap_arr), axis=0)


def run_shap_analysis(models, test_df: pd.DataFrame, output_dir: str, sample_size: int = 300):
    """
    Computes SHAP TreeExplainer values on sample test observations.
    Prefers direct lead-time inventory decision models ('lt_p50', 'lt_p90') with
    LT_FEATURE_COLS; normalizes each model's attributions before averaging equally.
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_df = test_df.sample(min(sample_size, len(test_df)), random_state=42)

    if isinstance(models, dict) and any(k in models for k in ("lt_p50", "lt_p90")):
        feature_cols = LT_FEATURE_COLS
        target_keys = [k for k in ("lt_p50", "lt_p90") if k in models]
        label = "LT P50+P90"
    elif isinstance(models, dict):
        feature_cols = FEATURE_COLS
        target_keys = [k for k in ("p50", "p90") if k in models]
        if not target_keys:
            target_keys = [next(iter(models))]
        label = "Daily P50+P90"
    else:
        feature_cols = FEATURE_COLS
        target_keys = None
        label = "Single Model"

    X_sample = sample_df[feature_cols]
    if target_keys is not None:
        per_model_shap = [_mean_abs_shap(models[k], X_sample) for k in target_keys]
        mean_abs_shap = np.mean(per_model_shap, axis=0)
        norm_shaps = [v / np.sum(v) if np.sum(v) > 0 else v for v in per_model_shap]
        contrib_pct = np.mean(norm_shaps, axis=0) * 100.0
    else:
        mean_abs_shap = _mean_abs_shap(models, X_sample)
        total_s = np.sum(mean_abs_shap)
        contrib_pct = (mean_abs_shap / total_s) * 100.0 if total_s > 0 else np.zeros_like(mean_abs_shap)

    df_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Abs_SHAP": np.round(mean_abs_shap, 4),
        "Contribution_Pct": np.round(contrib_pct, 2),
    }).sort_values(by="Contribution_Pct", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(output_dir, "shap_feature_importance.csv")
    df_importance.to_csv(csv_path, index=False)

    print("\n" + "="*55)
    print(f"   TOP 7 INVENTORY DECISION DRIVERS (SHAP {label})")
    print("="*55)
    for idx, row in df_importance.head(7).iterrows():
        print(f"  {idx+1}. {row['Feature']:<25} : {row['Contribution_Pct']}% impact")
    print("="*55 + "\n")

    return df_importance
