"""
Explainable AI (XAI) Module for Supply Chain Demand Governance
Computes SHAP (SHapley Additive exPlanations) values to interpret
model predictions, decompose promotional drivers, and audit algorithmic decisions.
"""

import os
import shap
import numpy as np
import pandas as pd
from src.models.lgbm_quantile import FEATURE_COLS

def run_shap_analysis(model, test_df: pd.DataFrame, output_dir: str, sample_size: int = 300):
    """
    Computes SHAP TreeExplainer values on sample test observations.
    Exports global feature importance and top decision attributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sample_df = test_df.sample(min(sample_size, len(test_df)), random_state=42)
    X_sample = sample_df[FEATURE_COLS].values
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_vals_arr = shap_values[0]
    else:
        shap_vals_arr = shap_values
        
    mean_abs_shap = np.mean(np.abs(shap_vals_arr), axis=0)
    
    df_importance = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Mean_Abs_SHAP": np.round(mean_abs_shap, 4)
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    total_shap = df_importance["Mean_Abs_SHAP"].sum()
    if total_shap > 0:
        df_importance["Contribution_Pct"] = np.round((df_importance["Mean_Abs_SHAP"] / total_shap) * 100, 2)
    else:
        df_importance["Contribution_Pct"] = 0.0
        
    csv_path = os.path.join(output_dir, "shap_feature_importance.csv")
    df_importance.to_csv(csv_path, index=False)
    
    print("\n" + "="*55)
    print("      TOP 7 DEMAND DRIVERS (SHAP EXPLAINABILITY)")
    print("="*55)
    for idx, row in df_importance.head(7).iterrows():
        print(f"  {idx+1}. {row['Feature']:<25} : {row['Contribution_Pct']}% impact")
    print("="*55 + "\n")
    
    return df_importance
