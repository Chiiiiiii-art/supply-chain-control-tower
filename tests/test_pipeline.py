"""
Automated Verification Suite for Supply Chain Control Tower Pipeline
Validates file integrity, schema compliance, monotonicity of quantiles, and cost savings.
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_exported_files():
    powerbi_dir = os.path.join(PROJECT_ROOT, "powerbi", "data")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    
    # 1. Check file existence
    expected_files = [
        os.path.join(powerbi_dir, "Fact_ForecastAndInventory.csv"),
        os.path.join(powerbi_dir, "Dim_Product.csv"),
        os.path.join(powerbi_dir, "Dim_Store.csv"),
        os.path.join(powerbi_dir, "Dim_Date.csv"),
        os.path.join(processed_dir, "Model_Benchmark_Summary.csv"),
        os.path.join(processed_dir, "shap_feature_importance.csv")
    ]
    for f in expected_files:
        assert os.path.exists(f), f"Missing file: {f}"
        print(f"  [PASS] File exists: {os.path.basename(f)}")

def test_data_integrity_and_monotonicity():
    fact_path = os.path.join(PROJECT_ROOT, "powerbi", "data", "Fact_ForecastAndInventory.csv")
    df_fact = pd.read_csv(fact_path)
    
    assert len(df_fact) == 1800, f"Expected 1800 test records, got {len(df_fact)}"
    assert df_fact.isna().sum().sum() == 0, "Fact table contains NaN values"
    
    # Monotonicity test: P10 <= P50 <= P90
    assert (df_fact["Pred_LGB_P10"] <= df_fact["Pred_LGB_P50"] + 1e-4).all(), "P10 > P50 detected in GBDT"
    assert (df_fact["Pred_LGB_P50"] <= df_fact["Pred_LGB_P90"] + 1e-4).all(), "P50 > P90 detected in GBDT"
    print("  [PASS] Quantile Monotonicity verified (P10 <= P50 <= P90).")
    
    # Cost savings test: Net cost reduction must be positive
    total_classic = df_fact["Total_Cost_Classic"].sum()
    total_ml = df_fact["Total_Cost_Quantile"].sum()
    savings = total_classic - total_ml
    assert savings > 0, "No positive financial savings generated"
    savings_pct = (savings / total_classic) * 100
    print(f"  [PASS] Financial Cost Reduction verified: ${savings:,.2f} ({savings_pct:.2f}% savings).")

def test_star_schema_keys():
    powerbi_dir = os.path.join(PROJECT_ROOT, "powerbi", "data")
    df_fact = pd.read_csv(os.path.join(powerbi_dir, "Fact_ForecastAndInventory.csv"))
    df_prod = pd.read_csv(os.path.join(powerbi_dir, "Dim_Product.csv"))
    df_store = pd.read_csv(os.path.join(powerbi_dir, "Dim_Store.csv"))
    
    assert set(df_fact["Item_ID"]).issubset(set(df_prod["Item_ID"])), "Fact Item_ID not in Dim_Product"
    assert set(df_fact["Store_ID"]).issubset(set(df_store["Store_ID"])), "Fact Store_ID not in Dim_Store"
    assert len(df_prod) == 10, f"Expected 10 unique SKUs, got {len(df_prod)}"
    assert len(df_store) == 3, f"Expected 3 unique Stores, got {len(df_store)}"
    print("  [PASS] Star Schema Referential Integrity verified.")

if __name__ == "__main__":
    print("\nRunning Automated Verification Suite...")
    test_exported_files()
    test_data_integrity_and_monotonicity()
    test_star_schema_keys()
    print("\nALL VERIFICATION TESTS PASSED!\n")
