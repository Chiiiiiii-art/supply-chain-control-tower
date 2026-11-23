"""
Automated Verification Suite for Supply Chain Control Tower Pipeline
Validates file integrity, zero leakage/extrapolation, quantile monotonicity,
DM statistical significance, 1/L cycle-normalized ablation hierarchy, and SHAP alignment.
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_exported_files():
    powerbi_dir = os.path.join(PROJECT_ROOT, "powerbi", "data")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")

    expected_files = [
        os.path.join(powerbi_dir, "Fact_ForecastAndInventory.csv"),
        os.path.join(powerbi_dir, "Dim_Product.csv"),
        os.path.join(powerbi_dir, "Dim_Store.csv"),
        os.path.join(powerbi_dir, "Dim_Date.csv"),
        os.path.join(processed_dir, "Model_Benchmark_Summary.csv"),
        os.path.join(processed_dir, "shap_feature_importance.csv"),
    ]
    for f in expected_files:
        assert os.path.exists(f), f"Missing file: {f}"
        print(f"  [PASS] File exists: {os.path.basename(f)}")


def test_leakage_and_boundary_integrity():
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    train_df = pd.read_csv(os.path.join(processed_dir, "train_features.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_features.csv"), keep_default_na=False)

    # Zero future promo leakage & zero collinear price duplicates
    forbidden_cols = {"lt_promo_days", "discount_amount", "discount_ratio"}
    assert forbidden_cols.isdisjoint(set(train_df.columns)), f"Forbidden cols in train: {forbidden_cols & set(train_df.columns)}"
    assert forbidden_cols.isdisjoint(set(test_df.columns)), f"Forbidden cols in test: {forbidden_cols & set(test_df.columns)}"

    # Exact integer L-day forward sums in test_df (no boundary scaling) and 135 masked train boundary rows
    assert len(test_df) == 1800, f"Expected 1800 test rows, got {len(test_df)}"
    assert test_df.isna().sum().sum() == 0, "Test features contain NaN values"
    assert np.allclose(test_df["lt_sales"].values, np.round(test_df["lt_sales"].values)), "Fractional lt_sales detected in test_features"
    masked_train_lt = int(train_df["lt_sales"].isna().sum())
    assert masked_train_lt == 135, f"Expected 135 masked train boundary rows, got {masked_train_lt}"
    print("  [PASS] Zero future promo leakage, zero collinearity, and exact L-day boundary integrity verified.")


def test_data_integrity_and_ablation():
    fact_path = os.path.join(PROJECT_ROOT, "powerbi", "data", "Fact_ForecastAndInventory.csv")
    df_fact = pd.read_csv(fact_path)

    assert len(df_fact) == 1800, f"Expected 1800 test records, got {len(df_fact)}"
    assert df_fact.isna().sum().sum() == 0, "Fact table contains NaN values"

    # Monotonicity test: P10 <= P50 <= P90 for both LightGBM and Deep ZINB-NLL
    assert (df_fact["Pred_LGB_P10"] <= df_fact["Pred_LGB_P50"] + 1e-4).all(), "P10 > P50 detected in LightGBM"
    assert (df_fact["Pred_LGB_P50"] <= df_fact["Pred_LGB_P90"] + 1e-4).all(), "P50 > P90 detected in LightGBM"
    assert (df_fact["Pred_Deep_P10"] <= df_fact["Pred_Deep_P50"] + 1e-4).all(), "P10 > P50 detected in Deep ZINB-NLL"
    assert (df_fact["Pred_Deep_P50"] <= df_fact["Pred_Deep_P90"] + 1e-4).all(), "P50 > P90 detected in Deep ZINB-NLL"
    print("  [PASS] Quantile Monotonicity verified (LightGBM & Deep ZINB-NLL P10 <= P50 <= P90).")

    # Causal rolling baseline test: Pred_Croston must vary across the 60-day test window
    croston_stds = df_fact.groupby(["Item_ID", "Store_ID"])["Pred_Croston"].std()
    assert (croston_stds > 1e-4).all(), "Static constant Pred_Croston detected; expected causal rolling SBA updates."
    print("  [PASS] Dynamic Causal Rolling Croston-SBA verified across all 30 SKU-Store series.")

    # 1/L cycle-normalized cost bounds & strict ablation hierarchy
    total_revenue = (df_fact["Actual_Sales"] * df_fact["Sell_Price"]).sum()
    total_classic = df_fact["Total_Cost_Classic"].sum()
    total_ml_gauss = df_fact["Total_Cost_ML_Gaussian"].sum()
    total_ml = df_fact["Total_Cost_Quantile"].sum()

    assert total_classic < 0.35 * total_revenue, (
        f"Classic cost ${total_classic:,.2f} exceeds 35% of 60-day revenue ${total_revenue:,.2f} (possible double counting)"
    )
    assert total_ml < total_ml_gauss < total_classic, (
        f"Ablation hierarchy broken: Quantile=${total_ml:,.2f}, ML-Gauss=${total_ml_gauss:,.2f}, Classic=${total_classic:,.2f}"
    )

    savings = total_classic - total_ml
    savings_pct = (savings / total_classic) * 100
    sla_classic = (1.0 - df_fact["Is_Stockout_Classic"].mean()) * 100
    sla_ml = (1.0 - df_fact["Is_Stockout_Quantile"].mean()) * 100
    assert sla_ml > sla_classic, f"ML Cycle SLA ({sla_ml:.1f}%) did not improve over Classic ({sla_classic:.1f}%)"
    print(
        f"  [PASS] Normalized Cost Ablation verified: Classic=${total_classic:,.2f} -> ML-Gauss=${total_ml_gauss:,.2f} "
        f"-> CQR Quantile=${total_ml:,.2f} ({savings_pct:.2f}% savings, Cycle SLA {sla_classic:.1f}% -> {sla_ml:.1f}%)."
    )


def test_benchmark_dm_and_shap_alignment():
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    df_bench = pd.read_csv(os.path.join(processed_dir, "Model_Benchmark_Summary.csv"))
    assert "DM_p_value" in df_bench.columns, "DM_p_value missing from Model_Benchmark_Summary.csv"

    for model_name in ("LightGBM Quantile (Pinball)", "Deep Probabilistic (ZINB-NLL)"):
        row = df_bench[df_bench["Model"] == model_name]
        assert len(row) == 1, f"Missing benchmark row for {model_name}"
        p_val = float(row["DM_p_value"].iloc[0])
        assert p_val < 0.05, f"DM test not statistically significant for {model_name}: p={p_val}"
    print("  [PASS] Diebold-Mariano statistical significance verified (p < 0.05 vs. Croston-SBA).")

    df_shap = pd.read_csv(os.path.join(processed_dir, "shap_feature_importance.csv"))
    assert "lead_time_days" in set(df_shap["Feature"]), "SHAP missing lead_time_days (not aligned with LT decision models)"
    promo_row = df_shap[df_shap["Feature"] == "is_promo"]
    assert len(promo_row) == 1 and float(promo_row["Contribution_Pct"].iloc[0]) > 0.5, (
        f"is_promo SHAP contribution collapsed: {promo_row.to_dict('records')}"
    )
    print(f"  [PASS] SHAP Lead-Time Decision Alignment verified (is_promo={float(promo_row['Contribution_Pct'].iloc[0]):.2f}%).")


def test_star_schema_keys():
    powerbi_dir = os.path.join(PROJECT_ROOT, "powerbi", "data")
    df_fact = pd.read_csv(os.path.join(powerbi_dir, "Fact_ForecastAndInventory.csv"))
    df_prod = pd.read_csv(os.path.join(powerbi_dir, "Dim_Product.csv"))
    df_store = pd.read_csv(os.path.join(powerbi_dir, "Dim_Store.csv"))
    df_date = pd.read_csv(os.path.join(powerbi_dir, "Dim_Date.csv"))

    assert set(df_fact["Item_ID"]).issubset(set(df_prod["Item_ID"])), "Fact Item_ID not in Dim_Product"
    assert set(df_fact["Store_ID"]).issubset(set(df_store["Store_ID"])), "Fact Store_ID not in Dim_Store"
    assert len(df_prod) == 10, f"Expected 10 unique SKUs, got {len(df_prod)}"
    assert len(df_store) == 3, f"Expected 3 unique Stores, got {len(df_store)}"
    assert len(df_date) == 730 and df_date["Date"].max() == "2025-12-30", (
        f"Dim_Date buffer leak: len={len(df_date)}, max={df_date['Date'].max()}"
    )

    # Verify static SKU cost accuracy (invariant to promo day on 2025-11-01)
    expected_cu_ratios = {
        "FOODS_1_001": 0.55, "FOODS_2_045": 0.45, "FOODS_3_120": 0.42,
        "HOBBIES_1_012": 0.48, "HOBBIES_2_008": 0.52, "HOUSEHOLD_1_050": 0.50,
        "HOUSEHOLD_1_180": 0.55, "HOUSEHOLD_2_032": 0.52, "FOODS_1_099": 0.48,
        "HOBBIES_1_340": 0.56,
    }
    for _, r in df_prod.iterrows():
        exp_cu = round(float(r["Base_Price"]) * expected_cu_ratios[r["Item_ID"]], 2)
        assert abs(float(r["Unit_Stockout_Cost"]) - exp_cu) < 1e-6, (
            f"Dim_Product promo corruption on {r['Item_ID']}: got {r['Unit_Stockout_Cost']}, expected {exp_cu}"
        )

    cr = df_prod["Unit_Stockout_Cost"] / (df_prod["Unit_Stockout_Cost"] + df_prod["Unit_Holding_Cost"])
    assert cr.nunique() >= 5 and (cr.max() - cr.min()) >= 0.10, (
        f"Critical fractiles collapsed: min={cr.min():.3f}, max={cr.max():.3f}"
    )
    assert cr.min() >= 0.70 and cr.max() <= 0.94, f"Critical fractiles out of calibrated [0.70, 0.94] range: {cr.tolist()}"
    print(f"  [PASS] Star Schema Integrity, Static Dim_Product Costs, & Heterogeneous CR ({cr.min():.3f} - {cr.max():.3f}) verified.")


if __name__ == "__main__":
    print("\nRunning Automated Verification Suite...")
    test_exported_files()
    test_leakage_and_boundary_integrity()
    test_data_integrity_and_ablation()
    test_benchmark_dm_and_shap_alignment()
    test_star_schema_keys()
    print("\nALL VERIFICATION TESTS PASSED!\n")
