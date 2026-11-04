"""
End-to-End Orchestrator Pipeline for Supply Chain Control Tower
Executes data ingestion, feature generation, multi-model probabilistic training,
OR Newsvendor inventory optimization, and Star Schema export for Power BI.
"""

import os
import sys
import time

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline.fetch_dataset import generate_benchmark_m5
from src.data_pipeline.feature_engineering import build_features
from src.models.baseline_croston import evaluate_baseline_models
from src.models.lgbm_quantile import train_quantile_lgbm
from src.models.deep_probabilistic import train_deep_probabilistic_model
from src.models.evaluate_benchmark import evaluate_all_models
from src.explainability.shap_explainer import run_shap_analysis
from src.inventory_engine.newsvendor_optimizer import compute_newsvendor_policies
from src.inventory_engine.export_star_schema import export_powerbi_tables

def main():
    print("="*75)
    print("   SUPPLY CHAIN CONTROL TOWER: PROBABILISTIC DEMAND & INVENTORY ENGINE")
    print("   Author: Alex Liang (National Central Univ / NUK / NKUST)")
    print("="*75 + "\n")
    
    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    powerbi_dir = os.path.join(PROJECT_ROOT, "powerbi", "data")
    
    # Step 1: Ingestion / Generation
    print("[1/6] Ingesting Walmart M5 supply chain benchmark dataset...")
    df_sales, df_calendar = generate_benchmark_m5(raw_dir)
    
    # Step 2: Feature Engineering
    print("\n[2/6] Building autoregressive lags, rolling statistics, and promo features...")
    train_df, test_df = build_features(raw_dir, processed_dir)
    
    # Step 3: Multi-Model Probabilistic Forecasting
    print("\n[3/6] Training & Evaluating Multi-Model Forecasting Engines...")
    time_stats = {}
    
    # 3.1 OR Baselines (Croston & MA)
    print("  -> Fitting Operations Research Baselines (Croston & MA-7)...")
    df_eval = evaluate_baseline_models(train_df, test_df)
    
    # 3.2 LightGBM Quantile Regression (P10, P50, P90)
    print("  -> Training LightGBM Pinball Quantile Models (P10, P50, P90)...")
    df_eval, lgb_models, lgb_times = train_quantile_lgbm(train_df, df_eval)
    time_stats.update(lgb_times)
    
    # 3.3 Deep Probabilistic Neural Network (PyTorch)
    print("  -> Training PyTorch Probabilistic Network (Gaussian NLL)...")
    df_eval, deep_times = train_deep_probabilistic_model(train_df, df_eval, epochs=20)
    time_stats.update(deep_times)
    
    # Step 4: Model Benchmarking
    print("\n[4/6] Generating Comprehensive Model Performance Benchmark...")
    df_benchmark = evaluate_all_models(df_eval, time_stats)
    df_benchmark.to_csv(os.path.join(processed_dir, "Model_Benchmark_Summary.csv"), index=False)
    
    # Step 5: Explainable AI (SHAP)
    print("\n[5/6] Executing SHAP Decision Attribution Analysis (Responsible AI)...")
    run_shap_analysis(lgb_models["p50"], test_df, processed_dir)
    
    # Step 6: OR Newsvendor Optimization & Star Schema Export
    print("\n[6/6] Solving Newsvendor Inventory Optimization & Exporting Star Schema...")
    df_optimized = compute_newsvendor_policies(df_eval)
    export_powerbi_tables(df_optimized, df_calendar, powerbi_dir)
    
    print("\n" + "="*75)
    print("   [SUCCESS] PIPELINE EXECUTION COMPLETE! READY FOR POWER BI IMPORT.")
    print(f"   Artifacts saved to: {powerbi_dir}")
    print("="*75)

if __name__ == "__main__":
    main()
