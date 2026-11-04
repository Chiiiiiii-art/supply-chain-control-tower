"""
Feature Engineering Module for Supply Chain Time-Series
Constructs autoregressive lags, rolling statistics, calendar dynamics,
and price elasticity features for multi-horizon demand forecasting.
"""

import os
import pandas as pd
import numpy as np

def build_features(raw_data_dir: str, processed_data_dir: str):
    """
    Load raw sales and calendar data, generate predictive features,
    and output train/test splits.
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    sales_path = os.path.join(raw_data_dir, "sales_transactions.csv")
    calendar_path = os.path.join(raw_data_dir, "calendar.csv")
    
    df_sales = pd.read_csv(sales_path)
    df_calendar = pd.read_csv(calendar_path)
    
    # Merge calendar
    df = pd.merge(df_sales, df_calendar, on="date", how="left")
    df["event_name"] = df["event_name"].fillna("None")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["item_id", "store_id", "date"]).reset_index(drop=True)
    
    # 1. Price dynamics
    df["discount_amount"] = df["base_price"] - df["sell_price"]
    df["discount_ratio"] = df["discount_amount"] / df["base_price"]
    
    # 2. Lag Features (per series)
    lag_cols = []
    for lag in [7, 14, 21, 28]:
        col = f"sales_lag_{lag}"
        df[col] = df.groupby(["item_id", "store_id"])["sales"].shift(lag)
        lag_cols.append(col)
        
    # 3. Rolling Window Statistics (shifted by 1 to prevent data leakage)
    for w in [7, 14, 28]:
        df[f"sales_rolling_mean_{w}"] = df.groupby(["item_id", "store_id"])["sales"].transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
        df[f"sales_rolling_std_{w}"] = df.groupby(["item_id", "store_id"])["sales"].transform(lambda x: x.shift(1).rolling(w, min_periods=3).std().fillna(0))
        
    # 4. Zero-demand frequency in past 28 days (intermittency signal)
    df["zero_ratio_28"] = df.groupby(["item_id", "store_id"])["sales"].transform(lambda x: (x.shift(1) == 0).rolling(28, min_periods=7).mean())
    
    # 5. Drop rows only where lags are NaN (first 28 days)
    df_clean = df.dropna(subset=lag_cols).reset_index(drop=True)
    
    # Train / Test split: Last 60 days as Holdout Test Set (30 series * 60 days = 1,800 test records)
    split_date = df_clean["date"].max() - pd.Timedelta(days=60)
    train_df = df_clean[df_clean["date"] <= split_date].copy()
    test_df = df_clean[df_clean["date"] > split_date].copy()
    
    train_df.to_csv(os.path.join(processed_data_dir, "train_features.csv"), index=False)
    test_df.to_csv(os.path.join(processed_data_dir, "test_features.csv"), index=False)
    
    print(f"Features built successfully. Train rows: {len(train_df):,}, Test rows: {len(test_df):,}")
    return train_df, test_df

if __name__ == "__main__":
    raw_dir = r"C:\Users\alexl\Desktop\Programs\supply-chain-control-tower\data\raw"
    proc_dir = r"C:\Users\alexl\Desktop\Programs\supply-chain-control-tower\data\processed"
    build_features(raw_dir, proc_dir)
