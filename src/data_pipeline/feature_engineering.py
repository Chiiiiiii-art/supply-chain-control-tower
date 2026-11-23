"""
Feature Engineering Module for Supply Chain Time-Series
Constructs autoregressive lags, rolling statistics, calendar dynamics,
and price elasticity features for multi-horizon demand forecasting.
"""

import os
import pandas as pd
import numpy as np


def build_features(raw_data_dir: str, processed_data_dir: str, buffer_days: int = 12, test_days: int = 60):
    """
    Load raw sales and calendar data, generate causal predictive features,
    exact L-day lead-time targets without boundary extrapolation or future promo leakage,
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
    
    # 1. State-matched SNAP eligibility & causal promotional persistence (no collinear price duplicates)
    df["is_snap"] = np.where(
        df["state_id"] == "CA",
        df["snap_CA"],
        np.where(df["state_id"] == "TX", df["snap_TX"], df["snap_WI"])
    )
    df["promo_rate_7"] = df.groupby(["item_id", "store_id"])["is_promo"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean().fillna(0.0)
    )

    # 2. Hierarchical Fixed-Effect Encodings (alphabetical deterministic codes)
    for raw_col, code_col in [
        ("item_id", "item_code"),
        ("dept_id", "dept_code"),
        ("cat_id", "cat_code"),
        ("store_id", "store_code"),
    ]:
        df[code_col] = pd.Categorical(df[raw_col], categories=sorted(df[raw_col].unique())).codes

    # 3. Autoregressive Lag Features (short-term + weekly seasonal per series)
    lag_cols = []
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        col = f"sales_lag_{lag}"
        df[col] = df.groupby(["item_id", "store_id"])["sales"].shift(lag)
        lag_cols.append(col)
        
    # 4. Rolling Window Statistics (shifted by 1 to prevent data leakage)
    for w in [7, 14, 28]:
        df[f"sales_rolling_mean_{w}"] = df.groupby(["item_id", "store_id"])["sales"].transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
        df[f"sales_rolling_std_{w}"] = df.groupby(["item_id", "store_id"])["sales"].transform(lambda x: x.shift(1).rolling(w, min_periods=3).std().fillna(0))
        
    # 5. Zero-demand frequency in past 28 days (intermittency signal)
    df["zero_ratio_28"] = df.groupby(["item_id", "store_id"])["sales"].transform(lambda x: (x.shift(1) == 0).rolling(28, min_periods=7).mean())
    
    # 6. Drop rows only where lags are NaN (first 28 days)
    df_clean = df.dropna(subset=lag_cols).reset_index(drop=True)

    # Split dates: exclude post-test realization buffer from test_df while using it for exact LT sums
    test_end = df_clean["date"].max() - pd.Timedelta(days=buffer_days)
    train_end = test_end - pd.Timedelta(days=test_days)

    # 7. Exact L-day lead-time target and known deterministic future calendar covariates
    #    Training windows that cross train_end are masked to NaN to prevent train->test leakage.
    lt_sales = np.full(len(df_clean), np.nan, dtype=float)
    lt_wknd = np.full(len(df_clean), np.nan, dtype=float)
    lt_snap = np.full(len(df_clean), np.nan, dtype=float)
    lt_event = np.full(len(df_clean), np.nan, dtype=float)

    train_end_np = np.datetime64(train_end)
    for (_, _), grp in df_clean.groupby(["item_id", "store_id"], sort=False):
        idx = grp.index.values
        lt = max(1, int(grp["lead_time_days"].iloc[0]))
        s_arr = grp["sales"].values.astype(float)
        w_arr = grp["is_weekend"].values.astype(float)
        sn_arr = grp["is_snap"].values.astype(float)
        ev_arr = grp["has_event"].values.astype(float)
        dts = grp["date"].values
        n = len(idx)
        for i in range(n):
            if i + lt <= n:
                if dts[i] <= train_end_np and dts[i + lt - 1] > train_end_np:
                    lt_sales[idx[i]] = np.nan
                else:
                    lt_sales[idx[i]] = float(np.sum(s_arr[i : i + lt]))
                lt_wknd[idx[i]] = float(np.sum(w_arr[i : i + lt]))
                lt_snap[idx[i]] = float(np.sum(sn_arr[i : i + lt]))
                lt_event[idx[i]] = float(np.sum(ev_arr[i : i + lt]))

    df_clean["lt_sales"] = lt_sales
    df_clean["lt_wknd_days"] = lt_wknd
    df_clean["lt_snap_days"] = lt_snap
    df_clean["lt_event_days"] = lt_event

    train_df = df_clean[df_clean["date"] <= train_end].copy().reset_index(drop=True)
    test_df = df_clean[(df_clean["date"] > train_end) & (df_clean["date"] <= test_end)].copy().reset_index(drop=True)

    train_df.to_csv(os.path.join(processed_data_dir, "train_features.csv"), index=False)
    test_df.to_csv(os.path.join(processed_data_dir, "test_features.csv"), index=False)
    
    print(f"Features built successfully. Train rows: {len(train_df):,} (valid LT: {train_df['lt_sales'].notna().sum():,}), Test rows: {len(test_df):,}")
    return train_df, test_df

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(project_root, "data", "raw")
    proc_dir = os.path.join(project_root, "data", "processed")
    build_features(raw_dir, proc_dir)
