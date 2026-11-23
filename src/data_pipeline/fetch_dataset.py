"""
Data Acquisition & Simulation Module for M5-Calibrated Synthetic Retail Benchmark
Generates multi-echelon retail time-series using a Zero-Inflated Autoregressive (AR(1))
Negative Binomial / Poisson DGP with Markov promotional campaigns, state-level SNAP
effects, calendar events, price elasticity, and calibrated Newsvendor cost ratios.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_benchmark_m5(output_dir: str, n_years: int = 2, seed: int = 42, buffer_days: int = 12):
    """
    Synthesize an M5-calibrated retail supply chain benchmark dataset
    spanning 10 SKUs across 3 Departments and 3 Stores over 2 years
    plus post-horizon buffer days for exact lead-time target realization.
    Uses a Zero-Inflated AR(1) overdispersed count DGP and 2-state Markov promotions.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    start_date = datetime(2024, 1, 1)
    n_days = n_years * 365 + buffer_days
    date_list = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # 1. Product Hierarchy (calibrated so CR = Cu/(Cu+Co) spans ~0.72 to 0.90)
    products = [
        {"item_id": "FOODS_1_001", "dept_id": "FOODS_1", "cat_id": "FOODS", "base_price": 3.99, "cu_ratio": 0.55, "holding_rate": 0.025, "lead_time": 3},
        {"item_id": "FOODS_2_045", "dept_id": "FOODS_2", "cat_id": "FOODS", "base_price": 7.50, "cu_ratio": 0.45, "holding_rate": 0.025, "lead_time": 2},
        {"item_id": "FOODS_3_120", "dept_id": "FOODS_3", "cat_id": "FOODS", "base_price": 2.25, "cu_ratio": 0.42, "holding_rate": 0.025, "lead_time": 2},
        {"item_id": "HOBBIES_1_012", "dept_id": "HOBBIES_1", "cat_id": "HOBBIES", "base_price": 14.99, "cu_ratio": 0.48, "holding_rate": 0.022, "lead_time": 7},
        {"item_id": "HOBBIES_2_008", "dept_id": "HOBBIES_2", "cat_id": "HOBBIES", "base_price": 24.50, "cu_ratio": 0.52, "holding_rate": 0.018, "lead_time": 10},
        {"item_id": "HOUSEHOLD_1_050", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD", "base_price": 9.99, "cu_ratio": 0.50, "holding_rate": 0.024, "lead_time": 5},
        {"item_id": "HOUSEHOLD_1_180", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD", "base_price": 12.80, "cu_ratio": 0.55, "holding_rate": 0.023, "lead_time": 5},
        {"item_id": "HOUSEHOLD_2_032", "dept_id": "HOUSEHOLD_2", "cat_id": "HOUSEHOLD", "base_price": 18.00, "cu_ratio": 0.52, "holding_rate": 0.024, "lead_time": 6},
        {"item_id": "FOODS_1_099", "dept_id": "FOODS_1", "cat_id": "FOODS", "base_price": 4.50, "cu_ratio": 0.48, "holding_rate": 0.025, "lead_time": 3},
        {"item_id": "HOBBIES_1_340", "dept_id": "HOBBIES_1", "cat_id": "HOBBIES", "base_price": 29.99, "cu_ratio": 0.56, "holding_rate": 0.018, "lead_time": 12},
    ]
    
    # 2. Stores
    stores = [
        {"store_id": "CA_1", "state_id": "CA", "store_factor": 1.25},
        {"store_id": "TX_1", "state_id": "TX", "store_factor": 1.00},
        {"store_id": "WI_1", "state_id": "WI", "store_factor": 0.85},
    ]
    
    # 3. Calendar & Event Data
    calendar_rows = []
    doy_list = []
    events = {
        (1, 1): "NewYear", (2, 14): "ValentinesDay", (7, 4): "IndependenceDay",
        (10, 31): "Halloween", (11, 26): "Thanksgiving", (12, 25): "Christmas"
    }
    
    for dt in date_list:
        event = events.get((dt.month, dt.day), "None")
        is_weekend = 1 if dt.weekday() in [5, 6] else 0
        snap_ca = 1 if dt.day in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] else 0
        snap_tx = 1 if dt.day in [1, 3, 5, 6, 7, 8, 10, 11, 12, 13] else 0
        snap_wi = 1 if dt.day in [2, 3, 5, 8, 9, 11, 12, 14, 15, 17] else 0
        
        doy_list.append(dt.timetuple().tm_yday)
        calendar_rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "weekday": dt.strftime("%A"),
            "is_weekend": is_weekend,
            "event_name": event,
            "has_event": 1 if event != "None" else 0,
            "snap_CA": snap_ca,
            "snap_TX": snap_tx,
            "snap_WI": snap_wi
        })
    df_calendar = pd.DataFrame(calendar_rows)
    df_calendar.to_csv(os.path.join(output_dir, "calendar.csv"), index=False)
    
    # 4. Generate Daily Sales Transactions (Zero-Inflated AR(1) NB-DGP)
    records = []
    snap_col_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    nb_shape = 6.0  # Overdispersion shape parameter for Gamma-Poisson (Negative Binomial)

    for store in stores:
        s_id = store["store_id"]
        st_id = store["state_id"]
        s_mult = store["store_factor"]
        snap_key = snap_col_map[st_id]
        
        for prod in products:
            item_id = prod["item_id"]
            base_p = prod["base_price"]
            base_demand = 8.0 if "FOODS" in item_id else (2.5 if "HOBBIES" in item_id else 4.0)
            zero_prob = 0.10 if "FOODS" in item_id else (0.45 if "HOBBIES" in item_id else 0.25)

            # Time-invariant SKU unit costs anchored to base price
            unit_stockout_cost = round(base_p * prod["cu_ratio"], 2)
            unit_holding_cost = round(base_p * prod["holding_rate"] * prod["lead_time"], 3)

            prev_promo = 0
            prev_sales = base_demand * s_mult
            
            for cal, doy in zip(calendar_rows, doy_list):
                d_str = cal["date"]
                is_wknd = cal["is_weekend"]
                has_evt = cal["has_event"]
                is_snap = cal[snap_key]
                
                # Smooth day-of-year seasonality peaking in late November/December
                seasonality = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 335) / 365.25)
                weekend_boost = 1.35 if (is_wknd and "FOODS" in item_id) else (1.15 if is_wknd else 1.0)
                event_boost = 1.60 if has_evt else 1.0
                snap_boost = (1.25 if "FOODS" in item_id else 1.05) if is_snap else 1.0
                
                # 2-state Markov promotional campaign (clustered promo windows)
                p_enter_or_stay = 0.65 if prev_promo == 1 else 0.06
                is_promo = 1 if np.random.rand() < p_enter_or_stay else 0
                prev_promo = is_promo

                price = base_p * (0.80 if is_promo else 1.0)
                price_elasticity = (base_p / price) ** 1.5
                
                # Autoregressive AR(1) conditional demand intensity
                base_rate = base_demand * s_mult * seasonality * weekend_boost * event_boost * snap_boost * price_elasticity
                expected_lambda = 0.55 * base_rate + 0.45 * prev_sales * (price_elasticity if is_promo else 1.0)
                
                # State-dependent zero-inflation (reduced intermittency during promos/events)
                eff_zero_prob = zero_prob * (0.35 if (is_promo or has_evt) else 1.0)
                if np.random.rand() < eff_zero_prob:
                    sales = 0
                else:
                    gamma_rate = np.random.gamma(shape=nb_shape, scale=max(0.5, expected_lambda) / nb_shape)
                    sales = int(np.random.poisson(lam=gamma_rate))
                prev_sales = float(sales)
                
                records.append({
                    "date": d_str,
                    "item_id": item_id,
                    "dept_id": prod["dept_id"],
                    "cat_id": prod["cat_id"],
                    "store_id": s_id,
                    "state_id": st_id,
                    "sell_price": round(price, 2),
                    "base_price": round(base_p, 2),
                    "is_promo": is_promo,
                    "lead_time_days": prod["lead_time"],
                    "unit_stockout_cost": unit_stockout_cost,
                    "unit_holding_cost": unit_holding_cost,
                    "sales": sales
                })
                
    df_sales = pd.DataFrame(records)
    df_sales.to_csv(os.path.join(output_dir, "sales_transactions.csv"), index=False)
    
    print(f"Generated {len(df_sales):,} sales transactions across {len(products)} SKUs and {len(stores)} Stores.")
    return df_sales, df_calendar

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(project_root, "data", "raw")
    generate_benchmark_m5(out_dir)
