"""
Data Acquisition & Ingestion Module for Walmart M5 Supply Chain Benchmark
Generates or ingests multi-echelon retail time-series with realistic intermittent demand,
promotions, calendar events, price elasticity, and cost parameters.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_benchmark_m5(output_dir: str, n_years: int = 2, seed: int = 42):
    """
    Synthesize an authentic Walmart M5-compatible supply chain dataset
    spanning 10 SKUs across 3 Departments and 3 Stores over 2 years.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    start_date = datetime(2024, 1, 1)
    n_days = n_years * 365
    date_list = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # 1. Product Hierarchy
    products = [
        {"item_id": "FOODS_1_001", "dept_id": "FOODS_1", "cat_id": "FOODS", "base_price": 3.99, "cu_ratio": 0.85, "holding_rate": 0.02, "lead_time": 3},
        {"item_id": "FOODS_2_045", "dept_id": "FOODS_2", "cat_id": "FOODS", "base_price": 7.50, "cu_ratio": 0.80, "holding_rate": 0.02, "lead_time": 2},
        {"item_id": "FOODS_3_120", "dept_id": "FOODS_3", "cat_id": "FOODS", "base_price": 2.25, "cu_ratio": 0.90, "holding_rate": 0.015, "lead_time": 2},
        {"item_id": "HOBBIES_1_012", "dept_id": "HOBBIES_1", "cat_id": "HOBBIES", "base_price": 14.99, "cu_ratio": 0.70, "holding_rate": 0.01, "lead_time": 7},
        {"item_id": "HOBBIES_2_008", "dept_id": "HOBBIES_2", "cat_id": "HOBBIES", "base_price": 24.50, "cu_ratio": 0.65, "holding_rate": 0.01, "lead_time": 10},
        {"item_id": "HOUSEHOLD_1_050", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD", "base_price": 9.99, "cu_ratio": 0.75, "holding_rate": 0.015, "lead_time": 5},
        {"item_id": "HOUSEHOLD_1_180", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD", "base_price": 12.80, "cu_ratio": 0.75, "holding_rate": 0.015, "lead_time": 5},
        {"item_id": "HOUSEHOLD_2_032", "dept_id": "HOUSEHOLD_2", "cat_id": "HOUSEHOLD", "base_price": 18.00, "cu_ratio": 0.68, "holding_rate": 0.012, "lead_time": 6},
        {"item_id": "FOODS_1_099", "dept_id": "FOODS_1", "cat_id": "FOODS", "base_price": 4.50, "cu_ratio": 0.88, "holding_rate": 0.02, "lead_time": 3},
        {"item_id": "HOBBIES_1_340", "dept_id": "HOBBIES_1", "cat_id": "HOBBIES", "base_price": 29.99, "cu_ratio": 0.60, "holding_rate": 0.008, "lead_time": 12},
    ]
    
    # 2. Stores
    stores = [
        {"store_id": "CA_1", "state_id": "CA", "store_factor": 1.25},
        {"store_id": "TX_1", "state_id": "TX", "store_factor": 1.00},
        {"store_id": "WI_1", "state_id": "WI", "store_factor": 0.85},
    ]
    
    # 3. Calendar & Event Data
    calendar_rows = []
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
    
    # 4. Generate Daily Sales Transactions
    records = []
    for store in stores:
        s_id = store["store_id"]
        s_mult = store["store_factor"]
        
        for prod in products:
            item_id = prod["item_id"]
            base_p = prod["base_price"]
            base_demand = 8.0 if "FOODS" in item_id else (2.5 if "HOBBIES" in item_id else 4.0)
            
            for cal in calendar_rows:
                d_str = cal["date"]
                month = cal["month"]
                is_wknd = cal["is_weekend"]
                has_evt = cal["has_event"]
                
                # Seasonality & Promo simulation
                seasonality = 1.0 + 0.25 * np.sin(2 * np.pi * month / 12)
                weekend_boost = 1.35 if (is_wknd and "FOODS" in item_id) else (1.15 if is_wknd else 1.0)
                event_boost = 1.60 if has_evt else 1.0
                
                # Price discount promo (15% chance of promo)
                is_promo = 1 if np.random.rand() < 0.15 else 0
                price = base_p * (0.80 if is_promo else 1.0)
                price_elasticity = (base_p / price) ** 1.5
                
                # Intermittent demand (zeros probability)
                zero_prob = 0.10 if "FOODS" in item_id else (0.45 if "HOBBIES" in item_id else 0.25)
                
                expected_lambda = base_demand * s_mult * seasonality * weekend_boost * event_boost * price_elasticity
                
                if np.random.rand() < zero_prob:
                    sales = 0
                else:
                    sales = int(np.random.poisson(lam=max(0.5, expected_lambda)))
                
                records.append({
                    "date": d_str,
                    "item_id": item_id,
                    "dept_id": prod["dept_id"],
                    "cat_id": prod["cat_id"],
                    "store_id": s_id,
                    "state_id": store["state_id"],
                    "sell_price": round(price, 2),
                    "base_price": round(base_p, 2),
                    "is_promo": is_promo,
                    "lead_time_days": prod["lead_time"],
                    "unit_stockout_cost": round(price * prod["cu_ratio"] * 2.5, 2), # Lost sales + goodwill penalty
                    "unit_holding_cost": round(price * prod["holding_rate"], 3),     # Daily holding cost rate
                    "sales": sales
                })
                
    df_sales = pd.DataFrame(records)
    df_sales.to_csv(os.path.join(output_dir, "sales_transactions.csv"), index=False)
    
    print(f"Generated {len(df_sales):,} sales transactions across {len(products)} SKUs and {len(stores)} Stores.")
    return df_sales, df_calendar

if __name__ == "__main__":
    out_dir = r"C:\Users\alexl\Desktop\Programs\supply-chain-control-tower\data\raw"
    generate_benchmark_m5(out_dir)
