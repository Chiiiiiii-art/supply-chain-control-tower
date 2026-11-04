"""
Deep Probabilistic Neural Network for Time-Series Demand Forecasting
TensorFlow/Keras architecture outputting parametric distribution parameters (Mean and Variance)
to yield continuous uncertainty intervals and quantile estimates.
"""

import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict

FEATURE_NUM_COLS = [
    "sell_price", "discount_amount", "discount_ratio", "is_promo",
    "sales_lag_7", "sales_lag_14", "sales_lag_21", "sales_lag_28",
    "sales_rolling_mean_7", "sales_rolling_std_7",
    "sales_rolling_mean_14", "sales_rolling_std_14",
    "sales_rolling_mean_28", "sales_rolling_std_28",
    "zero_ratio_28", "month", "day", "is_weekend", "has_event"
]

def build_keras_probabilistic_model(input_dim: int):
    """
    Keras Functional Model outputting 2 heads: Mean (mu) and Log-Variance (log_var).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(64, activation="relu")(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Dense(32, activation="relu")(h)
    
    # Non-negative mean demand
    mu = layers.Dense(1, activation="relu", name="mu")(h)
    # Clipped log variance
    log_var = layers.Dense(1, name="log_var")(h)
    
    outputs = layers.Concatenate(axis=-1)([mu, log_var])
    model = Model(inputs=inputs, outputs=outputs)
    
    def gaussian_nll(y_true, y_pred):
        m = y_pred[:, 0:1]
        lv = tf.clip_by_value(y_pred[:, 1:2], -4.0, 4.0)
        var = tf.exp(lv)
        loss = 0.5 * (lv + tf.square(y_true - m) / (var + 1e-6))
        return tf.reduce_mean(loss)
        
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss=gaussian_nll)
    return model

def train_deep_probabilistic_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 15,
    batch_size: int = 128
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Train Deep Probabilistic Network and output P10, P50, P90 predictions.
    """
    X_train = train_df[FEATURE_NUM_COLS].values.astype(np.float32)
    y_train = train_df["sales"].values.astype(np.float32).reshape(-1, 1)
    
    X_test = test_df[FEATURE_NUM_COLS].values.astype(np.float32)
    
    # Standardization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-6
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    t0 = time.time()
    try:
        model = build_keras_probabilistic_model(input_dim=len(FEATURE_NUM_COLS))
        model.fit(
            X_train_norm, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        train_time = time.time() - t0
        
        preds = model.predict(X_test_norm, verbose=0)
        mu_np = preds[:, 0]
        log_var_np = np.clip(preds[:, 1], -4.0, 4.0)
        sigma_np = np.sqrt(np.exp(log_var_np))
    except Exception as e:
        print(f"Deep network training exception ({e}), falling back to Scikit-Learn MLP...")
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42)
        mlp.fit(X_train_norm, y_train.ravel())
        train_time = time.time() - t0
        
        mu_np = np.clip(mlp.predict(X_test_norm), 0, None)
        # Residual std estimate
        residuals = y_train.ravel() - mlp.predict(X_train_norm)
        sigma_np = np.full_like(mu_np, max(0.5, np.std(residuals)))
        
    # Analytical Normal quantiles: Z_0.10 = -1.28155, Z_0.90 = +1.28155
    pred_p10 = np.clip(mu_np - 1.28155 * sigma_np, 0, None)
    pred_p50 = mu_np
    pred_p90 = np.clip(mu_np + 1.28155 * sigma_np, 0, None)
    
    test_preds = test_df.copy()
    test_preds["pred_deep_p10"] = np.round(pred_p10, 2)
    test_preds["pred_deep_p50"] = np.round(pred_p50, 2)
    test_preds["pred_deep_p90"] = np.round(pred_p90, 2)
    test_preds["pred_deep_mu"] = np.round(mu_np, 2)
    test_preds["pred_deep_sigma"] = np.round(sigma_np, 2)
    
    print(f"Trained Deep Probabilistic Model ({epochs} epochs) in {train_time:.2f}s.")
    return test_preds, {"train_time_deep": train_time}
