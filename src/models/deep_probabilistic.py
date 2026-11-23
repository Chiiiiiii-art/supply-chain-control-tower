"""
Deep Probabilistic Neural Network for Count Demand Forecasting (ZINB-NLL)
TensorFlow/Keras feedforward architecture with one-hot categorical encodings outputting
parametric Zero-Inflated Negative Binomial parameters (mean mu, shape r, zero-gate pi)
for discrete overdispersed intermittent demand intervals.
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from typing import Tuple, Dict
from src.models.lgbm_quantile import FEATURE_COLS, CATEGORICAL_COLS

CONT_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]


def _encode_tabular_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize continuous covariates and one-hot encode nominal hierarchy IDs."""
    X_tr_cont = train_df[CONT_COLS].values.astype(np.float32)
    X_te_cont = test_df[CONT_COLS].values.astype(np.float32)
    mean = np.mean(X_tr_cont, axis=0)
    std = np.std(X_tr_cont, axis=0) + 1e-6
    X_tr_norm = (X_tr_cont - mean) / std
    X_te_norm = (X_te_cont - mean) / std

    oh_tr, oh_te = [], []
    for col in CATEGORICAL_COLS:
        n_classes = int(train_df[col].max()) + 1
        eye = np.eye(n_classes, dtype=np.float32)
        oh_tr.append(eye[np.clip(train_df[col].values.astype(int), 0, n_classes - 1)])
        oh_te.append(eye[np.clip(test_df[col].values.astype(int), 0, n_classes - 1)])

    return np.hstack([X_tr_norm] + oh_tr), np.hstack([X_te_norm] + oh_te)


def build_keras_probabilistic_model(input_dim: int):
    """
    Keras Functional Model outputting 3 unconstrained logits mapped smoothly to
    ZINB parameters: count mean (mu), dispersion shape (r), and zero-inflation gate (pi).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(64, activation="relu")(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(0.15)(h)
    h = layers.Dense(32, activation="relu")(h)
    raw_params = layers.Dense(3, name="zinb_raw_params")(h)

    model = Model(inputs=inputs, outputs=raw_params)

    def zinb_nll(y_true, raw):
        mu = tf.nn.softplus(raw[:, 0:1]) + 1e-3
        r = 0.2 + 20.0 * tf.nn.sigmoid(raw[:, 1:2])
        pi = 0.85 * tf.nn.sigmoid(raw[:, 2:3])

        log_nb_0 = r * tf.math.log(r / (r + mu))
        ll_zero = tf.math.log(pi + (1.0 - pi) * tf.exp(log_nb_0) + 1e-8)
        ll_pos = (
            tf.math.log(1.0 - pi + 1e-8)
            + tf.math.lgamma(y_true + r)
            - tf.math.lgamma(y_true + 1.0)
            - tf.math.lgamma(r)
            + log_nb_0
            + y_true * tf.math.log(mu / (r + mu))
        )
        is_zero = tf.cast(tf.equal(y_true, 0.0), tf.float32)
        ll = is_zero * ll_zero + (1.0 - is_zero) * ll_pos
        return -tf.reduce_mean(ll)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss=zinb_nll,
    )
    return model


def _zinb_ppf(q: float, pi: np.ndarray, r: np.ndarray, p_nb: np.ndarray) -> np.ndarray:
    """Exact quantile function for Zero-Inflated Negative Binomial count distributions."""
    q_adj = np.clip((q - pi) / np.maximum(1e-6, 1.0 - pi), 0.0, 0.9999)
    return np.where(q <= pi, 0.0, nbinom.ppf(q_adj, n=r, p=p_nb)).astype(float)


def train_deep_probabilistic_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 25,
    batch_size: int = 128
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Train Deep Probabilistic ZINB-NLL Network and output expectation + P10, P50, P90 quantiles.
    """
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)

    X_train, X_test = _encode_tabular_features(train_df, test_df)
    y_train = train_df["sales"].values.astype(np.float32).reshape(-1, 1)

    train_dates = pd.to_datetime(train_df["date"])
    val_cut = train_dates.max() - pd.Timedelta(days=60)
    tr_mask = (train_dates <= val_cut).values
    val_mask = (train_dates > val_cut).values

    t0 = time.time()
    model = build_keras_probabilistic_model(input_dim=X_train.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    model.fit(
        X_train[tr_mask],
        y_train[tr_mask],
        validation_data=(X_train[val_mask], y_train[val_mask]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    train_time = time.time() - t0

    raw_preds = model.predict(X_test, verbose=0)
    mu_np = np.log1p(np.exp(np.clip(raw_preds[:, 0], -20.0, 20.0))) + 1e-3
    r_np = 0.2 + 20.0 / (1.0 + np.exp(-np.clip(raw_preds[:, 1], -20.0, 20.0)))
    pi_np = 0.85 / (1.0 + np.exp(-np.clip(raw_preds[:, 2], -20.0, 20.0)))

    p_nb = r_np / (r_np + mu_np)
    uncond_mu = (1.0 - pi_np) * mu_np
    uncond_var = (1.0 - pi_np) * (mu_np + (mu_np ** 2) / r_np + pi_np * (mu_np ** 2))
    sigma_np = np.sqrt(np.maximum(0.0, uncond_var))

    pred_p10 = _zinb_ppf(0.10, pi_np, r_np, p_nb)
    pred_p50 = np.maximum(pred_p10, _zinb_ppf(0.50, pi_np, r_np, p_nb))
    pred_p90 = np.maximum(pred_p50, _zinb_ppf(0.90, pi_np, r_np, p_nb))

    test_preds = test_df.copy()
    test_preds["pred_deep_p10"] = np.round(pred_p10, 2)
    test_preds["pred_deep_p50"] = np.round(pred_p50, 2)
    test_preds["pred_deep_p90"] = np.round(pred_p90, 2)
    test_preds["pred_deep_mu"] = np.round(uncond_mu, 2)
    test_preds["pred_deep_sigma"] = np.round(sigma_np, 2)

    print(f"Trained Deep Probabilistic ZINB-NLL Model (up to {epochs} epochs) in {train_time:.2f}s.")
    return test_preds, {"train_time_deep": train_time}
