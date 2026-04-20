"""
solcast_cnn_lstm.py
--------------------
Standalone CNN-LSTM hybrid forecaster for 1-24h ahead solar PV prediction
using Solcast weather data.

Architecture
------------
  Input (batch, 24, n_features)
    └─ Conv1D(64, k=3) + Conv1D(32, k=3) + MaxPooling1D
    └─ Bidirectional LSTM(128, return_sequences=True) + Dropout
    └─ LSTM(64) + Dropout
    └─ Dense(24)  →  24 horizon outputs (MIMO)

Run
---
    python solcast_cnn_lstm.py

Outputs
-------
    solcast_lstm_solar_model.keras — trained Keras model
    scaler_features.pkl           — fitted MinMaxScaler (features)
    scaler_target.pkl             — fitted MinMaxScaler (PowerOutput)
    predictions_test.csv          — per-row test predictions (pred_h1…pred_h24)
    results/metrics_per_horizon.csv
    results/figures/
"""

# ── Imports ────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info/warning logs
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow {tf.__version__}  |  GPU: {tf.config.list_physical_devices('GPU')}")

# ── Configuration ──────────────────────────────────────────────────────────────

CFG = {
    # Data
    "data_path":       "solcast_data.csv",
    "timestamp_col":   "timestamp",
    "target_col":      "PowerOutput",          # kW
    "train_frac":      0.70,
    "val_frac":        0.20,
    # test_frac = 1 - train_frac - val_frac = 0.10

    # Sequences
    "seq_len":         24,                     # look-back window (hours)
    "n_horizons":      24,                     # forecast horizons (hours ahead)

    # Model
    "cnn_filters":     [64, 32],
    "cnn_kernel":      3,
    "lstm_units":      [128, 64],
    "dropout":         0.2,
    "bidirectional":   True,                   # BiLSTM for first LSTM layer

    # Training
    "epochs":          100,
    "batch_size":      32,
    "learning_rate":   0.001,
    "early_stop_patience":  10,
    "reduce_lr_patience":   5,
    "reduce_lr_factor":     0.5,

    # Cloud regime threshold
    "cloud_clear_thresh": 30.0,               # % — below = clear sky

    # Output
    "model_path":      "solcast_lstm_solar_model.keras",
    "scaler_feat_path": "scaler_features.pkl",
    "scaler_tgt_path":  "scaler_target.pkl",
    "pred_csv_path":   "predictions_test.csv",
    "results_dir":     Path("results"),

    # Hyperparameter grid hints (for manual or Keras Tuner sweeps)
    "hp_grid": {
        "lstm_units":   [[64, 32], [128, 64], [256, 128]],
        "dropout":      [0.1, 0.2, 0.3],
        "n_layers":     [1, 2, 3],
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size":   [32, 64],
    },
}

RESULTS_DIR = CFG["results_dir"]
FIGS_DIR    = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Data Loading ────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load Solcast CSV and return a clean hourly DataFrame.

    Expected columns (case-insensitive):
        timestamp, DHI, DNI, GHI, Temperature, CloudCover,
        WindSpeed, Pressure, Dewpoint, PowerOutput
    """
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Parse timestamp → DatetimeIndex
    ts_col = CFG["timestamp_col"]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False)
    df = df.set_index(ts_col).sort_index()

    # Resample to hourly if finer resolution is supplied
    if df.index.inferred_freq not in ("H", "h", "60T", "60min"):
        df = df.resample("1h").mean()

    # Forward-fill then backward-fill short gaps (≤3 h)
    df = df.ffill(limit=3).bfill(limit=3)

    # Drop rows where the target is still NaN
    df = df.dropna(subset=[CFG["target_col"]])

    # Clip negative power to zero (sensor artefacts at night)
    df[CFG["target_col"]] = df[CFG["target_col"]].clip(lower=0.0)

    print(f"Loaded: {len(df):,} rows  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── 2. Feature Engineering ─────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific features to the hourly DataFrame."""
    df = df.copy()
    pv  = CFG["target_col"]
    ghi = "GHI"
    dhi = "DHI"
    cc  = "CloudCover"

    # ── Lag features (past 24h PowerOutput) ────────────────────────────────────
    for lag in range(1, 25):
        df[f"pv_lag_{lag}h"] = df[pv].shift(lag)

    # ── Rolling means ──────────────────────────────────────────────────────────
    for window in [3, 6, 24]:
        df[f"GHI_roll{window}"]   = df[ghi].shift(1).rolling(window, min_periods=1).mean()
        df[f"DHI_roll{window}"]   = df[dhi].shift(1).rolling(window, min_periods=1).mean()
        df[f"pv_roll{window}"]    = df[pv].shift(1).rolling(window, min_periods=1).mean()

    # ── Cyclical time encodings ────────────────────────────────────────────────
    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour        / 24.0)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour        / 24.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month       / 12.0)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month       / 12.0)

    # ── Cloud regime flag ──────────────────────────────────────────────────────
    if cc in df.columns:
        df["is_clear"] = (df[cc] < CFG["cloud_clear_thresh"]).astype(float)

    # ── Drop NaN rows introduced by lagging ───────────────────────────────────
    df = df.dropna()
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature columns (everything except the target)."""
    return [c for c in df.columns if c != CFG["target_col"]]


# ── 3. Train / Val / Test Split ────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames — chronological, no shuffle."""
    n = len(df)
    n_train = int(n * CFG["train_frac"])
    n_val   = int(n * CFG["val_frac"])
    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val]
    test  = df.iloc[n_train + n_val:]
    print(
        f"Split — train: {len(train):,}  val: {len(val):,}  test: {len(test):,} rows"
    )
    return train, val, test


# ── 4. Normalisation ───────────────────────────────────────────────────────────

def fit_scalers(
    train: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[MinMaxScaler, MinMaxScaler]:
    """Fit MinMaxScaler on train features and train target separately."""
    scaler_feat = MinMaxScaler(feature_range=(0, 1))
    scaler_tgt  = MinMaxScaler(feature_range=(0, 1))
    scaler_feat.fit(train[feature_cols].values)
    scaler_tgt.fit(train[[CFG["target_col"]]].values)
    return scaler_feat, scaler_tgt


def scale_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler_feat: MinMaxScaler,
    scaler_tgt: MinMaxScaler,
) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols]              = scaler_feat.transform(df[feature_cols].values)
    out[[CFG["target_col"]]]       = scaler_tgt.transform(df[[CFG["target_col"]]].values)
    return out


# ── 5. Sequence Generation ─────────────────────────────────────────────────────

def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int    = CFG["seq_len"],
    n_horizons: int = CFG["n_horizons"],
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Slide a window over the DataFrame to produce (X, y) arrays.

    Sequences with any NaN in y are dropped.  NaN feature values are
    zero-filled so they don't break training.

    Returns
    -------
    X : float32  (n_samples, seq_len, n_features)
    y : float32  (n_samples, n_horizons)
    idx : DatetimeIndex  — timestamp of the *first* forecast step
    """
    target_col = CFG["target_col"]
    # Zero-fill NaN features; keep target NaN so we can filter below
    X_arr = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y_arr = df[target_col].values.astype(np.float32)

    X_list, y_list, idx_list = [], [], []
    for i in range(len(df) - seq_len - n_horizons + 1):
        y = y_arr[i + seq_len : i + seq_len + n_horizons]
        if np.any(np.isnan(y)):           # skip sequences with missing targets
            continue
        X_list.append(X_arr[i : i + seq_len])
        y_list.append(y)
        idx_list.append(df.index[i + seq_len])

    if not X_list:
        raise ValueError("No valid sequences found — check for NaN in target column.")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y, pd.DatetimeIndex(idx_list)


# ── 6. Horizon-weighted loss (Step 2) ─────────────────────────────────────────

def horizon_weighted_mse(n_horizons: int = 24, max_weight: float = 2.5):
    """MSE loss with a linear ramp of per-horizon weights.

    Weight rises from 1.0 at h+1 to max_weight at h+24 and is
    mean-normalised so the overall loss magnitude stays comparable
    to plain MSE (no learning-rate adjustment required).

    Args:
        n_horizons:  number of output steps (default 24)
        max_weight:  weight assigned to the last horizon (default 2.5)
    """
    weights = np.linspace(1.0, max_weight, n_horizons).astype(np.float32)
    weights /= weights.mean()                          # keep loss scale stable
    w_tensor = tf.constant(weights, dtype=tf.float32)  # shape (n_horizons,)

    def loss_fn(y_true, y_pred):
        sq_err = tf.square(y_pred - y_true)            # (batch, n_horizons)
        return tf.reduce_mean(sq_err * w_tensor)        # scalar

    loss_fn.__name__ = "horizon_weighted_mse"
    return loss_fn


# ── 7. Model Architecture ──────────────────────────────────────────────────────

def build_cnn_lstm(
    seq_len: int,
    n_features: int,
    n_horizons: int,
    cnn_filters: list[int]  = CFG["cnn_filters"],
    cnn_kernel: int         = CFG["cnn_kernel"],
    lstm_units: list[int]   = CFG["lstm_units"],
    dropout: float          = CFG["dropout"],
    bidirectional: bool     = CFG["bidirectional"],
    learning_rate: float    = CFG["learning_rate"],
) -> keras.Model:
    """Build and compile the CNN-LSTM hybrid model.

    Extending to other architectures
    ---------------------------------
    - Pure BiLSTM:     set cnn_filters=[]
    - Transformer:     replace CNN+LSTM blocks with MultiHeadAttention (see below)
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="input_seq")
    x = inputs

    # ── CNN block: local pattern extraction ───────────────────────────────────
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=cnn_kernel,
            padding="same",
            activation="relu",
            name=f"conv_{i+1}",
        )(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)

    # ── LSTM block ────────────────────────────────────────────────────────────
    for j, units in enumerate(lstm_units):
        is_last = (j == len(lstm_units) - 1)
        return_seq = not is_last        # only last LSTM returns single vector
        if j == 0 and bidirectional:
            x = layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_seq),
                name=f"bilstm_{j+1}",
            )(x)
        else:
            x = layers.LSTM(units, return_sequences=return_seq, name=f"lstm_{j+1}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{j+1}")(x)

    # ── Output: MIMO — all 24 horizons at once ────────────────────────────────
    outputs = layers.Dense(n_horizons, name="forecast")(x)

    model = keras.Model(inputs, outputs, name="CNN_LSTM_Solar")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",     # Step 2 horizon-weighted loss did not improve results; reverted
        metrics=["mae"],
    )
    model.summary()
    return model


def build_attention_encoder_decoder(
    seq_len: int,
    n_features: int,
    n_horizons: int,
    cnn_filters: list[int]  = CFG["cnn_filters"],
    cnn_kernel: int         = CFG["cnn_kernel"],
    enc_units: int          = 128,
    dec_units: int          = 128,
    n_heads: int            = 4,
    key_dim: int            = 64,
    ff_dim: int             = 256,
    dropout: float          = CFG["dropout"],
    learning_rate: float    = CFG["learning_rate"],
) -> keras.Model:
    """CNN + BiLSTM Encoder with Multi-Head Cross-Attention decoder (Step 4).

    Architecture
    ------------
    Input (batch, seq_len, n_features)
      └─ Conv1D(64) + Conv1D(32) + MaxPooling1D       local pattern extraction
      └─ BiLSTM(128, return_sequences=True)            encoder states: all timesteps
      └─ MultiHeadAttention (4 heads)                  cross-attention over encoder
           query  = last encoder hidden (batch, 1, 256)
           key/value = all encoder states (batch, T, 256)
           → attended context (batch, 1, 256)
      └─ LayerNorm + residual + Feed-Forward           Transformer-style refinement
      └─ Dense(ff_dim, relu) + Dropout
      └─ Dense(n_horizons)                             MIMO: all 24 horizons

    Why this helps h+24
    -------------------
    The attention layer can learn to assign high weight to encoder states
    at t-24 (yesterday same hour) when forming the h+24 prediction, without
    relying on the LSTM to carry that signal through 24 recurrent steps.
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="input_seq")
    x = inputs

    # ── CNN: local temporal pattern extraction ────────────────────────────────
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=cnn_kernel,
            padding="same",
            activation="relu",
            name=f"enc_conv_{i+1}",
        )(x)
    x = layers.MaxPooling1D(pool_size=2, name="enc_maxpool")(x)

    # ── BiLSTM encoder: returns ALL hidden states ─────────────────────────────
    enc_out = layers.Bidirectional(
        layers.LSTM(enc_units, return_sequences=True),
        name="enc_bilstm",
    )(x)                                              # (batch, T//2, 2*enc_units)
    enc_out = layers.Dropout(dropout, name="enc_dropout")(enc_out)

    # ── Cross-Attention: query = last encoder state ───────────────────────────
    # Last hidden state summarises "now"; attention finds the most relevant
    # past timesteps (e.g. same hour yesterday) for each decode step.
    query = enc_out[:, -1:, :]                        # (batch, 1, 2*enc_units)
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=key_dim,
        dropout=dropout,
        name="cross_attention",
    )(query=query, value=enc_out, key=enc_out)        # (batch, 1, 2*enc_units)

    # Residual + LayerNorm (Transformer-style stabilisation)
    attn_out = layers.Add(name="attn_residual")([query, attn_out])
    attn_out = layers.LayerNormalization(name="attn_layernorm")(attn_out)

    # Squeeze attended context to 1-D
    ctx = layers.Flatten(name="ctx_flatten")(attn_out)   # (batch, 2*enc_units)

    # ── Feed-forward decoder head ─────────────────────────────────────────────
    x = layers.Dense(ff_dim, activation="relu", name="dec_ff1")(ctx)
    x = layers.Dropout(dropout, name="dec_dropout")(x)
    x = layers.Dense(ff_dim // 2, activation="relu", name="dec_ff2")(x)

    # ── MIMO output: all horizons at once ─────────────────────────────────────
    outputs = layers.Dense(n_horizons, name="forecast")(x)

    model = keras.Model(inputs, outputs, name="AttentionEncoderDecoder_Solar")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()
    return model


def build_transformer(
    seq_len: int,
    n_features: int,
    n_horizons: int,
    d_model: int   = 64,
    n_heads: int   = 4,
    ff_dim: int    = 128,
    dropout: float = 0.1,
    learning_rate: float = CFG["learning_rate"],
) -> keras.Model:
    """Optional Transformer alternative — swap in by calling this instead of build_cnn_lstm."""
    inputs = keras.Input(shape=(seq_len, n_features), name="input_seq")

    x = layers.Dense(d_model)(inputs)               # project to model dim
    x = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_horizons)(x)

    model = keras.Model(inputs, outputs, name="Transformer_Solar")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()
    return model


# ── 7. Training ────────────────────────────────────────────────────────────────

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> keras.callbacks.History:
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CFG["early_stop_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=CFG["reduce_lr_factor"],
            patience=CFG["reduce_lr_patience"],
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        callbacks=callbacks,
        shuffle=False,      # time series — preserve order
        verbose=1,
    )
    print(f"Best val_loss: {min(history.history['val_loss']):.5f}")
    return history


# ── 8. Evaluation ──────────────────────────────────────────────────────────────

def inverse_scale_y(
    y_norm: np.ndarray,
    scaler_tgt: MinMaxScaler,
) -> np.ndarray:
    """Inverse-transform a (n_samples, n_horizons) normalised array → kW."""
    n, h = y_norm.shape
    flat = y_norm.reshape(-1, 1)
    inv  = scaler_tgt.inverse_transform(flat).reshape(n, h)
    return np.clip(inv, 0.0, None)


def mape_score(y_true: np.ndarray, y_pred: np.ndarray, thresh: float = 1.0) -> float:
    """MAPE computed only where y_true > thresh (avoids division by near-zero at night)."""
    mask = y_true > thresh
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0)


def compute_metrics_per_horizon(
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
    n_horizons: int = CFG["n_horizons"],
) -> pd.DataFrame:
    """Compute RMSE / MAE / MAPE / R² for each forecast horizon (h+1 … h+24)."""
    rows = []
    for h in range(n_horizons):
        yt = y_true_kw[:, h]
        yp = y_pred_kw[:, h]
        valid = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[valid], yp[valid]
        rows.append({
            "horizon_h": h + 1,
            "RMSE_kW":   float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE_kW":    float(mean_absolute_error(yt, yp)),
            "MAPE_%":    mape_score(yt, yp),
            "R2":        float(r2_score(yt, yp)),
        })
    df_m = pd.DataFrame(rows).set_index("horizon_h")

    # Overall mean row
    mean_row = df_m.mean(numeric_only=True).rename("mean")
    df_m = pd.concat([df_m, mean_row.to_frame().T])
    return df_m


# ── 9. Plotting ────────────────────────────────────────────────────────────────

def plot_training_history(history: keras.callbacks.History) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"],     label="train_mae")
    axes[1].plot(history.history["val_mae"], label="val_mae")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(FIGS_DIR / "training_history.png", dpi=120)
    plt.close(fig)
    print(f"Saved: {FIGS_DIR / 'training_history.png'}")


def plot_forecast_vs_actual(
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
    idx: pd.DatetimeIndex,
    horizon: int = 1,
    n_show: int = 7 * 24,           # show last ~7 days
) -> None:
    h = horizon - 1
    yt = y_true_kw[-n_show:, h]
    yp = y_pred_kw[-n_show:, h]
    ts = idx[-n_show:]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts, yt, label="Actual",    linewidth=1.2, alpha=0.9)
    ax.plot(ts, yp, label="Predicted", linewidth=1.2, alpha=0.8, linestyle="--")
    ax.set_title(f"Forecast vs Actual — h+{horizon}  (last {n_show} hours of test set)")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    path = FIGS_DIR / f"forecast_vs_actual_h{horizon:02d}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scatter(
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
    horizon: int = 1,
) -> None:
    h = horizon - 1
    yt = y_true_kw[:, h]
    yp = y_pred_kw[:, h]
    r2 = r2_score(yt, yp)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=4, alpha=0.3, color="steelblue")
    lim = max(yt.max(), yp.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.0, label="Perfect")
    ax.set_xlabel("Actual (kW)")
    ax.set_ylabel("Predicted (kW)")
    ax.set_title(f"Scatter — h+{horizon}  (R²={r2:.4f})")
    ax.legend()
    plt.tight_layout()
    path = FIGS_DIR / f"scatter_h{horizon:02d}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_rmse_by_horizon(metrics_df: pd.DataFrame) -> None:
    df = metrics_df.drop(index="mean", errors="ignore")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(
        x=df.index.astype(str),
        y=df["RMSE_kW"].values,
        palette="Blues_d",
        ax=ax,
    )
    ax.set_title("RMSE by Forecast Horizon")
    ax.set_xlabel("Horizon (h ahead)")
    ax.set_ylabel("RMSE (kW)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = FIGS_DIR / "rmse_by_horizon.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_r2_by_horizon(metrics_df: pd.DataFrame) -> None:
    df = metrics_df.drop(index="mean", errors="ignore")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(
        x=df.index.astype(str),
        y=df["R2"].values,
        palette="Greens_d",
        ax=ax,
    )
    ax.axhline(0.90, color="red", linestyle="--", label="R²=0.90 target")
    ax.set_ylim(0, 1.05)
    ax.set_title("R² by Forecast Horizon")
    ax.set_xlabel("Horizon (h ahead)")
    ax.set_ylabel("R²")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    path = FIGS_DIR / "r2_by_horizon.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


# ── 10. Save Outputs ───────────────────────────────────────────────────────────

def save_predictions(
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
    idx: pd.DatetimeIndex,
    n_horizons: int = CFG["n_horizons"],
) -> None:
    data = {"timestamp": idx}
    for h in range(1, n_horizons + 1):
        data[f"actual_h{h}"]    = y_true_kw[:, h - 1]
        data[f"pred_h{h}"]      = y_pred_kw[:, h - 1]
    pd.DataFrame(data).set_index("timestamp").to_csv(CFG["pred_csv_path"])
    print(f"Saved: {CFG['pred_csv_path']}")


def save_scalers(
    scaler_feat: MinMaxScaler,
    scaler_tgt: MinMaxScaler,
) -> None:
    with open(CFG["scaler_feat_path"], "wb") as fh:
        pickle.dump(scaler_feat, fh)
    with open(CFG["scaler_tgt_path"], "wb") as fh:
        pickle.dump(scaler_tgt, fh)
    print(f"Saved scalers: {CFG['scaler_feat_path']}, {CFG['scaler_tgt_path']}")


# ── 11. Sample Data Generator (when solcast_data.csv is missing) ───────────────

def _generate_sample_data(n_hours: int = 8760) -> pd.DataFrame:
    """Generate synthetic 1-year hourly Solcast-schema data for local testing.

    NOT for production — real Solcast data will yield much higher accuracy.
    """
    print("WARNING: 'solcast_data.csv' not found — generating synthetic sample data.")
    rng  = np.random.default_rng(42)
    idx  = pd.date_range("2023-01-01", periods=n_hours, freq="1h")
    hour = idx.hour.to_numpy().astype(float)
    doy  = idx.dayofyear.to_numpy().astype(float)

    # Solar elevation proxy (sinusoidal over day)
    sun  = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, 1)
    # Seasonal GHI modulation (Sri Lanka: weaker Dec-Jan, stronger Mar-Apr)
    sea  = 0.85 + 0.15 * np.sin(2 * np.pi * (doy - 80) / 365)
    cloud = rng.uniform(0, 100, n_hours)
    cloud_factor = 1.0 - 0.6 * (cloud / 100)
    ghi  = np.clip(900 * sun * sea * cloud_factor + rng.normal(0, 20, n_hours), 0, None)
    dhi  = np.clip(ghi * 0.15 + rng.normal(0, 5, n_hours), 0, None)
    dni  = np.clip(ghi * 0.85 + rng.normal(0, 15, n_hours), 0, None)
    temp = 27 + 5 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 1.5, n_hours)
    pv   = np.clip(ghi * 0.003 * (1 - 0.004 * (temp - 25)), 0, None)   # rough kW estimate

    return pd.DataFrame({
        "timestamp":   idx,
        "DHI":         dhi,
        "DNI":         dni,
        "GHI":         ghi,
        "Temperature": temp,
        "CloudCover":  cloud,
        "WindSpeed":   rng.uniform(0, 8, n_hours),
        "Pressure":    1010 + rng.normal(0, 3, n_hours),
        "Dewpoint":    temp - rng.uniform(3, 8, n_hours),
        "PowerOutput": pv,
    })


# ── 12. Parquet data loaders (4-yr synthetic train + real test) ────────────────

# Paths to the pre-built pipeline artefacts
_SYNTH_PATH = Path("data/processed/synthetic_corrected_4yr.parquet")
_REAL_PATH  = Path("data/processed/feature_matrix_lstm.parquet")

# Feature columns shared by both datasets (after lag augmentation)
_CORE_FEATURES = [
    # Irradiance / cloud
    "ghi", "dni", "dhi", "cloud_opacity",
    "ghi_clearsky_ratio", "clearness_index_hourly",
    "clearsky_ghi",                             # deterministic clearsky GHI
    # Meteorological
    "air_temp", "relative_humidity", "surface_pressure", "dewpoint_temp",
    # Physics / pvlib clearsky model output
    "pvlib_ac_W", "cos_solar_zenith", "solar_elevation_deg",
    # Time cyclicals
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
    # Monsoon regime
    "monsoon_sw", "monsoon_ne", "monsoon_inter1", "monsoon_inter2",
    # Lag features (previous-day same-hour)
    "pv_lag24", "pv_lag48", "ghi_lag24", "clearness_lag24",
    # Future-weather summaries
    "ghi_fcast_mean_24h", "ghi_fcast_max_24h",
    "total_irradiance_ahead", "daylight_hours_ahead",
    # h+24 anchor features (Step 1 improvement)
    "clearness_nwp_h24",     # ghi_fcast_h24 / clearsky_ghi_at_h+24  (normalised cloud index)
    "pvlib_clearsky_h24",    # deterministic clearsky PV output at t+24
    "air_temp_fcast_h24",    # NWP temperature at exact h+24 horizon
]
# Add 24-horizon forecast columns
_CORE_FEATURES += [f"ghi_fcast_h{h}"           for h in range(1, 25)]
_CORE_FEATURES += [f"cloud_opacity_fcast_h{h}"  for h in range(1, 25)]


def _add_lag_and_summary_features(df: pd.DataFrame, pv_col: str) -> pd.DataFrame:
    """Add lag and forecast-summary features that the synthetic parquet lacks."""
    df = df.copy()

    pv = df[pv_col]
    df["pv_lag24"]       = pv.shift(24).clip(lower=0)
    df["pv_lag48"]       = pv.shift(48).clip(lower=0)
    if "ghi" in df.columns:
        df["ghi_lag24"]  = df["ghi"].shift(24).clip(lower=0)
    if "clearness_index_hourly" in df.columns:
        df["clearness_lag24"] = df["clearness_index_hourly"].shift(24).fillna(0.0)

    fcast_cols = [f"ghi_fcast_h{h}" for h in range(1, 25)
                  if f"ghi_fcast_h{h}" in df.columns]
    if fcast_cols:
        ghi_mat = df[fcast_cols].clip(lower=0)
        df["ghi_fcast_mean_24h"]     = ghi_mat.mean(axis=1)
        df["ghi_fcast_max_24h"]      = ghi_mat.max(axis=1)
        df["total_irradiance_ahead"] = ghi_mat.sum(axis=1)
        df["daylight_hours_ahead"]   = (ghi_mat > 50).sum(axis=1).astype(float)

    # ── Step 1: h+24 anchor features ─────────────────────────────────────────
    # clearness_nwp_h24: ratio of NWP GHI forecast to deterministic clearsky GHI
    # at t+24 — directly measures cloud attenuation at the forecast horizon.
    # clearsky_ghi.shift(-24) is safe: clearsky is purely astronomical (no leakage).
    if "clearsky_ghi" in df.columns and "ghi_fcast_h24" in df.columns:
        clearsky_h24 = df["clearsky_ghi"].shift(-24).fillna(0).clip(lower=0)
        df["clearness_nwp_h24"] = (
            df["ghi_fcast_h24"] / (clearsky_h24 + 1e-6)
        ).clip(0, 1.5).fillna(0)

    # pvlib_clearsky_h24: deterministic clearsky PV power 24 h ahead — anchors
    # the upper-bound prediction regardless of cloud cover forecast.
    if "pvlib_ac_W" in df.columns:
        df["pvlib_clearsky_h24"] = df["pvlib_ac_W"].shift(-24).fillna(0).clip(lower=0)

    return df


def load_parquet_splits(
    n_horizons: int = CFG["n_horizons"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and prepare train / val / test DataFrames from the pipeline parquets.

    Returns
    -------
    train_df : synthetic 4-year data (PowerOutput = pv_corrected_W in W → kW)
    val_df   : first 85% of real data (for early stopping)
    test_df  : last 15% of real data  (held-out evaluation)
    feat_cols: ordered list of feature columns present in all three
    """
    # ── Synthetic (training) ─────────────────────────────────────────────────
    synth = pd.read_parquet(_SYNTH_PATH)
    synth = _add_lag_and_summary_features(synth, pv_col="pv_corrected_W")
    synth["PowerOutput"] = synth["pv_corrected_W"] / 1000.0   # W → kW
    synth = synth.dropna(subset=["PowerOutput", "pv_lag24"])
    print(f"Synthetic train: {len(synth):,} rows  "
          f"({synth.index[0].date()} → {synth.index[-1].date()})")

    # ── Real LSTM feature matrix (val + test) ────────────────────────────────
    real = pd.read_parquet(_REAL_PATH)
    real["PowerOutput"] = real["pv_ac_W"] / 1000.0            # W → kW
    # Always run feature augmentation so Step-1 h+24 anchors are computed
    # for both synthetic and real datasets.
    real = _add_lag_and_summary_features(real, pv_col="pv_ac_W")
    real = real.dropna(subset=["PowerOutput", "pv_lag24"])
    print(f"Real data:       {len(real):,} rows  "
          f"({real.index[0].date()} → {real.index[-1].date()})")

    # Chronological 85/15 split on real data (val / test)
    n_real = len(real)
    split  = int(n_real * 0.85)
    val_df  = real.iloc[:split]
    test_df = real.iloc[split:]
    print(f"Real val: {len(val_df):,} rows  |  Real test: {len(test_df):,} rows")

    # ── Feature intersection ─────────────────────────────────────────────────
    feat_cols = [c for c in _CORE_FEATURES
                 if c in synth.columns and c in real.columns]
    missing_synth = [c for c in _CORE_FEATURES if c not in synth.columns]
    missing_real  = [c for c in _CORE_FEATURES if c not in real.columns]
    if missing_synth:
        print(f"  (dropped from synth — absent): {missing_synth}")
    if missing_real:
        print(f"  (dropped from real  — absent): {missing_real}")
    print(f"Shared feature columns: {len(feat_cols)}")

    return synth, val_df, test_df, feat_cols


# ── 13. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    seq_len    = CFG["seq_len"]
    n_horizons = CFG["n_horizons"]

    # ── Choose data path ───────────────────────────────────────────────────────
    if _SYNTH_PATH.exists() and _REAL_PATH.exists():
        print("=== Mode: train on 4-yr calibrated synthetic, test on real PV ===")
        train_df, val_df, test_df, feat_cols = load_parquet_splits(n_horizons)
        target_col = "PowerOutput"
    else:
        print("=== Mode: single CSV (fallback) ===")
        data_path = CFG["data_path"]
        if not Path(data_path).exists():
            df_raw = _generate_sample_data()
            df_raw.to_csv(data_path, index=False)
        df_raw = load_data(data_path)
        df = add_features(df_raw)
        feat_cols = select_feature_columns(df)
        train_df, val_df, test_df = chronological_split(df)
        target_col = CFG["target_col"]

    print(f"Features ({len(feat_cols)}): {feat_cols[:6]} …")

    # ── Fit scalers on SYNTHETIC (or train) only ───────────────────────────────
    scaler_feat = MinMaxScaler(feature_range=(0, 1))
    scaler_tgt  = MinMaxScaler(feature_range=(0, 1))
    scaler_feat.fit(train_df[feat_cols].values)
    scaler_tgt.fit(train_df[[target_col]].values)

    def _scale(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        feat_vals = df[feat_cols].fillna(0.0).values
        out[feat_cols]    = scaler_feat.transform(feat_vals)
        # Target: only scale non-NaN rows (NaN rows stay NaN → filtered in make_sequences)
        tgt_vals = df[[target_col]].values.astype(float)   # (n, 1)
        mask = ~np.isnan(tgt_vals.ravel())
        scaled_tgt = tgt_vals.copy()
        scaled_tgt[mask, 0] = scaler_tgt.transform(
            tgt_vals[mask].reshape(-1, 1)
        ).ravel()
        out[[target_col]] = scaled_tgt
        return out

    train_s = _scale(train_df)
    val_s   = _scale(val_df)
    test_s  = _scale(test_df)

    # ── Sequences ──────────────────────────────────────────────────────────────
    X_train, y_train, _        = make_sequences(train_s, feat_cols, seq_len, n_horizons)
    X_val,   y_val,   _        = make_sequences(val_s,   feat_cols, seq_len, n_horizons)
    X_test,  y_test,  idx_test = make_sequences(test_s,  feat_cols, seq_len, n_horizons)
    print(
        f"Sequences — train (synthetic): {X_train.shape}  "
        f"val (real): {X_val.shape}  test (real): {X_test.shape}"
    )

    # ── Build model ────────────────────────────────────────────────────────────
    n_features = X_train.shape[2]
    model = build_cnn_lstm(                    # best model: Step 1 (83 features)
        seq_len=seq_len,
        n_features=n_features,
        n_horizons=n_horizons,
    )

    # ── Train (on synthetic; early-stop on real val) ───────────────────────────
    history = train_model(model, X_train, y_train, X_val, y_val)
    plot_training_history(history)

    model.save(CFG["model_path"])
    print(f"Saved: {CFG['model_path']}")

    # ── Predict on real test set & inverse-scale ───────────────────────────────
    y_pred_norm = model.predict(X_test, batch_size=CFG["batch_size"], verbose=0)
    y_pred_kw   = inverse_scale_y(y_pred_norm, scaler_tgt)
    y_true_kw   = inverse_scale_y(y_test,      scaler_tgt)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics_df = compute_metrics_per_horizon(y_true_kw, y_pred_kw, n_horizons)
    metrics_df.to_csv(RESULTS_DIR / "metrics_per_horizon.csv")
    print("\n=== Per-horizon metrics (real test set) ===")
    print(metrics_df.to_string())

    mean = metrics_df.loc["mean"]
    print(
        f"\nMean — RMSE={mean['RMSE_kW']:.3f} kW  "
        f"MAE={mean['MAE_kW']:.3f} kW  "
        f"MAPE={mean['MAPE_%']:.2f}%  "
        f"R²={mean['R2']:.4f}"
    )
    if mean["R2"] >= 0.90:
        print("Target R² ≥ 0.90 ACHIEVED.")
    else:
        print(
            f"R²={mean['R2']:.4f} < 0.90 — try lstm_units=[256,128], "
            "dropout=0.1, or add a fine-tune pass on real val data."
        )

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_rmse_by_horizon(metrics_df)
    plot_r2_by_horizon(metrics_df)
    for h in [1, 6, 12, 24]:
        plot_forecast_vs_actual(y_true_kw, y_pred_kw, idx_test, horizon=h)
        plot_scatter(y_true_kw, y_pred_kw, horizon=h)

    # ── Save outputs ───────────────────────────────────────────────────────────
    save_predictions(y_true_kw, y_pred_kw, idx_test, n_horizons)
    save_scalers(scaler_feat, scaler_tgt)

    print("\n=== Done ===")
    print(f"  Model:       {CFG['model_path']}")
    print(f"  Predictions: {CFG['pred_csv_path']}")
    print(f"  Metrics:     {RESULTS_DIR / 'metrics_per_horizon.csv'}")
    print(f"  Figures:     {FIGS_DIR}/")

    print("\n=== Hyperparameter tuning grid ===")
    for k, v in CFG["hp_grid"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
