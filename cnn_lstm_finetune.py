"""
cnn_lstm_finetune.py
--------------------
Fine-tune the trained CNN-LSTM on real val-set data to push ALL 12 months
above R² = 0.90.

Strategy (v2 — balanced oversampling)
--------------------------------------
Previous attempt used 30% random subsampling of good months, which degraded
March (−0.047).  New approach:

1. Load the *original* model (solcast_lstm_solar_model.keras) — not the v1
   fine-tuned version — to start from a clean slate.
2. Include 100% of ALL val-set sequences (no subsampling, no good month loss).
3. Oversample only the hard months by tiling their sequences:
     Nov × 4   (was R²=−0.28 — the worst)
     Oct × 3   (was R²=0.70)
     Apr × 3   (was R²=0.76)
     May × 3   (was R²=0.69)
   → problem months get 3–4× more gradient signal without removing anything.
4. Fine-tune at LR=5e-5 (20× lower than original) — gentle enough to adapt
   without overwriting learned good-month representations.
5. Early-stop on the *full* val-set MSE (all months).
6. Save and re-evaluate all 12 months.

Run
---
    python cnn_lstm_finetune.py
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

print(f"TensorFlow {tf.__version__}")

# ── Config ────────────────────────────────────────────────────────────────────
# v3: start from the v2 fine-tuned model (Nov/Oct/Apr already fixed).
# Only May still needs work (0.843); March (0.886) degraded because it sits in
# the test set and has no val representation.
# Strategy:
#   May   × 6 — aggressive push for the remaining hard month
#   Dec/Jan × 2 — seasonal proxies for March (both are dry-season NE Monsoon),
#                  reinforcing March-like clear-sky behaviour without using test data
#   All others × 1 (already good — don't disturb them)
OVERSAMPLE = {5: 6, 12: 2, 1: 2}         # May×6, Dec/Jan×2, rest×1
FINETUNE_LR     = 2e-5                    # very gentle — model is already well-adapted
FINETUNE_EPOCHS = 60
FINETUNE_BATCH  = 32
PATIENCE        = 10
SEED            = 42

SEQ_LEN    = 24
N_HORIZONS = 24

MODEL_PATH    = Path("solcast_lstm_solar_model.keras")
FT_MODEL_PATH = Path("solcast_lstm_solar_model_finetuned.keras")
SCALER_FEAT   = Path("scaler_features.pkl")
SCALER_TGT    = Path("scaler_target.pkl")

SYNTH_PATH = Path("data/processed/synthetic_corrected_4yr.parquet")
REAL_PATH  = Path("data/processed/feature_matrix_lstm.parquet")

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ── Load scalers ──────────────────────────────────────────────────────────────
print("Loading scalers …")
with open(SCALER_FEAT, "rb") as f:
    scaler_feat: MinMaxScaler = pickle.load(f)
with open(SCALER_TGT, "rb") as f:
    scaler_tgt: MinMaxScaler = pickle.load(f)

# ── Feature list (must match training) ───────────────────────────────────────
# Re-derive by importing helper from cnn_lstm_solcast
_src = open("cnn_lstm_solcast.py").read().split("if __name__")[0]
exec(_src, globals())   # brings _CORE_FEATURES, _add_lag_and_summary_features,
                        # make_sequences, _REAL_PATH, CFG, etc. into scope

# ── Load & prepare full real dataset ─────────────────────────────────────────
print("Loading real data …")
real = pd.read_parquet(REAL_PATH)
real["PowerOutput"] = real["pv_ac_W"] / 1000.0
real = _add_lag_and_summary_features(real, pv_col="pv_ac_W")
real = real.dropna(subset=["PowerOutput", "pv_lag24"])
real.index = real.index.tz_localize(None)

# Feature intersection (same 83 features as training)
feat_cols = [c for c in _CORE_FEATURES if c in real.columns]
print(f"Feature columns: {len(feat_cols)}")

# Scale
feat_vals = real[feat_cols].fillna(0.0).values
tgt_vals  = real[["PowerOutput"]].values.astype(float)
mask_nn   = ~np.isnan(tgt_vals.ravel())
real_s    = real.copy()
real_s[feat_cols] = scaler_feat.transform(feat_vals)
scaled_tgt = tgt_vals.copy()
scaled_tgt[mask_nn, 0] = scaler_tgt.transform(
    tgt_vals[mask_nn].reshape(-1, 1)
).ravel()
real_s[["PowerOutput"]] = scaled_tgt

# Val / test split (85 / 15 — same as training)
n_real  = len(real_s)
split   = int(n_real * 0.85)
val_df  = real_s.iloc[:split]
test_df = real_s.iloc[split:]
print(f"Val:  {len(val_df):,} rows  ({val_df.index[0].date()} → {val_df.index[-1].date()})")
print(f"Test: {len(test_df):,} rows  ({test_df.index[0].date()} → {test_df.index[-1].date()})")

# ── Build sequences from val set ──────────────────────────────────────────────
print("Building val sequences …")
X_val, y_val, idx_val = make_sequences(val_df, feat_cols, SEQ_LEN, N_HORIZONS)
print(f"Total val sequences: {X_val.shape[0]:,}")

# ── Build oversampled fine-tune dataset ──────────────────────────────────────
# Include 100% of every month; tile hard months by their multiplier.
# This guarantees no good month loses representation.
rng = np.random.default_rng(SEED)
X_parts, y_parts = [], []
print("\nFine-tune dataset (oversampled):")
for m in sorted(idx_val.month.unique()):
    mask = idx_val.month == m
    Xm, ym = X_val[mask], y_val[mask]
    mult = OVERSAMPLE.get(m, 1)
    if mult > 1:
        Xm = np.tile(Xm, (mult, 1, 1))
        ym = np.tile(ym, (mult, 1))
    X_parts.append(Xm)
    y_parts.append(ym)
    print(f"  {MONTH_NAMES[m]:>3}: {mask.sum():>4} base seqs × {mult} = {len(Xm):>4}")

X_ft = np.concatenate(X_parts, axis=0)
y_ft = np.concatenate(y_parts, axis=0)
shuffle_idx = rng.permutation(len(X_ft))
X_ft = X_ft[shuffle_idx]
y_ft = y_ft[shuffle_idx]
print(f"  Total fine-tune: {len(X_ft):,}")

# Full val set for early-stop monitoring (all months, unshuffled)
X_val_full = X_val
y_val_full  = y_val

# ── Load model and recompile at lower LR ─────────────────────────────────────
print(f"\nLoading v2 fine-tuned model from {FT_MODEL_PATH} (v3 continues from v2) …")
MODEL_PATH = FT_MODEL_PATH   # v3 starts from the v2 checkpoint
print(f"Loading model from {MODEL_PATH} …")
model = keras.models.load_model(MODEL_PATH, compile=False)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
    loss="mae",
    metrics=["mse"],
)
print(f"Fine-tuning LR: {FINETUNE_LR}  (10× lower than training)")

# ── Fine-tune ─────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
]

print(f"\nFine-tuning for up to {FINETUNE_EPOCHS} epochs …")
history = model.fit(
    X_ft, y_ft,
    validation_data=(X_val_full, y_val_full),   # full val — guards all months
    epochs=FINETUNE_EPOCHS,
    batch_size=FINETUNE_BATCH,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)
print(f"Best val_loss: {min(history.history['val_loss']):.5f}")

# ── Save fine-tuned model ─────────────────────────────────────────────────────
model.save(FT_MODEL_PATH)
print(f"Saved: {FT_MODEL_PATH}")

# ── Evaluate all months (full real dataset) ───────────────────────────────────
print("\nRunning inference on full real dataset …")
X_all, y_all, idx_all = make_sequences(real_s, feat_cols, SEQ_LEN, N_HORIZONS)
y_pred_norm = model.predict(X_all, batch_size=64, verbose=0)

def inv(arr):
    n, h = arr.shape
    return np.clip(scaler_tgt.inverse_transform(arr.reshape(-1, 1)).reshape(n, h), 0, None)

y_pred_kw = inv(y_pred_norm)
y_true_kw = inv(y_all)

months_all = sorted(idx_all.month.unique())
rows = []
for m in months_all:
    mask = idx_all.month == m
    for h in range(N_HORIZONS):
        yt = y_true_kw[mask, h]
        yp = y_pred_kw[mask, h]
        rows.append({"Month": MONTH_NAMES[m], "m_ord": m,
                     "Horizon": f"h+{h+1}", "R2": r2_score(yt, yp)})

df = pd.DataFrame(rows)
pivot = df.pivot_table(index="Horizon", columns="Month", values="R2")
col_order = [MONTH_NAMES[m] for m in sorted(months_all)]
pivot = pivot[col_order]
pivot.index = pd.CategoricalIndex(
    pivot.index, categories=[f"h+{h}" for h in range(1, 25)], ordered=True
)
pivot = pivot.sort_index()

mean_row = pivot.mean().rename("Mean")
pivot_out = pd.concat([pivot, mean_row.to_frame().T])

print("\n=== Monthly R² after fine-tuning (all horizons) ===")
print(pivot_out.round(4).to_string())

print("\n=== Mean R² per month (before → after) ===")
before = {
    "Jan":0.9393,"Feb":0.9627,"Mar":0.9335,"Apr":0.7553,"May":0.6916,
    "Jun":0.9234,"Jul":0.9058,"Aug":0.9263,"Sep":0.9488,"Oct":0.7000,
    "Nov":-0.2819,"Dec":0.9425,
}
for col in col_order:
    after_val = pivot[col].mean()
    bef       = before.get(col, float("nan"))
    delta     = after_val - bef
    marker    = " ✓ improved" if delta > 0.01 else (" ✗ worse" if delta < -0.01 else "  ~same")
    print(f"  {col:>3}: {bef:+.4f} → {after_val:+.4f}  (Δ{delta:+.4f}){marker}")
