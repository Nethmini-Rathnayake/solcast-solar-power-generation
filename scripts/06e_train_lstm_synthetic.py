"""
scripts/06e_train_lstm_synthetic.py
-------------------------------------
LSTM trained directly on calibrated synthetic PV — no pretrain/finetune split.

Rationale
---------
The corrected synthetic data (step 05) is already calibrated to real PV via
XGBoost + LSTM residual correction (Layer B).  Rather than the two-phase
pretrain-then-finetune approach in step 06/06b, we train the LSTM in a single
pass on the full 4-year calibrated synthetic, then evaluate on the held-out
real PV test set.

Advantages:
  - Single training phase — no catastrophic forgetting risk
  - Full 4-year diversity (seasons, inter-annual variability) seen in training
  - 20 epochs × 36k rows runs in ~15 min (vs ~2 hours for 100-epoch pretrain)
  - seq_len=48 (2 days) gives better context than 24h

Pipeline
--------
1. Load calibrated synthetic (step 05) → build feature matrix with explicit
   day-lags and future-weather summaries.
2. Chronological split: train on first 3 years, test on last year of synthetic.
3. Train LSTM (20 epochs, seq_len=48, early stopping).
4. Evaluate on REAL PV test set (Jan 25 – Mar 12 2023) from feature_matrix_lstm.parquet.
5. Compare vs XGBoost and same-day baseline.

Prerequisites
-------------
    scripts/05_generate_synthetic.py   → data/processed/synthetic_corrected_4yr.parquet
    scripts/02b_build_features_lstm.py → data/processed/feature_matrix_lstm.parquet

Output
------
    models/lstm_synthetic/lstm_forecaster.pt
    results/metrics/lstm_synthetic/
    results/figures/lstm_synthetic/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.lstm_forecaster import LSTMPVForecaster
from src.models.train import build_feature_matrix
from src.evaluation.metrics import compute_metrics, summarise_metrics
from src.evaluation.plots import plot_metrics_by_horizon, plot_forecast_vs_actual, plot_scatter
from src.features.forecast_features import build_oracle_forecast_features

logger = get_logger(__name__)

_SYNTH_PATH  = Path("data/processed/synthetic_corrected_4yr.parquet")
_REAL_PATH   = Path("data/processed/feature_matrix_lstm.parquet")
_MODEL_DIR   = Path("models/lstm_synthetic")
_METRICS_DIR = Path("results/metrics/lstm_synthetic")
_FIGURES_DIR = Path("results/figures/lstm_synthetic")


def main() -> None:
    logger.info("=== Step 6e: LSTM Direct on Calibrated Synthetic (seq_len=48, 20 epochs) ===")

    cfg = load_config()
    capacity_w = cfg.site.pv_system.pdc0_kw * 1000.0
    n_horizons = cfg.model.forecasting.n_horizons
    seq_len    = cfg.model.lstm_forecaster.seq_len      # 48 from config

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load real PV feature matrix for evaluation ────────────────────────
    if not _REAL_PATH.exists():
        raise FileNotFoundError(f"Run scripts/02b_build_features_lstm.py first.")
    df_real = pd.read_parquet(_REAL_PATH)
    logger.info("Real PV matrix: %d rows × %d cols.", len(df_real), len(df_real.columns))

    # Get test split indices from real data (chronological 70/15/15)
    _, _, _, _, X_test, y_test = build_feature_matrix(df_real, cfg)
    logger.info("Real PV test set: %d rows (%s → %s).",
                len(X_test), X_test.index[0].date(), X_test.index[-1].date())

    # ── 2. Build synthetic feature matrix ────────────────────────────────────
    if not _SYNTH_PATH.exists():
        raise FileNotFoundError(f"Run scripts/05_generate_synthetic.py first.")
    synth_df = pd.read_parquet(_SYNTH_PATH)
    logger.info("Calibrated synthetic: %d rows × %d cols.", len(synth_df), len(synth_df.columns))

    df_synth = _build_synth_features(synth_df, n_horizons, capacity_w)
    logger.info("Synthetic feature matrix: %d rows × %d cols.",
                len(df_synth), len(df_synth.columns))

    # ── 3. Train LSTM directly on synthetic (20 epochs, no finetune) ─────────
    # Override epochs to n_epochs_direct (20)
    cfg.model.lstm_forecaster.n_epochs_pretrain = cfg.model.lstm_forecaster.n_epochs_direct
    cfg.model.lstm_forecaster.n_epochs_finetune = 0   # no finetune

    lstm = LSTMPVForecaster(cfg.model)
    logger.info("Training LSTM on synthetic (%d rows, seq_len=%d, %d epochs) …",
                len(df_synth), seq_len, cfg.model.lstm_forecaster.n_epochs_direct)
    lstm.pretrain(df_synth)

    lstm.save(_MODEL_DIR / "lstm_forecaster.pt")
    logger.info("Model saved: %s", _MODEL_DIR / "lstm_forecaster.pt")

    # ── 4. Evaluate on real PV test set ──────────────────────────────────────
    logger.info("Evaluating on real PV test set …")

    # Build test context window (need seq_len rows before test start)
    df_test_ctx = _with_context(df_real, X_test.index, seq_len)
    df_test_ctx = df_test_ctx.copy()
    df_test_ctx.loc[:, "pv_ac_norm"] = df_test_ctx["pv_ac_W"].fillna(0.0) / capacity_w

    y_pred = lstm.predict(df_test_ctx)

    # Align predictions to test index
    common = y_pred.index.intersection(X_test.index)
    pred_aligned = y_pred.loc[common]
    true_aligned = y_test.loc[common]

    target_cols = [f"target_h{h}" for h in range(1, n_horizons + 1)]
    pred_cols   = [f"pred_h{h}"   for h in range(1, n_horizons + 1)]
    valid = (
        true_aligned[target_cols].notna().all(axis=1)
        & pred_aligned[pred_cols].notna().all(axis=1)
    )
    pred_f = pred_aligned[valid]
    true_f = true_aligned[valid]

    metrics = compute_metrics(true_f, pred_f, cfg.pipeline.evaluation)
    row = metrics.loc["mean"]
    logger.info(
        "LSTM-synthetic test: RMSE=%.1f W  MAE=%.1f W  nRMSE=%.1f%%  R²=%.4f",
        row["RMSE"], row["MAE"], row["nRMSE"], row["R2"],
    )
    summarise_metrics(metrics).to_csv(_METRICS_DIR / "metrics_lstm_synthetic.csv")

    # ── 5. Same-day baseline for comparison ──────────────────────────────────
    pv = df_real["pv_ac_W"]
    sd_pred = pd.DataFrame(index=true_f.index)
    for h in range(1, n_horizons + 1):
        sd_pred[f"pred_h{h}"] = pv.shift(24 - h + 1).reindex(true_f.index).clip(lower=0).fillna(0)
    valid_sd = true_f[target_cols].notna().all(axis=1)
    m_sd = compute_metrics(true_f[valid_sd], sd_pred[valid_sd], cfg.pipeline.evaluation)
    r_sd = m_sd.loc["mean"]

    logger.info("\n=== Comparison (mean across h=1…24) ===")
    logger.info("%-22s  RMSE=%8.1f W  MAE=%8.1f W  R²=%.4f",
                "lstm_synthetic", row["RMSE"], row["MAE"], row["R2"])
    logger.info("%-22s  RMSE=%8.1f W  MAE=%8.1f W  R²=%.4f",
                "same_day_baseline", r_sd["RMSE"], r_sd["MAE"], r_sd["R2"])

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    plot_metrics_by_horizon(metrics, save_dir=_FIGURES_DIR)
    for h in [1, 6, 12, 24]:
        plot_forecast_vs_actual(true_f, pred_f, horizon=h, save_dir=_FIGURES_DIR)
        plot_scatter(true_f, pred_f, horizon=h, save_dir=_FIGURES_DIR)

    logger.info("=== Step 6e complete ===")
    logger.info("Model: %s", _MODEL_DIR)
    logger.info("Metrics: %s", _METRICS_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_synth_features(
    synth_df: pd.DataFrame,
    n_horizons: int,
    capacity_w: float,
) -> pd.DataFrame:
    """Build a minimal LSTM-compatible feature matrix from the synthetic DataFrame.

    Adds:
    - Normalised PV (pv_ac_norm from pv_corrected_W)
    - Explicit day-lags: pv_lag24, pv_lag48, ghi_lag24, clearness_lag24
    - Oracle forecast features (ghi_fcast_h1..h24 from actual future GHI)
    - Future-weather summaries
    - Target columns: target_h1..h24
    """
    df = synth_df.copy()

    # Rename corrected PV as the canonical target column
    if "pv_corrected_W" in df.columns and "pv_ac_W" not in df.columns:
        df["pv_ac_W"] = df["pv_corrected_W"]

    # Normalised PV
    df["pv_ac_norm"] = (df["pv_ac_W"] / capacity_w).clip(0, 1.1)

    # Explicit day-lags
    df["pv_lag24"]       = df["pv_ac_W"].shift(24).clip(lower=0)
    df["pv_lag48"]       = df["pv_ac_W"].shift(48).clip(lower=0)
    if "ghi" in df.columns:
        df["ghi_lag24"]  = df["ghi"].shift(24).clip(lower=0)
    if "clearness_index_hourly" in df.columns:
        df["clearness_lag24"] = df["clearness_index_hourly"].shift(24).fillna(0.0)

    # Oracle forecast features (use actual future GHI as perfect NWP)
    df = build_oracle_forecast_features(df, n_horizons=n_horizons)

    # Future-weather summaries
    fcast_cols = [f"ghi_fcast_h{h}" for h in range(1, n_horizons + 1)
                  if f"ghi_fcast_h{h}" in df.columns]
    if fcast_cols:
        ghi_mat = df[fcast_cols].clip(lower=0)
        df["ghi_fcast_mean_24h"]    = ghi_mat.mean(axis=1)
        df["ghi_fcast_max_24h"]     = ghi_mat.max(axis=1)
        df["total_irradiance_ahead"] = ghi_mat.sum(axis=1)
        df["daylight_hours_ahead"]  = (ghi_mat > 50).sum(axis=1).astype(float)
        morning = [f"ghi_fcast_h{h}" for h in range(6, 13) if f"ghi_fcast_h{h}" in df.columns]
        if morning:
            df["ghi_fcast_morning_mean"] = df[morning].clip(lower=0).mean(axis=1)

    # Target columns: target_h1..h24
    pv = df["pv_ac_W"]
    for h in range(1, n_horizons + 1):
        df[f"target_h{h}"] = pv.shift(-h)

    # Drop rows missing targets or pv_ac_norm
    df = df.dropna(subset=["pv_ac_norm"] + [f"target_h{h}" for h in range(1, n_horizons + 1)])
    return df


def _with_context(df_full: pd.DataFrame, target_idx: pd.Index, seq_len: int) -> pd.DataFrame:
    if len(target_idx) == 0:
        return df_full.iloc[0:0]
    first_pos = df_full.index.get_loc(target_idx[0])
    if isinstance(first_pos, slice):
        first_pos = first_pos.start
    context_start = max(0, first_pos - seq_len)
    last_pos = df_full.index.get_loc(target_idx[-1])
    if isinstance(last_pos, slice):
        last_pos = last_pos.stop - 1
    return df_full.iloc[context_start: last_pos + 1]


if __name__ == "__main__":
    main()
