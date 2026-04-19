"""
scripts/06b_train_lstm_only.py
--------------------------------
LSTM-only PV Forecaster — full training without lag features.

Rationale
---------
The LSTM encoder learns temporal dependencies from the raw historical
sequence (pv_ac_norm, weather features, solar position) directly through
its recurrent state.  Hand-crafted lag features (lag_1h … lag_168h,
rolling_mean_*) are redundant for an LSTM and are therefore excluded.

Pipeline
--------
1. Load LSTM feature matrix (step 02b — no lag features).
2. Chronological 70/15/15 split.
3. Pre-train LSTM on 4-year corrected synthetic PV (step 05 output).
4. Fine-tune LSTM on real PV training set (extended epochs).
5. Evaluate on test set: LSTM vs same-day baseline vs persistence.
6. Save model and metrics.

Prerequisites
-------------
    scripts/02b_build_features_lstm.py → data/processed/feature_matrix_lstm.parquet
    scripts/05_generate_synthetic.py   → data/processed/synthetic_corrected_4yr.parquet

Output
------
    models/lstm_only/lstm_forecaster.pt
    results/metrics/lstm_only/  (per-horizon CSV + figures)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.lstm_forecaster import LSTMPVForecaster
from src.models.train import build_feature_matrix
from src.evaluation.metrics import compute_metrics, summarise_metrics
from src.evaluation.plots import (
    plot_metrics_by_horizon,
    plot_forecast_vs_actual,
    plot_scatter,
)

logger = get_logger(__name__)

_FEAT_PATH  = Path("data/processed/feature_matrix_lstm.parquet")
_SYNTH_PATH = Path("data/processed/synthetic_corrected_4yr.parquet")
_MODEL_DIR  = Path("models/lstm_only")
_METRICS_DIR = Path("results/metrics/lstm_only")
_FIGURES_DIR = Path("results/figures/lstm_only")


def main() -> None:
    logger.info("=== Step 6b: LSTM-Only Forecaster Training (no lag features) ===")

    cfg  = load_config()
    pipe = cfg.pipeline

    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load LSTM feature matrix (no lags) ────────────────────────────────
    if not _FEAT_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {_FEAT_PATH}. "
            "Run scripts/02b_build_features_lstm.py first."
        )
    df_real = pd.read_parquet(_FEAT_PATH)
    logger.info(
        "LSTM feature matrix: %d rows × %d columns.", len(df_real), len(df_real.columns)
    )
    lag_cols = [c for c in df_real.columns if c.startswith("lag_") or c.startswith("rolling_")]
    if lag_cols:
        logger.warning("Unexpected lag columns still present: %s", lag_cols)
    else:
        logger.info("Confirmed: no lag/rolling features in matrix.")

    # ── 2. Chronological train / val / test split ─────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = build_feature_matrix(df_real, cfg)
    capacity_w = cfg.site.pv_system.pdc0_kw * 1000.0
    seq_len    = cfg.model.lstm_forecaster.seq_len

    logger.info("Split: train=%d | val=%d | test=%d rows.",
                len(X_train), len(X_val), len(X_test))

    # ── 3. Load 4-year synthetic for pre-training ────────────────────────────
    if not _SYNTH_PATH.exists():
        raise FileNotFoundError(
            f"Corrected synthetic not found: {_SYNTH_PATH}. "
            "Run scripts/05_generate_synthetic.py first."
        )
    df_synth = pd.read_parquet(_SYNTH_PATH)
    logger.info("Synthetic PV: %d rows × %d columns.", len(df_synth), len(df_synth.columns))

    # ── 4. Pre-train LSTM on 4-year synthetic ─────────────────────────────────
    logger.info("Pre-training LSTM on synthetic 4-year data …")
    lstm = LSTMPVForecaster(cfg.model)
    lstm.pretrain(df_synth)

    # ── 5. Fine-tune LSTM on real PV training set ────────────────────────────
    # Use extended epochs — no lag features means LSTM must rely entirely on
    # its recurrent state; more fine-tuning helps adaptation.
    logger.info("Fine-tuning LSTM on real PV training set …")
    df_real_train = df_real.loc[X_train.index].copy()

    # Override fine-tune epochs: double the config value (up to 60)
    original_epochs = cfg.model.lstm_forecaster.n_epochs_finetune
    extended_epochs = max(original_epochs * 2, 60)
    cfg.model.lstm_forecaster.n_epochs_finetune = extended_epochs
    logger.info("Fine-tune epochs: %d (extended from %d).", extended_epochs, original_epochs)

    lstm.finetune(df_real_train)

    # ── 6. Save model ─────────────────────────────────────────────────────────
    lstm.save(_MODEL_DIR / "lstm_forecaster.pt")
    logger.info("Model saved: %s", _MODEL_DIR / "lstm_forecaster.pt")

    # ── 7. Evaluate on test set ───────────────────────────────────────────────
    logger.info("Evaluating on test set …")

    df_test_ctx = _with_context(df_real, X_test.index, seq_len)
    df_test_ctx.loc[:, "pv_ac_norm"] = df_test_ctx["pv_ac_W"].fillna(0.0) / capacity_w

    y_pred_lstm = lstm.predict(df_test_ctx)

    # Align to test index
    test_common = y_pred_lstm.index.intersection(X_test.index)
    pred_aligned = y_pred_lstm.loc[test_common]
    true_aligned = y_test.loc[test_common]

    # Drop rows with any NaN in targets or preds
    n_horizons = cfg.model.forecasting.n_horizons
    target_cols = [f"target_h{h}" for h in range(1, n_horizons + 1)]
    pred_cols   = [f"pred_h{h}"   for h in range(1, n_horizons + 1)]
    valid = (
        true_aligned[target_cols].notna().all(axis=1)
        & pred_aligned[pred_cols].notna().all(axis=1)
    )
    pred_final = pred_aligned[valid]
    true_final = true_aligned[valid]

    metrics_lstm = compute_metrics(true_final, pred_final, pipe.evaluation)
    row = metrics_lstm.loc["mean"]
    logger.info(
        "LSTM-only test metrics: RMSE=%.1f W  MAE=%.1f W  nRMSE=%.1f%%  R²=%.4f",
        row["RMSE"], row["MAE"], row["nRMSE"], row["R2"],
    )
    summarise_metrics(metrics_lstm).to_csv(_METRICS_DIR / "metrics_lstm_only.csv")
    logger.info("Metrics saved: %s", _METRICS_DIR / "metrics_lstm_only.csv")

    # ── 8. Baselines ─────────────────────────────────────────────────────────
    baselines = _compute_baselines(true_final, df_real, n_horizons, capacity_w)

    logger.info("\n=== LSTM-Only Model Comparison (mean across h=1…24) ===")
    logger.info("%-18s  %8s  %8s  %8s  %6s", "Model", "RMSE (W)", "MAE (W)", "nRMSE %", "R²")
    logger.info("-" * 60)

    all_models = {"lstm_only": (metrics_lstm, pred_final)}
    for bname, (bmets, _) in baselines.items():
        all_models[bname] = (bmets, None)

    for name, (mdf, _) in all_models.items():
        r = mdf.loc["mean"]
        logger.info("%-18s  %8.1f  %8.1f  %7.1f%%  %.4f",
                    name, r["RMSE"], r["MAE"], r["nRMSE"], r["R2"])

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    plot_metrics_by_horizon(metrics_lstm, save_dir=_FIGURES_DIR)
    for h in [1, 6, 12, 24]:
        plot_forecast_vs_actual(true_final, pred_final, horizon=h, save_dir=_FIGURES_DIR)
        plot_scatter(true_final, pred_final, horizon=h, save_dir=_FIGURES_DIR)

    # Per-horizon RMSE comparison CSV
    rmse_rows = {}
    for name, (mdf, _) in all_models.items():
        rmse_rows[name] = mdf["RMSE"]
    rmse_df = pd.DataFrame(rmse_rows)
    rmse_df.to_csv(_METRICS_DIR / "rmse_by_horizon.csv")
    logger.info("Per-horizon RMSE saved: %s", _METRICS_DIR / "rmse_by_horizon.csv")

    logger.info("=== Step 6b complete ===")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _with_context(
    df_full: pd.DataFrame,
    target_idx: pd.Index,
    seq_len: int,
) -> pd.DataFrame:
    if len(target_idx) == 0:
        return df_full.iloc[0:0]
    first_pos = df_full.index.get_loc(target_idx[0])
    if isinstance(first_pos, slice):
        first_pos = first_pos.start
    context_start = max(0, first_pos - seq_len)
    last_pos = df_full.index.get_loc(target_idx[-1])
    if isinstance(last_pos, slice):
        last_pos = last_pos.stop - 1
    return df_full.iloc[context_start : last_pos + 1]


def _compute_baselines(
    y_true: pd.DataFrame,
    df_full: pd.DataFrame,
    n_horizons: int,
    capacity_w: float,
) -> dict:
    """Compute same-day and persistence baselines aligned to y_true.index."""
    from src.evaluation.metrics import compute_metrics
    from src.utils.config import load_config
    cfg  = load_config()

    baselines = {}

    # ── Same-day baseline (same hour, 24h before) ────────────────────────────
    pv = df_full["pv_ac_W"]
    same_day_pred = pd.DataFrame(index=y_true.index)
    for h in range(1, n_horizons + 1):
        # predicted t+h = actual t+h-24 (same hour yesterday)
        shifted = pv.shift(24 - h + 1)
        same_day_pred[f"pred_h{h}"] = shifted.reindex(y_true.index).clip(lower=0)
    same_day_pred = same_day_pred.fillna(0.0)
    valid_sd = y_true[[f"target_h{h}" for h in range(1, n_horizons + 1)]].notna().all(axis=1)
    m = compute_metrics(y_true[valid_sd], same_day_pred[valid_sd], cfg.pipeline.evaluation)
    summarise_metrics(m).to_csv(_METRICS_DIR / "metrics_same_day.csv")
    baselines["same_day"] = (m, same_day_pred)

    # ── Persistence baseline ─────────────────────────────────────────────────
    pers_pred = pd.DataFrame(index=y_true.index)
    for h in range(1, n_horizons + 1):
        # predicted t+h = actual t (persist current value)
        pers_pred[f"pred_h{h}"] = pv.reindex(y_true.index).fillna(0.0).values
    m = compute_metrics(y_true[valid_sd], pers_pred[valid_sd], cfg.pipeline.evaluation)
    summarise_metrics(m).to_csv(_METRICS_DIR / "metrics_persistence.csv")
    baselines["persistence"] = (m, pers_pred)

    return baselines


if __name__ == "__main__":
    main()
