"""
scripts/06_train_hybrid.py
---------------------------
Pipeline Step 6: Train the heterogeneous XGBoost-DMS + LSTM hybrid forecaster.

What this script does
---------------------
1. Loads the 1-year real PV hourly feature matrix (step 02 output).
2. Splits chronologically into train / val / test (same 70/15/15 as step 03).
3. Trains XGBoost DMS on the real PV training set.
4. Pre-trains the LSTM on the 4-year corrected synthetic PV (step 05 output).
5. Fine-tunes the LSTM on the real PV training set.
6. Optimises the blending weight α on the validation set.
7. Evaluates the hybrid (+ individual components) on the test set.
8. Saves all model artifacts to models/hybrid/.

Prerequisites
-------------
    scripts/02_build_features.py   → data/processed/feature_matrix_hourly.parquet
    scripts/05_generate_synthetic.py → data/processed/synthetic_corrected_4yr.parquet

Run from the project root:
    python scripts/06_train_hybrid.py

Output
------
    models/hybrid/
      alpha.pkl               — optimised α scalar
      xgb/                    — XGBoost DMS model artifacts
      lstm_forecaster.pt      — LSTM forecaster checkpoint

    results/metrics/
      metrics_hybrid.csv
      metrics_xgb_only.csv
      metrics_lstm_only.csv

    results/figures/
      forecast_vs_actual_hybrid_h*.png
      scatter_hybrid_h*.png
      metrics_by_horizon_hybrid.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.gradient_boost import XGBoostDMSForecaster
from src.models.lstm_forecaster import LSTMPVForecaster
from src.models.hybrid_forecaster import HybridForecaster
from src.models.train import build_feature_matrix
from src.evaluation.metrics import compute_metrics, summarise_metrics
from src.evaluation.plots import (
    plot_metrics_by_horizon,
    plot_forecast_vs_actual,
    plot_scatter,
)

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Step 6: Train Hybrid Forecaster ===")

    cfg  = load_config()
    pipe = cfg.pipeline

    # Save hybrid results in a dedicated sub-directory so they do not
    # overwrite the XGBoost-only outputs from step 04.
    figures_dir = Path(pipe.paths.figures_dir) / "hybrid"
    metrics_dir = Path(pipe.paths.metrics_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load 1-year real PV hourly feature matrix ─────────────────────────
    feat_path = Path(pipe.paths.processed_hourly)
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {feat_path}. "
            "Run scripts/02_build_features.py first."
        )
    df_real = pd.read_parquet(feat_path)
    logger.info(
        "Real PV feature matrix: %d rows × %d columns.", len(df_real), len(df_real.columns)
    )

    # ── 2. Chronological train / val / test split ─────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = build_feature_matrix(
        df_real, cfg
    )
    logger.info(
        "Split: train=%d | val=%d | test=%d rows.",
        len(X_train), len(X_val), len(X_test),
    )

    # ── 3. Train XGBoost DMS on real PV ──────────────────────────────────────
    logger.info("Training XGBoost DMS …")
    xgb_model = XGBoostDMSForecaster(cfg.model)
    xgb_model.fit(X_train, y_train, X_val, y_val)

    # ── 4. Load 4-year corrected synthetic for LSTM pre-training ─────────────
    synth_path = Path("data/processed/synthetic_corrected_4yr.parquet")
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Corrected synthetic data not found: {synth_path}. "
            "Run scripts/05_generate_synthetic.py first."
        )
    df_synth = pd.read_parquet(synth_path)
    logger.info(
        "Corrected synthetic PV: %d rows × %d columns.", len(df_synth), len(df_synth.columns)
    )

    # ── 5. Pre-train LSTM on 4-year corrected synthetic ───────────────────────
    logger.info("Pre-training LSTM on 4-year corrected synthetic PV …")
    lstm_model = LSTMPVForecaster(cfg.model)
    lstm_model.pretrain(df_synth)

    # ── 6. Fine-tune LSTM on 1-year real PV (training set only) ──────────────
    logger.info("Fine-tuning LSTM on 1-year real PV (train split) …")
    # Reconstruct the training rows of df_real with pv_ac_W attached
    df_real_train = df_real.loc[X_train.index].copy()
    # pv_ac_W may have been excluded from X_train features; re-attach from df_real
    if "pv_ac_W" not in df_real_train.columns:
        df_real_train["pv_ac_W"] = df_real.loc[X_train.index, "pv_ac_W"]
    lstm_model.finetune(df_real_train)

    # ── 7. Assemble hybrid and optimise α on validation set ──────────────────
    logger.info("Assembling hybrid forecaster and optimising alpha …")
    hybrid = HybridForecaster(cfg.model)
    hybrid.attach_xgb(xgb_model)
    hybrid.attach_lstm(lstm_model)

    # For LSTM prediction we need the context window (seq_len rows) prepended,
    # and pv_ac_norm must exist so LSTM_SEQUENCE_FEATURES can be sliced.
    seq_len    = cfg.model.lstm_forecaster.seq_len
    capacity_w = cfg.site.pv_system.pdc0_kw * 1000.0

    df_real_val_ctx  = _with_context(df_real, X_val.index,  seq_len)
    df_real_test_ctx = _with_context(df_real, X_test.index, seq_len)

    for df_tmp in [df_real_val_ctx, df_real_test_ctx]:
        df_tmp.loc[:, "pv_ac_norm"] = df_tmp["pv_ac_W"].fillna(0.0) / capacity_w

    hybrid.optimize_alpha(X_val, df_real_val_ctx, y_val)

    # ── 8. Save all model artifacts ───────────────────────────────────────────
    hybrid_dir = Path("models/hybrid")
    hybrid.save(hybrid_dir)
    logger.info("Hybrid forecaster saved to: %s", hybrid_dir)

    # ── 9. Evaluate on test set ───────────────────────────────────────────────
    logger.info("Evaluating on test set …")

    y_pred_hybrid = hybrid.predict(X_test, df_real_test_ctx)
    y_pred_xgb    = xgb_model.predict(X_test)
    y_pred_lstm   = lstm_model.predict(df_real_test_ctx)

    # Align LSTM predictions to test index (LSTM NaN rows during warm-up)
    lstm_test_idx = y_pred_lstm.index.intersection(X_test.index)
    y_pred_lstm_aligned = pd.DataFrame(index=X_test.index)
    for h in range(1, cfg.model.forecasting.n_horizons + 1):
        col = f"pred_h{h}"
        y_pred_lstm_aligned[col] = np.nan
        if col in y_pred_lstm.columns:
            y_pred_lstm_aligned.loc[lstm_test_idx, col] = y_pred_lstm.loc[lstm_test_idx, col]

    models_eval = {
        "hybrid":    y_pred_hybrid,
        "xgb_only":  y_pred_xgb,
        "lstm_only": y_pred_lstm_aligned,
    }

    all_metrics: dict[str, pd.DataFrame] = {}
    for name, y_pred in models_eval.items():
        aligned_pred, aligned_true = _align(y_pred, y_test)
        if len(aligned_pred) == 0:
            logger.warning("No aligned rows for model '%s'. Skipping metrics.", name)
            continue
        metrics_df = compute_metrics(aligned_true, aligned_pred, pipe.evaluation)
        all_metrics[name] = metrics_df
        csv_path = metrics_dir / f"metrics_{name}.csv"
        summarise_metrics(metrics_df).to_csv(csv_path)
        logger.info("Metrics saved: %s", csv_path)

    # ── 10. Print comparison table ────────────────────────────────────────────
    logger.info("\n=== Hybrid Model Comparison (mean across all horizons) ===")
    for name, mdf in all_metrics.items():
        row = mdf.loc["mean"]
        logger.info(
            "%-15s  RMSE=%7.1f W  MAE=%7.1f W  nRMSE=%5.1f%%  R²=%.3f",
            name,
            row["RMSE"],
            row["MAE"],
            row["nRMSE"],
            row["R2"],
        )

    # ── 11. Save plots ────────────────────────────────────────────────────────
    if "hybrid" in all_metrics:
        plot_metrics_by_horizon(all_metrics["hybrid"], save_dir=figures_dir)
        n_horizons = cfg.model.forecasting.n_horizons
        aligned_pred, aligned_true = _align(y_pred_hybrid, y_test)
        for h in [1, 6, 12, 24]:
            if h <= n_horizons:
                plot_forecast_vs_actual(
                    aligned_true, aligned_pred, horizon=h, save_dir=figures_dir
                )
                plot_scatter(
                    aligned_true, aligned_pred, horizon=h, save_dir=figures_dir
                )

    logger.info("=== Step 6 complete ===")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _with_context(
    df_full: pd.DataFrame,
    target_idx: pd.Index,
    seq_len: int,
) -> pd.DataFrame:
    """Return a DataFrame slice that includes ``seq_len`` warm-up rows before
    the first row of ``target_idx``.

    This ensures the LSTM has enough history to produce predictions for every
    row in ``target_idx`` without a cold-start gap.
    """
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


def _align(
    y_pred: pd.DataFrame,
    y_true: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pred, true) aligned on common index, both NaN-free."""
    common = y_pred.index.intersection(y_true.index)
    pred = y_pred.loc[common].copy()
    true = y_true.loc[common].copy()
    # Drop rows where any target is NaN
    n_horizons = len([c for c in true.columns if c.startswith("target_h")])
    target_cols = [f"target_h{h}" for h in range(1, n_horizons + 1)]
    valid = true[target_cols].notna().all(axis=1)
    return pred[valid], true[valid]


if __name__ == "__main__":
    main()
