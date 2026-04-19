"""
scripts/07_evaluate_hybrid.py
------------------------------
Pipeline Step 7: Evaluate the hybrid forecaster against all baselines.

What this script does
---------------------
1. Loads the saved hybrid model artifacts from models/hybrid/.
2. Loads the real PV hourly feature matrix and recreates the test split
   (identical 70/15/15 chronological split used in step 06).
3. Generates predictions from five models:
     hybrid         — α * XGBoost + (1-α) * LSTM
     xgb_dms        — XGBoost DMS only (from hybrid sub-model)
     lstm           — LSTM only (from hybrid sub-model)
     persistence    — last observed value repeated for h=1…24
     same_day       — same clock-hour from yesterday (seasonal persistence)
4. Computes per-horizon metrics (RMSE, MAE, MBE, MAPE, nRMSE, R²).
5. Saves metrics CSVs to results/metrics/.
6. Saves comparison plots to results/figures/hybrid/.

Prerequisites
-------------
    scripts/06_train_hybrid.py  (produces models/hybrid/ artifacts)

Run from the project root:
    python scripts/07_evaluate_hybrid.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.gradient_boost import XGBoostDMSForecaster
from src.models.lstm_forecaster import LSTMPVForecaster
from src.models.hybrid_forecaster import HybridForecaster
from src.models.baseline import PersistenceBaseline, SameDayBaseline
from src.models.train import build_feature_matrix
from src.evaluation.metrics import compute_metrics, compute_daytime_metrics, summarise_metrics
from src.evaluation.plots import (
    plot_metrics_by_horizon,
    plot_forecast_vs_actual,
    plot_scatter,
)

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Step 7: Hybrid Forecaster Evaluation ===")

    cfg  = load_config()
    pipe = cfg.pipeline
    n_horizons = cfg.model.forecasting.n_horizons

    figures_dir = Path(pipe.paths.figures_dir) / "hybrid"
    metrics_dir = Path(pipe.paths.metrics_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load feature matrix and recreate the test split ───────────────────
    feat_path = Path(pipe.paths.processed_hourly)
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {feat_path}. "
            "Run scripts/02_build_features.py first."
        )
    df_real = pd.read_parquet(feat_path)
    logger.info("Feature matrix: %d rows × %d columns.", len(df_real), len(df_real.columns))

    X_train, y_train, X_val, y_val, X_test, y_test = build_feature_matrix(df_real, cfg)
    logger.info(
        "Test split: %d rows  (%s → %s).",
        len(X_test),
        X_test.index[0].date(),
        X_test.index[-1].date(),
    )

    # ── 2. Load hybrid model ──────────────────────────────────────────────────
    hybrid_dir = Path("models/hybrid")
    if not hybrid_dir.exists():
        raise FileNotFoundError(
            f"Hybrid model directory not found: {hybrid_dir}. "
            "Run scripts/06_train_hybrid.py first."
        )

    xgb_model  = XGBoostDMSForecaster(cfg.model)
    lstm_model = LSTMPVForecaster(cfg.model)
    hybrid     = HybridForecaster(cfg.model)
    hybrid.attach_xgb(xgb_model).attach_lstm(lstm_model)
    hybrid.load(hybrid_dir)

    logger.info("Hybrid model loaded. alpha=%.3f", hybrid.alpha)

    # ── 3. Build test context window for LSTM ────────────────────────────────
    seq_len    = cfg.model.lstm_forecaster.seq_len
    capacity_w = cfg.site.pv_system.pdc0_kw * 1000.0
    df_test_ctx = _with_context(df_real, X_test.index, seq_len)
    df_test_ctx.loc[:, "pv_ac_norm"] = df_test_ctx["pv_ac_W"].fillna(0.0) / capacity_w

    # ── 4. Generate predictions ───────────────────────────────────────────────
    logger.info("Generating predictions …")

    y_pred_hybrid = hybrid.predict(X_test, df_test_ctx)
    y_pred_xgb    = xgb_model.predict(X_test)
    y_pred_lstm   = _align_lstm_to_index(lstm_model.predict(df_test_ctx), X_test.index, n_horizons)

    # Baselines (need pv_ac_W column)
    test_with_pv = X_test.copy()
    if "pv_ac_W" not in test_with_pv.columns:
        test_with_pv["pv_ac_W"] = df_real.loc[X_test.index, "pv_ac_W"]

    y_pred_persistence = PersistenceBaseline(n_horizons=n_horizons).predict(test_with_pv)
    y_pred_same_day    = SameDayBaseline(n_horizons=n_horizons).predict(test_with_pv)

    # ── 5. Compute metrics for every model ────────────────────────────────────
    models_to_eval = {
        "hybrid":      y_pred_hybrid,
        "xgb_dms":     y_pred_xgb,
        "lstm":        y_pred_lstm,
        "persistence": y_pred_persistence,
        "same_day":    y_pred_same_day,
    }

    # Daytime mask: use pvlib simulation to identify generation hours
    pvlib_col = "pvlib_ac_W" if "pvlib_ac_W" in df_real.columns else None
    if pvlib_col is not None:
        daytime_mask = df_real[pvlib_col] > 0
    else:
        logger.warning("pvlib_ac_W not found — daytime mask unavailable.")
        daytime_mask = None

    all_metrics: dict[str, pd.DataFrame] = {}
    all_metrics_day: dict[str, pd.DataFrame] = {}

    for name, y_pred in models_to_eval.items():
        aligned_pred, aligned_true = _align(y_pred, y_test)
        if len(aligned_pred) == 0:
            logger.warning("No aligned rows for '%s' — skipped.", name)
            continue

        # All-hour metrics
        mdf = compute_metrics(aligned_true, aligned_pred, pipe.evaluation)
        all_metrics[name] = mdf
        summarise_metrics(mdf).to_csv(metrics_dir / f"metrics_{name}.csv")

        # Daytime-only metrics
        if daytime_mask is not None:
            mdf_day = compute_daytime_metrics(
                aligned_true, aligned_pred, daytime_mask, pipe.evaluation
            )
            all_metrics_day[name] = mdf_day
            summarise_metrics(mdf_day).to_csv(metrics_dir / f"metrics_{name}_daytime.csv")

        logger.info("Saved metrics for: %s", name)

    # ── 6. Print comparison tables ────────────────────────────────────────────
    logger.info("\n--- All-hour metrics ---")
    _print_comparison(all_metrics)
    if all_metrics_day:
        logger.info("\n--- Daytime-only metrics (pvlib_ac_W > 0) ---")
        _print_comparison(all_metrics_day)

    # ── 7. Save plots ─────────────────────────────────────────────────────────
    logger.info("Generating plots …")

    if "hybrid" in all_metrics:
        plot_metrics_by_horizon(all_metrics["hybrid"], save_dir=figures_dir)

    aligned_pred_hybrid, aligned_true_hybrid = _align(y_pred_hybrid, y_test)
    for h in [1, 6, 12, 24]:
        if h <= n_horizons and "hybrid" in all_metrics:
            plot_forecast_vs_actual(
                aligned_true_hybrid, aligned_pred_hybrid,
                horizon=h, save_dir=figures_dir,
            )
            plot_scatter(
                aligned_true_hybrid, aligned_pred_hybrid,
                horizon=h, save_dir=figures_dir,
            )

    # ── 8. Horizon-level RMSE comparison CSV ─────────────────────────────────
    _save_rmse_comparison(all_metrics, metrics_dir, suffix="")
    if all_metrics_day:
        _save_rmse_comparison(all_metrics_day, metrics_dir, suffix="_daytime")

    logger.info("=== Step 7 complete ===")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _with_context(
    df_full: pd.DataFrame,
    target_idx: pd.Index,
    seq_len: int,
) -> pd.DataFrame:
    """Prepend ``seq_len`` warm-up rows before the first row of target_idx."""
    if len(target_idx) == 0:
        return df_full.iloc[0:0]
    first_pos = df_full.index.get_loc(target_idx[0])
    if isinstance(first_pos, slice):
        first_pos = first_pos.start
    last_pos = df_full.index.get_loc(target_idx[-1])
    if isinstance(last_pos, slice):
        last_pos = last_pos.stop - 1
    context_start = max(0, first_pos - seq_len)
    return df_full.iloc[context_start : last_pos + 1]


def _align_lstm_to_index(
    lstm_pred: pd.DataFrame,
    target_idx: pd.Index,
    n_horizons: int,
) -> pd.DataFrame:
    """Re-index LSTM predictions to match target_idx (NaN where absent)."""
    out = pd.DataFrame(np.nan, index=target_idx,
                       columns=[f"pred_h{h}" for h in range(1, n_horizons + 1)])
    common = lstm_pred.index.intersection(target_idx)
    for col in out.columns:
        if col in lstm_pred.columns:
            out.loc[common, col] = lstm_pred.loc[common, col]
    return out


def _align(
    y_pred: pd.DataFrame,
    y_true: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pred, true) on common index with no NaN targets."""
    common = y_pred.index.intersection(y_true.index)
    target_cols = [c for c in y_true.columns if c.startswith("target_h")]
    valid = y_true.loc[common, target_cols].notna().all(axis=1)
    idx = common[valid]
    return y_pred.loc[idx], y_true.loc[idx]


def _print_comparison(all_metrics: dict[str, pd.DataFrame]) -> None:
    """Log a formatted comparison table (mean across all horizons)."""
    header = f"{'Model':<18}  {'RMSE (W)':>9}  {'MAE (W)':>9}  {'MBE (W)':>9}  {'MAPE (%)':>9}  {'nRMSE (%)':>10}  {'R²':>7}"
    sep    = "-" * len(header)
    logger.info("\n=== Test-Set Metrics (mean across h=1…24) ===")
    logger.info(header)
    logger.info(sep)
    for name, mdf in all_metrics.items():
        row = mdf.loc["mean"]
        logger.info(
            "%-18s  %9.1f  %9.1f  %9.1f  %9.2f  %10.2f  %7.4f",
            name,
            row["RMSE"],
            row["MAE"],
            row["MBE"],
            row.get("MAPE", float("nan")),
            row["nRMSE"],
            row["R2"],
        )
    logger.info(sep)


def _save_rmse_comparison(
    all_metrics: dict[str, pd.DataFrame],
    metrics_dir: Path,
    suffix: str = "",
) -> None:
    """Save per-horizon RMSE for all models to a single comparison CSV."""
    frames = []
    for name, mdf in all_metrics.items():
        s = mdf["RMSE"].drop(index="mean", errors="ignore").rename(name)
        frames.append(s)
    if frames:
        cmp = pd.concat(frames, axis=1)
        cmp.index.name = "horizon"
        out = metrics_dir / f"rmse_comparison{suffix}.csv"
        cmp.to_csv(out)
        logger.info("Per-horizon RMSE comparison saved: %s", out)


if __name__ == "__main__":
    main()
