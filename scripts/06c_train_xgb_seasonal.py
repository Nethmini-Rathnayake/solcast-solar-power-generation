"""
scripts/06c_train_xgb_seasonal.py
------------------------------------
Train XGBoost with seasonal / yearly lag features and compare against the
standard XGBoost (step 02 / 06 output).

What this adds over the standard XGBoost
-----------------------------------------
  lag_2190h     — same hour ~3 months ago
  lag_4380h     — same hour ~6 months ago
  lag_8760h     — same hour ~1 year ago
  rolling_mean_720h — 30-day trailing mean
  pvlib_lag_8760h   — pvlib physics from same hour last year

These features expose the annual monsoon cycle directly to XGBoost,
which otherwise only sees the weekly pattern (lag_168h).

Prerequisites
-------------
    scripts/02c_build_features_xgb_seasonal.py
        → data/processed/feature_matrix_xgb_seasonal.parquet

Output
------
    models/xgb_seasonal/          — saved XGBoost models
    results/metrics/xgb_seasonal/ — per-horizon metrics CSV
    results/figures/xgb_seasonal/ — horizon plots + scatter
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.gradient_boost import XGBoostDMSForecaster
from src.models.train import build_feature_matrix
from src.evaluation.metrics import compute_metrics, summarise_metrics
from src.evaluation.plots import (
    plot_metrics_by_horizon,
    plot_forecast_vs_actual,
    plot_scatter,
)

logger = get_logger(__name__)

_FEAT_PATH   = Path("data/processed/feature_matrix_xgb_seasonal.parquet")
_MODEL_DIR   = Path("models/xgb_seasonal")
_METRICS_DIR = Path("results/metrics/xgb_seasonal")
_FIGURES_DIR = Path("results/figures/xgb_seasonal")

# Standard XGBoost metrics for side-by-side comparison
_STANDARD_METRICS = Path("results/metrics/metrics_xgb_dms.csv")


def main() -> None:
    logger.info("=== Step 6c: XGBoost with Seasonal Lags ===")

    cfg  = load_config()
    pipe = cfg.pipeline

    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load seasonal feature matrix ──────────────────────────────────────
    if not _FEAT_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {_FEAT_PATH}. "
            "Run scripts/02c_build_features_xgb_seasonal.py first."
        )
    df = pd.read_parquet(_FEAT_PATH)
    seasonal_cols = [c for c in df.columns
                     if any(c.startswith(p) for p in
                            ("lag_2190", "lag_4380", "lag_8760",
                             "rolling_mean_720", "pvlib_lag_8760"))]
    logger.info(
        "Seasonal feature matrix: %d rows × %d columns "
        "(%d seasonal lag cols).",
        len(df), len(df.columns), len(seasonal_cols),
    )
    logger.info("Seasonal columns: %s", seasonal_cols)

    # ── 2. Train / val / test split ───────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = build_feature_matrix(df, cfg)
    logger.info("Split: train=%d | val=%d | test=%d rows.",
                len(X_train), len(X_val), len(X_test))

    # ── 3. Train XGBoost ──────────────────────────────────────────────────────
    logger.info("Training XGBoost DMS with seasonal lags …")
    xgb = XGBoostDMSForecaster(cfg.model)
    xgb.fit(X_train, y_train, X_val, y_val)

    # ── 4. Save model ─────────────────────────────────────────────────────────
    xgb.save(_MODEL_DIR)
    logger.info("Model saved: %s", _MODEL_DIR)

    # ── 5. Evaluate on test set ───────────────────────────────────────────────
    logger.info("Evaluating on test set …")
    y_pred = xgb.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, pipe.evaluation)
    summarise_metrics(metrics).to_csv(_METRICS_DIR / "metrics_xgb_seasonal.csv")
    logger.info("Metrics saved: %s", _METRICS_DIR / "metrics_xgb_seasonal.csv")

    # ── 6. Feature importance ─────────────────────────────────────────────────
    _save_feature_importance(xgb, X_train, seasonal_cols)

    # ── 7. Print comparison ───────────────────────────────────────────────────
    row = metrics.loc["mean"]
    logger.info("\n=== XGBoost Seasonal Lags — Test Results ===")
    logger.info("RMSE=%.1f W  MAE=%.1f W  nRMSE=%.1f%%  R²=%.4f",
                row["RMSE"], row["MAE"], row["nRMSE"], row["R2"])

    if _STANDARD_METRICS.exists():
        std = pd.read_csv(_STANDARD_METRICS, index_col=0)
        std_row = std.loc["mean"] if "mean" in std.index else std.iloc[-1]
        logger.info("\n=== Comparison vs Standard XGBoost ===")
        logger.info("%-25s  %8s  %8s  %7s  %6s",
                    "Model", "RMSE (W)", "MAE (W)", "nRMSE %", "R²")
        logger.info("-" * 60)
        logger.info("%-25s  %8.1f  %8.1f  %6.1f%%  %.4f",
                    "XGBoost (standard)",
                    float(std_row["RMSE"]), float(std_row["MAE"]),
                    float(std_row["nRMSE"]), float(std_row["R2"]))
        logger.info("%-25s  %8.1f  %8.1f  %6.1f%%  %.4f",
                    "XGBoost (seasonal lags)",
                    row["RMSE"], row["MAE"], row["nRMSE"], row["R2"])

        rmse_delta = row["RMSE"] - float(std_row["RMSE"])
        r2_delta   = row["R2"]   - float(std_row["R2"])
        logger.info(
            "\nΔ RMSE = %+.1f W  |  Δ R² = %+.4f  "
            "(%s with seasonal lags)",
            rmse_delta, r2_delta,
            "IMPROVED" if rmse_delta < 0 else "WORSE",
        )

    # Per-horizon RMSE comparison CSV
    rmse_seasonal = metrics["RMSE"].rename("xgb_seasonal")
    if _STANDARD_METRICS.exists():
        std_full = pd.read_csv(_STANDARD_METRICS, index_col=0)
        rmse_std = std_full["RMSE"].rename("xgb_standard")
        rmse_compare = pd.concat([rmse_std, rmse_seasonal], axis=1)
    else:
        rmse_compare = rmse_seasonal.to_frame()
    rmse_compare.to_csv(_METRICS_DIR / "rmse_comparison.csv")
    logger.info("Per-horizon RMSE comparison: %s", _METRICS_DIR / "rmse_comparison.csv")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_metrics_by_horizon(metrics, save_dir=_FIGURES_DIR)
    for h in [1, 6, 12, 24]:
        plot_forecast_vs_actual(y_test, y_pred, horizon=h, save_dir=_FIGURES_DIR)
        plot_scatter(y_test, y_pred, horizon=h, save_dir=_FIGURES_DIR)

    logger.info("=== Step 6c complete ===")


def _save_feature_importance(
    xgb: XGBoostDMSForecaster,
    X_train: pd.DataFrame,
    seasonal_cols: list[str],
) -> None:
    """Log average feature importance with seasonal cols highlighted."""
    try:
        import numpy as np
        importances = []
        for model in xgb._models.values():
            imp = model.get_booster().get_fscore()
            importances.append(imp)

        # Average across all 24 horizon models
        all_features = set()
        for imp in importances:
            all_features.update(imp.keys())

        avg_imp = {}
        for f in all_features:
            avg_imp[f] = np.mean([imp.get(f, 0) for imp in importances])

        imp_series = pd.Series(avg_imp).sort_values(ascending=False)
        imp_df = imp_series.reset_index()
        imp_df.columns = ["feature", "avg_importance"]
        imp_df["is_seasonal"] = imp_df["feature"].isin(seasonal_cols)
        imp_df.to_csv(_METRICS_DIR / "feature_importance.csv", index=False)

        logger.info("\nTop 15 features (avg across 24 horizons):")
        for _, row in imp_df.head(15).iterrows():
            tag = "  ← SEASONAL" if row["is_seasonal"] else ""
            logger.info("  %-35s  %.1f%s", row["feature"], row["avg_importance"], tag)

        seasonal_imp = imp_df[imp_df["is_seasonal"]]
        if not seasonal_imp.empty:
            logger.info("\nSeasonal feature importances:")
            for _, row in seasonal_imp.iterrows():
                rank = imp_df.index[imp_df["feature"] == row["feature"]][0] + 1
                logger.info(
                    "  %-35s  %.1f  (rank #%d / %d)",
                    row["feature"], row["avg_importance"], rank, len(imp_df),
                )
    except Exception as exc:
        logger.warning("Feature importance logging failed: %s", exc)


if __name__ == "__main__":
    main()
