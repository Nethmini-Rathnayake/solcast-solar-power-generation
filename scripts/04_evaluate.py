"""
scripts/04_evaluate.py
-----------------------
Pipeline Step 4: Evaluate all models and generate result plots and metrics.

What this script does
---------------------
1. Loads the test set and trained XGBoost model.
2. Generates predictions from XGBoost, persistence, and same-day baselines.
3. Computes per-horizon metrics (RMSE, MAE, MBE, MAPE, nRMSE, R²).
4. Saves metrics tables to results/metrics/ as CSV.
5. Saves evaluation plots to results/figures/.

Run from the project root:
    python scripts/04_evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.gradient_boost import XGBoostDMSForecaster
from src.models.baseline import PersistenceBaseline, SameDayBaseline
from src.models.train import build_feature_matrix, _select_feature_cols
from src.evaluation.metrics import compute_metrics, summarise_metrics
from src.evaluation.plots import (
    plot_metrics_by_horizon,
    plot_forecast_vs_actual,
    plot_scatter,
    plot_feature_importance,
)

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Step 4: Evaluation ===")

    cfg = load_config()
    pipe = cfg.pipeline
    n_horizons = cfg.model.forecasting.n_horizons

    figures_dir = Path(pipe.paths.figures_dir)
    metrics_dir = Path(pipe.paths.metrics_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Load feature matrix and re-split to get test set ─────────────────────
    feat_path = Path(pipe.paths.processed_hourly)
    df = pd.read_parquet(feat_path)

    X_train, y_train, X_val, y_val, X_test, y_test = build_feature_matrix(
        df, cfg
    )
    logger.info("Test set: %d rows.", len(X_test))

    # ── Load trained XGBoost model ────────────────────────────────────────────
    models_dir = Path(pipe.paths.models_dir)
    xgb_model = XGBoostDMSForecaster(cfg.model)
    xgb_model.load(models_dir)

    # ── Generate predictions ──────────────────────────────────────────────────
    y_pred_xgb = xgb_model.predict(X_test)

    # Baselines need the full DataFrame context for lag-based prediction
    # Reconstruct test_df with pv_ac_W for baseline computation
    test_with_pv = X_test.copy()
    if "pv_ac_W" not in test_with_pv.columns:
        # Recover from original df (pv_ac_W excluded from features in train.py)
        test_with_pv["pv_ac_W"] = df.loc[X_test.index, "pv_ac_W"]

    persistence = PersistenceBaseline(n_horizons=n_horizons)
    same_day = SameDayBaseline(n_horizons=n_horizons)
    y_pred_persistence = persistence.predict(test_with_pv)
    y_pred_same_day = same_day.predict(test_with_pv)

    # ── Compute metrics ───────────────────────────────────────────────────────
    models_to_evaluate = {
        "xgboost_dms": y_pred_xgb,
        "persistence": y_pred_persistence,
        "same_day": y_pred_same_day,
    }

    all_metrics: dict[str, pd.DataFrame] = {}
    for model_name, y_pred in models_to_evaluate.items():
        metrics_df = compute_metrics(y_test, y_pred, pipe.evaluation)
        all_metrics[model_name] = metrics_df

        # Save CSV
        csv_path = metrics_dir / f"metrics_{model_name}.csv"
        summarise_metrics(metrics_df).to_csv(csv_path)
        logger.info("Metrics saved: %s", csv_path)

    # ── Print comparison summary ──────────────────────────────────────────────
    logger.info("\n=== Model Comparison (mean across all horizons) ===")
    for name, mdf in all_metrics.items():
        row = mdf.loc["mean"]
        logger.info(
            "%-20s RMSE=%.1f W  MAE=%.1f W  nRMSE=%.1f%%  R²=%.3f",
            name,
            row["RMSE"],
            row["MAE"],
            row["nRMSE"],
            row["R2"],
        )

    # ── Generate plots ────────────────────────────────────────────────────────
    # Metrics comparison (XGBoost only)
    plot_metrics_by_horizon(all_metrics["xgboost_dms"], save_dir=figures_dir)

    # Forecast vs actual for horizons 1, 6, 12, 24
    for h in [1, 6, 12, 24]:
        if h <= n_horizons:
            plot_forecast_vs_actual(
                y_test, y_pred_xgb, horizon=h, save_dir=figures_dir
            )
            plot_scatter(y_test, y_pred_xgb, horizon=h, save_dir=figures_dir)

    # Feature importance
    importance_df = xgb_model.feature_importances()
    plot_feature_importance(importance_df, top_n=20, save_dir=figures_dir)

    logger.info("=== Step 4 complete ===")


if __name__ == "__main__":
    main()
