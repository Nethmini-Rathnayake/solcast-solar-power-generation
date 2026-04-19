"""
scripts/08_predict_live.py
---------------------------
Live 24h-ahead PV forecast using the trained hybrid model + Solcast forecast API.

What this script does
---------------------
1. Loads the trained hybrid model from models/hybrid/.
2. Reads the latest rows from data/processed/feature_matrix_hourly.parquet
   (or a live data source) to build the current feature vector.
3. Calls the Open-Meteo NWP API to get 24h-ahead weather forecasts (free, no key).
4. Injects the forecast weather into the feature vector.
5. Generates a 24h-ahead PV forecast using the hybrid model.
6. Prints the forecast to the console and saves it to
   results/forecasts/forecast_<timestamp>.csv.

Prerequisites
-------------
    scripts/06_train_hybrid.py  (models/hybrid/ must exist)
    No API key needed — Open-Meteo is completely free.
    scripts/02_build_features.py (feature matrix must exist)

Run from the project root:
    python scripts/08_predict_live.py

Optional arguments
------------------
    --decision-time   ISO datetime for the decision time t
                      (default: last available row in the feature matrix)
    --no-cache        Force fresh API fetch (ignore cached forecast)

Example
-------
    python scripts/08_predict_live.py
    python scripts/08_predict_live.py --decision-time 2023-03-15T08:00:00
"""

import argparse
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
from src.models.train import _select_feature_cols
from src.data.nwp_forecast import make_nwp_client_from_cfg
from src.features.forecast_features import (
    build_live_forecast_features,
    get_forecast_feature_cols,
)

logger = get_logger(__name__)


def main() -> None:
    args = _parse_args()

    logger.info("=== Live 24h-Ahead PV Forecast ===")

    cfg  = load_config()
    pipe = cfg.pipeline
    site = cfg.site

    # ── 1. Load feature matrix ────────────────────────────────────────────────
    feat_path = Path(pipe.paths.processed_hourly)
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {feat_path}. "
            "Run scripts/02_build_features.py first."
        )
    df = pd.read_parquet(feat_path)
    logger.info("Feature matrix loaded: %d rows.", len(df))

    # ── 2. Select decision time t ─────────────────────────────────────────────
    if args.decision_time:
        t = pd.Timestamp(args.decision_time, tz="Asia/Colombo")
        if t not in df.index:
            raise KeyError(
                f"Decision time {t} not found in feature matrix index. "
                "Use a timestamp that appears in the data."
            )
    else:
        t = df.index[-1]
        logger.info("Using last available row: %s", t)

    logger.info("Decision time t = %s", t)

    # ── 3. Fetch Solcast forecast ─────────────────────────────────────────────
    n_horizons = cfg.model.forecasting.n_horizons
    seq_len    = cfg.model.lstm_forecaster.seq_len

    client = make_nwp_client_from_cfg(cfg)
    if args.no_cache and client._cache_dir:
        _clear_forecast_cache(client._cache_dir, site.latitude, site.longitude)

    logger.info("Fetching NWP forecast for (%.4f, %.4f) …", site.latitude, site.longitude)
    forecast_df = client.fetch(
        lat=site.latitude,
        lon=site.longitude,
        hours=n_horizons,
    )
    logger.info(
        "Forecast received: %d hours  (%s → %s)",
        len(forecast_df),
        forecast_df.index[0].strftime("%Y-%m-%d %H:%M"),
        forecast_df.index[-1].strftime("%Y-%m-%d %H:%M"),
    )

    # ── 4. Build feature row for t ────────────────────────────────────────────
    # Identify the feature columns used at training time
    n_horizons_model = cfg.model.forecasting.n_horizons
    target_cols = [f"target_h{h}" for h in range(1, n_horizons_model + 1)]
    feature_cols = _select_feature_cols(df, target_cols)

    # Single-row DataFrame at decision time t
    row = df.loc[[t]].copy()

    # Inject live forecast weather
    row = build_live_forecast_features(row, forecast_df, n_horizons=n_horizons)

    # Ensure all expected feature columns are present (fill new fcast cols if needed)
    fcast_cols = get_forecast_feature_cols(n_horizons)
    for col in fcast_cols:
        if col not in feature_cols:
            feature_cols.append(col)
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0.0

    X_live = row[feature_cols].fillna(0.0)

    # ── 5. Load hybrid model ──────────────────────────────────────────────────
    hybrid_dir = Path("models/hybrid")
    if not hybrid_dir.exists():
        raise FileNotFoundError(
            f"Hybrid model not found: {hybrid_dir}. "
            "Run scripts/06_train_hybrid.py first."
        )

    xgb_model  = XGBoostDMSForecaster(cfg.model)
    lstm_model = LSTMPVForecaster(cfg.model)
    hybrid     = HybridForecaster(cfg.model)
    hybrid.attach_xgb(xgb_model).attach_lstm(lstm_model)
    hybrid.load(hybrid_dir)
    logger.info("Hybrid model loaded. alpha=%.3f", hybrid.alpha)

    # ── 6. Build LSTM sequence context window ─────────────────────────────────
    # LSTM needs seq_len historical rows ending at t, plus forecast columns.
    t_pos = df.index.get_loc(t)
    if isinstance(t_pos, slice):
        t_pos = t_pos.start
    ctx_start = max(0, t_pos - seq_len + 1)
    df_ctx = df.iloc[ctx_start : t_pos + 1].copy()

    # Add pv_ac_norm for LSTM
    capacity_w = site.pv_system.pdc0_kw * 1000.0
    df_ctx["pv_ac_norm"] = df_ctx["pv_ac_W"].fillna(0.0) / capacity_w

    # Inject live forecast features into every row of the context window
    # (LSTM only reads the forecast from the last row at prediction time)
    df_ctx = build_live_forecast_features(df_ctx, forecast_df, n_horizons=n_horizons)

    # ── 7. Generate forecast ──────────────────────────────────────────────────
    pred_df = hybrid.predict(X_live, df_ctx)

    # ── 8. Build output table ─────────────────────────────────────────────────
    horizon_times = pd.date_range(
        start=t + pd.Timedelta(hours=1),
        periods=n_horizons,
        freq="1h",
        tz="Asia/Colombo",
    )
    pred_vals = pred_df.iloc[0].values if len(pred_df) > 0 else np.zeros(n_horizons)

    output = pd.DataFrame(
        {
            "horizon":      [f"h+{h}" for h in range(1, n_horizons + 1)],
            "forecast_time": horizon_times,
            "predicted_W":   pred_vals.clip(min=0.0),
            "predicted_kW":  (pred_vals / 1000.0).clip(min=0.0),
        }
    )

    # ── 9. Display ────────────────────────────────────────────────────────────
    _print_forecast(output, t)

    # ── 10. Save ──────────────────────────────────────────────────────────────
    out_dir = Path("results/forecasts")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_str  = t.strftime("%Y%m%d_%H%M")
    out_csv = out_dir / f"forecast_{ts_str}.csv"
    output.to_csv(out_csv, index=False)
    logger.info("Forecast saved: %s", out_csv)

    logger.info("=== Live forecast complete ===")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a live 24h PV forecast.")
    parser.add_argument(
        "--decision-time",
        type=str,
        default=None,
        help="ISO8601 datetime for decision time t (default: last row in feature matrix)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force a fresh Solcast API fetch (ignore cached forecast)",
    )
    return parser.parse_args()


def _clear_forecast_cache(cache_dir: Path, lat: float, lon: float) -> None:
    pattern = f"forecast_{lat:.4f}_{lon:.4f}_*.json"
    for p in cache_dir.glob(pattern):
        p.unlink()
        logger.info("Cache cleared: %s", p)


def _print_forecast(output: pd.DataFrame, decision_time: pd.Timestamp) -> None:
    print(f"\n{'─' * 55}")
    print(f"  24h-Ahead PV Forecast  |  Decision time: {decision_time.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"{'─' * 55}")
    print(f"  {'Horizon':<8}  {'Forecast Time':<22}  {'kW':>7}")
    print(f"  {'─'*7}  {'─'*21}  {'─'*7}")
    for _, row in output.iterrows():
        bar_len = int(row["predicted_kW"] / 5)   # 1 char per 5 kW
        bar = "█" * min(bar_len, 30)
        print(f"  {row['horizon']:<8}  {row['forecast_time'].strftime('%Y-%m-%d %H:%M'):<22}  "
              f"{row['predicted_kW']:>7.1f}  {bar}")
    print(f"{'─' * 55}")
    peak = output["predicted_kW"].max()
    total_kwh = output["predicted_kW"].sum()   # hourly, so kWh = kW × 1h
    print(f"  Peak: {peak:.1f} kW  |  Day total: {total_kwh:.0f} kWh")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    main()
