"""
scripts/02b_build_features_lstm.py
------------------------------------
Pipeline Step 2b: Build the LSTM feature matrix.

Produces a feature matrix containing:

  - Cyclical time encodings and solar position
  - Solcast weather / irradiance columns
  - Physics-derived features (pvlib, clearness index)
  - Monsoon category
  - NWP multi-horizon forecast context  (ghi_fcast_h1..h24, etc.)
  - Explicit previous-day same-hour features:
      pv_lag24, pv_lag48, ghi_lag24, clearness_lag24
      (very useful for day-ahead forecasting even with an LSTM)
  - Future-weather summary features:
      ghi_fcast_mean_24h, ghi_fcast_max_24h,
      ghi_fcast_morning_mean (h6..h12), cloud_fcast_afternoon_mean (h12..h18),
      daylight_hours_ahead, total_irradiance_ahead
  - Target columns  (target_h1 … target_h24)

No rolling_* columns are included.

Output
------
    data/processed/feature_matrix_lstm.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.features.aggregation import aggregate_to_hourly
from src.features.time_features import add_time_features
from src.features.weather_features import add_weather_features
from src.features.physics_features import add_physics_features
from src.features.monsoon import add_monsoon_features
from src.features.lag_features import build_target_matrix   # targets only, no lags
from src.features.forecast_features import (
    build_oracle_forecast_features,
    build_nwp_forecast_features,
)
from src.data.nwp_historical import load_nwp_history

logger = get_logger(__name__)

_OUT_PATH = Path("data/processed/feature_matrix_lstm.parquet")


def main() -> None:
    logger.info("=== Step 2b: Feature Engineering (LSTM — no lag features) ===")

    cfg  = load_config()
    pipe = cfg.pipeline

    # ── 1. Load interim 5-min data ────────────────────────────────────────────
    interim_path = Path(pipe.paths.interim_5min)
    if not interim_path.exists():
        raise FileNotFoundError(
            f"Interim file not found: {interim_path}. "
            "Run scripts/01_prepare_data.py first."
        )
    df_5min = pd.read_parquet(interim_path)
    logger.info("Loaded interim data: %d rows.", len(df_5min))

    # ── 2. Aggregate to hourly ────────────────────────────────────────────────
    df_hourly = aggregate_to_hourly(df_5min)

    # ── 3. Time and solar-position features ───────────────────────────────────
    df_hourly = add_time_features(df_hourly, cfg.site)

    # ── 4. Weather ratio features ─────────────────────────────────────────────
    df_hourly = add_weather_features(df_hourly, pipe)

    # ── 5. Physics features ───────────────────────────────────────────────────
    df_hourly = add_physics_features(df_hourly)

    # ── 6. Monsoon features ───────────────────────────────────────────────────
    df_hourly = add_monsoon_features(df_hourly)

    # ── 7. NWP multi-horizon forecast context ─────────────────────────────────
    # (same columns used as LSTM forecast context at inference)
    nwp_history = load_nwp_history(cfg)
    if nwp_history is not None:
        logger.info("Using NWP history for forecast features.")
        df_hourly = build_nwp_forecast_features(
            df_hourly, nwp_history,
            n_horizons=cfg.model.forecasting.n_horizons,
        )
    else:
        logger.warning("NWP history not found — falling back to oracle forecast features.")
        df_hourly = build_oracle_forecast_features(
            df_hourly,
            n_horizons=cfg.model.forecasting.n_horizons,
        )

    # ── 8. Explicit previous-day same-hour features ───────────────────────────
    # Even though the LSTM sees sequences, explicit previous-day features
    # are especially powerful for day-ahead (h+24) prediction.
    pv_col = "pv_ac_W"
    if pv_col in df_hourly.columns:
        df_hourly["pv_lag24"]       = df_hourly[pv_col].shift(24).clip(lower=0)
        df_hourly["pv_lag48"]       = df_hourly[pv_col].shift(48).clip(lower=0)
    if "ghi" in df_hourly.columns:
        df_hourly["ghi_lag24"]      = df_hourly["ghi"].shift(24).clip(lower=0)
    if "clearness_index_hourly" in df_hourly.columns:
        df_hourly["clearness_lag24"] = df_hourly["clearness_index_hourly"].shift(24).fillna(0.0)

    logger.info("Added explicit previous-day lag features (lag24, lag48).")

    # ── 9. Future-weather summary features ────────────────────────────────────
    # Aggregate the h1..h24 forecast columns into intuitive summaries.
    # These help XGBoost and LSTM alike for day-ahead energy shape prediction.
    n_horizons = cfg.model.forecasting.n_horizons
    fcast_ghi_cols   = [f"ghi_fcast_h{h}"   for h in range(1, n_horizons + 1)
                        if f"ghi_fcast_h{h}" in df_hourly.columns]
    fcast_cloud_cols = [f"cloud_opacity_fcast_h{h}" for h in range(1, n_horizons + 1)
                        if f"cloud_opacity_fcast_h{h}" in df_hourly.columns]

    if fcast_ghi_cols:
        ghi_mat = df_hourly[fcast_ghi_cols].clip(lower=0)
        df_hourly["ghi_fcast_mean_24h"]   = ghi_mat.mean(axis=1)
        df_hourly["ghi_fcast_max_24h"]    = ghi_mat.max(axis=1)
        df_hourly["total_irradiance_ahead"] = ghi_mat.sum(axis=1)
        df_hourly["daylight_hours_ahead"] = (ghi_mat > 50).sum(axis=1).astype(float)

        # Morning (h6..h12) and afternoon (h12..h18) subsets
        morning_cols = [f"ghi_fcast_h{h}" for h in range(6, 13)
                        if f"ghi_fcast_h{h}" in df_hourly.columns]
        if morning_cols:
            df_hourly["ghi_fcast_morning_mean"] = df_hourly[morning_cols].clip(lower=0).mean(axis=1)

    if fcast_cloud_cols:
        afternoon_cols = [f"cloud_opacity_fcast_h{h}" for h in range(12, 19)
                          if f"cloud_opacity_fcast_h{h}" in df_hourly.columns]
        if afternoon_cols:
            df_hourly["cloud_fcast_afternoon_mean"] = df_hourly[afternoon_cols].mean(axis=1)

    summary_cols = [c for c in df_hourly.columns if c.startswith("ghi_fcast_mean")
                    or c.startswith("ghi_fcast_max") or c.startswith("total_irradiance")
                    or c.startswith("daylight_hours") or c.startswith("ghi_fcast_morning")
                    or c.startswith("cloud_fcast_afternoon")]
    logger.info("Added %d future-weather summary features: %s", len(summary_cols), summary_cols)

    # ── 10. Target matrix ─────────────────────────────────────────────────────
    df_final = build_target_matrix(
        df_hourly,
        n_horizons=cfg.model.forecasting.n_horizons,
    )

    # ── 11. Save ──────────────────────────────────────────────────────────────
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(_OUT_PATH)

    lag_cols = [c for c in df_final.columns if c.startswith("rolling_")]
    logger.info(
        "LSTM feature matrix saved: %s | %d rows × %d columns | "
        "explicit lags: pv_lag24/48, ghi_lag24, clearness_lag24 | "
        "weather summaries: %d cols",
        _OUT_PATH, len(df_final), len(df_final.columns), len(summary_cols),
    )
    logger.info("=== Step 2b complete ===")


if __name__ == "__main__":
    main()
