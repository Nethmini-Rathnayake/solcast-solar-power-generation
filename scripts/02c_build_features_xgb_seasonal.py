"""
scripts/02c_build_features_xgb_seasonal.py
--------------------------------------------
Pipeline Step 2c: Build the XGBoost feature matrix with seasonal / yearly lags.

Adds on top of the standard feature matrix (step 02):
  - lag_2190h  — same hour ~3 months ago  (quarter-year)
  - lag_4380h  — same hour ~6 months ago  (half-year)
  - lag_8760h  — same hour ~1 year ago    (annual cycle)
  - rolling_mean_720h — 30-day trailing mean
  - pvlib_lag_8760h   — pvlib physics prediction from same hour last year

Deep lags that predate the real PV window are filled from the 4-year
corrected synthetic PV (pv_corrected_W, r=0.918 with real).

Output
------
    data/processed/feature_matrix_xgb_seasonal.parquet

Prerequisites
-------------
    scripts/01_prepare_data.py        → data/interim/aligned_5min.parquet
    scripts/05_generate_synthetic.py  → data/processed/synthetic_corrected_4yr.parquet
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
from src.features.lag_features import add_lag_features, build_target_matrix
from src.features.lag_features_seasonal import add_seasonal_lag_features
from src.features.forecast_features import (
    build_oracle_forecast_features,
    build_nwp_forecast_features,
)
from src.data.nwp_historical import load_nwp_history

logger = get_logger(__name__)

_SYNTH_PATH = Path("data/processed/synthetic_corrected_4yr.parquet")
_OUT_PATH   = Path("data/processed/feature_matrix_xgb_seasonal.parquet")


def main() -> None:
    logger.info("=== Step 2c: Feature Engineering (XGBoost + Seasonal Lags) ===")

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

    # ── 7. Standard short-term lag features (t-1 … t-168) ────────────────────
    df_hourly = add_lag_features(df_hourly, pipe.features)
    logger.info("Standard lags added.")

    # ── 8. Seasonal / yearly lag features ────────────────────────────────────
    synth_df = None
    if _SYNTH_PATH.exists():
        synth_df = pd.read_parquet(_SYNTH_PATH)
        logger.info(
            "Synthetic data loaded for deep-lag fill: %d rows.", len(synth_df)
        )
    else:
        logger.warning(
            "Synthetic data not found at %s. "
            "Deep lags will be forward-filled from real data only. "
            "Run scripts/05_generate_synthetic.py for best results.",
            _SYNTH_PATH,
        )

    df_hourly = add_seasonal_lag_features(df_hourly, synth_df=synth_df)
    logger.info("Seasonal lags added.")

    # ── 9. NWP multi-horizon forecast context ─────────────────────────────────
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

    # ── 10. Target matrix ─────────────────────────────────────────────────────
    df_final = build_target_matrix(
        df_hourly,
        n_horizons=cfg.model.forecasting.n_horizons,
    )

    # ── 11. Save ──────────────────────────────────────────────────────────────
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(_OUT_PATH)

    seasonal_cols = [c for c in df_final.columns
                     if any(c.startswith(p) for p in
                            ("lag_2190", "lag_4380", "lag_8760",
                             "rolling_mean_720", "pvlib_lag_8760"))]
    logger.info(
        "XGBoost seasonal feature matrix saved: %s | %d rows × %d columns",
        _OUT_PATH, len(df_final), len(df_final.columns),
    )
    logger.info("New seasonal columns (%d): %s", len(seasonal_cols), seasonal_cols)
    logger.info("=== Step 2c complete ===")


if __name__ == "__main__":
    main()
