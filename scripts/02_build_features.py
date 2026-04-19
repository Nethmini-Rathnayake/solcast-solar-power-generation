"""
scripts/02_build_features.py
------------------------------
Pipeline Step 2: Build the hourly feature matrix.

What this script does
---------------------
1. Loads the cleaned 5-min interim parquet.
2. Aggregates to hourly resolution (mean power, mean irradiance).
3. Adds time/solar-position features.
4. Adds Solcast-derived weather ratio features.
5. Adds physics-derived features (pvlib residual, clearness index).
6. Adds lag features and rolling statistics.
7. Constructs the 24-step target matrix and drops incomplete rows.
8. Saves the hourly feature matrix to data/processed/feature_matrix_hourly.parquet.

Run from the project root:
    python scripts/02_build_features.py

Output
------
    data/processed/feature_matrix_hourly.parquet
    ~6,000–7,000 rows × ~60+ columns
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
from src.features.lag_features import add_lag_features, build_target_matrix
from src.features.forecast_features import (
    build_oracle_forecast_features,
    build_nwp_forecast_features,
)
from src.data.nwp_historical import load_nwp_history

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Step 2: Feature Engineering ===")

    cfg = load_config()
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
    print("after load:", len(df_5min))

    # ── 2. Aggregate to hourly ────────────────────────────────────────────────
    df_hourly = aggregate_to_hourly(df_5min)
    print("after aggregation:", len(df_hourly))

    # ── 3. Time and solar-position features ───────────────────────────────────
    df_hourly = add_time_features(df_hourly, cfg.site)
    print("after time features:", len(df_hourly))

    # ── 4. Weather ratio features ─────────────────────────────────────────────
    df_hourly = add_weather_features(df_hourly, pipe)
    print("after weather features:", len(df_hourly))

    # ── 5. Physics features ───────────────────────────────────────────────────
    df_hourly = add_physics_features(df_hourly)
    print("after physics features:", len(df_hourly))

    # ── 6. Lag features and rolling statistics ────────────────────────────────
    df_hourly = add_lag_features(df_hourly, pipe.features)
    print("after lag features:", len(df_hourly))

    # ── 6b. Forecast features: NWP history (preferred) or oracle (fallback) ───
    # NWP history aligns training with inference — both use Open-Meteo ERA5.
    # Oracle (shifted Solcast actuals) is used if the NWP parquet is absent;
    # run scripts/00_download_nwp_history.py once to enable NWP mode.
    nwp_history = load_nwp_history(cfg)
    if nwp_history is not None:
        logger.info("Using NWP history for forecast features (train/inference aligned).")
        df_hourly = build_nwp_forecast_features(
            df_hourly,
            nwp_history,
            n_horizons=cfg.model.forecasting.n_horizons,
        )
    else:
        logger.warning(
            "NWP history not found at %s. "
            "Falling back to oracle forecast features. "
            "Run scripts/00_download_nwp_history.py to fix this.",
            cfg.pipeline.paths.nwp_history,
        )
        df_hourly = build_oracle_forecast_features(
            df_hourly,
            n_horizons=cfg.model.forecasting.n_horizons,
        )
    print("after forecast features:", len(df_hourly))

    print("\nTop NaN counts before target creation:")
    print(df_hourly.isna().sum().sort_values(ascending=False).head(30))

    # ── 7. Target matrix ──────────────────────────────────────────────────────
    df_final = build_target_matrix(
        df_hourly,
        n_horizons=cfg.model.forecasting.n_horizons,
    )
    print("after target creation:", len(df_hourly))

    print("\nTop NaN counts after target creation:")
    print(df_final.isna().sum().sort_values(ascending=False).head(30))  

    # ── 8. Save ───────────────────────────────────────────────────────────────
    out_path = Path(pipe.paths.processed_hourly)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out_path)

    logger.info(
        "Feature matrix saved: %s | %d rows × %d columns",
        out_path,
        len(df_final),
        len(df_final.columns),
    )
    logger.info("=== Step 2 complete ===")

    
    print(df_hourly.isna().sum().sort_values(ascending=False).head(30)
)


if __name__ == "__main__":
    main()
