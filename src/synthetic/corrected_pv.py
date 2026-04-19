"""
src/synthetic/corrected_pv.py
------------------------------
Orchestrates the full synthetic PV generation and residual correction pipeline.

Pipeline
--------
1. Load all Solcast data (2020–2024, ~4 years, 5-min).
2. Run pvlib PVWatts simulation → 5-min synthetic PV.
3. Aggregate 5-min → hourly (mean power = proportional to hourly energy).
4. Add time, weather, physics, and monsoon features.
5. For the 1-year overlap window (Apr 2022 – Apr 2023):
     a. Build residual feature matrix.
     b. Train XGBoost + LSTM residual correction models.
6. Apply both correction models to the full 4-year hourly period.
7. Combine XGBoost and LSTM corrections (equal weight by default).
8. Output:
     ``pv_corrected_W`` = pvlib_ac_W + combined_residual_pred_W
     Clipped to [0, capacity_w] and zeroed at nighttime.

Output DataFrame columns (hourly, full 4-year span)
----------------------------------------------------
  datetime_local (index)
  pvlib_ac_W              : raw pvlib simulation
  pv_corrected_W          : residual-corrected synthetic PV (W)
  ghi, dni, dhi           : Solcast irradiance (hourly mean)
  air_temp, relative_humidity, cloud_opacity
  clearness_index_hourly
  monsoon_category
  + all other Solcast and feature columns

Design note on combining XGBoost + LSTM residuals
--------------------------------------------------
Both models are independently trained; their predictions are averaged:
    combined_residual = 0.5 * xgb_residual + 0.5 * lstm_residual

The LSTM warm-up period (first seq_len hours) has NaN LSTM predictions;
those rows fall back to XGBoost-only correction automatically.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data.solcast import load_solcast
from src.data.nwp_historical import load_nwp_history
from src.features.aggregation import aggregate_to_hourly
from src.features.time_features import add_time_features
from src.features.weather_features import add_weather_features
from src.features.physics_features import add_physics_features
from src.features.monsoon import add_monsoon_features
from src.features.forecast_features import (
    build_nwp_forecast_features,
    build_oracle_forecast_features,
)
from src.physics.pvlib_model import run_pvlib_simulation
from src.synthetic.residual_features import build_residual_features
from src.synthetic.xgb_residual import XGBResidualModel
from src.synthetic.lstm_residual import LSTMResidualModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Local time window for the real-PV overlap period
_OVERLAP_START = "2022-04-01"
_OVERLAP_END   = "2023-04-01"


def build_corrected_synthetic_pv(
    cfg: SimpleNamespace,
    real_pv_hourly: pd.DataFrame,
) -> tuple[pd.DataFrame, XGBResidualModel, LSTMResidualModel]:
    """Run the full synthetic PV generation and residual correction pipeline.

    Parameters
    ----------
    cfg:
        Full ``PipelineConfig`` (site, model, pipeline namespaces).
    real_pv_hourly:
        Hourly real PV DataFrame (index: tz-aware Asia/Colombo,
        columns include ``pv_ac_W``).  Used only for the overlap window.

    Returns
    -------
    (corrected_hourly_df, xgb_residual_model, lstm_residual_model)
    """
    pipe = cfg.pipeline

    # ── 1. Load full Solcast (4+ years, 5-min) ───────────────────────────────
    logger.info("Loading full Solcast data for 4-year simulation …")
    solcast_full = load_solcast(pipe.paths.external_solcast_dir)

    # ── 2. pvlib simulation on full Solcast period ────────────────────────────
    logger.info("Running pvlib simulation on full Solcast period …")
    sim_5min = run_pvlib_simulation(solcast_full, cfg.site)

    # ── 3. Aggregate 5-min → hourly ───────────────────────────────────────────
    logger.info("Aggregating to hourly …")
    hourly = aggregate_to_hourly(sim_5min)

    # ── 4. Add feature columns ────────────────────────────────────────────────
    hourly = add_time_features(hourly, cfg.site)
    hourly = add_weather_features(hourly, pipe)
    hourly = add_physics_features(hourly)
    hourly = add_monsoon_features(hourly)

    # ── 5. Extract overlap window and attach real PV ──────────────────────────
    overlap_mask = (
        (hourly.index >= pd.Timestamp(_OVERLAP_START, tz="Asia/Colombo"))
        & (hourly.index <  pd.Timestamp(_OVERLAP_END,   tz="Asia/Colombo"))
    )
    df_overlap = hourly[overlap_mask].copy()

    # Join real PV into the overlap (left join: synthetic index is the authority)
    real_aligned = real_pv_hourly[["pv_ac_W"]].reindex(df_overlap.index)
    df_overlap["pv_ac_W"] = real_aligned["pv_ac_W"]

    logger.info(
        "Overlap window: %d hourly rows | %d with real PV values.",
        len(df_overlap),
        df_overlap["pv_ac_W"].notna().sum(),
    )

    # ── 6a. Build residual feature matrices ──────────────────────────────────
    # XGBoost: daytime-only tabular rows (no sequence continuity needed)
    df_residual_xgb = build_residual_features(df_overlap, daytime_only=True)

    # LSTM: full hourly rows (continuous sequences), daytime flag for targets
    df_residual_lstm = build_residual_features(df_overlap, daytime_only=False)

    # ── 6b. Train XGBoost residual model ─────────────────────────────────────
    logger.info("Training XGBoost residual model …")
    xgb_model = XGBResidualModel(cfg.model.xgboost)
    xgb_model.fit(df_residual_xgb)

    # ── 6c. Train LSTM residual model ─────────────────────────────────────────
    logger.info("Training LSTM residual model …")
    lstm_model = LSTMResidualModel(cfg.model.lstm_residual)
    lstm_model.fit(df_residual_lstm)

    # ── 7. Apply corrections to full 4-year period ───────────────────────────
    logger.info("Applying residual corrections to full 4-year period …")

    # Prepare full feature DataFrame for prediction
    full_feat = build_residual_features_for_inference(hourly)

    xgb_residual  = xgb_model.predict(full_feat)
    lstm_residual = lstm_model.predict(full_feat)

    # Combine: average where both available; fall back to XGBoost alone
    combined_residual = _combine_residuals(xgb_residual, lstm_residual)

    # ── 8. Compute corrected PV ───────────────────────────────────────────────
    capacity_w = cfg.site.pv_system.pdc0_kw * 1_000.0

    hourly["pv_corrected_W"] = (
        hourly["pvlib_ac_W"] + combined_residual
    ).clip(lower=0.0, upper=capacity_w)

    # Zero out nighttime (pvlib says zero → corrected should also be zero)
    nighttime = hourly["pvlib_ac_W"] <= 0.0
    hourly.loc[nighttime, "pv_corrected_W"] = 0.0

    n_corrected = (hourly["pv_corrected_W"] > 0).sum()
    logger.info(
        "Corrected PV generated: %d total hourly rows | %d daytime rows.",
        len(hourly),
        n_corrected,
    )

    # ── 9. Add NWP forecast features for LSTM pretraining ────────────────────
    # LSTM_FORECAST_COLS (ghi_fcast_h1..h24, cloud_opacity_fcast_h1..h24)
    # must be present in the synthetic parquet so that LSTM pretraining gets
    # the same forecast context as the real-data fine-tuning step.
    n_horizons = getattr(cfg.model.forecasting, "n_horizons", 24)
    nwp_history = load_nwp_history(cfg)
    if nwp_history is not None:
        logger.info("Adding NWP forecast features to synthetic data …")
        hourly = build_nwp_forecast_features(hourly, nwp_history, n_horizons=n_horizons)
    else:
        logger.info(
            "NWP history not found — using oracle forecast features "
            "for LSTM pretraining context …"
        )
        hourly = build_oracle_forecast_features(hourly, n_horizons=n_horizons)

    return hourly, xgb_model, lstm_model


def build_residual_features_for_inference(hourly: pd.DataFrame) -> pd.DataFrame:
    """Prepare the feature matrix for inference (no real PV column needed)."""
    from src.synthetic.residual_features import _add_normalised_cols

    df = hourly.copy()
    df = _add_normalised_cols(df)
    # Add monsoon if not already present
    if "monsoon_category" not in df.columns:
        from src.features.monsoon import add_monsoon_features
        df = add_monsoon_features(df)
    return df


def _combine_residuals(
    xgb_pred: pd.Series,
    lstm_pred: pd.Series,
    xgb_weight: float = 0.5,
) -> pd.Series:
    """Weighted average of XGBoost and LSTM residual predictions.

    Where LSTM has NaN (warm-up period), fall back to XGBoost only.
    """
    combined = xgb_weight * xgb_pred + (1 - xgb_weight) * lstm_pred.fillna(xgb_pred)
    # Any remaining NaN → zero correction
    return combined.fillna(0.0)
