"""
src/features/forecast_features.py
-----------------------------------
Build horizon-matched forecast weather features for training and inference.

The core insight
----------------
The model at decision time t cannot know the sky state at t+12 from lag
features alone — it regresses to the seasonal mean.  With NWP forecast
inputs the model knows *predicted* GHI/cloud_opacity/air_temp at every
future hour, turning an extrapolation task into a calibration task.

Training (NWP history mode — preferred)
----------------------------------------
``build_nwp_forecast_features(df_hourly, nwp_df)`` uses Open-Meteo ERA5
historical data as the training forecast.  For each decision row at time t
and horizon h, ``ghi_fcast_h{h}[t]`` = ERA5 ``ghi`` at t+h.

This aligns the training distribution with the live inference distribution
(same data source, same biases) so the model learns how to interpret
Open-Meteo output rather than perfect-oracle values.

Training (oracle mode — fallback)
----------------------------------
``build_oracle_forecast_features(df_hourly)`` uses the actual future Solcast
values shifted backward.  Used automatically when the NWP history parquet
does not exist.  Gives an upper bound on forecast-aware performance.

Inference (live mode)
---------------------
``build_live_forecast_features(df_row, forecast_df)`` maps Open-Meteo live
API output into the same flat columns at inference time.

Feature naming
--------------
Flat columns added to the feature matrix (one per decision row t):
    ghi_fcast_h1  … ghi_fcast_h24          (W/m²)
    cloud_opacity_fcast_h1 … _h24          (%)
    air_temp_fcast_h1 … _h24               (°C)

Total: 72 additional columns for XGBoost.

For the LSTM, only GHI and cloud_opacity are used as the forecast context
(48 values) — they are the primary drivers of solar output variance.
Normalized before injection into the LSTM FC layer:
    ghi          / 1000 → [0, ~1.2]
    cloud_opacity / 100  → [0, 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Feature column configuration ─────────────────────────────────────────────

# Weather columns to shift forward (for XGBoost flat features)
FORECAST_WEATHER_COLS: list[str] = ["ghi", "cloud_opacity", "air_temp"]

# Subset used for LSTM forecast context + their normalization scales
LSTM_FORECAST_WEATHER: list[str] = ["ghi", "cloud_opacity"]
LSTM_FORECAST_SCALES:  list[float] = [1000.0, 100.0]

N_HORIZONS = 24


def get_forecast_feature_cols(n_horizons: int = N_HORIZONS) -> list[str]:
    """Return all flat forecast feature column names (for XGBoost).

    Returns
    -------
    list[str]
        e.g. ['ghi_fcast_h1', …, 'ghi_fcast_h24',
               'cloud_opacity_fcast_h1', …, 'air_temp_fcast_h24']
    """
    cols = []
    for col in FORECAST_WEATHER_COLS:
        for h in range(1, n_horizons + 1):
            cols.append(f"{col}_fcast_h{h}")
    return cols


def get_lstm_forecast_cols(n_horizons: int = N_HORIZONS) -> list[str]:
    """Return the forecast column names used for LSTM context injection.

    Returns
    -------
    list[str]
        ghi_fcast_h1 … h24,  cloud_opacity_fcast_h1 … h24  (48 total)
    """
    cols = []
    for col in LSTM_FORECAST_WEATHER:
        for h in range(1, n_horizons + 1):
            cols.append(f"{col}_fcast_h{h}")
    return cols


# Pre-computed so lstm_forecaster can import without calling functions
LSTM_FORECAST_COLS: list[str] = get_lstm_forecast_cols()
N_LSTM_FCAST_FEATURES: int    = len(LSTM_FORECAST_COLS)   # 48


# ── NWP history training features (preferred) ────────────────────────────────

def build_nwp_forecast_features(
    df_hourly: pd.DataFrame,
    nwp_df: pd.DataFrame,
    n_horizons: int = N_HORIZONS,
) -> pd.DataFrame:
    """Add NWP-aligned forecast features using Open-Meteo historical data.

    For each decision row at time t and horizon h:
        ``col_fcast_h{h}``[t]  =  Open-Meteo ERA5 ``col`` at timestamp t+h

    This builds the same columns as the oracle approach but uses NWP-quality
    data (with real model errors) instead of perfect-forecast proxy values.
    Training and inference therefore see the same data source and error
    distribution.

    Parameters
    ----------
    df_hourly:
        Hourly feature DataFrame (training data, tz-aware index).
    nwp_df:
        Open-Meteo historical DataFrame from ``NWPHistoricalClient.fetch()``.
        Must cover the full date range of ``df_hourly`` plus 24 hours ahead.
        Columns: ``ghi``, ``cloud_opacity``, ``air_temp`` (and others).
    n_horizons:
        Number of forecast horizons (must match model config).

    Returns
    -------
    pd.DataFrame
        ``df_hourly`` with 72 additional NWP forecast feature columns.
        Rows where NWP data is unavailable for t+h are filled with the
        oracle fallback (actual Solcast value shifted back).
    """
    df = df_hourly.copy()

    # Reindex NWP to the hourly grid; gaps get NaN (handled below)
    nwp_aligned = nwp_df.reindex(df.index)

    added = 0
    for col in FORECAST_WEATHER_COLS:
        if col not in nwp_aligned.columns:
            logger.warning(
                "NWP history missing column '%s' — using oracle fallback.", col
            )
            for h in range(1, n_horizons + 1):
                src = df[col].shift(-h) if col in df.columns else pd.Series(0.0, index=df.index)
                df[f"{col}_fcast_h{h}"] = src
            continue

        for h in range(1, n_horizons + 1):
            # Shift NWP data: value at t+h appears at row t
            nwp_shifted = nwp_aligned[col].shift(-h)
            # Where NWP is NaN (coverage gap), fall back to oracle
            if col in df.columns:
                oracle = df[col].shift(-h)
                nwp_shifted = nwp_shifted.fillna(oracle)
            df[f"{col}_fcast_h{h}"] = nwp_shifted
            added += 1

    logger.info(
        "NWP forecast features added: %d columns for %d horizons.",
        added, n_horizons,
    )
    return df


# ── Oracle training features (fallback) ──────────────────────────────────────

def build_oracle_forecast_features(
    df_hourly: pd.DataFrame,
    n_horizons: int = N_HORIZONS,
) -> pd.DataFrame:
    """Add oracle (perfect) forecast features by shifting actual values forward.

    For each weather column w and horizon h:
        ``w_fcast_h{h}``[t]  =  actual ``w``[t+h]

    Used during training where true future weather is known.
    The last ``n_horizons`` rows will have NaN in these columns; they are
    already removed by ``build_target_matrix()``.

    Parameters
    ----------
    df_hourly:
        Hourly DataFrame with ``ghi``, ``cloud_opacity``, ``air_temp`` columns.
    n_horizons:
        Number of horizons to construct.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 72 additional forecast feature columns.
    """
    df = df_hourly.copy()

    added = 0
    for col in FORECAST_WEATHER_COLS:
        if col not in df.columns:
            logger.warning("Column '%s' not found — forecast features for it will be 0.", col)
            for h in range(1, n_horizons + 1):
                df[f"{col}_fcast_h{h}"] = 0.0
            continue
        for h in range(1, n_horizons + 1):
            df[f"{col}_fcast_h{h}"] = df[col].shift(-h)
            added += 1

    logger.info(
        "Oracle forecast features added: %d columns for %d horizons.",
        added, n_horizons,
    )
    return df


# ── Live inference features ───────────────────────────────────────────────────

def build_live_forecast_features(
    df_row: pd.DataFrame,
    forecast_df: pd.DataFrame,
    n_horizons: int = N_HORIZONS,
) -> pd.DataFrame:
    """Map Solcast API forecast output into the flat feature row.

    Parameters
    ----------
    df_row:
        Single-row (or multi-row) feature DataFrame for the current
        decision time(s).  Will be augmented with forecast columns in-place.
    forecast_df:
        DataFrame from ``SolcastForecastClient.fetch()``, indexed by future
        timestamps (tz-aware Asia/Colombo), columns include ``ghi``,
        ``cloud_opacity``, ``air_temp``.
    n_horizons:
        Must match the number of horizons the model was trained with.

    Returns
    -------
    pd.DataFrame
        ``df_row`` with ``ghi_fcast_h1`` … ``air_temp_fcast_h24`` columns added.
    """
    df = df_row.copy()
    fcast = forecast_df.head(n_horizons).reset_index(drop=True)

    for col in FORECAST_WEATHER_COLS:
        for h in range(1, n_horizons + 1):
            fcast_col = f"{col}_fcast_h{h}"
            if col in fcast.columns and (h - 1) < len(fcast):
                df[fcast_col] = float(fcast.loc[h - 1, col])
            else:
                df[fcast_col] = 0.0

    return df


# ── LSTM forecast context helpers ─────────────────────────────────────────────

def extract_lstm_forecast_context(
    df: pd.DataFrame,
    n_horizons: int = N_HORIZONS,
) -> np.ndarray:
    """Extract and normalize the LSTM forecast context array.

    Parameters
    ----------
    df:
        DataFrame that contains the LSTM forecast columns
        (``ghi_fcast_h1``…``cloud_opacity_fcast_h24``).

    Returns
    -------
    np.ndarray of shape (n_rows, N_LSTM_FCAST_FEATURES)
        Normalized values: ghi columns / 1000, cloud_opacity columns / 100.
        Returns zeros array if columns are absent.
    """
    lstm_cols = get_lstm_forecast_cols(n_horizons)
    if not any(c in df.columns for c in lstm_cols):
        return np.zeros((len(df), len(lstm_cols)), dtype=np.float32)

    out = np.zeros((len(df), len(lstm_cols)), dtype=np.float32)
    for i, col in enumerate(lstm_cols):
        if col in df.columns:
            out[:, i] = df[col].fillna(0.0).values

    # Normalize in blocks
    n = n_horizons
    for j, (_, scale) in enumerate(zip(LSTM_FORECAST_WEATHER, LSTM_FORECAST_SCALES)):
        start = j * n
        out[:, start : start + n] /= scale

    return out
