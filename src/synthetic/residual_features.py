"""
src/synthetic/residual_features.py
------------------------------------
Build the feature matrix used to train and apply the residual correction
models (XGBoost + LSTM).

What the residual captures
--------------------------
  residual = real_pv_hourly - synthetic_pv_hourly

This scalar encodes everything the pvlib/Solcast physical model missed:
  - Soiling losses       (systematic, season/rain-dependent)
  - Local shading        (time-of-day and declination dependent)
  - Inverter degradation (slow drift, mostly captured by season features)
  - Measurement noise    (random, zero-mean — unlearnable, absorbed by model)

The residual correction model learns to predict this offset from features
that correlate with the above effects, so that when we apply it to the
4-year synthetic PV we get output that "behaves" like real PV.

Feature set (no wind speed — not available in Solcast)
------------------------------------------------------
  Weather (hourly):
    ghi, dni, dhi, air_temp, relative_humidity, cloud_opacity

  Physics proxy (hourly, from pvlib simulation):
    pvlib_ac_kw, clearness_index_hourly

  Time / cyclical:
    hour_sin, hour_cos, doy_sin, doy_cos, month_sin, month_cos

  Monsoon (for Sri Lanka seasonal loss patterns):
    monsoon_category (int 0-3, for XGBoost)
    monsoon_sw, monsoon_ne, monsoon_inter1, monsoon_inter2 (one-hot, for LSTM)

Target
------
  residual_W = pv_ac_W - pvlib_ac_W   (in Watts, signed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.monsoon import add_monsoon_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Columns included in the tabular feature matrix (XGBoost) ─────────────────
TABULAR_FEATURE_COLS = [
    "ghi",
    "dni",
    "dhi",
    "air_temp",
    "relative_humidity",
    "cloud_opacity",
    "pvlib_ac_kW",
    "clearness_index_hourly",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
    "monsoon_category",
]

# ── Columns included per timestep in the LSTM input sequence ─────────────────
SEQUENCE_FEATURE_COLS = [
    "ghi_norm",
    "air_temp_norm",
    "relative_humidity_norm",
    "cloud_opacity_norm",
    "clearness_index_norm",
    "pvlib_ac_norm",
    "hour_sin",
    "hour_cos",
]

RESIDUAL_COL = "residual_W"


def build_residual_features(
    hourly_df: pd.DataFrame,
    daytime_only: bool = True,
) -> pd.DataFrame:
    """Compute residual target and add all correction features.

    Parameters
    ----------
    hourly_df:
        Hourly DataFrame containing at minimum:
          ``pv_ac_W``         — observed AC output (real plant, W)
          ``pvlib_ac_W``      — pvlib simulated AC output (W)
          ``ghi``, ``dni``, ``dhi``
          ``air_temp``, ``relative_humidity``, ``cloud_opacity``
          ``pvlib_ac_kW``     — pvlib AC in kW
          ``clearness_index_hourly``
          ``hour_sin``, ``hour_cos``, ``doy_sin``, ``doy_cos``
          ``month_sin``, ``month_cos``
    daytime_only:
        If True (default), return only daytime rows (pvlib_ac_W > 100 W).
        Use True for XGBoost (tabular — no sequence continuity needed).
        If False, return all hourly rows with a boolean ``is_daytime``
        column so the LSTM dataset can filter valid *targets* while still
        using continuous time sequences.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
          ``residual_W``, ``is_daytime``, ``monsoon_*``,
          normalised columns for LSTM sequences.
    """
    df = hourly_df.copy()

    # ── Daytime flag ──────────────────────────────────────────────────────────
    daytime_mask = (
        df["pv_ac_W"].notna()
        & df["pvlib_ac_W"].notna()
        & (df["pvlib_ac_W"] > 100.0)
    )
    df["is_daytime"] = daytime_mask.astype(int)

    # ── Residual target (0 at nighttime — not used for LSTM loss) ─────────────
    df[RESIDUAL_COL] = (df["pv_ac_W"] - df["pvlib_ac_W"]).where(daytime_mask, 0.0)

    # ── Monsoon features ──────────────────────────────────────────────────────
    df = add_monsoon_features(df)

    # ── Normalised columns for LSTM sequences ─────────────────────────────────
    df = _add_normalised_cols(df)

    if daytime_only:
        n_before = len(df)
        df = df[daytime_mask].copy()
        logger.info(
            "Residual feature matrix: %d daytime rows retained (from %d total).",
            len(df), n_before,
        )
    else:
        logger.info(
            "Residual feature matrix (full): %d total rows | %d daytime.",
            len(df), daytime_mask.sum(),
        )

    return df


def _add_normalised_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised versions of key columns for LSTM input.

    All resulting columns are NaN-free (filled with 0.0) so they are safe
    to use directly in PyTorch tensors without further processing.
    """
    pairs = [
        ("ghi",               "ghi_norm",               1000.0),
        ("air_temp",          "air_temp_norm",           50.0),
        ("relative_humidity", "relative_humidity_norm",  100.0),
        ("cloud_opacity",     "cloud_opacity_norm",      100.0),
        ("clearness_index_hourly", "clearness_index_norm", 1.1),
        ("pvlib_ac_W",        "pvlib_ac_norm",           None),   # dynamic
    ]
    for src, dst, scale in pairs:
        if src not in df.columns:
            df[dst] = 0.0
            continue
        if scale is None:
            col_max = df[src].max()
            df[dst] = df[src] / col_max if col_max > 0 else 0.0
        else:
            df[dst] = df[src] / scale
        # Fill NaN with 0 — nighttime or missing values become neutral inputs
        df[dst] = df[dst].fillna(0.0)
    return df
