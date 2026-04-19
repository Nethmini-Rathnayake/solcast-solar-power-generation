"""
src/features/weather_features.py
----------------------------------
Select and rename Solcast weather/irradiance columns for the feature matrix.

This module is intentionally thin — Solcast columns are already numeric and
at the correct resolution after ``aggregation.py``.  The role here is to:
  1. Confirm that expected columns are present.
  2. Add derived irradiance features that are physically meaningful.
  3. Log what is available.

Derived features added
----------------------
  - ``ghi_clearsky_ratio``    : GHI / clearsky_GHI  (= clearness_index at
                                hourly scale; re-computed here for explicitness)
  - ``dni_clearsky_ratio``    : DNI / clearsky_DNI
  - ``diffuse_fraction``      : DHI / GHI  (sky diffuse fraction; high during
                                overcast conditions)

These ratios expose cloud state independently of the absolute irradiance level,
which helps the model generalise across seasons.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum clearsky value to avoid division by near-zero
_CLEARSKY_FLOOR_WM2 = 10.0
_GHI_FLOOR_WM2 = 10.0


def add_weather_features(
    df: pd.DataFrame,
    pipeline_cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Add and validate Solcast-derived weather features.

    Parameters
    ----------
    df:
        Hourly DataFrame with Solcast columns already present.
    pipeline_cfg:
        The ``features`` namespace from ``pipeline.yaml``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional derived irradiance ratio columns.
    """
    df = df.copy()
    feat_cfg = pipeline_cfg.features

    # ── Check expected irradiance columns ────────────────────────────────────
    irr_cols = list(feat_cfg.solcast_irradiance_cols)
    met_cols = list(feat_cfg.solcast_met_cols)
    expected = irr_cols + met_cols

    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning(
            "Expected Solcast columns not found in DataFrame (will be NaN): %s",
            missing,
        )

    # ── Derived irradiance ratios ─────────────────────────────────────────────
    if "ghi" in df.columns and "clearsky_ghi" in df.columns:
        df["ghi_clearsky_ratio"] = np.where(
            df["clearsky_ghi"] >= _CLEARSKY_FLOOR_WM2,
            df["ghi"] / df["clearsky_ghi"].clip(lower=_CLEARSKY_FLOOR_WM2),
            np.nan,
        ).clip(0, 1.1)

    if "dni" in df.columns and "clearsky_dni" in df.columns:
        df["dni_clearsky_ratio"] = np.where(
            df["clearsky_dni"] >= _CLEARSKY_FLOOR_WM2,
            df["dni"] / df["clearsky_dni"].clip(lower=_CLEARSKY_FLOOR_WM2),
            np.nan,
        ).clip(0, 1.1)

    if "dhi" in df.columns and "ghi" in df.columns:
        df["diffuse_fraction"] = np.where(
            df["ghi"] >= _GHI_FLOOR_WM2,
            df["dhi"] / df["ghi"].clip(lower=_GHI_FLOOR_WM2),
            np.nan,
        ).clip(0, 1.0)

    logger.debug(
        "Weather features added: ghi_clearsky_ratio, dni_clearsky_ratio, "
        "diffuse_fraction."
    )
    return df
