"""
src/features/physics_features.py
----------------------------------
Physics-derived features for the hourly feature matrix.

These features encode the relationship between Solcast-simulated irradiance
(what the sun provides) and the pvlib-simulated PV output (what an ideal
plant would produce), relative to what was actually measured.

Features added
--------------
  - ``pvlib_ac_kW``            : pvlib simulated AC output (kW) — provides
                                  the model with a physics-prior for each hour.
  - ``pvlib_residual``         : actual / simulated  (dimensionless).
                                  Values < 1 indicate real-world losses (soiling,
                                  clipping, communication faults).
                                  Values > 1 indicate over-performance or
                                  measurement offsets.
                                  NaN during nighttime.
  - ``clearness_index_hourly`` : hourly-resolution clearness index
                                  (GHI / clearsky_GHI), renamed for clarity.

Viva note
---------
The pvlib_residual feature is the key calibration signal.  If the residual
has a systematic pattern (e.g. lower in summer due to soiling), the ML model
learns to correct for it.  This is what distinguishes a physics-informed
ML model from a black-box regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum simulated AC power to compute a meaningful residual (W)
_PVLIB_MIN_W = 500.0


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-derived features to the hourly feature matrix.

    Expects ``pvlib_ac_W``, ``pv_ac_W``, and optionally ``clearness_index``
    to be present in ``df`` (produced by the aggregation step).

    Parameters
    ----------
    df:
        Hourly DataFrame after aggregation and time/weather feature steps.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional physics feature columns.
    """
    df = df.copy()

    # ── pvlib AC in kW (more interpretable than W at this scale) ─────────────
    if "pvlib_ac_W" in df.columns:
        df["pvlib_ac_kW"] = df["pvlib_ac_W"] / 1_000.0
    else:
        logger.warning(
            "pvlib_ac_W not found — physics features will be NaN. "
            "Run pvlib simulation before feature engineering."
        )
        df["pvlib_ac_kW"] = np.nan

    # ── pvlib residual (actual / simulated) ───────────────────────────────────
    # Only meaningful during daytime when pvlib output is substantial.
    if "pvlib_ac_W" in df.columns and "pv_ac_W" in df.columns:
        daytime_sim = df["pvlib_ac_W"] >= _PVLIB_MIN_W
        df["pvlib_residual"] = np.where(
            daytime_sim,
            df["pv_ac_W"] / df["pvlib_ac_W"].clip(lower=_PVLIB_MIN_W),
            np.nan,
        )
        # Clip to physically reasonable range
        df["pvlib_residual"] = df["pvlib_residual"].clip(0, 2.0)
    else:
        df["pvlib_residual"] = np.nan

    # ── Hourly clearness index (rename for clarity) ────────────────────────
    if "clearness_index" in df.columns:
        df["clearness_index_hourly"] = df["clearness_index"]
    elif "ghi" in df.columns and "clearsky_ghi" in df.columns:
        daytime = df["clearsky_ghi"] >= 10.0
        df["clearness_index_hourly"] = np.where(
            daytime,
            (df["ghi"] / df["clearsky_ghi"].clip(lower=10.0)).clip(0, 1.1),
            np.nan,
        )
    else:
        df["clearness_index_hourly"] = np.nan

    logger.debug(
        "Physics features added: pvlib_ac_kW, pvlib_residual, "
        "clearness_index_hourly."
    )
    return df
