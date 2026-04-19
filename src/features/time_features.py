"""
src/features/time_features.py
-------------------------------
Time-based and solar-position features for the hourly feature matrix.

Features added
--------------
  Cyclical encodings (sin/cos pairs prevent discontinuity at wrap-around):
    - ``hour_sin``, ``hour_cos``           : hour of day  (0–23)
    - ``month_sin``, ``month_cos``         : month of year (1–12)
    - ``doy_sin``, ``doy_cos``             : day of year   (1–366)

  Solar position (from pvlib, computed at each hourly timestamp):
    - ``solar_elevation_deg``              : apparent solar elevation (°)
    - ``solar_azimuth_deg``                : solar azimuth (°, N=0, clockwise)
    - ``cos_solar_zenith``                 : cos(zenith) — linear irradiance proxy

  Calendar flags:
    - ``is_daytime``                       : 1 if solar elevation > 0

These features give the model direct access to the astronomical state of the
sun, which is the dominant driver of PV output variability.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_time_features(
    df: pd.DataFrame,
    site_cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Add time-based and solar-position features to the hourly DataFrame.

    Parameters
    ----------
    df:
        Hourly DataFrame with a tz-aware DatetimeIndex (Asia/Colombo).
    site_cfg:
        The top-level ``site`` namespace from ``site.yaml`` (contains
        ``site.latitude``, ``site.longitude``, ``site.timezone``).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional feature columns appended.
    """
    df = df.copy()
    idx = df.index

    # ── Cyclical time encodings ───────────────────────────────────────────────
    hour = idx.hour
    month = idx.month
    doy = idx.day_of_year

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365)
    df["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365)

    # ── Solar position ────────────────────────────────────────────────────────
    location = Location(
        latitude=site_cfg.latitude,
        longitude=site_cfg.longitude,
        tz=site_cfg.timezone,
        altitude=site_cfg.elevation_m,
        name=site_cfg.name,

    )
    solar_pos = location.get_solarposition(idx)

    df["solar_elevation_deg"] = solar_pos["apparent_elevation"].values
    df["solar_azimuth_deg"] = solar_pos["azimuth"].values
    # cos(zenith) is the projection factor used in irradiance calculations
    zenith_rad = np.radians(solar_pos["apparent_zenith"].values)
    df["cos_solar_zenith"] = np.clip(np.cos(zenith_rad), 0, None)
    # ── Daytime flag ─────────────────────────────────────────────────────────
    df["is_daytime"] = (df["solar_elevation_deg"] > 0).astype(int)

    logger.debug("Time features added: %d rows.", len(df))
    return df
