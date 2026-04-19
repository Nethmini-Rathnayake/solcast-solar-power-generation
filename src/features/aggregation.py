"""
src/features/aggregation.py
-----------------------------
Resample the cleaned 5-minute aligned DataFrame to hourly resolution.

Design rationale
----------------
The model operates at hourly resolution (24 forecast horizons, h+1 … h+24).
Features and targets are therefore computed on hourly aggregates.

Aggregation rules
-----------------
  - Power (W):        mean  — average power over the hour is proportional
                               to energy delivered (Wh = W_mean × 1 h).
  - Irradiance (W/m²): mean — standard practice for hourly solar resource.
  - Cloud opacity:    mean  — captures mean cloud cover during the hour.
  - Temperature (°C): mean  — representative mid-hour value.
  - Clearness index:  mean  — daytime only (NaN rows excluded automatically
                               by pandas mean).
  - ``data_ok``:      min   — hour is marked valid only if ALL 5-min rows
                               within it are valid.  This is conservative but
                               prevents training on partially corrupted hours.

Output
------
Hourly DataFrame with DatetimeIndex frequency "1h", still tz-aware
(Asia/Colombo).  Each row represents the 60-minute period ending at the
timestamp (period_end convention, consistent with the 5-min source data).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Columns aggregated by mean
_MEAN_COLS = [
    "pv_ac_W",
    "pvlib_ac_W",
    "pvlib_dc_W",
    "poa_global_Wm2",
    "cell_temp_C",
    "clearness_index",
    "ghi",
    "dni",
    "dhi",
    "gti",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
    "clearsky_gti",
    "cloud_opacity",
    "air_temp",
    "dewpoint_temp",
    "relative_humidity",
    "surface_pressure",
    "albedo",
    "min_air_temp",
    "max_air_temp",
    "pm10",
    "pm2.5",
]

# Columns aggregated by min (quality flag: True only if all 5-min rows ok)
_MIN_COLS = ["data_ok"]


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample the 5-minute DataFrame to hourly.

    Parameters
    ----------
    df:
        Cleaned 5-min aligned DataFrame with a tz-aware DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame (``freq="1h"``).  Index name remains
        ``datetime_local``.  Only columns present in ``df`` are included.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")
    if df.index.tz is None:
        raise ValueError("DatetimeIndex must be tz-aware (Asia/Colombo).")

    mean_present = [c for c in _MEAN_COLS if c in df.columns]
    min_present = [c for c in _MIN_COLS if c in df.columns]

    agg_dict: dict[str, str] = {}
    for col in mean_present:
        agg_dict[col] = "mean"
    for col in min_present:
        agg_dict[col] = "min"

    # weather_type: take the mode (most frequent category in the hour)
    has_weather_type = "weather_type" in df.columns

    hourly = df[list(agg_dict.keys())].resample("1h").agg(agg_dict)

    if has_weather_type:
        weather_mode = (
            df["weather_type"]
            .resample("1h")
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        )
        hourly["weather_type"] = weather_mode

    logger.info(
        "Aggregated 5-min → hourly: %d rows → %d rows (%.1f%% retained).",
        len(df),
        len(hourly),
        100.0 * len(hourly) * 12 / len(df),
    )
    return hourly
