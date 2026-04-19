"""
src/features/monsoon.py
------------------------
Sri Lanka monsoon season classification and feature encoding.

Sri Lanka experiences four distinct meteorological seasons driven by the
Inter-Tropical Convergence Zone (ITCZ) and large-scale monsoon circulation.
Each season has a different cloud-cover regime that directly affects solar
irradiance and PV output:

  SW Monsoon         (May–Sep)    : heavy cloud, high humidity, low GHI
  NE Monsoon         (Dec–Feb)    : moderate cloud, drier in west/south
  First Inter-Monsoon (Mar–Apr)   : convective instability, afternoon clouds
  Second Inter-Monsoon (Oct–Nov)  : transitional, mixed convection

Encoding
--------
Two representations are provided:

1. ``monsoon_category`` (int, 0–3): ordinal label suitable for XGBoost,
   which can split on the integer directly.

2. One-hot columns ``monsoon_sw``, ``monsoon_ne``, ``monsoon_inter1``,
   ``monsoon_inter2``: appropriate for LSTM input where the embedding
   must not imply ordinal distance between seasons.

Usage
-----
    from src.features.monsoon import add_monsoon_features
    df = add_monsoon_features(df)
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Season → integer mapping ──────────────────────────────────────────────────
_CATEGORY_MAP = {
    "SW_monsoon": 0,
    "NE_monsoon": 1,
    "first_inter_monsoon": 2,
    "second_inter_monsoon": 3,
}

# Month → season name
_MONTH_TO_SEASON: dict[int, str] = {
    1:  "NE_monsoon",
    2:  "NE_monsoon",
    3:  "first_inter_monsoon",
    4:  "first_inter_monsoon",
    5:  "SW_monsoon",
    6:  "SW_monsoon",
    7:  "SW_monsoon",
    8:  "SW_monsoon",
    9:  "SW_monsoon",
    10: "second_inter_monsoon",
    11: "second_inter_monsoon",
    12: "NE_monsoon",
}


def get_monsoon_season(month: int) -> str:
    """Return the monsoon season name for a given calendar month (1–12)."""
    return _MONTH_TO_SEASON[month]


def add_monsoon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add monsoon season features to the hourly feature matrix.

    Adds both an integer category column (for XGBoost) and one-hot columns
    (for LSTM input sequences).

    Parameters
    ----------
    df:
        DataFrame with a tz-aware DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        ``monsoon_category`` (int, 0–3),
        ``monsoon_sw``, ``monsoon_ne``, ``monsoon_inter1``, ``monsoon_inter2``
        (int, 0 or 1).
    """
    df = df.copy()
    month = df.index.month

    season_names = pd.Series(
        [_MONTH_TO_SEASON[m] for m in month],
        index=df.index,
    )

    df["monsoon_category"] = season_names.map(_CATEGORY_MAP).astype(int)

    # One-hot columns
    df["monsoon_sw"]     = (season_names == "SW_monsoon").astype(int)
    df["monsoon_ne"]     = (season_names == "NE_monsoon").astype(int)
    df["monsoon_inter1"] = (season_names == "first_inter_monsoon").astype(int)
    df["monsoon_inter2"] = (season_names == "second_inter_monsoon").astype(int)

    logger.debug(
        "Monsoon features added. Season distribution:\n%s",
        season_names.value_counts().to_string(),
    )
    return df
