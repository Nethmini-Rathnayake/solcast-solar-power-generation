"""
src/features/lag_features.py
------------------------------
Lag features, rolling statistics, and multi-step target construction for the
hourly XGBoost Direct Multi-Step (DMS) forecaster.

Cross-repo compatibility
------------------------
Target column name (``pv_ac_W``) and target column naming convention
(``target_h1`` … ``target_h24``) are kept identical to the
``solar-generation-forecasting`` (NASA POWER) repository so that evaluation
scripts and notebooks can be reused across both repos.

Forecasting strategy: Direct Multi-Step (DMS)
---------------------------------------------
One XGBoost model is trained per horizon h ∈ {1, …, 24}.
Model h predicts:
    ŷ(t+h) = f_h(X_t)
where X_t contains only information available at decision time t (lags of
observed pv_ac_W, current-hour weather features, pvlib simulation).

Lag groups
----------
  Recent (short memory):   t-1 … t-6    (last 6 hours)
  Daily lag:               t-24          (same hour yesterday)
  Two-day lag:             t-48          (same hour 2 days ago)
  Weekly lag:              t-168         (same hour last week — captures
                                          weekly cloud / rain cycle patterns)

Rolling statistics (trailing windows, no lookahead)
----------------------------------------------------
  pv_ac_W rolling mean:  6h, 24h
  pv_ac_W rolling std:   24h

Target columns
--------------
  ``target_h1`` … ``target_h24`` : pv_ac_W at t+1 … t+24 (in Watts)
  Rows where any target is NaN are dropped (end of series, data gaps).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lag depths (hours before decision time t)
_LAG_HOURS: list[int] = [1, 2, 3, 4, 5, 6, 24, 48, 168]

# Rolling window lengths (hours)
_ROLLING_WINDOWS: list[int] = [6, 24]

# Number of forecast horizons
_N_HORIZONS: int = 24

# Target column name (shared with solar-generation-forecasting repo)
_TARGET_COL: str = "pv_ac_W"


def add_lag_features(
    df: pd.DataFrame,
    cfg: SimpleNamespace | None = None,
) -> pd.DataFrame:
    """Add lag and rolling statistic features for pv_ac_W.

    Parameters
    ----------
    df:
        Hourly DataFrame with a ``pv_ac_W`` column.
    cfg:
        Optional ``features`` namespace from ``pipeline.yaml``.  If provided,
        ``lag_hours`` and ``rolling_windows`` override the module defaults.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional lag and rolling stat columns.
        Rows at the start of the series (within the longest lag window) will
        have NaN in the new columns — handled when building the final matrix.
    """
    df = df.copy()

    lag_hours = _LAG_HOURS
    rolling_windows = _ROLLING_WINDOWS

    if cfg is not None:
        lag_hours = list(getattr(cfg, "lag_hours", lag_hours))
        rolling_windows = list(getattr(cfg, "rolling_windows", rolling_windows))

    if _TARGET_COL not in df.columns:
        raise KeyError(
            f"Column '{_TARGET_COL}' not found.  "
            "Run aggregation before lag feature construction."
        )

    # ── Lag features ──────────────────────────────────────────────────────────
    for h in lag_hours:
        df[f"lag_{h}h"] = df[_TARGET_COL].shift(h)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in rolling_windows:
        # min_periods=w//2 allows partial windows at the start of the series
        df[f"rolling_mean_{w}h"] = (
            df[_TARGET_COL]
            .shift(1)   # shift(1): use only information before time t
            .rolling(window=w, min_periods=w // 2)
            .mean()
        )
        df[f"rolling_std_{w}h"] = (
            df[_TARGET_COL]
            .shift(1)
            .rolling(window=w, min_periods=w // 2)
            .std()
        )

    logger.debug(
        "Lag features added: %d lags, rolling windows %s.",
        len(lag_hours),
        rolling_windows,
    )
    return df


def build_target_matrix(
    df: pd.DataFrame,
    n_horizons: int = _N_HORIZONS,
) -> pd.DataFrame:
    """Append target columns target_h1 … target_hN and drop incomplete rows.

    Parameters
    ----------
    df:
        Hourly DataFrame with lag features already added.
    n_horizons:
        Number of forecast horizons.  Default: 24.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``target_h1`` … ``target_hN`` columns.
        Rows where any target is NaN are dropped (tail of the series and
        gaps longer than the interpolation limit).
    """
    df = df.copy()

    target_cols: list[str] = []
    for h in range(1, n_horizons + 1):
        col = f"target_h{h}"
        df[col] = df[_TARGET_COL].shift(-h)
        target_cols.append(col)

    # Drop rows where any target is NaN (no leakage, no partial targets)
    n_before = len(df)
    df = df.dropna(subset=target_cols)
    n_dropped = n_before - len(df)

    logger.info(
        "Target matrix built: %d rows retained, %d dropped (NaN targets).",
        len(df),
        n_dropped,
    )
    return df
