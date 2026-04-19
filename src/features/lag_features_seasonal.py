"""
src/features/lag_features_seasonal.py
---------------------------------------
Extended lag features for XGBoost — adds yearly and seasonal lags on top of
the standard short-term lags in ``lag_features.py``.

Why seasonal / yearly lags help XGBoost
-----------------------------------------
XGBoost has no recurrent memory.  Every temporal pattern must be explicitly
encoded as a column.  Standard lags (t-1 … t-168) capture intra-day and
weekly cycles.  Seasonal / yearly lags additionally capture:

  - ``lag_2190h``  (~91 days / 3 months) — same hour last quarter.
    Bridges inter-monsoon → SW-monsoon transition (and vice-versa).
  - ``lag_4380h``  (~182 days / 6 months) — same hour half-year ago.
    Directly contrasts the two monsoon peaks (SW May–Sep, NE Oct–Jan).
  - ``lag_8760h``  (365 days / 1 year)  — same hour last year.
    Strongest seasonal signal: captures annual cloud-cover cycle.

Rolling monthly statistic
--------------------------
  - ``rolling_mean_720h`` — 30-day trailing mean of pv_ac_W.
    Captures slow seasonal drift better than the 24h / 6h windows.

Data-length problem & synthetic fill
--------------------------------------
The real PV dataset spans only ~1 year (Apr 2022 – Mar 2023).
  • lag_2190h needs data from ~Jan 2022 — partially available.
  • lag_4380h needs data from ~Oct 2021 — not available in real PV.
  • lag_8760h needs data from Apr 2021   — not available in real PV.

To avoid all-NaN columns, these deep lags are filled from the 4-year
corrected synthetic PV (``pv_corrected_W``) where the real PV is absent.
Synthetic r = 0.918 with real, so it is a reasonable lag proxy.

After the fill, any remaining NaN is forward-filled then zero-filled so
that XGBoost never receives NaN inputs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features.lag_features import add_lag_features, build_target_matrix
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Seasonal lag depths (hours) ───────────────────────────────────────────────
_SEASONAL_LAGS: list[int] = [
    2190,   # ~3 months  (quarter-year)
    4380,   # ~6 months  (half-year)
    8760,   # ~12 months (full year)
]

_ROLLING_MONTHLY: list[int] = [720]   # 30-day window

_TARGET_COL = "pv_ac_W"
_SYNTH_COL  = "pv_corrected_W"


def add_seasonal_lag_features(
    df: pd.DataFrame,
    synth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add seasonal / yearly lag features to the hourly real-PV DataFrame.

    Parameters
    ----------
    df:
        Hourly real-PV DataFrame; must contain ``pv_ac_W``.
        Index: tz-aware DatetimeIndex.
    synth_df:
        Optional 4-year corrected synthetic DataFrame (output of step 05).
        Must contain ``pv_corrected_W`` with the same timezone.
        Used to fill deep lags that fall before the start of real data.
        If None, deep lags are left as NaN and forward-filled.

    Returns
    -------
    pd.DataFrame with additional lag and rolling-stat columns.
    """
    df = df.copy()

    if _TARGET_COL not in df.columns:
        raise KeyError(f"Column '{_TARGET_COL}' not found.")

    # ── Build unified PV series: real where available, synthetic elsewhere ──
    if synth_df is not None and _SYNTH_COL in synth_df.columns:
        # Reindex synthetic to hourly grid; only keep hours NOT covered by real
        synth_pv = synth_df[_SYNTH_COL].reindex(
            pd.date_range(
                start=synth_df.index.min(),
                end=df.index.max(),
                freq="h",
                tz=df.index.tz,
            )
        )
        # Merge: real takes priority; fill gaps with synthetic
        combined_pv = df[_TARGET_COL].combine_first(synth_pv)
        logger.info(
            "Seasonal lags: combined PV series — %d real + %d synthetic hours.",
            df[_TARGET_COL].notna().sum(),
            (combined_pv.notna() & df[_TARGET_COL].isna()).sum(),
        )
    else:
        combined_pv = df[_TARGET_COL]
        logger.warning(
            "No synthetic data supplied — deep lags (t-4380, t-8760) will be "
            "mostly NaN and will be forward-filled."
        )

    # ── Seasonal lag features ────────────────────────────────────────────────
    for h in _SEASONAL_LAGS:
        col = f"lag_{h}h"
        shifted = combined_pv.shift(h)
        # Align back to df.index
        df[col] = shifted.reindex(df.index)
        n_valid = df[col].notna().sum()
        logger.info(
            "  lag_%dh: %d / %d rows filled (%.1f%%).",
            h, n_valid, len(df), 100.0 * n_valid / len(df),
        )

    # ── Monthly rolling mean ─────────────────────────────────────────────────
    for w in _ROLLING_MONTHLY:
        col = f"rolling_mean_{w}h"
        df[col] = (
            combined_pv
            .reindex(df.index)
            .shift(1)                              # no lookahead
            .rolling(window=w, min_periods=w // 4) # allow up to 75% missing
            .mean()
        )

    # ── pvlib yearly lag (pure physics — no synthetic uncertainty) ────────────
    if "pvlib_ac_W" in df.columns:
        pvlib_series = synth_df["pvlib_ac_W"].reindex(
            pd.date_range(
                start=synth_df.index.min(),
                end=df.index.max(),
                freq="h",
                tz=df.index.tz,
            )
        ) if synth_df is not None else df["pvlib_ac_W"]

        df["pvlib_lag_8760h"] = pvlib_series.shift(8760).reindex(df.index)
        logger.info("  pvlib_lag_8760h added.")

    # ── Forward-fill then zero-fill remaining NaN ────────────────────────────
    seasonal_cols = (
        [f"lag_{h}h" for h in _SEASONAL_LAGS]
        + [f"rolling_mean_{w}h" for w in _ROLLING_MONTHLY]
        + (["pvlib_lag_8760h"] if "pvlib_ac_W" in df.columns else [])
    )
    df[seasonal_cols] = df[seasonal_cols].ffill().fillna(0.0)

    logger.info("Seasonal lag features added: %s", seasonal_cols)
    return df
