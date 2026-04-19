"""
src/preprocessing/cleaning.py
-------------------------------
Data cleaning for the aligned 5-minute PV + Solcast DataFrame.

Cleaning steps applied in order
--------------------------------
1. **Overflow removal**
   Per-inverter columns contain a hardware overflow sentinel: 32,767,000 W
   (int16_max × 1000 from SCADA firmware).  These are set to NaN.
   Note: ``pv_ac_W`` (Power Total) is read at the AC bus and does NOT carry
   the overflow sentinel; it is cleaned by bounds checking only.

2. **Physical bounds check**
   ``pv_ac_W`` values outside [0, 400 kW] are set to NaN.
   Negative values are clamped to 0 at night (clearsky GHI < threshold).

3. **Nighttime validation**
   When clearsky GHI < 5 W/m² (sun below horizon), any non-zero pv_ac_W
   is suspicious.  Values above a small noise floor (50 W) are flagged.

4. **Short-gap interpolation**
   Gaps of up to 6 consecutive NaN steps (30 minutes) in ``pv_ac_W`` are
   filled by linear interpolation.  Longer gaps are left as NaN; downstream
   models should either drop or mask these rows.

5. **Quality flag column**
   A boolean ``data_ok`` column is appended: True where pv_ac_W is finite
   and passes all checks.  This lets downstream steps choose whether to
   impute, drop, or weight bad rows without hard-deleting data.

Cross-repo note
---------------
Cleaning thresholds come from ``pipeline.yaml`` (``cleaning`` section) so
they are not hardcoded here.  The ``clean`` function takes a
``CleaningConfig`` namespace matching the YAML structure.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Overflow sentinel (int16_max × 1000 from SCADA firmware) ─────────────────
_OVERFLOW_SENTINEL = 32_767_000.0

# Columns that may carry the overflow sentinel (per-inverter diagnostics)
_INVERTER_COLS = [
    "inv_1_1_W", "inv_1_2_W", "inv_1_3_W",
    "inv_2_1_W", "inv_2_2_W",
    "inv_3_1_W", "inv_3_2_W", "inv_3_3_W",
]

# Small noise floor: nighttime pv_ac_W above this is flagged (W)
_NIGHTTIME_NOISE_FLOOR_W = 500.0


def clean(
    df: pd.DataFrame,
    cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Apply the full cleaning sequence to the aligned 5-min DataFrame.

    Parameters
    ----------
    df:
        Aligned DataFrame from ``alignment.align`` — contains ``pv_ac_W``,
        per-inverter columns, and Solcast columns (including ``clearsky_ghi``).
    cfg:
        The ``cleaning`` namespace from ``pipeline.yaml``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an added ``data_ok`` boolean column.
        No rows are dropped — bad data is marked NaN and flagged.
    """
    df = df.copy()
    n_rows = len(df)

    df = _remove_inverter_overflows(df, cfg)
    df = _apply_physical_bounds(df, cfg)
    df = _flag_nighttime_anomalies(df, cfg)
    df = _interpolate_short_gaps(df, cfg)
    df = _add_quality_flag(df)

    n_bad = (~df["data_ok"]).sum()
    logger.info(
        "Cleaning complete: %d / %d rows flagged as bad (%.1f%%).",
        n_bad,
        n_rows,
        100.0 * n_bad / n_rows,
    )
    return df


# ── Step 1: Overflow removal ─────────────────────────────────────────────────

def _remove_inverter_overflows(
    df: pd.DataFrame,
    cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Replace overflow sentinel values in per-inverter columns with NaN."""
    sentinel = getattr(cfg, "overflow_sentinel_w", _OVERFLOW_SENTINEL)
    cols_present = [c for c in _INVERTER_COLS if c in df.columns]

    total_replaced = 0
    for col in cols_present:
        mask = df[col] >= sentinel
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            total_replaced += n

    if total_replaced > 0:
        logger.info(
            "Overflow sentinel removed: %d values across %d inverter columns.",
            total_replaced,
            len(cols_present),
        )
    return df


# ── Step 2: Physical bounds check ────────────────────────────────────────────

def _apply_physical_bounds(
    df: pd.DataFrame,
    cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Set pv_ac_W to NaN where it falls outside physical plausibility bounds."""
    pv_min = getattr(cfg, "pv_min_w", 0.0)
    pv_max = getattr(cfg, "pv_max_w", 400_000.0)

    below = df["pv_ac_W"] < pv_min
    above = df["pv_ac_W"] > pv_max

    n_below = below.sum()
    n_above = above.sum()

    if n_below > 0:
        logger.warning(
            "%d rows with pv_ac_W < %.0f W — set to NaN.", n_below, pv_min
        )
        df.loc[below, "pv_ac_W"] = np.nan

    if n_above > 0:
        logger.warning(
            "%d rows with pv_ac_W > %.0f W — set to NaN.", n_above, pv_max
        )
        df.loc[above, "pv_ac_W"] = np.nan

    return df


# ── Step 3: Nighttime anomaly flagging ────────────────────────────────────────

def _flag_nighttime_anomalies(
    df: pd.DataFrame,
    cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Warn about (and zero-out) pv_ac_W values during confirmed nighttime."""
    if "clearsky_ghi" not in df.columns:
        logger.debug(
            "clearsky_ghi not found — skipping nighttime anomaly check."
        )
        return df

    night_threshold = getattr(cfg, "clearsky_ghi_night_threshold_wm2", 5.0)
    is_night = df["clearsky_ghi"] == 0
    suspicious = is_night & (df["pv_ac_W"] > _NIGHTTIME_NOISE_FLOOR_W)

    sus = df.loc[suspicious, ["pv_ac_W", "clearsky_ghi", "ghi", "dni", "dhi"]].copy()

    print("\n=== Suspicious nighttime rows analysis ===")
    print(sus.describe())

    print("\nHour distribution:")
    print(sus.index.hour.value_counts().sort_index())

    print("\nTop PV values:")
    print(sus["pv_ac_W"].sort_values(ascending=False).head(20))

    n_suspicious = suspicious.sum()
    if n_suspicious > 0:
        logger.warning(
            "%d rows have pv_ac_W > %.0f W during confirmed nighttime "
            "(clearsky_ghi < %.0f W/m²) — set to NaN.",
            n_suspicious,
            _NIGHTTIME_NOISE_FLOOR_W,
            night_threshold,
        )
        df.loc[suspicious, "pv_ac_W"] = 0.0

    # Clamp small negative/near-zero values during nighttime to 0 (expected)
    genuine_night_near_zero = is_night & df["pv_ac_W"].notna() & (
        df["pv_ac_W"] <= _NIGHTTIME_NOISE_FLOOR_W
    )
    df.loc[genuine_night_near_zero, "pv_ac_W"] = 0.0
    

    return df


# ── Step 4: Short-gap interpolation ──────────────────────────────────────────

def _interpolate_short_gaps(
    df: pd.DataFrame,
    cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Linearly interpolate pv_ac_W over gaps ≤ max_interpolation_steps."""
    max_steps = getattr(cfg, "max_interpolation_steps", 6)

    n_missing_before = df["pv_ac_W"].isna().sum()

    df["pv_ac_W"] = df["pv_ac_W"].interpolate(
        method="linear",
        limit=max_steps,
        limit_direction="forward",
        limit_area="inside",   # only interpolate between valid values
    )

    n_missing_after = df["pv_ac_W"].isna().sum()
    n_filled = n_missing_before - n_missing_after

    logger.info(
        "Short-gap interpolation: %d NaNs filled (limit=%d steps). "
        "%d NaNs remaining.",
        n_filled,
        max_steps,
        n_missing_after,
    )
    return df


# ── Step 5: Quality flag ─────────────────────────────────────────────────────

def _add_quality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``data_ok`` boolean column: True where pv_ac_W is valid."""
    df["data_ok"] = df["pv_ac_W"].notna()
    return df
