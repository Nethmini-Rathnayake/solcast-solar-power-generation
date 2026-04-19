"""
src/data/alignment.py
----------------------
Timestamp alignment of local PV data and Solcast weather data.

Both datasets are at 5-minute resolution and use the period_end convention.
After timezone conversion in their respective loaders, both have a
tz-aware (Asia/Colombo) DatetimeIndex named ``datetime_local``.

This module:
  1. Validates that the expected overlap window exists (April 2022 – April 2023).
  2. Performs an inner join on the 5-minute timestamp index.
  3. Reports coverage statistics and warns about large gaps.

Design note
-----------
An inner join is used so that every row in the aligned dataset has both local
PV measurements and Solcast weather observations.  Rows that appear in only
one source (e.g. Solcast data outside the local measurement window) are
discarded here; they can later be used for standalone pvlib simulation if
needed.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Expected overlap window (local time) ─────────────────────────────────────
_EXPECTED_START = "2022-04-01"
_EXPECTED_END = "2023-04-01"

# Warn if more than this fraction of expected rows are missing after alignment
_MISSING_ROW_WARN_THRESHOLD = 0.05   # 5 %


def align(
    local_pv: pd.DataFrame,
    solcast: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join local PV and Solcast DataFrames on the 5-minute timestamp.

    Parameters
    ----------
    local_pv:
        Output of ``load_local_pv`` — tz-aware (Asia/Colombo) DatetimeIndex,
        contains ``pv_ac_W`` and per-inverter columns.
    solcast:
        Output of ``load_solcast`` — tz-aware (Asia/Colombo) DatetimeIndex,
        contains irradiance and meteorological columns.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns from both sources, indexed by
        ``datetime_local``.  Only rows present in both sources are kept.

    Raises
    ------
    ValueError
        If the resulting aligned DataFrame is empty (no timestamp overlap).
    """
    _check_index_compatibility(local_pv, solcast)

    logger.info(
        "Aligning datasets: local PV (%d rows) ↔ Solcast (%d rows)",
        len(local_pv),
        len(solcast),
    )

    # Inner join preserves only timestamps present in both sources
    aligned = local_pv.join(solcast, how="inner", rsuffix="_solcast")

    if aligned.empty:
        raise ValueError(
            "Aligned DataFrame is empty — no timestamp overlap between "
            "local PV data and Solcast data.  Check timezone conversion."
        )

    _log_coverage_report(local_pv, solcast, aligned)
    return aligned


def _check_index_compatibility(
    local_pv: pd.DataFrame,
    solcast: pd.DataFrame,
) -> None:
    """Verify both DataFrames have tz-aware DatetimeIndexes."""
    for name, df in (("local_pv", local_pv), ("solcast", solcast)):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"Expected DatetimeIndex for '{name}', got {type(df.index).__name__}."
            )
        if df.index.tz is None:
            raise ValueError(
                f"Index of '{name}' is timezone-naive. "
                "Run the respective loader to attach Asia/Colombo timezone."
            )


def _log_coverage_report(
    local_pv: pd.DataFrame,
    solcast: pd.DataFrame,
    aligned: pd.DataFrame,
) -> None:
    """Log a coverage summary after alignment."""
    # Expected 5-min rows in the overlap window
    expected_start = pd.Timestamp(_EXPECTED_START, tz="Asia/Colombo")
    expected_end = pd.Timestamp(_EXPECTED_END, tz="Asia/Colombo")

    # Number of 5-min steps in one year
    expected_rows = int(
        (expected_end - expected_start).total_seconds() / 300
    )

    actual_rows = len(aligned)
    missing_fraction = max(0.0, 1.0 - actual_rows / expected_rows)

    logger.info(
        "Alignment complete: %d rows retained | span %s → %s",
        actual_rows,
        aligned.index[0],
        aligned.index[-1],
    )
    logger.info(
        "Coverage: %.1f%% of expected annual window "
        "(%d / ~%d rows).",
        100.0 * (1 - missing_fraction),
        actual_rows,
        expected_rows,
    )

    if missing_fraction > _MISSING_ROW_WARN_THRESHOLD:
        logger.warning(
            "%.1f%% of expected rows are missing after alignment. "
            "Investigate timestamp offsets or data gaps.",
            100.0 * missing_fraction,
        )

    # Log any columns that ended up duplicated (rsuffix collision)
    dup_cols = [c for c in aligned.columns if c.endswith("_solcast")]
    if dup_cols:
        logger.debug(
            "Duplicate column suffixes from join (safe to ignore if not "
            "expected to be unique): %s",
            dup_cols,
        )
