"""
src/data/local_pv.py
---------------------
Loader for the local measured PV plant data from the University of Moratuwa
Smartgrid Lab.

Data characteristics
--------------------
- File:       data/raw/Smartgrid lab solar PV data.csv
- Resolution: 5-minute intervals, period_end convention
- Coverage:   2022-04-01 00:05:00 → 2023-04-01 00:00:00 (local Sri Lanka time)
- Target col: "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"
              renamed to ``pv_ac_W`` for cross-repo compatibility

Per-system note
---------------
The plant has three sub-arrays (System 1, 2, 3).  ``Power Total`` is read at
the AC bus and already aggregates all three systems.  Individual inverter
columns contain a hardware overflow sentinel (32,767,000 W = int16_max × 1000)
that must be cleaned before use — this is handled in ``src/preprocessing/``.

Output
------
DataFrame with:
  - DatetimeIndex (tz-aware, Asia/Colombo, name="datetime_local")
  - ``pv_ac_W``  : total AC output in Watts (float)
  - per-inverter power columns retained for diagnostics (prefixed ``raw_``)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Column names in the raw CSV ───────────────────────────────────────────────
_DATETIME_COL = "datetime"
_TARGET_RAW = "PV Hybrid Plant - PV SYSTEM - PV - Power Total (W)"

# Per-inverter power columns retained for diagnostics
_INVERTER_COLS = {
    "PV Hybrid Plant - PV SYSTEM - PV - PV System 1 - PV Inverter 1.1 Power (W)": "inv_1_1_W",
    "PV Hybrid Plant - PV SYSTEM - PV - PV System 1 - PV Inverter 1.2 Power (W)": "inv_1_2_W",
    "PV Hybrid Plant - PV SYSTEM - PV - PV System 1 - PV Inverter 1.3 Power (W)": "inv_1_3_W",
    "PV Hybrid Plant - PV SYSTEM - PV - PV System 2 - PV Inverter 2.1 Power (W)": "inv_2_1_W",
    "PV Hybrid Plant - PV SYSTEM - PV - PV System 2 - PV Inverter 2.2 Power (W)": "inv_2_2_W",
    "PV Hybrid Plant - PV SYSTEM - PV - PV System 3 - PV Inverter 3.1 Power (W)": "inv_3_1_W",
    "PV Hybrid Plant - PV SYSTEM - PV - PV System 3 - PV Inverter 3.2 Power (W)": "inv_3_2_W",
    "PV Hybrid Plant - PV SYSTEM - PV - PV System 3 - PV Inverter 3.3 Power (W)": "inv_3_3_W",
}

_TIMEZONE = "Asia/Colombo"


def load_local_pv(file_path: str | Path) -> pd.DataFrame:
    """Load and minimally parse the raw local PV plant CSV.

    Parameters
    ----------
    file_path:
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (tz-aware, Asia/Colombo).
        Columns: ``pv_ac_W`` + per-inverter diagnostics.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``file_path``.
    KeyError
        If expected columns are missing from the CSV.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Local PV file not found: {file_path}")

    logger.info("Loading local PV data from: %s", file_path)

    # Read only the columns we need to keep memory low
    usecols = [_DATETIME_COL, _TARGET_RAW] + list(_INVERTER_COLS.keys())
    df = pd.read_csv(file_path, usecols=usecols, low_memory=False)

    # ── Timestamp handling ────────────────────────────────────────────────────
    # Raw timestamps are naive local time (Asia/Colombo, UTC+5:30).
    # Localise (not convert) to make them tz-aware.
    df[_DATETIME_COL] = pd.to_datetime(df[_DATETIME_COL])
    df[_DATETIME_COL] = df[_DATETIME_COL].dt.tz_localize(_TIMEZONE)
    df = df.set_index(_DATETIME_COL)
    df.index.name = "datetime_local"

    # ── Rename columns ────────────────────────────────────────────────────────
    df = df.rename(columns={_TARGET_RAW: "pv_ac_W"})
    df = df.rename(columns=_INVERTER_COLS)

    # ── Cast to float ─────────────────────────────────────────────────────────
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Sort and validate ─────────────────────────────────────────────────────
    df = df.sort_index()
    _validate(df)

    logger.info(
        "Local PV data loaded: %d rows | %s → %s",
        len(df),
        df.index[0],
        df.index[-1],
    )
    return df


def _validate(df: pd.DataFrame) -> None:
    """Basic sanity checks on the loaded DataFrame."""
    if "pv_ac_W" not in df.columns:
        raise KeyError("Target column 'pv_ac_W' missing after rename.")

    n_missing = df["pv_ac_W"].isna().sum()
    if n_missing > 0:
        logger.warning(
            "%d missing values in pv_ac_W (%.1f%% of rows) — will be handled in cleaning.",
            n_missing,
            100.0 * n_missing / len(df),
        )

    # Check for unexpected duplicates
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        logger.warning(
            "%d duplicate timestamps found in local PV data.", n_dup
        )
