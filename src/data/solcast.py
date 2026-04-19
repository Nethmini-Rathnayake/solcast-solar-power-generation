"""
src/data/solcast.py
--------------------
Loader for the Solcast historical irradiance and meteorological data.

Data characteristics
--------------------
- Files:       data/external/solcast_weather_data_YYYY.csv (4 files)
- Resolution:  5-minute intervals (period = PT5M), period_end convention
- Timestamp:   UTC-aware (``+00:00`` suffix in period_end column)
- Coverage:    2020-01-01 → 2024-02-29 UTC
- Variables:   ghi, dni, dhi, gti, clearsky_*, cloud_opacity, air_temp,
               relative_humidity, surface_pressure, dewpoint_temp, albedo,
               weather_type, azimuth, pm10, pm2.5

Output
------
DataFrame with:
  - DatetimeIndex (tz-aware, Asia/Colombo, name="datetime_local")
    converted from the UTC period_end timestamps
  - All Solcast variables as columns
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Column names in the raw Solcast CSV ──────────────────────────────────────
_PERIOD_END_COL = "period_end"
_PERIOD_COL = "period"
_EXPECTED_PERIOD = "PT5M"
_TARGET_TIMEZONE = "Asia/Colombo"

# Canonical output column list (subset we carry through the pipeline)
_KEEP_COLS = [
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
    "weather_type",
    "min_air_temp",
    "max_air_temp",
    "pm10",
    "pm2.5",
    "azimuth",
]


def load_solcast(solcast_dir: str | Path) -> pd.DataFrame:
    """Load, concatenate, and parse all Solcast CSV files in a directory.

    All files matching the pattern ``solcast_weather_data_*.csv`` are loaded
    and combined.  UTC timestamps are converted to Asia/Colombo local time.

    Parameters
    ----------
    solcast_dir:
        Directory containing the Solcast CSV files.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (tz-aware, Asia/Colombo).
        Columns: all variables listed in ``_KEEP_COLS`` (where present).

    Raises
    ------
    FileNotFoundError
        If no matching CSV files are found in ``solcast_dir``.
    """
    solcast_dir = Path(solcast_dir)
    if not solcast_dir.is_dir():
        raise FileNotFoundError(f"Solcast directory not found: {solcast_dir}")

    csv_files = sorted(solcast_dir.glob("solcast_weather_data_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No solcast_weather_data_*.csv files found in: {solcast_dir}"
        )

    logger.info("Found %d Solcast file(s): %s", len(csv_files), [f.name for f in csv_files])

    frames: list[pd.DataFrame] = []
    for fpath in csv_files:
        df_part = _load_single_file(fpath)
        frames.append(df_part)
        logger.debug("  Loaded %s: %d rows", fpath.name, len(df_part))

    df = pd.concat(frames, axis=0)

    # ── Deduplicate (files may share rows at year boundaries) ─────────────────
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_dup = n_before - len(df)
    if n_dup > 0:
        logger.info("Removed %d duplicate Solcast timestamps.", n_dup)

    df = df.sort_index()

    # ── Retain only known columns ─────────────────────────────────────────────
    available = [c for c in _KEEP_COLS if c in df.columns]
    missing = [c for c in _KEEP_COLS if c not in df.columns]
    if missing:
        logger.warning("Solcast columns not found (skipped): %s", missing)
    df = df[available]

    logger.info(
        "Solcast data loaded: %d rows | %s → %s",
        len(df),
        df.index[0],
        df.index[-1],
    )
    return df


def _load_single_file(fpath: Path) -> pd.DataFrame:
    """Load and parse a single Solcast CSV file.

    Parameters
    ----------
    fpath:
        Path to the Solcast CSV.

    Returns
    -------
    pd.DataFrame with a tz-aware (Asia/Colombo) DatetimeIndex.
    """
    df = pd.read_csv(fpath, low_memory=False)

    if _PERIOD_END_COL not in df.columns:
        raise KeyError(
            f"Expected '{_PERIOD_END_COL}' column in {fpath.name}. "
            f"Found: {list(df.columns)}"
        )

    # Validate all rows are 5-min intervals
    if _PERIOD_COL in df.columns:
        unexpected = df[df[_PERIOD_COL] != _EXPECTED_PERIOD]
        if len(unexpected) > 0:
            logger.warning(
                "%s: %d rows with unexpected period (not %s). Rows dropped.",
                fpath.name,
                len(unexpected),
                _EXPECTED_PERIOD,
            )
            df = df[df[_PERIOD_COL] == _EXPECTED_PERIOD]
        df = df.drop(columns=[_PERIOD_COL])

    # ── Parse UTC timestamp and convert to local time ─────────────────────────
    # Solcast period_end is UTC-aware (e.g. "2022-01-01 00:05:00+00:00")
    df[_PERIOD_END_COL] = pd.to_datetime(df[_PERIOD_END_COL], utc=True)
    df[_PERIOD_END_COL] = df[_PERIOD_END_COL].dt.tz_convert(_TARGET_TIMEZONE)
    df = df.set_index(_PERIOD_END_COL)
    df.index.name = "datetime_local"

    # Cast numeric columns
    for col in df.columns:
        if col != "weather_type":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
