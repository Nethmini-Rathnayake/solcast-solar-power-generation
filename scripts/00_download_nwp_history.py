"""
scripts/00_download_nwp_history.py
------------------------------------
One-time download: fetch Open-Meteo historical NWP data for the full
training period and save to data/external/nwp_history.parquet.

Run this ONCE before the feature-engineering pipeline (step 02).
The parquet file is then used by scripts/02_build_features.py to build
NWP-aligned forecast features instead of the oracle-shifted actuals.

Why run this separately
-----------------------
The download takes ~30–60 seconds for a 4-year period (one API call per
year).  Keeping it as a separate script means steps 01–06 can be re-run
without hitting the API again.

Date range
----------
The script infers the date range automatically from the Solcast external
data files.  You can override with --start and --end if needed.

Run from the project root:
    python scripts/00_download_nwp_history.py

Output
------
    data/external/nwp_history.parquet
    ~35,000–40,000 rows × 6 columns (hourly, tz-aware Asia/Colombo)
    Columns: ghi, dni, dhi, cloud_opacity, air_temp, relative_humidity
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.solcast import load_solcast
from src.data.nwp_historical import NWPHistoricalClient

logger = get_logger(__name__)

# Extra buffer days on each side so the NWP history covers the full
# lag + horizon window at both ends of the feature matrix.
_DATE_BUFFER_DAYS = 30


def main() -> None:
    args = _parse_args()
    logger.info("=== Step 0: Download NWP Historical Data ===")

    cfg  = load_config()
    pipe = cfg.pipeline
    site = cfg.site

    # ── Determine date range ──────────────────────────────────────────────────
    if args.start and args.end:
        start_str = args.start
        end_str   = args.end
        logger.info("Using user-supplied date range: %s → %s", start_str, end_str)
    else:
        logger.info("Inferring date range from Solcast data …")
        start_str, end_str = _infer_date_range(pipe, _DATE_BUFFER_DAYS)
        logger.info("Inferred range: %s → %s", start_str, end_str)

    # ── Download ──────────────────────────────────────────────────────────────
    client = NWPHistoricalClient(timezone=site.timezone)
    df = client.fetch(
        lat=site.latitude,
        lon=site.longitude,
        start=start_str,
        end=end_str,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(pipe.paths.nwp_history)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    logger.info(
        "NWP history saved: %s | %d rows | %d columns",
        out_path,
        len(df),
        len(df.columns),
    )

    # ── Quick summary ─────────────────────────────────────────────────────────
    logger.info(
        "Coverage: %s → %s",
        df.index[0].strftime("%Y-%m-%d %H:%M"),
        df.index[-1].strftime("%Y-%m-%d %H:%M"),
    )
    logger.info(
        "Mean GHI: %.1f W/m²  |  Mean cloud_opacity: %.1f%%  |  Mean air_temp: %.1f°C",
        df["ghi"].mean(),
        df["cloud_opacity"].mean(),
        df["air_temp"].mean(),
    )
    logger.info("=== Step 0 complete ===")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_date_range(pipe, buffer_days: int) -> tuple[str, str]:
    """Infer start/end dates from the Solcast external data files."""
    solcast_dir = Path(pipe.paths.external_solcast_dir)
    csv_files   = sorted(solcast_dir.glob("solcast_weather_data_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No Solcast CSV files found in {solcast_dir}. "
            "Cannot infer date range.  Use --start and --end instead."
        )

    # Peek at first and last file to get the overall date range
    first_df = pd.read_csv(csv_files[0],  usecols=["period_end"], nrows=1)
    last_df  = pd.read_csv(csv_files[-1], usecols=["period_end"])

    t_start = (
        pd.to_datetime(first_df["period_end"].iloc[0], utc=True)
        - pd.Timedelta(days=buffer_days)
    )
    t_end = (
        pd.to_datetime(last_df["period_end"].iloc[-1], utc=True)
        + pd.Timedelta(days=buffer_days)
    )

    # Open-Meteo archive has a ~5-day lag; cap end at yesterday
    yesterday = pd.Timestamp.utcnow() - pd.Timedelta(days=5)
    t_end = min(t_end, yesterday)

    return t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Open-Meteo historical NWP data for the training period."
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: inferred from Solcast files)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: inferred from Solcast files)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
