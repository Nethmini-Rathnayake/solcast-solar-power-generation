"""
src/data/nwp_historical.py
---------------------------
Open-Meteo historical archive client.

Downloads past hourly NWP weather data via the Open-Meteo Archive API
(backed by ERA5 reanalysis) for any date range.  The returned DataFrame
uses the same column names as ``nwp_forecast.py`` so the two clients are
drop-in substitutes.

Why this matters for training
------------------------------
The oracle approach (shifting actual Solcast data back) gives the model
*perfect* future weather during training.  At inference the model receives
Open-Meteo NWP output instead.  This train/inference distribution mismatch
means the model under-exploits the forecast signal.

Replacing the oracle with Open-Meteo historical data (ERA5) trains the
model on the same data source it will see at inference — so the errors,
biases, and seasonal patterns of the NWP system are learned rather than
ignored.

ERA5 vs live NWP
----------------
ERA5 is a reanalysis (best-fit retrospective model run), not a real-time
forecast.  For tropical sites like Sri Lanka, ERA5 closely tracks the live
Open-Meteo ``best_match`` output for short-range horizons (1–24 h), so it
is a realistic training proxy.

API endpoint
------------
    GET https://archive-api.open-meteo.com/v1/archive
    Parameters: latitude, longitude, start_date, end_date,
                hourly (comma-separated variables), timezone

No API key required.  Rate limit: 10,000 requests/day (free).

Usage
-----
    from src.data.nwp_historical import NWPHistoricalClient

    client = NWPHistoricalClient()
    df = client.fetch(
        lat=6.7912, lon=79.9005,
        start="2020-01-01", end="2024-03-31",
    )
    df.to_parquet("data/external/nwp_history.parquet")
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── API config ────────────────────────────────────────────────────────────────
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_HOURLY_VARS = [
    "shortwave_radiation",    # GHI  (W/m²)
    "direct_radiation",       # DNI  (W/m²)
    "diffuse_radiation",      # DHI  (W/m²)
    "cloud_cover",            # cloud opacity proxy  (%)
    "temperature_2m",         # air temperature      (°C)
    "relative_humidity_2m",   # relative humidity    (%)
]

# Same rename map as nwp_forecast.py — columns are interchangeable
_RENAME = {
    "shortwave_radiation":   "ghi",
    "direct_radiation":      "dni",
    "diffuse_radiation":     "dhi",
    "cloud_cover":           "cloud_opacity",
    "temperature_2m":        "air_temp",
    "relative_humidity_2m":  "relative_humidity",
}

NWP_HISTORY_COLS = ["ghi", "dni", "dhi", "cloud_opacity", "air_temp", "relative_humidity"]

# Maximum rows per API request (~1 year of hourly data = 8,760 rows)
# Open-Meteo accepts multi-year requests; we chunk by year to be polite.
_CHUNK_YEARS = 1


class NWPHistoricalClient:
    """Open-Meteo archive client for multi-year historical NWP data.

    Parameters
    ----------
    timezone:
        IANA timezone for the returned index.  Must match the site timezone.
    """

    def __init__(self, timezone: str = "Asia/Colombo") -> None:
        self._tz = timezone

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Download hourly historical NWP data for a date range.

        Parameters
        ----------
        lat, lon:
            Site coordinates (decimal degrees).
        start, end:
            Date strings in ``YYYY-MM-DD`` format (inclusive).

        Returns
        -------
        pd.DataFrame
            Indexed by ``time`` (tz-aware, hourly).
            Columns: ``ghi``, ``dni``, ``dhi``, ``cloud_opacity``,
            ``air_temp``, ``relative_humidity``.
        """
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)

        # Chunk by year so each request is manageable
        chunks: list[pd.DataFrame] = []
        chunk_start = start_dt
        while chunk_start <= end_dt:
            chunk_end = min(
                chunk_start + pd.DateOffset(years=_CHUNK_YEARS) - pd.Timedelta(days=1),
                end_dt,
            )
            logger.info(
                "Fetching NWP history: %s → %s …",
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            raw = self._call_api(
                lat, lon,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            chunks.append(self._parse(raw))
            chunk_start = chunk_end + pd.Timedelta(days=1)

        df = pd.concat(chunks).sort_index()
        # Drop any duplicate timestamps from chunk boundary overlap
        df = df[~df.index.duplicated(keep="first")]

        logger.info(
            "NWP history downloaded: %d hourly rows  (%s → %s)",
            len(df),
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )
        return df

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _call_api(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
    ) -> dict:
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start,
            "end_date":   end,
            "hourly":     ",".join(_HOURLY_VARS),
            "timezone":   self._tz,
        }
        try:
            resp = requests.get(_ARCHIVE_URL, params=params, timeout=60)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            raise RuntimeError(
                f"Open-Meteo archive API returned HTTP {status}.  "
                f"Check date range ({start} → {end}) and coordinates."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Open-Meteo archive request failed: {exc}") from exc

        return resp.json()

    def _parse(self, data: dict) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            raise ValueError("Open-Meteo archive returned empty or malformed response.")

        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])

        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(self._tz)
        else:
            df["time"] = df["time"].dt.tz_convert(self._tz)

        df = df.set_index("time").sort_index()
        df = df.rename(columns=_RENAME)

        for col in NWP_HISTORY_COLS:
            if col not in df.columns:
                df[col] = 0.0

        for col in ("ghi", "dni", "dhi"):
            df[col] = df[col].clip(lower=0.0)

        return df[NWP_HISTORY_COLS]


# ── Convenience factory ───────────────────────────────────────────────────────

def make_nwp_historical_client_from_cfg(cfg: SimpleNamespace) -> NWPHistoricalClient:
    return NWPHistoricalClient(timezone=cfg.site.timezone)


def load_nwp_history(cfg: SimpleNamespace) -> pd.DataFrame:
    """Load the cached NWP history parquet file.

    Parameters
    ----------
    cfg:
        Full ``PipelineConfig``.

    Returns
    -------
    pd.DataFrame or None if the file does not exist.
    """
    path = Path(cfg.pipeline.paths.nwp_history)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    logger.info("NWP history loaded: %d rows from %s", len(df), path)
    return df
