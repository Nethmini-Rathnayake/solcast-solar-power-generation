"""
src/data/nwp_forecast.py
-------------------------
Open-Meteo NWP forecast client.

Fetches free, no-key hourly NWP weather forecasts from the Open-Meteo API
and returns them as a tz-aware (Asia/Colombo) DataFrame with the same column
names used throughout the feature pipeline.

Why Open-Meteo
--------------
- Completely free, no API key or registration required.
- Covers the full globe including Sri Lanka.
- Returns hourly data up to 16 days ahead.
- Backed by multiple NWP models (GFS, ECMWF IFS, ICON, etc.); the default
  ensemble model (`best_match`) blends whichever is most skilful for the
  requested location.

Column mapping
--------------
  Open-Meteo variable        → our column name  (units preserved)
  ─────────────────────────────────────────────────────────────
  shortwave_radiation         → ghi              (W/m²)
  direct_radiation            → dni              (W/m²)
  diffuse_radiation           → dhi              (W/m²)
  cloud_cover                 → cloud_opacity    (%)
  temperature_2m              → air_temp         (°C)
  relative_humidity_2m        → relative_humidity (%)

``cloud_cover`` (0–100 %) and Solcast's ``cloud_opacity`` (0–100 %) are not
identical (cloud_cover is fractional cloud area; cloud_opacity is
transmittance-based) but they are strongly correlated and are used
identically by the model.

Training vs inference
---------------------
During training the oracle approach is used: actual future Solcast values
are shifted back in time as perfect forecasts.  At inference the NWP values
replace them.  The distribution mismatch (perfect oracle vs NWP error) means
the model conservatively under-uses the forecast signal — this is acceptable
and still far better than no future information at all.

Caching
-------
Responses are cached to a local JSON file with a configurable TTL (default
30 min).  Re-use the cache when running the live script repeatedly without
burning bandwidth.

Usage
-----
    from src.data.nwp_forecast import NWPForecastClient

    client = NWPForecastClient()
    df = client.fetch(lat=6.7912, lon=79.9005, hours=24)
    # Returns DataFrame indexed by hour (Asia/Colombo, tz-aware), columns:
    # ghi, dni, dhi, cloud_opacity, air_temp, relative_humidity
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── API config ────────────────────────────────────────────────────────────────
_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Variables to request from Open-Meteo
_OPENMETEO_VARS = [
    "shortwave_radiation",    # GHI  (W/m²)
    "direct_radiation",       # DNI  (W/m²)
    "diffuse_radiation",      # DHI  (W/m²)
    "cloud_cover",            # cloud opacity proxy  (%)
    "temperature_2m",         # air temperature      (°C)
    "relative_humidity_2m",   # relative humidity    (%)
]

# Rename Open-Meteo columns → pipeline column names
_RENAME = {
    "shortwave_radiation":  "ghi",
    "direct_radiation":     "dni",
    "diffuse_radiation":    "dhi",
    "cloud_cover":          "cloud_opacity",
    "temperature_2m":       "air_temp",
    "relative_humidity_2m": "relative_humidity",
}

# Final columns in return order
FORECAST_COLS = ["ghi", "dni", "dhi", "cloud_opacity", "air_temp", "relative_humidity"]

_DEFAULT_CACHE_TTL_S = 1800   # 30 minutes


class NWPForecastClient:
    """Open-Meteo NWP forecast client.  No API key required.

    Parameters
    ----------
    cache_dir:
        Directory for caching responses.  No caching if None.
    cache_ttl_s:
        Seconds before a cached response is considered stale.
    timezone:
        IANA timezone for returned index.
    model:
        Open-Meteo NWP model identifier.  ``"best_match"`` (default) lets
        Open-Meteo select the highest-skill model for the location.
        Other options: ``"gfs_seamless"``, ``"ecmwf_ifs025"``,
        ``"icon_seamless"``.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        cache_ttl_s: int = _DEFAULT_CACHE_TTL_S,
        timezone: str = "Asia/Colombo",
        model: str = "best_match",
    ) -> None:
        self._cache_dir   = Path(cache_dir) if cache_dir else None
        self._cache_ttl_s = cache_ttl_s
        self._tz          = timezone
        self._model       = model

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        lat: float,
        lon: float,
        hours: int = 24,
    ) -> pd.DataFrame:
        """Fetch hourly NWP forecast.

        Parameters
        ----------
        lat, lon:
            Site coordinates (decimal degrees).
        hours:
            Number of forecast hours to return (max 384 = 16 days).

        Returns
        -------
        pd.DataFrame
            Indexed by ``time`` (tz-aware, Asia/Colombo, hourly).
            Columns: ``ghi``, ``dni``, ``dhi``, ``cloud_opacity``,
            ``air_temp``, ``relative_humidity``.
            Only the first ``hours`` rows are returned.
        """
        cache_key = f"nwp_{lat:.4f}_{lon:.4f}_{hours}_{self._model}.json"

        if self._cache_dir is not None:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached.head(hours)

        raw = self._call_api(lat, lon, hours)
        df  = self._parse(raw)

        if self._cache_dir is not None:
            self._save_cache(cache_key, raw)

        df = df.head(hours)
        logger.info(
            "NWP forecast fetched: %d rows  (%s → %s)",
            len(df),
            df.index[0].strftime("%Y-%m-%d %H:%M"),
            df.index[-1].strftime("%Y-%m-%d %H:%M"),
        )
        return df

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _call_api(self, lat: float, lon: float, hours: int) -> dict:
        # Open-Meteo uses forecast_days (ceil hours/24, max 16)
        forecast_days = min(16, max(1, -(-hours // 24)))   # ceil division

        params = {
            "latitude":     lat,
            "longitude":    lon,
            "hourly":       ",".join(_OPENMETEO_VARS),
            "timezone":     self._tz,
            "forecast_days": forecast_days,
            "models":       self._model,
        }

        try:
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            raise RuntimeError(
                f"Open-Meteo API returned HTTP {status}.  "
                "Check coordinates and network connectivity."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Open-Meteo API request failed: {exc}") from exc

        return resp.json()

    def _parse(self, data: dict) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            raise ValueError("Open-Meteo returned an empty or malformed response.")

        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])

        # Open-Meteo honours the `timezone` param → times are already local
        # but returned as naive strings; localise to Asia/Colombo
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(self._tz)
        else:
            df["time"] = df["time"].dt.tz_convert(self._tz)

        df = df.set_index("time").sort_index()
        df = df.rename(columns=_RENAME)

        # Ensure all output columns present; fill missing with 0
        for col in FORECAST_COLS:
            if col not in df.columns:
                logger.warning("Open-Meteo missing column '%s'. Filled with 0.", col)
                df[col] = 0.0

        # Clip irradiance to non-negative (can be slightly negative at night)
        for col in ("ghi", "dni", "dhi"):
            df[col] = df[col].clip(lower=0.0)

        return df[FORECAST_COLS]

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        path = self._cache_dir / key
        if not path.exists():
            return None
        age_s = time.time() - path.stat().st_mtime
        if age_s > self._cache_ttl_s:
            logger.debug("NWP cache stale (%.0f s old); will re-fetch.", age_s)
            return None
        logger.info("Using cached NWP forecast (%.0f s old).", age_s)
        return self._parse(json.loads(path.read_text()))

    def _save_cache(self, key: str, data: dict) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / key).write_text(json.dumps(data))


# ── Convenience factory ───────────────────────────────────────────────────────

def make_nwp_client_from_cfg(cfg: SimpleNamespace) -> NWPForecastClient:
    """Build an ``NWPForecastClient`` from the pipeline config.

    Reads ``cfg.pipeline.forecast.cache_dir``,
    ``cfg.pipeline.forecast.cache_ttl_s``, and
    ``cfg.pipeline.forecast.nwp_model``.
    """
    forecast_cfg = getattr(cfg.pipeline, "forecast", None)
    cache_dir    = getattr(forecast_cfg, "cache_dir",   "data/external/forecast_cache")
    cache_ttl_s  = getattr(forecast_cfg, "cache_ttl_s", 1800)
    model        = getattr(forecast_cfg, "nwp_model",   "best_match")
    return NWPForecastClient(
        cache_dir=cache_dir,
        cache_ttl_s=int(cache_ttl_s),
        model=model,
    )
