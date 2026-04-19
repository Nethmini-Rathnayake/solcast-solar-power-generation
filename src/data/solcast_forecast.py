"""
src/data/solcast_forecast.py
-----------------------------
Solcast Forecast API client.

Fetches 24h-ahead hourly solar irradiance and weather forecasts for the site
and returns them as a tz-aware (Asia/Colombo) DataFrame that can be fed
directly into the forecast feature pipeline.

API notes
---------
Solcast provides two endpoint families:

  Free / Hobbyist (world_radiation):
    GET /world_radiation/forecasts?latitude=...&longitude=...&api_key=...
    Up to 10 calls/day.  Returns 14-day ahead at ≤ hourly resolution.

  Commercial (data API with resource_id):
    GET /data/forecast/radiation_and_weather?resource_id=...&api_key=...
    Higher rate limits.  The resource_id is the UUID assigned to your site
    in the Solcast dashboard.

Configure via environment variables or configs/pipeline.yaml:
    SOLCAST_API_KEY   — required
    SOLCAST_RESOURCE_ID — optional; enables the commercial endpoint

Caching
-------
Responses are optionally cached to a local JSON file with a configurable TTL
(default 30 min).  This prevents burning API quota during development and
lets the live prediction script re-use the last fetch when running offline.

Usage
-----
    from src.data.solcast_forecast import SolcastForecastClient

    client = SolcastForecastClient(api_key="<key>")
    df = client.fetch(lat=6.7912, lon=79.9005, hours=24)
    # Returns DataFrame indexed by period_end (Asia/Colombo), columns:
    # ghi, dni, dhi, cloud_opacity, air_temp, relative_humidity
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Endpoint constants ────────────────────────────────────────────────────────
_BASE_URL            = "https://api.solcast.com.au"
_WORLD_RADIATION_EP  = "/world_radiation/forecasts"
_COMMERCIAL_EP       = "/data/forecast/radiation_and_weather"

# Columns to retain from the API response
FORECAST_API_COLS = [
    "ghi", "dni", "dhi",
    "cloud_opacity",
    "air_temp",
    "relative_humidity",
]

# Default cache TTL (seconds); re-use cached response if younger than this
_DEFAULT_CACHE_TTL_S = 1800   # 30 minutes


class SolcastForecastClient:
    """HTTP client for the Solcast forecast API.

    Parameters
    ----------
    api_key:
        Solcast API key.  Falls back to ``SOLCAST_API_KEY`` env var if None.
    resource_id:
        Optional site resource UUID (commercial plan).  Falls back to
        ``SOLCAST_RESOURCE_ID`` env var.  When set, the commercial endpoint
        is used instead of the free world_radiation endpoint.
    cache_dir:
        Directory for caching API responses.  No caching if None.
    cache_ttl_s:
        Seconds before a cached response is considered stale.
    timezone:
        Target timezone for the returned index.
    """

    def __init__(
        self,
        api_key: str | None = None,
        resource_id: str | None = None,
        cache_dir: str | Path | None = None,
        cache_ttl_s: int = _DEFAULT_CACHE_TTL_S,
        timezone: str = "Asia/Colombo",
    ) -> None:
        self._api_key = api_key or os.environ.get("SOLCAST_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Solcast API key is required.  "
                "Pass api_key= or set the SOLCAST_API_KEY environment variable."
            )
        self._resource_id = resource_id or os.environ.get("SOLCAST_RESOURCE_ID", "")
        self._cache_dir   = Path(cache_dir) if cache_dir else None
        self._cache_ttl_s = cache_ttl_s
        self._tz          = timezone

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        lat: float,
        lon: float,
        hours: int = 24,
        period: str = "PT60M",
    ) -> pd.DataFrame:
        """Fetch the Solcast hourly forecast.

        Parameters
        ----------
        lat, lon:
            Site coordinates (decimal degrees).
        hours:
            Forecast horizon in hours (max 336 for commercial, 168 free).
        period:
            ISO-8601 period string: ``PT60M`` (hourly), ``PT30M``, ``PT5M``.

        Returns
        -------
        pd.DataFrame
            Indexed by ``period_end`` (tz-aware, Asia/Colombo).
            Columns: ``ghi``, ``dni``, ``dhi``, ``cloud_opacity``,
            ``air_temp``, ``relative_humidity``.
            Only the first ``hours`` rows are returned.
        """
        cache_key = f"forecast_{lat:.4f}_{lon:.4f}_{hours}_{period}.json"

        # ── Try cache first ───────────────────────────────────────────────────
        if self._cache_dir is not None:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # ── Call API ─────────────────────────────────────────────────────────
        raw = self._call_api(lat, lon, hours, period)
        df  = self._parse(raw)

        # ── Store in cache ────────────────────────────────────────────────────
        if self._cache_dir is not None:
            self._save_cache(cache_key, raw)

        logger.info(
            "Solcast forecast fetched: %d rows, %s → %s (Asia/Colombo)",
            len(df),
            df.index[0].strftime("%Y-%m-%d %H:%M"),
            df.index[-1].strftime("%Y-%m-%d %H:%M"),
        )
        return df

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _call_api(
        self,
        lat: float,
        lon: float,
        hours: int,
        period: str,
    ) -> dict:
        """Make the HTTP GET request and return raw JSON dict."""
        if self._resource_id:
            url    = _BASE_URL + _COMMERCIAL_EP
            params = {
                "resource_id": self._resource_id,
                "hours":       hours,
                "period":      period,
                "api_key":     self._api_key,
            }
        else:
            url    = _BASE_URL + _WORLD_RADIATION_EP
            params = {
                "latitude":  lat,
                "longitude": lon,
                "hours":     hours,
                "period":    period,
                "api_key":   self._api_key,
            }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            raise RuntimeError(
                f"Solcast API returned HTTP {status}.  "
                "Check your API key, quota, and coordinates."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Solcast API request failed: {exc}"
            ) from exc

        return resp.json()

    def _parse(self, data: dict) -> pd.DataFrame:
        """Convert raw API JSON to a clean DataFrame."""
        # Both endpoints return a list under 'forecasts'
        records = data.get("forecasts", [])
        if not records:
            raise ValueError(
                "Solcast API returned an empty forecast list.  "
                "Check coordinates and API plan."
            )

        df = pd.DataFrame(records)
        df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
        df = df.set_index("period_end").sort_index()
        df.index = df.index.tz_convert(self._tz)

        # Keep only the columns we need; fill missing with 0
        available = [c for c in FORECAST_API_COLS if c in df.columns]
        missing   = [c for c in FORECAST_API_COLS if c not in df.columns]
        if missing:
            logger.warning("Solcast response missing columns: %s. Filled with 0.", missing)

        df = df[available].copy()
        for c in missing:
            df[c] = 0.0

        return df[FORECAST_API_COLS]

    def _load_cache(self, key: str) -> pd.DataFrame | None:
        path = self._cache_dir / key
        if not path.exists():
            return None
        age_s = time.time() - path.stat().st_mtime
        if age_s > self._cache_ttl_s:
            logger.debug("Cache stale (%.0f s old); will re-fetch.", age_s)
            return None
        logger.info("Using cached Solcast forecast (%.0f s old).", age_s)
        return self._parse(json.loads(path.read_text()))

    def _save_cache(self, key: str, data: dict) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / key).write_text(json.dumps(data))


# ── Convenience factory ───────────────────────────────────────────────────────

def make_client_from_cfg(cfg: SimpleNamespace) -> SolcastForecastClient:
    """Build a ``SolcastForecastClient`` from the pipeline config.

    Reads ``cfg.pipeline.forecast.cache_dir`` and the
    ``SOLCAST_API_KEY`` / ``SOLCAST_RESOURCE_ID`` environment variables.

    Parameters
    ----------
    cfg:
        Full ``PipelineConfig`` (site, model, pipeline namespaces).
    """
    forecast_cfg = getattr(cfg.pipeline, "forecast", None)
    cache_dir    = getattr(forecast_cfg, "cache_dir", "data/external/forecast_cache") \
                   if forecast_cfg else "data/external/forecast_cache"

    return SolcastForecastClient(cache_dir=cache_dir)
