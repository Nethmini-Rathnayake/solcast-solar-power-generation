"""
src/physics/pvlib_model.py
---------------------------
Physics-based PV simulation using pvlib and Solcast irradiance inputs.

Why physics-first?
------------------
A pure ML model trained on weather → PV can learn the right correlations but
gives no guarantee of physical plausibility.  Running a pvlib simulation first
provides:
  - A *clear-sky baseline*: what the plant would produce under ideal optics.
  - A *clearness index* (GHI / clearsky_GHI): a normalised cloud signal.
  - A *pvlib residual* (actual / simulated): captures the gap between ideal
    physics and real-world behaviour (soiling, clipping, inverter losses,
    shading, degradation).  This residual is then a feature for the ML model.

Model choice: PVWatts
---------------------
PVWatts is used rather than the full CEC database chain because:
  - No nameplate module/inverter database lookup required.
  - Parameters (pdc0, gamma_pdc, eta_inv) are interpretable and configurable.
  - Accuracy is comparable for system-level forecasting (as opposed to
    component-level diagnosis).
  - It is easy to explain in a viva context.

Irradiance source
-----------------
Solcast provides GHI, DNI, DHI directly (5-min resolution, site-specific).
This is converted to Plane-of-Array (POA) irradiance using the Perez
transposition model inside pvlib.

Cell temperature
----------------
The Faiman model is used for cell temperature estimation.  It requires only
air temperature and wind speed; since Solcast does not provide wind speed,
a nominal value of 1.0 m/s is assumed.
  # TODO: incorporate a wind speed source if available.

Output columns added
--------------------
  - ``pvlib_ac_W``        : simulated AC output (W)
  - ``pvlib_dc_W``        : simulated DC output (W)
  - ``poa_global_Wm2``    : plane-of-array irradiance (W/m²)
  - ``cell_temp_C``       : estimated cell temperature (°C)
  - ``clearness_index``   : GHI / clearsky_GHI (dimensionless, 0–1+)
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import faiman

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Assumed wind speed when not available in dataset (m/s)
_DEFAULT_WIND_SPEED_MS = 1.0

# Minimum clearsky GHI to compute a meaningful clearness index (W/m²)
_CLEARSKY_MIN_WM2 = 10.0


def run_pvlib_simulation(
    df: pd.DataFrame,
    site_cfg: SimpleNamespace,
) -> pd.DataFrame:
    """Add pvlib simulation columns to the aligned 5-min DataFrame.

    Parameters
    ----------
    df:
        Cleaned aligned DataFrame with Solcast columns (ghi, dni, dhi,
        clearsky_ghi, air_temp) and a tz-aware DatetimeIndex.
    site_cfg:
        The ``site`` and ``pv_system`` namespaces from ``site.yaml``.
        Expected attributes: latitude, longitude, timezone, elevation_m,
        pv_system.pdc0_kw, pv_system.gamma_pdc, pv_system.eta_inv_nom,
        pv_system.tilt_deg, pv_system.azimuth_deg,
        pv_system.temp_model_params.u0, pv_system.temp_model_params.u1.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        ``pvlib_ac_W``, ``pvlib_dc_W``, ``poa_global_Wm2``,
        ``cell_temp_C``, ``clearness_index``.
    """
    df = df.copy()
    pv_sys = site_cfg.pv_system

    # ── Build pvlib Location ──────────────────────────────────────────────────
    location = Location(
        latitude=site_cfg.latitude,
        longitude=site_cfg.longitude,
        tz=site_cfg.timezone,
        altitude=site_cfg.elevation_m,
        name=site_cfg.name,
    )
    logger.info(
        "pvlib simulation starting: %s (lat=%.4f, lon=%.4f)",
        location.name,
        location.latitude,
        location.longitude,
    )

    # ── Solar position ────────────────────────────────────────────────────────
    solar_pos = location.get_solarposition(df.index)

    # ── Plane-of-Array (POA) irradiance ───────────────────────────────────────
    # Use Solcast GHI/DNI/DHI directly (site-specific satellite-derived values)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=pv_sys.tilt_deg,
        surface_azimuth=pv_sys.azimuth_deg,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=df["dni"],
        ghi=df["ghi"],
        dhi=df["dhi"],
        model="perez",
        # Perez requires extra_radiation; pvlib computes it from the index
        dni_extra=pvlib.irradiance.get_extra_radiation(df.index),
        airmass=pvlib.atmosphere.get_relative_airmass(
            solar_pos["apparent_zenith"]
        ),
    )
    df["poa_global_Wm2"] = poa["poa_global"].clip(lower=0)

    # ── Cell temperature (Faiman model) ───────────────────────────────────────
    wind_speed = pd.Series(
        _DEFAULT_WIND_SPEED_MS, index=df.index, dtype=float
    )
    u0 = pv_sys.temp_model_params.u0
    u1 = pv_sys.temp_model_params.u1

    df["cell_temp_C"] = faiman(
        poa_global=df["poa_global_Wm2"],
        temp_air=df["air_temp"],
        wind_speed=wind_speed,
        u0=u0,
        u1=u1,
    )

    # ── PVWatts DC model ──────────────────────────────────────────────────────
    # P_dc = pdc0 × (poa_global / 1000) × [1 + gamma_pdc × (T_cell − 25)]
    pdc0_w = pv_sys.pdc0_kw * 1_000.0   # kW → W
    df["pvlib_dc_W"] = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=df["poa_global_Wm2"],
        temp_cell=df["cell_temp_C"],
        pdc0=pdc0_w,
        gamma_pdc=pv_sys.gamma_pdc,
        temp_ref=25.0,
    ).clip(lower=0)

    # ── PVWatts AC model ──────────────────────────────────────────────────────
    # Simplified: P_ac = P_dc × eta_inv, clipped at AC nameplate
    pac_max_w = pdc0_w   # assume AC capacity ≈ DC nameplate
    df["pvlib_ac_W"] = (
        df["pvlib_dc_W"] * pv_sys.eta_inv_nom
    ).clip(lower=0, upper=pac_max_w)

    # ── Clearness index ───────────────────────────────────────────────────────
    # k_t = GHI / clearsky_GHI (dimensionless; 1.0 = perfectly clear sky)
    # Set to NaN during nighttime to avoid division by near-zero.
    if "clearsky_ghi" in df.columns:
        daytime_mask = df["clearsky_ghi"] >= _CLEARSKY_MIN_WM2
        df["clearness_index"] = np.where(
            daytime_mask,
            df["ghi"] / df["clearsky_ghi"].clip(lower=_CLEARSKY_MIN_WM2),
            np.nan,
        )
        # Clip to physical range: clearness index rarely exceeds 1.1
        df["clearness_index"] = df["clearness_index"].clip(upper=1.1)
    else:
        logger.warning(
            "clearsky_ghi not found — clearness_index will not be computed."
        )
        df["clearness_index"] = np.nan

    logger.info(
        "pvlib simulation complete. "
        "Peak simulated AC: %.1f kW | Mean daytime AC: %.1f kW",
        df["pvlib_ac_W"].max() / 1_000,
        df.loc[df["pvlib_ac_W"] > 1_000, "pvlib_ac_W"].mean() / 1_000,
    )
    return df
