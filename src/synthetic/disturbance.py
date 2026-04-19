"""
src/synthetic/disturbance.py
------------------------------
Layer C — Real-world disturbance injection for synthetic PV generation.

Implements the NREL-inspired upgrade to the physics + ML bias-corrected
synthetic output.  Each function takes the corrected power series and a
weather DataFrame, and returns a modified series with one type of disturbance
injected.

The full pipeline (applied in order):

    P_phys  →  [Layer A: pvlib]
    P_corr  →  [Layer B: XGBoost + LSTM residual]
    P_deg   →  apply_degradation()
    P_soil  →  apply_soiling()
    P_avail →  apply_outages()
    P_noise →  apply_weather_noise()
    P_final →  apply_clipping()

All functions are pure (return a new Series) and accept a ``seed`` argument
for reproducibility.

Disturbance models (NREL/TP-5K00-86459 methodology)
----------------------------------------------------
Degradation
    Slow multiplicative decay: P_deg(t) = P_corr(t) × (1 − r_d × t_years)
    Rate sampled from truncated normal (μ=0.75%/yr, σ=0.5%/yr) per NREL
    PV Fleets 2022 CDF.  Fixed default: 0.0075.

Soiling — monsoon-aware sawtooth (Sri Lanka)
    Loss accumulates each dry hour at a monthly-scaled rate to reflect
    Sri Lanka's two monsoon seasons (SW: May–Sep, NE: Nov–Jan) and the
    dry inter-monsoon periods (Feb–Apr, Oct).
    Rain proxy: cloud_opacity AND relative_humidity exceed thresholds.
    Recovery intensity scales with cloud_opacity (heavy rain → full recovery,
    moderate rain → partial), matching NREL's rainfall-magnitude approach.

Outages / curtailment (NREL fleet statistics)
    Poisson-distributed outage events.
    Count: Poisson(λ=8.6/yr); duration: exponential(μ=74 h = 3.1 days).
    Curtailment: Poisson(λ=10/yr), duration 2–4 h, cap at 75–90% of AC.

Weather-dependent noise (NREL heteroscedastic model)
    Noise σ scales with both irradiance level AND cloud variability
    (hour-to-hour change in cloud_opacity as a proxy for within-hour
    irradiance variability, per NREL 2D lookup table concept).
    Noise is zero at night.

Clipping
    Hard AC cap at system capacity.

Stochastic parameter sampling
    ``sample_disturbance_params(seed)`` draws all disturbance parameters from
    their NREL-derived distributions so each call to ``generate_variant``
    with ``stochastic=True`` produces a unique realistic realisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Sri Lanka monsoon seasonality ─────────────────────────────────────────────
# Monthly multiplier on the base soiling accumulation rate.
# Calibrated against Colombo climatological monthly rainfall normals
# (Sri Lanka Meteorological Department):
#
#   Month | Rainfall | Season
#   ------+----------+------------------------------------------
#   Jan   |   89 mm  | NE monsoon fading → relatively dry
#   Feb   |   69 mm  | DRIEST month → peak soiling
#   Mar   |  112 mm  | Pre-monsoon onset, moderate showers
#   Apr   |  231 mm  | Pre-monsoon convective PEAK (3rd rainiest)
#   May   |  371 mm  | SW monsoon onset — heaviest month
#   Jun   |  224 mm  | SW monsoon (declines after May peak)
#   Jul   |  135 mm  | SW monsoon waning — drier than expected
#   Aug   |  109 mm  | Relatively dry despite "monsoon season"
#   Sep   |  160 mm  | SW monsoon tail
#   Oct   |  348 mm  | 2nd inter-monsoon — 2nd rainiest month
#   Nov   |  315 mm  | 2nd inter-monsoon / early NE monsoon
#   Dec   |  179 mm  | NE monsoon
#
# Key corrections vs previous version:
#   • Apr (1.50→0.80): very rainy — was wrongly treated as dry
#   • Oct (0.80→0.20): 2nd rainiest month — was badly underestimated
#   • Nov (0.70→0.40): very rainy — was underestimated
#   • Jul–Aug (0.20→1.30–1.45): drier than Jun peak — was overestimated as wet
#   • Jan (0.80→1.55): drier than Jul–Aug — was underestimated
MONSOON_RATE_MULTIPLIERS: dict[int, float] = {
    1:  1.55,   # Jan  — NE monsoon fading, relatively dry (89 mm)
    2:  1.65,   # Feb  — driest month, peak soiling accumulation (69 mm)
    3:  1.45,   # Mar  — pre-monsoon, moderate showers begin (112 mm)
    4:  0.80,   # Apr  — pre-monsoon convective peak, very rainy (231 mm)
    5:  0.10,   # May  — SW monsoon onset, heaviest rainfall (371 mm)
    6:  0.85,   # Jun  — SW monsoon (lower than May peak) (224 mm)
    7:  1.30,   # Jul  — SW monsoon waning, drier mid-season (135 mm)
    8:  1.45,   # Aug  — relatively dry despite monsoon label (109 mm)
    9:  1.20,   # Sep  — SW monsoon tail (160 mm)
    10: 0.20,   # Oct  — 2nd inter-monsoon, 2nd rainiest month (348 mm)
    11: 0.40,   # Nov  — 2nd inter-monsoon / early NE monsoon (315 mm)
    12: 1.10,   # Dec  — NE monsoon, drying towards Jan (179 mm)
}


# ── Monsoon onset profile helper ──────────────────────────────────────────────

def _sigmoid(x: np.ndarray | float, center: float, width: float = 5.0) -> np.ndarray:
    """Logistic sigmoid centred at ``center`` with half-width ``width`` days."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(x, dtype=float) - center) / width))


def _build_monsoon_rate_profile(
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hourly soiling-rate multiplier with stochastic per-year monsoon timing.

    Starts from the climatological monthly means in ``MONSOON_RATE_MULTIPLIERS``
    then overlays two year-specific rain events per year, each with a randomly
    sampled onset and withdrawal date and a sigmoid (5-day) transition:

    Event 1 — SW monsoon heavy rain (Colombo SW wet-zone):
        Onset:      Day-of-year 135 ± 21  (mean May 15, ±3 weeks)
        Withdrawal: Day-of-year 270 ± 14  (mean Sep 27, ±2 weeks)
        Min rate during event: 0.08× base (near-full cleaning)

    Event 2 — 2nd inter-monsoon (Oct–Nov):
        Onset:      Day-of-year 280 ± 14  (mean Oct 7)
        Withdrawal: Day-of-year 330 ± 14  (mean Nov 26)
        Min rate during event: 0.15× base

    This creates visible year-to-year shape differences at the monthly level
    — matching the variability seen in measured PV data — rather than the same
    smooth curve every year.
    """
    # Base monthly multiplier for every hour
    profile = np.array(
        [MONSOON_RATE_MULTIPLIERS[int(m)] for m in index.month],
        dtype=np.float64,
    )

    doy = np.array([ts.timetuple().tm_yday for ts in index], dtype=np.float64)

    for year in sorted(set(index.year)):
        mask = index.year == year
        if not mask.any():
            continue
        d = doy[mask]

        # ── SW monsoon ────────────────────────────────────────────────────────
        sw_onset = float(np.clip(rng.normal(135, 21), 100, 165))
        sw_end   = float(np.clip(rng.normal(270, 14), 248, 295))
        # Depth rises at onset, falls at withdrawal (sigmoid blending)
        sw_depth = _sigmoid(d, sw_onset, 5.0) * (1.0 - _sigmoid(d, sw_end, 5.0))
        # Blend: during monsoon → rate drops to 0.08× base
        profile[mask] = profile[mask] * (1.0 - sw_depth) + 0.08 * sw_depth

        # ── 2nd inter-monsoon ─────────────────────────────────────────────────
        im2_onset = float(np.clip(rng.normal(280, 14), 260, 305))
        im2_end   = float(np.clip(rng.normal(330, 14), 308, 355))
        im2_depth = _sigmoid(d, im2_onset, 5.0) * (1.0 - _sigmoid(d, im2_end, 5.0))
        # Blend: during 2nd inter-monsoon → rate drops to 0.15× base
        profile[mask] = profile[mask] * (1.0 - im2_depth) + 0.15 * im2_depth

    return profile


# ── Degradation ───────────────────────────────────────────────────────────────

def apply_degradation(
    pv: pd.Series,
    rate_per_year: float = 0.0075,
) -> pd.Series:
    """Apply slow annual degradation.

    Parameters
    ----------
    pv:
        Hourly PV power series (W), tz-aware DatetimeIndex.
    rate_per_year:
        Annual degradation rate (fraction).  Default: 0.0075 = 0.75 %/year
        (NREL PV Fleets 2022 median).  Typical range: 0.003–0.040.

    Returns
    -------
    pd.Series — degraded power.
    """
    t0 = pv.index[0]
    t_years = (pv.index - t0).total_seconds() / (365.25 * 24 * 3600)
    factor = np.clip(1.0 - rate_per_year * t_years, 0.5, 1.0)
    result = pv * factor
    peak_loss_pct = (1 - factor[-1]) * 100
    logger.info(
        "Degradation applied: rate=%.2f%%/yr, peak loss at end=%.2f%%.",
        rate_per_year * 100, peak_loss_pct,
    )
    return result.rename(pv.name)


# ── Soiling (monsoon-aware) ───────────────────────────────────────────────────

def apply_soiling(
    pv: pd.Series,
    weather: pd.DataFrame,
    daily_rate: float = 0.0008,
    max_loss: float = 0.10,
    rain_cloud_threshold: float = 70.0,
    rain_humidity_threshold: float = 80.0,
    recovery_fraction: float = 0.80,
    monsoon_aware: bool = True,
    seed: int = 42,
) -> pd.Series:
    """Monsoon-aware sawtooth soiling model driven by a rain proxy.

    Soiling loss accumulates each dry hour at ``daily_rate`` scaled by a
    monthly monsoon multiplier (when ``monsoon_aware=True``).  The loss
    resets partially or fully on hours classified as "rain" based on
    cloud_opacity + relative_humidity thresholds.

    Recovery intensity mirrors NREL's rainfall-magnitude approach:
      - Heavy rain proxy (cloud_opacity > 90 %): full recovery (100 %)
      - Moderate rain proxy (70–90 %):           partial recovery (50–100 %)

    Parameters
    ----------
    pv:
        Hourly PV series (W).
    weather:
        DataFrame with ``cloud_opacity`` [%] and ``relative_humidity`` [%].
    daily_rate:
        Base soiling loss per dry hour (fraction).
        Default 0.0008 ≈ 0.6 %/week in the peak dry season.
    max_loss:
        Maximum soiling loss fraction (cap).  Default 10 %.
    rain_cloud_threshold:
        cloud_opacity (%) above which an hour is considered a rain event.
    rain_humidity_threshold:
        relative_humidity (%) above which an hour is considered a rain event.
    recovery_fraction:
        Maximum fraction of accumulated soiling removed per rain event.
        Actual recovery scales with cloud_opacity intensity.
    monsoon_aware:
        If True, scale the hourly rate by ``MONSOON_RATE_MULTIPLIERS`` for
        the calendar month — modelling Sri Lanka's seasonal rain patterns.
    seed:
        RNG seed for stochastic recovery jitter.

    Returns
    -------
    pd.Series — soiling-attenuated power.
    """
    rng = np.random.default_rng(seed)
    cloud    = weather["cloud_opacity"].reindex(pv.index).fillna(0.0)
    humidity = weather["relative_humidity"].reindex(pv.index).fillna(0.0)

    is_rain = (cloud > rain_cloud_threshold) & (humidity > rain_humidity_threshold)

    # Build per-hour rate multiplier: stochastic monsoon onset when enabled
    if monsoon_aware:
        rate_profile = daily_rate * _build_monsoon_rate_profile(pv.index, rng)
    else:
        rate_profile = np.full(len(pv), daily_rate, dtype=np.float64)

    soiling = np.zeros(len(pv))
    loss = 0.0
    cloud_vals = cloud.values

    for i in range(len(pv)):
        if is_rain.iloc[i]:
            # Recovery scales with cloud_opacity intensity (heavy rain = full)
            cloud_val = cloud_vals[i]
            if cloud_val > 90.0:
                # Heavy rain proxy → near-full recovery
                recovery = recovery_fraction * rng.uniform(0.90, 1.00)
            else:
                # Moderate rain → partial recovery (NREL: 50–100% for 0.5–3 mm)
                recovery = recovery_fraction * rng.uniform(0.50, 1.00)
            loss *= (1.0 - recovery)
            loss = max(loss, 0.0)
        else:
            loss = min(loss + rate_profile[i], max_loss)
        soiling[i] = loss

    result = pv.values * (1.0 - soiling)
    n_rain = int(is_rain.sum())
    mean_soiling_pct = soiling[soiling > 0].mean() * 100 if (soiling > 0).any() else 0.0
    logger.info(
        "Soiling applied: base_rate=%.4f%s, max_loss=%.1f%%, "
        "%d rain events, mean_soiling=%.2f%%.",
        daily_rate,
        " (stochastic monsoon onset)" if monsoon_aware else "",
        max_loss * 100,
        n_rain,
        mean_soiling_pct,
    )
    return pd.Series(result, index=pv.index, name=pv.name)


# ── Outages / curtailment ─────────────────────────────────────────────────────

def apply_outages(
    pv: pd.Series,
    outages_per_year: float = 8.6,
    severity_range: tuple[float, float] = (0.0, 1.0),
    mean_duration_hours: float = 74.4,
    curtailment_events_per_year: float = 10.0,
    curtailment_duration_hours: tuple[int, int] = (2, 4),
    curtailment_cap_range: tuple[float, float] = (0.75, 0.90),
    capacity_w: float | None = None,
    seed: int = 42,
) -> pd.Series:
    """Inject random outage and curtailment events (NREL fleet statistics).

    Outage statistics from NREL/TP-5K00-86459 Table 4:
      - Mean 8.6 outages/year (σ=8.3) → Poisson(λ=8.6)
      - Mean duration 3.1 days (σ=10.5) → exponential(μ=74.4 h)
    Curtailment statistics:
      - 40 events per 4-year dataset → ~10/year
      - ~3-hour events capping at 75–90 % of nominal AC

    Parameters
    ----------
    pv:
        Hourly PV series (W).
    outages_per_year:
        Expected number of outage events per year (Poisson rate).
        NREL default: 8.6.
    severity_range:
        (min, max) power reduction fraction per event.  1.0 = full shutdown.
    mean_duration_hours:
        Mean outage duration (hours) for exponential distribution.
        Default: 74.4 h = 3.1 days (NREL).
    curtailment_events_per_year:
        Expected number of curtailment events per year.  NREL default: ~10.
    curtailment_duration_hours:
        (min, max) curtailment event duration.  Default (2, 4) ≈ 3 h mean.
    curtailment_cap_range:
        (min, max) fraction of capacity at which curtailment caps output.
        Default (0.75, 0.90) → 80 % nominal AC.
    capacity_w:
        System capacity in W (used for curtailment).  If None, uses max(pv).
    seed:
        RNG seed.

    Returns
    -------
    pd.Series — power series with outage/curtailment events injected.
    """
    rng = np.random.default_rng(seed)
    result = pv.values.copy()
    n_hours = len(pv)
    n_years = n_hours / 8760.0
    cap_w = capacity_w if capacity_w is not None else float(pv.max())

    # ── Outages (exponential duration, NREL statistics) ───────────────────────
    n_outages = rng.poisson(outages_per_year * n_years)
    starts = rng.integers(0, n_hours, size=n_outages)
    total_outage_h = 0
    for s in starts:
        # Exponential duration: min 1 h, max 30 days (720 h)
        dur = int(np.clip(rng.exponential(scale=mean_duration_hours), 1, 720))
        sev = rng.uniform(severity_range[0], severity_range[1])
        end = min(s + dur, n_hours)
        result[s:end] *= (1.0 - sev)
        total_outage_h += dur

    # ── Curtailment (short events, NREL: ~3 h, 80 % cap) ─────────────────────
    n_curt = rng.poisson(curtailment_events_per_year * n_years)
    curt_starts = rng.integers(0, n_hours, size=n_curt)
    for s in curt_starts:
        dur = int(rng.integers(curtailment_duration_hours[0],
                               curtailment_duration_hours[1] + 1))
        cap_frac = rng.uniform(curtailment_cap_range[0], curtailment_cap_range[1])
        end = min(s + dur, n_hours)
        result[s:end] = np.minimum(result[s:end], cap_w * cap_frac)

    logger.info(
        "Outages injected: %d events (~%d hrs total) | "
        "%d curtailment events.",
        n_outages, total_outage_h, n_curt,
    )
    return pd.Series(result.clip(min=0.0), index=pv.index, name=pv.name)


# ── Weather-dependent noise ───────────────────────────────────────────────────

def apply_weather_noise(
    pv: pd.Series,
    weather: pd.DataFrame,
    base_sigma_frac: float = 0.02,
    seed: int = 42,
) -> pd.Series:
    """Add heteroscedastic Gaussian noise scaled by irradiance AND variability.

    Noise level combines two NREL-inspired drivers:
      1. GHI level   — low GHI → higher σ (more uncertain estimation)
      2. Cloud variability — rapid cloud_opacity change between hours
         (proxy for within-hour irradiance variability, per NREL 2D lookup)

    Tier base multipliers (GHI level):
      - Clear-sky midday (GHI > 700 W/m²):   scale × 1.0
      - Partly cloudy (GHI 200–700):          scale × 2.5 + variability boost
      - Overcast / sunrise-sunset (GHI < 200):scale × 5.0 + variability boost
      - Night (pv = 0):                        zero noise

    Variability boost = cloud_opacity_delta / 50  (capped at +3.0×)
    where cloud_opacity_delta is the absolute hour-to-hour change.

    Parameters
    ----------
    pv:
        Hourly PV series (W).
    weather:
        DataFrame with ``ghi`` and ``cloud_opacity`` columns.
    base_sigma_frac:
        Base noise as a fraction of current PV output.  Default 2 %.
    seed:
        RNG seed.

    Returns
    -------
    pd.Series — noisy power (clipped to ≥ 0).
    """
    rng = np.random.default_rng(seed)
    ghi   = weather["ghi"].reindex(pv.index).fillna(0.0)
    cloud = weather["cloud_opacity"].reindex(pv.index).fillna(0.0)

    # Hour-to-hour cloud variability (proxy for intra-hour irradiance variance)
    cloud_delta = cloud.diff().abs().fillna(0.0).values
    var_boost = np.minimum(cloud_delta / 50.0, 3.0)   # capped at +3× extra

    # GHI-tier base multiplier
    ghi_vals = ghi.values
    ghi_scale = np.where(
        pv.values <= 0,
        0.0,                                   # night → zero noise
        np.where(
            ghi_vals > 700,
            1.0 + var_boost * 0.5,             # clear sky: small variability bump
            np.where(
                ghi_vals > 200,
                2.5 + (cloud.values / 100.0) * 2.0 + var_boost,  # partly cloudy
                5.0 + var_boost,               # overcast / sunrise-sunset
            )
        )
    )

    sigma = base_sigma_frac * pv.values * ghi_scale
    noise = rng.normal(0.0, np.maximum(sigma, 0.0))
    result = (pv.values + noise).clip(min=0.0)

    noisy_frac = (sigma > 0).mean() * 100
    mean_sigma_pct = (sigma[sigma > 0] / pv.values[sigma > 0]).mean() * 100 if (sigma > 0).any() else 0.0
    logger.info(
        "Weather noise applied: base_sigma=%.1f%%, %.0f%% of hours affected, "
        "mean effective sigma=%.1f%% (with variability boost).",
        base_sigma_frac * 100, noisy_frac, mean_sigma_pct,
    )
    return pd.Series(result, index=pv.index, name=pv.name)


# ── Explicit clipping ─────────────────────────────────────────────────────────

def apply_clipping(
    pv: pd.Series,
    capacity_w: float,
) -> pd.Series:
    """Hard AC cap at system capacity (explicit inverter clipping).

    Parameters
    ----------
    pv:
        Hourly PV series (W).
    capacity_w:
        AC nameplate capacity in W.

    Returns
    -------
    pd.Series clipped to [0, capacity_w].
    """
    n_clipped = (pv > capacity_w).sum()
    result = pv.clip(lower=0.0, upper=capacity_w)
    logger.info(
        "Clipping applied: capacity=%.1f kW | %d hours clipped.",
        capacity_w / 1000, n_clipped,
    )
    return result.rename(pv.name)


# ── Stochastic parameter sampling ─────────────────────────────────────────────

def sample_disturbance_params(seed: int = 42) -> dict:
    """Sample disturbance parameters from NREL-derived distributions.

    Used by ``generate_variant(..., stochastic=True)`` to produce unique
    realisations of each variant rather than always using fixed defaults.

    Distributions
    -------------
    Degradation rate:
        Truncated normal (μ=0.0075, σ=0.005) clipped to [0.003, 0.040].
        Source: NREL PV Fleets 2022 CDF median −0.75 %/yr.

    Soiling rate multiplier:
        Chopped normal (μ=1.0, σ=0.30) clipped to [0.30, 2.50].
        Applied as a multiplicative scale on the base hourly rate.

    Soiling max loss:
        Uniform [0.05, 0.20] — varies by panel type / local dust.

    Outages per year:
        Gamma distribution approximating NREL mean=8.6, σ=8.3.
        shape = (mean/σ)² ≈ 1.07,  scale = σ²/mean ≈ 8.0.

    Curtailment per year:
        Normal (μ=10, σ=3) clipped to [2, 25].

    Noise base sigma fraction:
        Uniform [0.015, 0.035].

    Parameters
    ----------
    seed:
        RNG seed.

    Returns
    -------
    dict with keys: degradation_rate, soiling_multiplier, soiling_max_loss,
    outages_per_year, curtailment_per_year, noise_sigma.
    """
    rng = np.random.default_rng(seed)

    degradation_rate = float(np.clip(rng.normal(0.0075, 0.005), 0.003, 0.040))
    soiling_multiplier = float(np.clip(rng.normal(1.0, 0.30), 0.30, 2.50))
    soiling_max_loss = float(rng.uniform(0.05, 0.20))
    outages_per_year = float(np.clip(rng.gamma(shape=1.07, scale=8.0), 0.0, 30.0))
    curtailment_per_year = float(np.clip(rng.normal(10.0, 3.0), 2.0, 25.0))
    noise_sigma = float(rng.uniform(0.015, 0.035))

    logger.info(
        "Stochastic params sampled (seed=%d): "
        "deg=%.2f%%/yr | soil_mult=%.2f | soil_max=%.0f%% | "
        "outages=%.1f/yr | curtail=%.1f/yr | noise_sigma=%.1f%%",
        seed,
        degradation_rate * 100, soiling_multiplier, soiling_max_loss * 100,
        outages_per_year, curtailment_per_year, noise_sigma * 100,
    )
    return {
        "degradation_rate":    degradation_rate,
        "soiling_multiplier":  soiling_multiplier,
        "soiling_max_loss":    soiling_max_loss,
        "outages_per_year":    outages_per_year,
        "curtailment_per_year": curtailment_per_year,
        "noise_sigma":         noise_sigma,
    }


# ── Orchestrator ──────────────────────────────────────────────────────────────

# Predefined variant configurations
VARIANT_CONFIGS: dict[str, dict] = {
    "A_clean_baseline": {
        "description": "Corrected synthetic only (no disturbances)",
        "degradation": False,
        "soiling":     False,
        "outages":     False,
        "noise":       False,
    },
    "B_degradation": {
        "description": "Annual degradation only (NREL median 0.75%/year)",
        "degradation": True,  "degradation_rate": 0.0075,
        "soiling":     False,
        "outages":     False,
        "noise":       False,
    },
    "C_soiling": {
        "description": "Monsoon-aware soiling only (Sri Lanka sawtooth)",
        "degradation": False,
        "soiling":     True,
        "outages":     False,
        "noise":       False,
    },
    "D_outages": {
        "description": "Random outages and curtailment (NREL fleet statistics)",
        "degradation": False,
        "soiling":     False,
        "outages":     True,
        "noise":       False,
    },
    "E_noise": {
        "description": "Weather+variability-dependent noise only",
        "degradation": False,
        "soiling":     False,
        "outages":     False,
        "noise":       True,
    },
    "F_realistic": {
        "description": "Full realistic scenario (all disturbances combined)",
        "degradation": True,  "degradation_rate": 0.0075,
        "soiling":     True,
        "outages":     True,
        "noise":       True,
    },
}


def generate_variant(
    corrected_df: pd.DataFrame,
    variant_name: str,
    capacity_w: float,
    seed: int = 42,
    stochastic: bool = False,
) -> pd.Series:
    """Apply the named disturbance variant to the corrected PV series.

    Parameters
    ----------
    corrected_df:
        Full 4-year hourly DataFrame with ``pv_corrected_W``,
        ``ghi``, ``cloud_opacity``, ``relative_humidity``.
    variant_name:
        One of the keys in ``VARIANT_CONFIGS``.
    capacity_w:
        System AC capacity in W.
    seed:
        RNG seed for reproducibility.
    stochastic:
        If True, sample disturbance parameters from NREL-derived distributions
        using ``sample_disturbance_params(seed)`` rather than fixed defaults.
        Each unique seed produces a different realistic realisation.

    Returns
    -------
    pd.Series — final PV power (W) for this variant.
    """
    if variant_name not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant_name}'. "
            f"Available: {list(VARIANT_CONFIGS.keys())}"
        )

    cfg = VARIANT_CONFIGS[variant_name]
    pv = corrected_df["pv_corrected_W"].copy()

    logger.info(
        "Generating variant '%s'%s: %s",
        variant_name,
        " [stochastic]" if stochastic else "",
        cfg["description"],
    )

    # Optionally sample parameters from NREL distributions
    params: dict = sample_disturbance_params(seed) if stochastic else {}

    # Layer C1: Degradation
    if cfg.get("degradation"):
        rate = params.get("degradation_rate", cfg.get("degradation_rate", 0.0075))
        pv = apply_degradation(pv, rate_per_year=rate)

    # Layer C2: Soiling (monsoon-aware by default)
    if cfg.get("soiling"):
        base_rate = 0.0008 * params.get("soiling_multiplier", 1.0)
        max_loss  = params.get("soiling_max_loss", 0.10)
        pv = apply_soiling(
            pv, corrected_df,
            daily_rate=base_rate,
            max_loss=max_loss,
            monsoon_aware=True,
            seed=seed,
        )

    # Layer C3: Outages / curtailment
    if cfg.get("outages"):
        pv = apply_outages(
            pv,
            outages_per_year=params.get("outages_per_year", 8.6),
            curtailment_events_per_year=params.get("curtailment_per_year", 10.0),
            capacity_w=capacity_w,
            seed=seed + 1,
        )

    # Layer C4: Weather-dependent noise (with variability boost)
    if cfg.get("noise"):
        pv = apply_weather_noise(
            pv, corrected_df,
            base_sigma_frac=params.get("noise_sigma", 0.02),
            seed=seed + 2,
        )

    # Layer C5: Hard clipping
    pv = apply_clipping(pv, capacity_w)

    return pv
