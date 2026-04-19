"""
scripts/05b_generate_variants.py
----------------------------------
Pipeline Step 5b: Generate synthetic PV variants with real-world disturbances.

Upgrades the corrected synthetic output (step 05) with Layer C disturbances
inspired by the NREL synthetic data methodology:

  Variant A — Clean baseline          (corrected only, no disturbances)
  Variant B — Degradation             (0.6%/year annual decay)
  Variant C — Soiling                 (weather-driven sawtooth)
  Variant D — Outages/curtailment     (random availability events)
  Variant E — Weather-dependent noise (heteroscedastic Gaussian)
  Variant F — Realistic combined      (all disturbances together)

Each variant is saved as a separate parquet file so they can be loaded
independently for training or analysis.

Statistical validation (Upgrade 10)
-------------------------------------
Compares each synthetic variant against real PV on:
  - Daily energy distribution
  - Hourly power distribution
  - Monthly mean daytime power
  - Ramp-rate distribution
  - Residual (synthetic - real) distribution

Outputs
-------
  data/processed/synthetic_variants/
    variant_A_clean_baseline.parquet
    variant_B_degradation.parquet
    variant_C_soiling.parquet
    variant_D_outages.parquet
    variant_E_noise.parquet
    variant_F_realistic.parquet
    variants_summary.csv          ← mean daytime power + RMSE vs real per variant

  results/figures/synthetic_variants/
    01_variant_comparison_monthly.png
    02_hourly_distribution.png
    03_ramp_rate_distribution.png
    04_daily_energy_histogram.png
    05_residual_distribution.png

Prerequisites
-------------
    scripts/05_generate_synthetic.py  → data/processed/synthetic_corrected_4yr.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.synthetic.disturbance import generate_variant, VARIANT_CONFIGS

logger = get_logger(__name__)

_SYNTH_PATH  = Path("data/processed/synthetic_corrected_4yr.parquet")
_REAL_PATH   = Path("data/processed/feature_matrix_hourly.parquet")
_OUT_DIR     = Path("data/processed/synthetic_variants")
_FIGURES_DIR = Path("results/figures/synthetic_variants")

OVERLAP_START = "2022-04-01"
OVERLAP_END   = "2023-03-12"

VARIANT_COLORS = {
    "A_clean_baseline": "#3A7FD5",
    "B_degradation":    "#2ECC71",
    "C_soiling":        "#E67E22",
    "D_outages":        "#9B59B6",
    "E_noise":          "#1ABC9C",
    "F_realistic":      "#E05C5C",
}
REAL_COLOR = "#2C3E50"


def main() -> None:
    logger.info("=== Step 5b: Generate Synthetic Variants (Layer C Disturbances) ===")

    cfg = load_config()
    capacity_w = cfg.site.pv_system.pdc0_kw * 1000.0

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load base data ────────────────────────────────────────────────────────
    if not _SYNTH_PATH.exists():
        raise FileNotFoundError(
            f"Corrected synthetic not found: {_SYNTH_PATH}. "
            "Run scripts/05_generate_synthetic.py first."
        )
    synth_df = pd.read_parquet(_SYNTH_PATH)
    real_df  = pd.read_parquet(_REAL_PATH) if _REAL_PATH.exists() else None

    logger.info(
        "Synthetic base: %d rows (%s → %s).",
        len(synth_df),
        synth_df.index.min().date(),
        synth_df.index.max().date(),
    )
    if real_df is not None:
        logger.info("Real PV: %d rows.", len(real_df))

    # ── Generate all variants ─────────────────────────────────────────────────
    variants: dict[str, pd.Series] = {}
    for name in VARIANT_CONFIGS:
        pv = generate_variant(synth_df, name, capacity_w=capacity_w, seed=42)
        variants[name] = pv
        # Save to parquet (attach to a copy of synth_df for context columns)
        out_df = synth_df.copy()
        out_df["pv_variant_W"] = pv
        out_path = _OUT_DIR / f"variant_{name}.parquet"
        out_df.to_parquet(out_path)
        logger.info("Saved: %s", out_path)

    # ── Statistical summary ───────────────────────────────────────────────────
    summary = _compute_summary(variants, real_df, synth_df, capacity_w)
    summary.to_csv(_OUT_DIR / "variants_summary.csv")
    logger.info("\n%s", summary.to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_monthly_comparison(variants, real_df, synth_df)
    _plot_hourly_distribution(variants, real_df, synth_df)
    _plot_ramp_rates(variants, real_df, synth_df)
    _plot_daily_energy(variants, real_df, synth_df)
    _plot_residual_distribution(variants, real_df, synth_df)

    logger.info("=== Step 5b complete ===")
    logger.info("Variants saved: %s", _OUT_DIR)
    logger.info("Figures saved:  %s", _FIGURES_DIR)


# ── Statistical helpers ────────────────────────────────────────────────────────

def _compute_summary(
    variants: dict[str, pd.Series],
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
    capacity_w: float,
) -> pd.DataFrame:
    """Compute per-variant summary statistics vs real PV."""
    rows = []

    # Real PV stats (overlap period)
    if real_df is not None:
        r = real_df["pv_ac_W"]
        real_daytime = r[r > 0]
        real_stats = {
            "mean_daytime_kW": real_daytime.mean() / 1000,
            "peak_kW":         r.max() / 1000,
            "daily_energy_kWh_mean": _daily_energy(r).mean(),
            "rmse_vs_real_kW": 0.0,
        }
        rows.append({"variant": "real", "description": "Measured real PV", **real_stats})

    for name, pv in variants.items():
        desc = VARIANT_CONFIGS[name]["description"]
        daytime = pv[pv > 0]
        row = {
            "variant":     name,
            "description": desc,
            "mean_daytime_kW":        daytime.mean() / 1000,
            "peak_kW":                pv.max() / 1000,
            "daily_energy_kWh_mean":  _daily_energy(pv).mean(),
            "rmse_vs_real_kW":        np.nan,
        }
        # RMSE vs real (overlap period only)
        if real_df is not None:
            common = pv.index.intersection(real_df.index)
            common = common[
                (common >= pd.Timestamp(OVERLAP_START, tz=synth_df.index.tz))
                & (common <= pd.Timestamp(OVERLAP_END,   tz=synth_df.index.tz))
            ]
            if len(common) > 0:
                diff = pv.loc[common] - real_df["pv_ac_W"].loc[common]
                row["rmse_vs_real_kW"] = float(np.sqrt(np.nanmean(diff**2))) / 1000
        rows.append(row)

    return pd.DataFrame(rows).set_index("variant")


def _daily_energy(pv: pd.Series) -> pd.Series:
    """Daily energy in kWh (sum of hourly W / 1000)."""
    return pv.clip(lower=0).resample("D").sum() / 1000


def _daytime_only(pv: pd.Series, synth_df: pd.DataFrame) -> pd.Series:
    """Mask to daytime hours (pvlib > 0)."""
    mask = synth_df["pvlib_ac_W"] > 0
    return pv[mask]


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_monthly_comparison(
    variants: dict[str, pd.Series],
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 5))

    # Each variant — monthly mean daytime power
    for name, pv in variants.items():
        daytime_mask = synth_df["pvlib_ac_W"] > 0
        monthly = (pv[daytime_mask] / 1000).resample("ME").mean()
        lw = 2.5 if name == "F_realistic" else 1.2
        alpha = 0.95 if name in ("A_clean_baseline", "F_realistic") else 0.65
        ax.plot(monthly.index, monthly.values,
                color=VARIANT_COLORS[name], linewidth=lw, alpha=alpha,
                label=f"{name.split('_',1)[1].replace('_',' ').title()}")

    # Real PV (overlap)
    if real_df is not None:
        r_monthly = (real_df["pv_ac_W"][real_df["pv_ac_W"] > 0] / 1000).resample("ME").mean()
        ax.plot(r_monthly.index, r_monthly.values,
                color=REAL_COLOR, linewidth=2.5, linestyle="--",
                label="Real PV (measured)", zorder=5)

    ax.set_title("Monthly Mean Daytime Power — All Variants vs Real PV",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean daytime power (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(_FIGURES_DIR / "01_variant_comparison_monthly.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 01_variant_comparison_monthly.png")


def _plot_hourly_distribution(
    variants: dict[str, pd.Series],
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, synth_df["pv_corrected_W"].max() / 1000, 60)

    for name, pv in variants.items():
        daytime = _daytime_only(pv, synth_df) / 1000
        ax.hist(daytime, bins=bins, density=True, alpha=0.35,
                color=VARIANT_COLORS[name],
                label=name.split("_", 1)[1].replace("_", " ").title())

    if real_df is not None:
        r = real_df["pv_ac_W"][real_df["pv_ac_W"] > 0] / 1000
        ax.hist(r, bins=bins, density=True, alpha=0.0,
                histtype="step", color=REAL_COLOR, linewidth=2.0,
                label="Real PV")

    ax.set_xlabel("Power (kW)")
    ax.set_ylabel("Density")
    ax.set_title("Daytime Power Distribution — All Variants vs Real PV",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(_FIGURES_DIR / "02_hourly_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 02_hourly_distribution.png")


def _plot_ramp_rates(
    variants: dict[str, pd.Series],
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cap_kw = synth_df["pv_corrected_W"].max() / 1000
    bins = np.linspace(-cap_kw * 0.5, cap_kw * 0.5, 80)

    for name, pv in variants.items():
        ramp = pv.diff().dropna() / 1000
        ax.hist(ramp, bins=bins, density=True, alpha=0.35,
                color=VARIANT_COLORS[name],
                label=name.split("_", 1)[1].replace("_", " ").title())

    if real_df is not None:
        r_ramp = real_df["pv_ac_W"].diff().dropna() / 1000
        ax.hist(r_ramp, bins=bins, density=True, alpha=0.0,
                histtype="step", color=REAL_COLOR, linewidth=2.0,
                label="Real PV")

    ax.set_xlabel("Hourly ramp rate (kW/h)")
    ax.set_ylabel("Density")
    ax.set_title("Ramp-Rate Distribution — All Variants vs Real PV",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(_FIGURES_DIR / "03_ramp_rate_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 03_ramp_rate_distribution.png")


def _plot_daily_energy(
    variants: dict[str, pd.Series],
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 2500, 60)

    for name, pv in variants.items():
        de = _daily_energy(pv)
        ax.hist(de, bins=bins, density=True, alpha=0.35,
                color=VARIANT_COLORS[name],
                label=name.split("_", 1)[1].replace("_", " ").title())

    if real_df is not None:
        r_de = _daily_energy(real_df["pv_ac_W"])
        ax.hist(r_de, bins=bins, density=True, alpha=0.0,
                histtype="step", color=REAL_COLOR, linewidth=2.0,
                label="Real PV")

    ax.set_xlabel("Daily energy (kWh)")
    ax.set_ylabel("Density")
    ax.set_title("Daily Energy Histogram — All Variants vs Real PV",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(_FIGURES_DIR / "04_daily_energy_histogram.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 04_daily_energy_histogram.png")


def _plot_residual_distribution(
    variants: dict[str, pd.Series],
    real_df: pd.DataFrame | None,
    synth_df: pd.DataFrame,
) -> None:
    if real_df is None:
        logger.warning("No real PV data — skipping residual distribution plot.")
        return

    common_idx = synth_df.index.intersection(real_df.index)
    tz = synth_df.index.tz
    overlap = common_idx[
        (common_idx >= pd.Timestamp(OVERLAP_START, tz=tz))
        & (common_idx <= pd.Timestamp(OVERLAP_END, tz=tz))
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    real_vals = real_df["pv_ac_W"].loc[overlap].values / 1000
    cap_kw = synth_df["pv_corrected_W"].max() / 1000
    bins = np.linspace(-cap_kw * 0.5, cap_kw * 0.5, 80)

    for name, pv in variants.items():
        residual = (pv.loc[overlap].values / 1000) - real_vals
        ax.hist(residual, bins=bins, density=True, alpha=0.35,
                color=VARIANT_COLORS[name],
                label=name.split("_", 1)[1].replace("_", " ").title())

    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Residual: synthetic − real (kW)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution (overlap period) — Synthetic variants vs Real PV",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(_FIGURES_DIR / "05_residual_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: 05_residual_distribution.png")


if __name__ == "__main__":
    main()
