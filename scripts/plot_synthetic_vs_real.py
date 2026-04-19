"""
scripts/plot_synthetic_vs_real.py
----------------------------------
Plots synthetic (pvlib + residual-corrected) vs real PV data over the
overlap period (Apr 2022 – Mar 2023) and the full synthetic 4-year span.

Panels
------
1. Full 4-year synthetic time-series (monthly aggregated daily max)
2. Overlap: daily average kW — real vs synthetic side-by-side
3. Overlap: scatter plot real vs synthetic (hourly, daytime only)
4. A zoomed 7-day window (one week in July 2022) showing hourly profiles
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

OUT_DIR = Path("results/figures/synthetic_vs_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY_KW = 305.0
OVERLAP_START = "2022-04-01"
OVERLAP_END   = "2023-03-12"
ZOOM_START    = "2022-07-04"
ZOOM_END      = "2022-07-10"

REAL_COLOR  = "#E05C5C"
SYNTH_COLOR = "#3A7FD5"
PVLIB_COLOR = "#A8C97F"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
synth_full = pd.read_parquet("data/processed/synthetic_corrected_4yr.parquet")
real_full  = pd.read_parquet("data/processed/feature_matrix_hourly.parquet")

# Convert to kW
synth_full["pv_corrected_kW"] = synth_full["pv_corrected_W"] / 1000
synth_full["pvlib_ac_kW"]     = synth_full["pvlib_ac_W"]     / 1000
real_full["pv_ac_kW"]         = real_full["pv_ac_W"]         / 1000

# Overlap slices
s_ov = synth_full.loc[OVERLAP_START:OVERLAP_END].copy()
r_ov = real_full.copy()

# ── Figure 1: Full 4-year synthetic overview ──────────────────────────────────
print("Figure 1: 4-year overview …")
fig, ax = plt.subplots(figsize=(14, 4))

# Monthly mean daytime power
monthly_synth = (
    synth_full.loc[synth_full["pvlib_ac_kW"] > 0, "pv_corrected_kW"]
    .resample("ME").mean()
)
monthly_pvlib = (
    synth_full.loc[synth_full["pvlib_ac_kW"] > 0, "pvlib_ac_kW"]
    .resample("ME").mean()
)

ax.fill_between(monthly_pvlib.index, monthly_pvlib.values,
                alpha=0.25, color=PVLIB_COLOR, label="pvlib (physics only)")
ax.plot(monthly_pvlib.index, monthly_pvlib.values,
        color=PVLIB_COLOR, linewidth=1.2)
ax.plot(monthly_synth.index, monthly_synth.values,
        color=SYNTH_COLOR, linewidth=2, label="Corrected synthetic")

# Mark overlap window
ax.axvspan(pd.Timestamp(OVERLAP_START, tz="Asia/Colombo"),
           pd.Timestamp(OVERLAP_END,   tz="Asia/Colombo"),
           alpha=0.12, color=REAL_COLOR, label="Real PV overlap window")

ax.set_title("4-Year Synthetic PV — Monthly Mean Daytime Power", fontsize=13, fontweight="bold")
ax.set_ylabel("Mean daytime power (kW)")
ax.set_xlabel("")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30, ha="right")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "01_full_4yr_synthetic_overview.png", dpi=150)
plt.close(fig)

# ── Figure 2: Daily mean — real vs synthetic over overlap period ──────────────
print("Figure 2: Overlap daily mean …")
s_daily = s_ov.loc[s_ov["pvlib_ac_kW"] > 0, "pv_corrected_kW"].resample("D").mean()
r_daily = r_ov.loc[r_ov["pv_ac_kW"]    > 0, "pv_ac_kW"       ].resample("D").mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(s_daily.index, s_daily.values,
        color=SYNTH_COLOR, linewidth=1.4, alpha=0.9, label="Synthetic (corrected)")
ax.plot(r_daily.index, r_daily.values,
        color=REAL_COLOR,  linewidth=1.4, alpha=0.9, label="Real PV (measured)")
ax.set_title("Overlap Period: Daily Mean Daytime Power — Real vs Synthetic",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Daily mean power (kW)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=30, ha="right")
ax.legend(fontsize=10)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "02_overlap_daily_mean.png", dpi=150)
plt.close(fig)

# ── Figure 3: Scatter hourly real vs synthetic (daytime only) ─────────────────
print("Figure 3: Scatter …")
# Align on common index
common_idx = s_ov.index.intersection(r_ov.index)
s_h = s_ov.loc[common_idx, "pv_corrected_kW"]
r_h = r_ov.loc[common_idx, "pv_ac_kW"]

# Daytime only (real PV > 0 or synthetic > 0)
daytime = (s_h > 0) | (r_h > 0)
s_h = s_h[daytime]
r_h = r_h[daytime]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(r_h, s_h, alpha=0.15, s=5, color=SYNTH_COLOR, rasterized=True)

# Identity line
lim = max(r_h.max(), s_h.max()) * 1.02
ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="1:1 line")

# Regression line (use only finite, non-NaN pairs)
mask = np.isfinite(r_h.values) & np.isfinite(s_h.values)
coeff = np.polyfit(r_h.values[mask], s_h.values[mask], 1)
x_line = np.linspace(0, lim, 200)
ax.plot(x_line, np.polyval(coeff, x_line),
        color=REAL_COLOR, linewidth=1.5, label=f"Fit: y={coeff[0]:.2f}x+{coeff[1]:.1f}")

# Correlation (on finite pairs only)
both_finite = np.isfinite(r_h.values) & np.isfinite(s_h.values)
corr = np.corrcoef(r_h.values[both_finite], s_h.values[both_finite])[0, 1]
rmse = np.sqrt(np.nanmean((s_h.values - r_h.values) ** 2))
ax.text(0.05, 0.92, f"r = {corr:.3f}\nRMSE = {rmse:.1f} kW",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_xlabel("Real PV power (kW)", fontsize=11)
ax.set_ylabel("Synthetic PV power (kW)", fontsize=11)
ax.set_title("Hourly Scatter: Real vs Synthetic (daytime, overlap period)",
             fontsize=12, fontweight="bold")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend(fontsize=9)
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(OUT_DIR / "03_scatter_real_vs_synthetic.png", dpi=150)
plt.close(fig)

# ── Figure 4: 7-day zoomed hourly profiles ────────────────────────────────────
print("Figure 4: 7-day zoom …")
s_zoom = s_ov.loc[ZOOM_START:ZOOM_END, "pv_corrected_kW"]
r_zoom = r_ov.loc[ZOOM_START:ZOOM_END, "pv_ac_kW"]
p_zoom = s_ov.loc[ZOOM_START:ZOOM_END, "pvlib_ac_kW"]

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(p_zoom.index, p_zoom.values, alpha=0.2,
                color=PVLIB_COLOR, label="pvlib (physics only)")
ax.plot(s_zoom.index, s_zoom.values,
        color=SYNTH_COLOR, linewidth=1.8, label="Synthetic (corrected)")
ax.plot(r_zoom.index, r_zoom.values,
        color=REAL_COLOR, linewidth=1.8, linestyle="--", label="Real PV")

ax.set_title(f"7-Day Hourly Profile: Real vs Synthetic  ({ZOOM_START} → {ZOOM_END})",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Power (kW)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.legend(fontsize=10)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "04_7day_zoom_profiles.png", dpi=150)
plt.close(fig)

# ── Figure 5: Monthly box plots (real vs synthetic) ───────────────────────────
print("Figure 5: Monthly box plots …")
s_ov_day = s_ov.loc[s_ov["pvlib_ac_kW"] > 0, "pv_corrected_kW"].copy()
r_ov_day = r_ov.loc[r_ov["pv_ac_kW"] > 0, "pv_ac_kW"].copy()

months = pd.date_range(OVERLAP_START, OVERLAP_END, freq="MS")
month_labels = [m.strftime("%b %Y") for m in months]
real_by_month  = [r_ov_day[r_ov_day.index.month == m.month].values for m in months]
synth_by_month = [s_ov_day[s_ov_day.index.month == m.month].values for m in months]

fig, ax = plt.subplots(figsize=(15, 5))
x = np.arange(len(months))
w = 0.35

bp1 = ax.boxplot(real_by_month,  positions=x - w/2, widths=w*0.9,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor=REAL_COLOR,  alpha=0.7),
                  medianprops=dict(color="white", linewidth=2),
                  whiskerprops=dict(color=REAL_COLOR),
                  capprops=dict(color=REAL_COLOR))
bp2 = ax.boxplot(synth_by_month, positions=x + w/2, widths=w*0.9,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor=SYNTH_COLOR, alpha=0.7),
                  medianprops=dict(color="white", linewidth=2),
                  whiskerprops=dict(color=SYNTH_COLOR),
                  capprops=dict(color=SYNTH_COLOR))

ax.set_xticks(x)
ax.set_xticklabels(month_labels, rotation=30, ha="right")
ax.set_ylabel("Daytime power (kW)")
ax.set_title("Monthly Distribution: Real vs Synthetic PV Power (daytime hours)",
             fontsize=13, fontweight="bold")
from matplotlib.patches import Patch
ax.legend([Patch(facecolor=REAL_COLOR,  alpha=0.7),
           Patch(facecolor=SYNTH_COLOR, alpha=0.7)],
          ["Real PV", "Synthetic (corrected)"], fontsize=10, loc="upper right")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "05_monthly_boxplots.png", dpi=150)
plt.close(fig)

print(f"\nAll figures saved to: {OUT_DIR}")
print("\nSummary statistics (daytime hours, overlap period):")
print(f"  Real      — mean: {r_ov['pv_ac_kW'].mean():.1f} kW  peak: {r_ov['pv_ac_kW'].max():.1f} kW")
print(f"  Synthetic — mean: {s_ov['pv_corrected_kW'].mean():.1f} kW  peak: {s_ov['pv_corrected_kW'].max():.1f} kW")
print(f"  Pearson r (hourly daytime): {corr:.4f}")
print(f"  RMSE (hourly daytime): {rmse:.2f} kW")
