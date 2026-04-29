"""
cnn_lstm_visualize.py
---------------------
Visualization suite for the CNN-LSTM solar PV forecasting results.

Generates 8 figures in results/figures/analysis/:
  1. synth_vs_real_full.png      — full-period overlay (daily means)
  2. synth_vs_real_weekly.png    — one representative week per season
  3. synth_vs_real_daily.png     — one representative day per season
  4. seasonal_full.png           — real data seasonal full-period (daily means)
  5. seasonal_weekly.png         — real data: one week per season
  6. seasonal_daily.png          — real data: mean diurnal profile per season
  7. daytime_timeseries.png      — daytime-only actual vs predicted (h+1 & h+24)
  8. best_worst_months.png       — best and worst months (time series + scatter)

Run
---
    python cnn_lstm_visualize.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SYNTH_PATH = Path("data/processed/synthetic_corrected_4yr.parquet")
REAL_PATH  = Path("data/processed/feature_matrix_lstm.parquet")
PRED_PATH  = Path("predictions_test.csv")
OUT_DIR    = Path("results/figures/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
SYNTH_COLOR   = "#2196F3"   # blue
REAL_COLOR    = "#FF5722"   # deep orange
ACTUAL_COLOR  = "#1B5E20"   # dark green
PRED_COLOR    = "#F57F17"   # amber
SEASON_COLORS = ["#1565C0", "#2E7D32", "#F57F17", "#6A1B9A"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})

# ── Sri Lanka seasons ──────────────────────────────────────────────────────────
SEASONS = {
    "SW Monsoon (May–Sep)":       [5, 6, 7, 8, 9],
    "Inter-monsoon 2 (Oct)":      [10],
    "NE Monsoon (Nov–Feb)":       [11, 12, 1, 2],
    "Inter-monsoon 1 (Mar–Apr)":  [3, 4],
}
SEASON_NAMES = list(SEASONS.keys())

# ── Helpers ────────────────────────────────────────────────────────────────────

def _strip_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Remove timezone info for plotting (keeps wall-clock local time)."""
    if idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def _season_of(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "Unknown"


def _pick_week(series: pd.Series, month: int, year: int | None = None) -> pd.Series:
    """Return one full Mon–Sun week from a given month (prefer middle of month)."""
    mask = series.index.month == month
    if year is not None:
        mask &= series.index.year == year
    sub = series[mask]
    if len(sub) < 7 * 24:
        return sub
    # Find Monday closest to day 10 of the month
    for day in range(10, 20):
        candidates = sub[sub.index.day == day]
        if not candidates.empty:
            start = candidates.index[0]
            end   = start + pd.Timedelta(days=7)
            return sub.loc[start:end]
    return sub.iloc[:7 * 24]


def _pick_day(series: pd.Series, month: int, year: int | None = None,
              prefer_clear: bool = True) -> pd.Series:
    """Return one 24-hour period from the given month.

    If prefer_clear=True, pick the day with the highest peak PV (clear sky day).
    """
    mask = series.index.month == month
    if year is not None:
        mask &= series.index.year == year
    sub = series[mask]
    if sub.empty:
        return sub
    daily_max = sub.resample("1D").max()
    if prefer_clear:
        best_day = daily_max.idxmax().date()
    else:
        best_day = daily_max.idxmin().date()
    day_data = sub[sub.index.date == best_day]
    return day_data


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
synth    = pd.read_parquet(SYNTH_PATH)
synth_pv = synth["pv_corrected_W"] / 1000.0
synth_pv.index = _strip_tz(synth_pv.index)

real    = pd.read_parquet(REAL_PATH)
real_pv = real["pv_ac_W"] / 1000.0
real_pv.index = _strip_tz(real_pv.index)

preds = pd.read_csv(PRED_PATH, parse_dates=["timestamp"], index_col="timestamp")
preds.index = _strip_tz(preds.index)

actual_h1  = preds["actual_h1"]
pred_h1    = preds["pred_h1"]
actual_h24 = preds["actual_h24"]
pred_h24   = preds["pred_h24"]


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Synthetic vs Real: Full Period overlaid (monthly mean by month-of-year
#             + overlapping daily means on shared time axis)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 1: Synthetic vs Real — full period overlaid …")

# ── Panel A: overlapping period on shared time axis ───────────────────────────
# Synthetic covers Jan 2020–Feb 2024; real covers Apr 2022–Mar 2023.
# Restrict synthetic to the real-data period so they can share one x-axis.
overlap_start = real_pv.index[0]
overlap_end   = real_pv.index[-1]
synth_overlap = synth_pv.loc[overlap_start:overlap_end]

synth_daily_ov = synth_overlap.resample("1D").mean()
real_daily     = real_pv.resample("1D").mean()

# ── Panel B: mean monthly profile (Jan–Dec) averaged across all available years ─
synth_monthly_profile = synth_pv.groupby(synth_pv.index.month).mean()
real_monthly_profile  = real_pv.groupby(real_pv.index.month).mean()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("Synthetic vs Real PV Output — Overlapping Comparison", fontsize=14, fontweight="bold")

# — Shared time axis (overlapping period) —
ax1.plot(synth_daily_ov.index, synth_daily_ov.values,
         color=SYNTH_COLOR, linewidth=0.9, alpha=0.7, label="Synthetic (daily mean)")
ax1.fill_between(synth_daily_ov.index, 0, synth_daily_ov.values,
                 color=SYNTH_COLOR, alpha=0.15)
ax1.plot(real_daily.index, real_daily.values,
         color=REAL_COLOR, linewidth=0.9, alpha=0.7, label="Real (daily mean)")
ax1.fill_between(real_daily.index, 0, real_daily.values,
                 color=REAL_COLOR, alpha=0.15)
# Monthly rolling means on top
synth_roll = synth_daily_ov.rolling(30, center=True).mean()
real_roll  = real_daily.rolling(30, center=True).mean()
ax1.plot(synth_roll.index, synth_roll.values,
         color=SYNTH_COLOR, linewidth=2.5, label="Synthetic (30-day mean)", zorder=4)
ax1.plot(real_roll.index, real_roll.values,
         color=REAL_COLOR, linewidth=2.5, label="Real (30-day mean)", zorder=4)
n_real = len(real_pv)
split_time = real_pv.index[int(n_real * 0.85)]
ax1.axvline(split_time, color="black", linestyle="--", linewidth=1.2,
            label=f"Val/Test split ({split_time.strftime('%b %Y')})")
ax1.set_title("Daily Means — Overlapping Period (Apr 2022 – Mar 2023)")
ax1.set_ylabel("PV Output (kW)")
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper right", fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
fig.autofmt_xdate()

# — Monthly profile (Jan–Dec) —
x = np.arange(1, 13)
width = 0.35
ax2.bar(x - width/2, synth_monthly_profile.values, width,
        color=SYNTH_COLOR, alpha=0.75, label=f"Synthetic (Jan 2020–Feb 2024)")
ax2.bar(x + width/2, real_monthly_profile.values, width,
        color=REAL_COLOR, alpha=0.75, label="Real (Apr 2022–Mar 2023)")
ax2.plot(x - width/2, synth_monthly_profile.values,
         color=SYNTH_COLOR, linewidth=2.0, marker="o", markersize=5, zorder=4)
ax2.plot(x + width/2, real_monthly_profile.values,
         color=REAL_COLOR, linewidth=2.0, marker="o", markersize=5, zorder=4)
ax2.set_xticks(x)
ax2.set_xticklabels(month_labels)
ax2.set_title("Mean Hourly PV Output by Calendar Month — Synthetic vs Real")
ax2.set_ylabel("Mean PV Output (kW)")
ax2.set_ylim(bottom=0)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "synth_vs_real_full.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'synth_vs_real_full.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Synthetic vs Real: One Week per Season (overlaid on same axes)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Synthetic vs Real — weekly overlaid …")

SEASON_EXAMPLE_MONTHS = {
    "SW Monsoon (May–Sep)":       (6, 2022),
    "Inter-monsoon 2 (Oct)":      (10, 2022),
    "NE Monsoon (Nov–Feb)":       (1, 2023),
    "Inter-monsoon 1 (Mar–Apr)":  (3, 2023),
}

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle("Synthetic vs Real PV — Representative Week per Season (Overlaid)", fontsize=14, fontweight="bold")
axes_flat = axes.flatten()

for idx, (season, (month, year)) in enumerate(SEASON_EXAMPLE_MONTHS.items()):
    synth_year = year if year in synth_pv.index.year else 2022
    synth_week = _pick_week(synth_pv, month, synth_year)
    real_week  = _pick_week(real_pv,  month, year)

    # Trim both to the same length (min of the two)
    n = min(len(synth_week), len(real_week), 7 * 24)
    s_arr = synth_week.values[:n]
    r_arr = real_week.values[:n]
    x     = np.arange(n)

    ax = axes_flat[idx]
    ax.fill_between(x, 0, s_arr, color=SYNTH_COLOR, alpha=0.25)
    ax.fill_between(x, 0, r_arr, color=REAL_COLOR,  alpha=0.25)
    ax.plot(x, s_arr, color=SYNTH_COLOR, linewidth=1.3, label="Synthetic", zorder=3)
    ax.plot(x, r_arr, color=REAL_COLOR,  linewidth=1.3, label="Real",      zorder=3)

    # Day separator lines
    n_days = n // 24
    for d in range(1, n_days + 1):
        ax.axvline(d * 24, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

    # Day labels centred in each day
    if not real_week.empty:
        day_start = real_week.index[0]
        ax.set_xticks([i * 24 + 12 for i in range(n_days)])
        ax.set_xticklabels(
            [(day_start + pd.Timedelta(days=i)).strftime("%a %d %b") for i in range(n_days)],
            fontsize=9, rotation=25, ha="right"
        )

    ax.set_title(f"{season}")
    ax.set_ylabel("PV Output (kW)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "synth_vs_real_weekly.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'synth_vs_real_weekly.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Synthetic vs Real: One Day per Season (overlaid on same axes)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Synthetic vs Real — daily overlaid …")

def _to_24h(s: pd.Series) -> np.ndarray:
    arr = np.zeros(24)
    for ts, v in s.items():
        arr[ts.hour] = v
    return arr

hours = np.arange(24)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Synthetic vs Real PV — Clear-Sky Day per Season (Overlaid)", fontsize=14, fontweight="bold")
axes_flat = axes.flatten()

for idx, (season, (month, year)) in enumerate(SEASON_EXAMPLE_MONTHS.items()):
    synth_year = year if year in synth_pv.index.year else 2022
    synth_day  = _pick_day(synth_pv, month, synth_year, prefer_clear=True)
    real_day   = _pick_day(real_pv,  month, year,       prefer_clear=True)

    synth_arr = _to_24h(synth_day.iloc[:24])
    real_arr  = _to_24h(real_day.iloc[:24])

    synth_label = synth_day.index[0].strftime("%d %b %Y") if not synth_day.empty else "N/A"
    real_label  = real_day.index[0].strftime("%d %b %Y")  if not real_day.empty  else "N/A"

    ax = axes_flat[idx]
    ax.fill_between(hours, 0, synth_arr, color=SYNTH_COLOR, alpha=0.25)
    ax.fill_between(hours, 0, real_arr,  color=REAL_COLOR,  alpha=0.25)
    ax.plot(hours, synth_arr, color=SYNTH_COLOR, linewidth=2.2, marker="o",
            markersize=4, label=f"Synthetic ({synth_label})", zorder=3)
    ax.plot(hours, real_arr,  color=REAL_COLOR,  linewidth=2.2, marker="s",
            markersize=4, label=f"Real ({real_label})",      zorder=3)

    ax.set_title(f"{season}")
    ax.set_ylabel("PV Output (kW)")
    ax.set_xlabel("Hour of day")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "synth_vs_real_daily.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'synth_vs_real_daily.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Seasonal Full Period (real data, daily means)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: Seasonal full period (real) …")

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=False)
fig.suptitle("Real PV Output by Season — Full Period (Daily Means)", fontsize=14, fontweight="bold")

for row, (season, months) in enumerate(SEASONS.items()):
    mask = real_pv.index.month.isin(months)
    sub  = real_pv[mask]
    sub_daily = sub.resample("1D").mean()

    ax = axes[row]
    ax.plot(sub_daily.index, sub_daily.values,
            color=SEASON_COLORS[row], linewidth=0.9, alpha=0.85)
    ax.fill_between(sub_daily.index, 0, sub_daily.values,
                    color=SEASON_COLORS[row], alpha=0.2)

    # Rolling 7-day mean
    roll7 = sub_daily.rolling(7, center=True).mean()
    ax.plot(roll7.index, roll7.values,
            color=SEASON_COLORS[row], linewidth=2.5, label="7-day rolling mean")

    ax.set_title(season)
    ax.set_ylabel("PV Output (kW)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()

plt.tight_layout()
fig.savefig(OUT_DIR / "seasonal_full.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'seasonal_full.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Seasonal Weekly (real data, one week per season)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: Seasonal weekly (real) …")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Real PV Output — Representative Week per Season", fontsize=14, fontweight="bold")
axes_flat = axes.flatten()

for idx, (season, (month, year)) in enumerate(SEASON_EXAMPLE_MONTHS.items()):
    week = _pick_week(real_pv, month, year)
    ax   = axes_flat[idx]

    if not week.empty:
        n_pts = len(week)
        ax.plot(range(n_pts), week.values, color=SEASON_COLORS[idx], linewidth=1.2)
        ax.fill_between(range(n_pts), 0, week.values, color=SEASON_COLORS[idx], alpha=0.25)
        # Day separators
        for d in range(1, n_pts // 24 + 1):
            ax.axvline(d * 24, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
        n_days = n_pts // 24
        ax.set_xticks([i * 24 + 12 for i in range(n_days)])
        day_start = week.index[0]
        ax.set_xticklabels(
            [(day_start + pd.Timedelta(days=i)).strftime("%a %d %b") for i in range(n_days)],
            fontsize=9, rotation=30, ha="right"
        )

    ax.set_title(f"{season}")
    ax.set_ylabel("PV Output (kW)")
    ax.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(OUT_DIR / "seasonal_weekly.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'seasonal_weekly.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Seasonal Daily Profile (mean diurnal curve per season)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6: Seasonal mean diurnal profiles (real) …")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Mean Diurnal PV Profile by Season — Real Data", fontsize=14, fontweight="bold")

for idx, (season, months) in enumerate(SEASONS.items()):
    mask = real_pv.index.month.isin(months)
    sub  = real_pv[mask]
    # Mean and std by hour of day
    hourly_mean = sub.groupby(sub.index.hour).mean()
    hourly_std  = sub.groupby(sub.index.hour).std()
    hours = hourly_mean.index

    ax.plot(hours, hourly_mean.values, color=SEASON_COLORS[idx],
            linewidth=2.5, label=season, marker="o", markersize=4)
    ax.fill_between(hours,
                    (hourly_mean - hourly_std).clip(lower=0),
                    hourly_mean + hourly_std,
                    color=SEASON_COLORS[idx], alpha=0.12)

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Mean PV Output (kW)")
ax.set_xticks(range(0, 24, 2))
ax.set_xlim(0, 23)
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Save a second version as subplots (one per season)
plt.tight_layout()
fig.savefig(OUT_DIR / "seasonal_daily.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'seasonal_daily.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Daytime-Only Time Series (actual vs predicted)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 7: Daytime-only time series …")

daytime_mask = actual_h1 > 1.0   # daytime = actual PV > 1 kW

actual_dt_h1  = actual_h1[daytime_mask]
pred_dt_h1    = pred_h1[daytime_mask]
actual_dt_h24 = actual_h24[daytime_mask]
pred_dt_h24   = pred_h24[daytime_mask]

r2_h1  = r2_score(actual_dt_h1,  pred_dt_h1)
r2_h24 = r2_score(actual_dt_h24, pred_dt_h24)

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
fig.suptitle("Daytime-Only Actual vs Predicted PV Output (CNN-LSTM, Step 1)", fontsize=14, fontweight="bold")

for ax, act, prd, label, r2 in [
    (axes[0], actual_dt_h1,  pred_dt_h1,  "h+1",  r2_h1),
    (axes[1], actual_dt_h24, pred_dt_h24, "h+24", r2_h24),
]:
    ax.plot(act.index, act.values,  color=ACTUAL_COLOR, linewidth=1.0,
            label="Actual", alpha=0.9)
    ax.plot(prd.index, prd.values,  color=PRED_COLOR, linewidth=1.0,
            label=f"Predicted ({label})", alpha=0.85, linestyle="--")
    ax.fill_between(act.index, act.values, prd.values,
                    where=(prd.values < act.values), alpha=0.15,
                    color="red", label="Under-prediction")
    ax.fill_between(act.index, act.values, prd.values,
                    where=(prd.values >= act.values), alpha=0.15,
                    color="blue", label="Over-prediction")
    ax.set_title(f"Daytime — {label}  (R² = {r2:.4f})")
    ax.set_ylabel("PV Output (kW)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate()

plt.tight_layout()
fig.savefig(OUT_DIR / "daytime_timeseries.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'daytime_timeseries.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Best and Worst Months
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 8: Best and worst months …")

# Compute R² per month from h+1 predictions
monthly_r2 = {}
for month in preds.index.month.unique():
    mask_m  = preds.index.month == month
    yt = actual_h1[mask_m].values
    yp = pred_h1[mask_m].values
    monthly_r2[month] = r2_score(yt, yp)

best_month  = max(monthly_r2, key=monthly_r2.get)
worst_month = min(monthly_r2, key=monthly_r2.get)

MONTH_NAMES = {1:"January", 2:"February", 3:"March", 4:"April", 5:"May",
               6:"June", 7:"July", 8:"August", 9:"September",
               10:"October", 11:"November", 12:"December"}

print(f"  Monthly R²: { {MONTH_NAMES[m]: f'{r:.4f}' for m, r in monthly_r2.items()} }")
print(f"  Best:  {MONTH_NAMES[best_month]}  R²={monthly_r2[best_month]:.4f}")
print(f"  Worst: {MONTH_NAMES[worst_month]} R²={monthly_r2[worst_month]:.4f}")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Best vs Worst Month — CNN-LSTM Forecasting Performance",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.4, 2.5, 2.5], hspace=0.5, wspace=0.3)

# ── Monthly R² bar chart (top row, spanning both columns) ─────────────────────
ax_bar = fig.add_subplot(gs[0, :])
months_sorted = sorted(monthly_r2.keys())
colors_bar = ["#2E7D32" if m == best_month else
              "#B71C1C" if m == worst_month else "#90A4AE"
              for m in months_sorted]
bars = ax_bar.bar(
    [MONTH_NAMES[m] for m in months_sorted],
    [monthly_r2[m] for m in months_sorted],
    color=colors_bar, edgecolor="white", linewidth=0.8
)
ax_bar.axhline(0.90, color="black", linestyle="--", linewidth=1.0, label="R²=0.90 target")
ax_bar.set_ylim(0.85, 1.0)
ax_bar.set_ylabel("R²")
ax_bar.set_title("Monthly R² (h+1) — Test Set (Jan–Mar 2023)")
ax_bar.legend(fontsize=9)
for bar, m in zip(bars, months_sorted):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{monthly_r2[m]:.4f}", ha="center", va="bottom", fontsize=9)

def _plot_month(ax_ts, ax_sc, month, label, ts_color, sc_color):
    mask_m = preds.index.month == month
    yt = actual_h1[mask_m]
    yp = pred_h1[mask_m]
    r2 = r2_score(yt.values, yp.values)
    rmse = float(np.sqrt(mean_squared_error(yt.values, yp.values)))

    # Time series
    ax_ts.plot(yt.index, yt.values, color=ACTUAL_COLOR, linewidth=1.2, label="Actual", alpha=0.9)
    ax_ts.plot(yp.index, yp.values, color=ts_color,    linewidth=1.2,
               label="Predicted", alpha=0.85, linestyle="--")
    ax_ts.set_title(f"{label} — {MONTH_NAMES[month]} (R²={r2:.4f}, RMSE={rmse:.1f} kW)")
    ax_ts.set_ylabel("PV Output (kW)")
    ax_ts.set_ylim(bottom=0)
    ax_ts.legend(loc="upper right", fontsize=9)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_ts.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    fig.autofmt_xdate()

    # Scatter
    lim = max(yt.max(), yp.max()) * 1.05
    ax_sc.scatter(yt.values, yp.values, s=6, alpha=0.4, color=sc_color)
    ax_sc.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="Perfect")
    ax_sc.set_xlabel("Actual (kW)")
    ax_sc.set_ylabel("Predicted (kW)")
    ax_sc.set_title(f"Scatter — {MONTH_NAMES[month]}")
    ax_sc.set_xlim(0, lim)
    ax_sc.set_ylim(0, lim)
    ax_sc.legend(fontsize=9)
    ax_sc.text(0.05, 0.92, f"R²={r2:.4f}", transform=ax_sc.transAxes,
               fontsize=11, fontweight="bold", color=sc_color)

ax_ts_best  = fig.add_subplot(gs[1, 0])
ax_sc_best  = fig.add_subplot(gs[1, 1])
ax_ts_worst = fig.add_subplot(gs[2, 0])
ax_sc_worst = fig.add_subplot(gs[2, 1])

_plot_month(ax_ts_best,  ax_sc_best,  best_month,  "Best Month",  "#2E7D32", "#1B5E20")
_plot_month(ax_ts_worst, ax_sc_worst, worst_month, "Worst Month", "#B71C1C", "#7F0000")

fig.savefig(OUT_DIR / "best_worst_months.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'best_worst_months.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Real vs Predicted: Full Test Period (overlaid)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 9: Real vs Predicted — full period overlaid …")

# Use h+1 for continuous time series; test covers Jan 24 – Mar 11 2023
real_daily_full   = real_pv.resample("1D").mean()
pred_daily_h1     = pred_h1.resample("1D").mean()
actual_daily_h1   = actual_h1.resample("1D").mean()

# Monthly mean profile (Jan–Mar) for actual & predicted
months_test = sorted(preds.index.month.unique())
actual_monthly  = actual_h1.groupby(actual_h1.index.month).mean()
pred_monthly_h1 = pred_h1.groupby(pred_h1.index.month).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("Real vs Predicted PV Output — Full Test Period (CNN-LSTM, h+1 & h+24)",
             fontsize=14, fontweight="bold")

# — Shared time axis: full real + predicted test overlay —
ax1.plot(real_daily_full.index, real_daily_full.values,
         color=REAL_COLOR, linewidth=0.8, alpha=0.5, label="Real (all data, daily mean)")
ax1.fill_between(real_daily_full.index, 0, real_daily_full.values,
                 color=REAL_COLOR, alpha=0.1)
ax1.plot(actual_daily_h1.index, actual_daily_h1.values,
         color=ACTUAL_COLOR, linewidth=1.2, label="Actual — test set", zorder=3)
ax1.plot(pred_daily_h1.index,   pred_daily_h1.values,
         color=PRED_COLOR, linewidth=1.2, linestyle="--", label="Predicted h+1 — test set", zorder=3)
# h+24 daily mean overlaid
pred_daily_h24   = pred_h24.resample("1D").mean()
actual_daily_h24 = actual_h24.resample("1D").mean()
ax1.plot(pred_daily_h24.index, pred_daily_h24.values,
         color="#9C27B0", linewidth=1.2, linestyle=":", label="Predicted h+24 — test set", zorder=3)

n_real = len(real_pv)
split_time = real_pv.index[int(n_real * 0.85)]
ax1.axvline(split_time, color="black", linestyle="--", linewidth=1.2,
            label=f"Test set start ({split_time.strftime('%d %b %Y')})")
ax1.set_title("Daily Mean PV — Full Real Period with Test Set Predictions Overlaid")
ax1.set_ylabel("PV Output (kW)")
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper left", fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
fig.autofmt_xdate()

# — Monthly mean profile (Jan–Mar) —
x = np.arange(len(months_test))
width = 0.35
month_lbls = [MONTH_NAMES[m] for m in months_test]
ax2.bar(x - width/2, [actual_monthly[m]  for m in months_test], width,
        color=ACTUAL_COLOR, alpha=0.75, label="Actual")
ax2.bar(x + width/2, [pred_monthly_h1[m] for m in months_test], width,
        color=PRED_COLOR,   alpha=0.75, label="Predicted (h+1)")
ax2.plot(x - width/2, [actual_monthly[m]  for m in months_test],
         color=ACTUAL_COLOR, linewidth=2.0, marker="o", markersize=6, zorder=4)
ax2.plot(x + width/2, [pred_monthly_h1[m] for m in months_test],
         color=PRED_COLOR,   linewidth=2.0, marker="s", markersize=6, zorder=4)
# Annotate R² per month
for xi, m in zip(x, months_test):
    r2 = monthly_r2[m]
    ax2.text(xi, max(actual_monthly[m], pred_monthly_h1[m]) + 1.5,
             f"R²={r2:.3f}", ha="center", fontsize=9, color="black")
ax2.set_xticks(x)
ax2.set_xticklabels(month_lbls)
ax2.set_title("Mean Hourly PV by Month — Actual vs Predicted (h+1)")
ax2.set_ylabel("Mean PV Output (kW)")
ax2.set_ylim(bottom=0)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "real_vs_pred_full.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'real_vs_pred_full.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Real vs Predicted: One Week per Test Month (overlaid)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 10: Real vs Predicted — weekly overlaid …")

# Rebuild hourly actual/predicted as indexed Series aligned to real timestamps
# pred_h1 timestamp = time of h+1 prediction
actual_h1_ts = actual_h1.copy()
pred_h1_ts   = pred_h1.copy()

TEST_MONTHS = {
    "January 2023":  (1, 2023),
    "February 2023": (2, 2023),
    "March 2023":    (3, 2023),
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Real vs Predicted PV — Representative Week per Month (h+1, Overlaid)",
             fontsize=14, fontweight="bold")

for idx, (label, (month, year)) in enumerate(TEST_MONTHS.items()):
    act_week = _pick_week(actual_h1_ts, month, year)
    prd_week = _pick_week(pred_h1_ts,   month, year)

    # Align on shared index
    shared_idx = act_week.index.intersection(prd_week.index)
    act_w = act_week.loc[shared_idx]
    prd_w = prd_week.loc[shared_idx]
    n = len(shared_idx)

    ax = axes[idx]
    ax.fill_between(range(n), 0, act_w.values, color=ACTUAL_COLOR, alpha=0.2)
    ax.fill_between(range(n), 0, prd_w.values, color=PRED_COLOR,   alpha=0.2)
    ax.plot(range(n), act_w.values, color=ACTUAL_COLOR, linewidth=1.3,
            label="Actual", zorder=3)
    ax.plot(range(n), prd_w.values, color=PRED_COLOR,   linewidth=1.3,
            linestyle="--", label="Predicted (h+1)", zorder=3)

    # Day separators
    n_days = n // 24
    for d in range(1, n_days + 1):
        ax.axvline(d * 24, color="grey", linewidth=0.6, linestyle="--", alpha=0.4)

    if not shared_idx.empty:
        day_start = shared_idx[0]
        ax.set_xticks([i * 24 + 12 for i in range(n_days)])
        ax.set_xticklabels(
            [(day_start + pd.Timedelta(days=i)).strftime("%a %d") for i in range(n_days)],
            fontsize=9, rotation=25, ha="right"
        )

    r2_w = r2_score(act_w.values, prd_w.values) if len(act_w) > 1 else float("nan")
    ax.set_title(f"{label}\n(R²={r2_w:.4f})")
    ax.set_ylabel("PV Output (kW)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "real_vs_pred_weekly.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'real_vs_pred_weekly.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 11 — Real vs Predicted: One Day per Test Month (overlaid, h+1 & h+24)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 11: Real vs Predicted — daily overlaid (h+1 & h+24) …")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Real vs Predicted PV — Clear-Sky Day per Month (h+1 top, h+24 bottom, Overlaid)",
             fontsize=14, fontweight="bold")

for col, (label, (month, year)) in enumerate(TEST_MONTHS.items()):
    for row, (horizon_name, act_ser, prd_ser) in enumerate([
        ("h+1",  actual_h1, pred_h1),
        ("h+24", actual_h24, pred_h24),
    ]):
        act_day = _pick_day(act_ser, month, year, prefer_clear=True)
        prd_day = _pick_day(prd_ser, month, year, prefer_clear=True)

        act_arr = _to_24h(act_day.iloc[:24])
        prd_arr = _to_24h(prd_day.iloc[:24])

        act_label = act_day.index[0].strftime("%d %b %Y") if not act_day.empty else "N/A"

        ax = axes[row, col]
        ax.fill_between(hours, 0, act_arr, color=ACTUAL_COLOR, alpha=0.2)
        ax.fill_between(hours, 0, prd_arr, color=PRED_COLOR,   alpha=0.2)
        ax.plot(hours, act_arr, color=ACTUAL_COLOR, linewidth=2.2,
                marker="o", markersize=4, label=f"Actual ({act_label})", zorder=3)
        ax.plot(hours, prd_arr, color=PRED_COLOR,   linewidth=2.2,
                marker="s", markersize=4, linestyle="--",
                label=f"Predicted ({horizon_name})", zorder=3)

        valid = (act_arr > 0) | (prd_arr > 0)
        r2_d = r2_score(act_arr[valid], prd_arr[valid]) if valid.sum() > 2 else float("nan")
        ax.set_title(f"{label} — {horizon_name}\n(R²={r2_d:.4f})")
        ax.set_ylabel("PV Output (kW)")
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 3))
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "real_vs_pred_daily.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'real_vs_pred_daily.png'}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to {OUT_DIR}/")
print("  1. synth_vs_real_full.png")
print("  2. synth_vs_real_weekly.png")
print("  3. synth_vs_real_daily.png")
print("  4. seasonal_full.png")
print("  5. seasonal_weekly.png")
print("  6. seasonal_daily.png")
print("  7. daytime_timeseries.png")
print("  8. best_worst_months.png")
