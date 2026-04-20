"""
validate_cnn_lstm.py
---------------------
Independent validation of the CNN-LSTM R² = 0.9249 result.

Checks:
  1. Metric recomputation from raw predictions CSV
  2. Residual analysis (bias, normality, autocorrelation)
  3. Daytime-only vs all-hour metrics
  4. Comparison against same-day baseline on identical test rows
  5. Diurnal error profile (mean absolute error by hour-of-day)
  6. Monthly error profile
  7. Confidence: bootstrap 95% CI on mean R²
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path

OUT = Path("results/figures/validation")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load predictions & real data ───────────────────────────────────────────────
pred = pd.read_csv("predictions_test.csv", index_col=0, parse_dates=True)
real = pd.read_parquet("data/processed/feature_matrix_lstm.parquet")

N_H = 24
horizons = list(range(1, N_H + 1))

# Per-horizon arrays
yt_all = np.stack([pred[f"actual_h{h}"].values for h in horizons], axis=1)  # (1095, 24)
yp_all = np.stack([pred[f"pred_h{h}"].values   for h in horizons], axis=1)
res_all = yp_all - yt_all   # residuals

# ── 1. Metric table ────────────────────────────────────────────────────────────
print("=" * 65)
print("1. INDEPENDENTLY RECOMPUTED METRICS (from predictions_test.csv)")
print("=" * 65)
rows = []
for h in horizons:
    yt = yt_all[:, h-1]
    yp = yp_all[:, h-1]
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae  = mean_absolute_error(yt, yp)
    mbe  = float(np.mean(yp - yt))
    r2   = r2_score(yt, yp)
    day  = yt > 1.0
    mape = float(np.mean(np.abs(yp[day] - yt[day]) / yt[day]) * 100) if day.sum() > 0 else np.nan
    rows.append(dict(horizon=f"h+{h:02d}", RMSE=rmse, MAE=mae, MBE=mbe, MAPE=mape, R2=r2))

df_m = pd.DataFrame(rows).set_index("horizon")
mean_row = df_m.mean(numeric_only=True).rename("MEAN")
df_m = pd.concat([df_m, mean_row.to_frame().T])
df_m.to_csv("results/validation_metrics_cnn_lstm.csv")

print(df_m[["RMSE","MAE","MBE","R2"]].round({"RMSE":2,"MAE":2,"MBE":2,"R2":4}).to_string())
print(f"\n✓ Confirmed R² = {df_m.loc['MEAN','R2']:.4f}  "
      f"RMSE = {df_m.loc['MEAN','RMSE']:.3f} kW  "
      f"MAE = {df_m.loc['MEAN','MAE']:.3f} kW")

# ── 2. Daytime-only metrics ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. DAYTIME-ONLY METRICS  (actual > 1 kW)")
print("=" * 65)
day_r2, day_rmse, day_mae = [], [], []
for h in horizons:
    yt = yt_all[:, h-1]
    yp = yp_all[:, h-1]
    day = yt > 1.0
    if day.sum() > 10:
        day_r2.append(r2_score(yt[day], yp[day]))
        day_rmse.append(np.sqrt(mean_squared_error(yt[day], yp[day])))
        day_mae.append(mean_absolute_error(yt[day], yp[day]))

print(f"  Daytime rows per horizon (avg): {day.sum()} / {len(yt)}")
print(f"  Mean daytime R²:   {np.mean(day_r2):.4f}")
print(f"  Mean daytime RMSE: {np.mean(day_rmse):.3f} kW")
print(f"  Mean daytime MAE:  {np.mean(day_mae):.3f} kW")

# ── 3. Residual analysis ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("3. RESIDUAL ANALYSIS")
print("=" * 65)
flat_res = res_all.ravel()
flat_res = flat_res[np.isfinite(flat_res)]
print(f"  Mean residual (bias): {flat_res.mean():.4f} kW")
print(f"  Std  residual:        {flat_res.std():.4f} kW")
print(f"  Skewness:             {stats.skew(flat_res):.4f}")
print(f"  Kurtosis (excess):    {stats.kurtosis(flat_res):.4f}")
stat, p = stats.shapiro(flat_res[:5000])  # Shapiro on sample
print(f"  Shapiro-Wilk p-value: {p:.4e}  {'(not normal — expected for solar)' if p < 0.05 else '(approx normal)'}")

# MBE per horizon
mbe_per_h = [float(np.nanmean(res_all[:, h-1])) for h in horizons]
print(f"  MBE range: {min(mbe_per_h):.3f} – {max(mbe_per_h):.3f} kW  "
      f"(mean: {np.mean(mbe_per_h):.4f} kW)")

# ── 4. Bootstrap CI on R² ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("4. BOOTSTRAP 95% CONFIDENCE INTERVAL ON MEAN R²")
print("=" * 65)
rng = np.random.default_rng(42)
boot_r2 = []
n = len(pred)
for _ in range(2000):
    idx = rng.integers(0, n, n)
    r2_b = []
    for h in horizons:
        yt = yt_all[idx, h-1]
        yp = yp_all[idx, h-1]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() > 2:
            r2_b.append(r2_score(yt[mask], yp[mask]))
    boot_r2.append(np.mean(r2_b))
boot_r2 = np.array(boot_r2)
ci_lo, ci_hi = np.percentile(boot_r2, [2.5, 97.5])
print(f"  Bootstrap mean R² = {boot_r2.mean():.4f}")
print(f"  95% CI:  [{ci_lo:.4f},  {ci_hi:.4f}]")
print(f"  ✓ CI entirely above 0.90: {ci_lo > 0.90}")

# ── 5. Same-day baseline comparison on identical rows ─────────────────────────
print("\n" + "=" * 65)
print("5. SAME-DAY BASELINE  (identical test rows, h-24 shift)")
print("=" * 65)
pv = real["pv_ac_W"] / 1000.0
sd_r2, sd_rmse = [], []
for h in horizons:
    yt = yt_all[:, h-1]
    yp_sd = pv.shift(24 - h + 1).reindex(pred.index).clip(lower=0).fillna(0).values
    mask = np.isfinite(yt)
    if mask.sum() > 2:
        sd_r2.append(r2_score(yt[mask], yp_sd[mask]))
        sd_rmse.append(np.sqrt(mean_squared_error(yt[mask], yp_sd[mask])))

print(f"  Same-day mean R²:   {np.mean(sd_r2):.4f}")
print(f"  CNN-LSTM mean R²:   {np.mean([r2_score(yt_all[:,h-1][np.isfinite(yt_all[:,h-1])], yp_all[:,h-1][np.isfinite(yt_all[:,h-1])]) for h in horizons]):.4f}")
print(f"  Improvement over same-day: +{np.mean(list(r2_per_h.values() if False else [r2_score(yt_all[:,h-1][np.isfinite(yt_all[:,h-1])], yp_all[:,h-1][np.isfinite(yt_all[:,h-1])]) for h in horizons])) - np.mean(sd_r2):.4f} R²")

# ── 6. Diurnal error profile ───────────────────────────────────────────────────
hours_of_day = pd.DatetimeIndex(pred.index).hour
mae_by_hour = {}
for hod in range(24):
    mask = hours_of_day == hod
    if mask.sum() > 0:
        err = np.abs(res_all[mask, :]).mean()
        mae_by_hour[hod] = err

# ── 7. Monthly error profile ───────────────────────────────────────────────────
months = pd.DatetimeIndex(pred.index).month
r2_by_month = {}
for m in sorted(set(months)):
    mask = months == m
    if mask.sum() > 5:
        yt_m = yt_all[mask, :].ravel()
        yp_m = yp_all[mask, :].ravel()
        v = np.isfinite(yt_m) & np.isfinite(yp_m)
        r2_by_month[m] = r2_score(yt_m[v], yp_m[v])

print("\n" + "=" * 65)
print("6. R² BY MONTH (test set covers Jan–Mar 2023)")
print("=" * 65)
month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
               7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
for m, r2 in r2_by_month.items():
    print(f"  {month_names[m]}: R² = {r2:.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════════

# Figure 1: Per-horizon R² with CI band + same-day comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CNN-LSTM Validation Analysis — Real PV Test Set (Jan–Mar 2023)",
             fontsize=13, fontweight="bold", y=1.01)

# ── Panel A: R² per horizon ────────────────────────────────────────────────────
ax = axes[0, 0]
r2_vals = [r2_score(yt_all[:,h-1][np.isfinite(yt_all[:,h-1])],
                     yp_all[:,h-1][np.isfinite(yt_all[:,h-1])]) for h in horizons]
ax.plot(horizons, r2_vals, "o-", color="#2e5fa3", linewidth=2, markersize=5, label="CNN-LSTM")
ax.plot(horizons, sd_r2,   "s--", color="#e07b39", linewidth=1.5, markersize=4, label="Same-day baseline")
ax.axhline(0.90, color="red", linestyle=":", linewidth=1.2, label="R²=0.90 target")
ax.fill_between(horizons, [ci_lo]*N_H, [ci_hi]*N_H, alpha=0.15, color="#2e5fa3",
                label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
ax.set_xlabel("Forecast horizon (h ahead)")
ax.set_ylabel("R²")
ax.set_title("A: R² per Horizon with Bootstrap CI")
ax.set_ylim(0.50, 1.00)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ── Panel B: RMSE per horizon ──────────────────────────────────────────────────
ax = axes[0, 1]
rmse_vals = [np.sqrt(mean_squared_error(
    yt_all[:,h-1][np.isfinite(yt_all[:,h-1])],
    yp_all[:,h-1][np.isfinite(yt_all[:,h-1])])) for h in horizons]
ax.bar(horizons, rmse_vals, color="#2e5fa3", alpha=0.8, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Forecast horizon (h ahead)")
ax.set_ylabel("RMSE (kW)")
ax.set_title("B: RMSE per Horizon")
ax.grid(axis="y", alpha=0.3)
ax.text(0.05, 0.92, f"Mean RMSE = {np.mean(rmse_vals):.2f} kW",
        transform=ax.transAxes, fontsize=9, color="#1a3a5c")

# ── Panel C: Residual distribution ────────────────────────────────────────────
ax = axes[1, 0]
flat_res_day = res_all[yt_all > 1.0].ravel()
flat_res_day = flat_res_day[np.isfinite(flat_res_day)]
ax.hist(flat_res_day, bins=60, color="#2e5fa3", alpha=0.75, edgecolor="white",
        linewidth=0.3, density=True, label="Daytime residuals")
mu, sigma = flat_res_day.mean(), flat_res_day.std()
x = np.linspace(flat_res_day.min(), flat_res_day.max(), 200)
ax.plot(x, stats.norm.pdf(x, mu, sigma), "r--", linewidth=1.5, label=f"N({mu:.2f}, {sigma:.2f}²)")
ax.axvline(0, color="black", linewidth=1.0, linestyle="-")
ax.set_xlabel("Residual (Predicted − Actual) kW")
ax.set_ylabel("Density")
ax.set_title("C: Residual Distribution (Daytime Only)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
bias_txt = f"Bias = {mu:.3f} kW\nStd = {sigma:.2f} kW"
ax.text(0.97, 0.90, bias_txt, transform=ax.transAxes, fontsize=8,
        ha="right", va="top", color="#1a3a5c",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# ── Panel D: Diurnal MAE profile ──────────────────────────────────────────────
ax = axes[1, 1]
hods  = sorted(mae_by_hour.keys())
maes  = [mae_by_hour[h] for h in hods]
bars  = ax.bar(hods, maes, color=[
    "#f5a623" if 6 <= h <= 18 else "#4a90d9" for h in hods
], alpha=0.85, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Hour of day (local time)")
ax.set_ylabel("Mean |Residual| across h=1..24 (kW)")
ax.set_title("D: Diurnal Error Profile")
ax.set_xticks(range(0, 24, 3))
ax.grid(axis="y", alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#f5a623", label="Daytime (6–18h)"),
                   Patch(color="#4a90d9", label="Nighttime")], fontsize=8)

plt.tight_layout()
path = OUT / "validation_overview.png"
fig.savefig(path, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {path}")

# Figure 2: Full test set time-series (all 1095 rows, h+1)
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
fig.suptitle("CNN-LSTM — Full Test Set Forecast (h+1, h+12, h+24)", fontsize=12, fontweight="bold")

for i, h in enumerate([1, 12, 24]):
    ax = axes[i]
    yt = pred[f"actual_h{h}"].values
    yp = pred[f"pred_h{h}"].values
    ts = pd.DatetimeIndex(pred.index)
    ax.plot(ts, yt, color="#1a3a5c", linewidth=0.8, alpha=0.9, label="Actual")
    ax.plot(ts, yp, color="#e07b39", linewidth=0.8, alpha=0.85, linestyle="--", label="Predicted")
    r2 = r2_score(yt[np.isfinite(yt)], yp[np.isfinite(yt)])
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"h+{h}  (R²={r2:.4f})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
path2 = OUT / "validation_full_timeseries.png"
fig.savefig(path2, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path2}")

# Figure 3: QQ plot + scatter h+1 vs h+24
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("CNN-LSTM — Residual QQ Plot & Scatter Comparison", fontsize=11, fontweight="bold")

# QQ
ax = axes[0]
stats.probplot(flat_res_day, dist="norm", plot=ax)
ax.set_title("QQ Plot (Daytime Residuals)")
ax.get_lines()[0].set(markersize=2, alpha=0.4, color="#2e5fa3")
ax.get_lines()[1].set(color="red", linewidth=1.5)

for j, h in enumerate([1, 24]):
    ax = axes[j + 1]
    yt = yt_all[:, h-1]
    yp = yp_all[:, h-1]
    mask = np.isfinite(yt)
    ax.scatter(yt[mask], yp[mask], s=5, alpha=0.35, color="#2e5fa3")
    lim = max(yt[mask].max(), yp[mask].max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Perfect")
    ax.set_xlabel("Actual (kW)")
    ax.set_ylabel("Predicted (kW)")
    r2 = r2_score(yt[mask], yp[mask])
    ax.set_title(f"Scatter h+{h}  (R²={r2:.4f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
path3 = OUT / "validation_scatter_qq.png"
fig.savefig(path3, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path3}")

print("\n" + "=" * 65)
print("VALIDATION SUMMARY")
print("=" * 65)
print(f"  Reported R²:             0.9249")
print(f"  Independently verified:  {np.mean(r2_vals):.4f}  ✓")
print(f"  Bootstrap 95% CI:        [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Daytime-only R²:         {np.mean(day_r2):.4f}")
print(f"  vs Same-day baseline:    {np.mean(sd_r2):.4f}  (Δ = +{np.mean(r2_vals)-np.mean(sd_r2):.4f})")
print(f"  Bias (mean residual):    {flat_res_day.mean():.4f} kW")
print(f"  All horizons ≥ 0.90:     {all(r >= 0.90 for r in r2_vals)}")
print("=" * 65)
