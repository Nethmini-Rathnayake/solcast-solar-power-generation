"""
build_openmeteo_features.py
---------------------------
Fetch Open-Meteo ERA5 historical data for the PV_Logger site and build the
full 83-feature matrix used by the CNN-LSTM — replacing Solcast as the
weather data source.

Steps
-----
1. Fetch hourly historical (Jan 2020 – yesterday) from Open-Meteo archive API
2. Derive irradiance ratios, pvlib AC output, time/monsoon features
3. Add NWP forecast columns (shift current GHI/cloud forward h+1..h+24)
4. Add lag features (pv_lag24, pv_lag48, ghi_lag24, clearness_lag24)
   using PV_Logger actuals where available, 2022 proxy otherwise
5. Save to data/processed/feature_matrix_openmeteo.parquet

Run
---
    python build_openmeteo_features.py
"""

from __future__ import annotations

import os
import sys
import warnings
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

warnings.filterwarnings("ignore")

# ── Site config ───────────────────────────────────────────────────────────────
LAT       = 6.77
LON       = 79.88
ALT       = 20          # metres (from site.yaml)
TZ        = "Asia/Colombo"
PDC0      = 6000.0      # W — 6 kWp system
GAMMA     = -0.0037
ETA_INV   = 0.96
TILT      = 10
AZIMUTH   = 180

OUT_PATH  = Path("data/processed/feature_matrix_openmeteo.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

loc = pvlib.location.Location(LAT, LON, TZ, ALT)
tp  = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

# ── Step 1: Fetch Open-Meteo historical ───────────────────────────────────────
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE  = "2020-01-01"
END_DATE    = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

HOURLY_VARS = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "dew_point_2m",
    "wind_speed_10m",
]

print(f"Fetching Open-Meteo historical: {START_DATE} → {END_DATE} …")
params = {
    "latitude":   LAT,
    "longitude":  LON,
    "start_date": START_DATE,
    "end_date":   END_DATE,
    "hourly":     ",".join(HOURLY_VARS),
    "timezone":   TZ,
}
r = requests.get(ARCHIVE_URL, params=params, timeout=120)
r.raise_for_status()
raw = r.json()["hourly"]

df = pd.DataFrame(raw)
df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(TZ)
df = df.set_index("time").sort_index()

# Rename to CNN-LSTM column names (must match _CORE_FEATURES exactly)
df = df.rename(columns={
    "shortwave_radiation":      "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation":        "dhi",
    "cloud_cover":              "cloud_opacity",
    "temperature_2m":           "air_temp",
    "relative_humidity_2m":     "relative_humidity",
    "dew_point_2m":             "dewpoint_temp",
    "wind_speed_10m":           "wind_speed",
})
df[["ghi","dni","dhi"]] = df[["ghi","dni","dhi"]].clip(lower=0)
print(f"Downloaded {len(df)} hourly rows  ({df.index.min().date()} → {df.index.max().date()})")

# ── Step 2: Clearsky, pvlib, derived features ─────────────────────────────────
print("Deriving pvlib and irradiance features …")

# Clearsky
idx_tz = df.index
cs   = loc.get_clearsky(idx_tz, model="ineichen")
solp = loc.get_solarposition(idx_tz)

df["clearsky_ghi"]           = cs["ghi"].values
df["cos_solar_zenith"]       = np.cos(np.radians(solp["zenith"].values)).clip(0)
df["solar_elevation_deg"]    = solp["elevation"].values
df["clearness_index_hourly"] = (df["ghi"] / (cs["ghi"].values + 1e-6)).clip(0, 1.5)
df["ghi_clearsky_ratio"]     = df["clearness_index_hourly"]

# pvlib AC output
poa_d = np.maximum(
    df["dni"].values * np.cos(np.radians(
        pvlib.irradiance.aoi(TILT, AZIMUTH,
                             solp["apparent_zenith"].values,
                             solp["azimuth"].values))), 0)
poa_g = (poa_d
         + df["dhi"].values * (1 + np.cos(np.radians(TILT))) / 2
         + df["ghi"].values * 0.25 * (1 - np.cos(np.radians(TILT))) / 2)
cell_t = pvlib.temperature.sapm_cell(poa_g, df["air_temp"].values, df["wind_speed"].values, **tp)
dc  = pvlib.pvsystem.pvwatts_dc(poa_g, cell_t, pdc0=PDC0, gamma_pdc=GAMMA)
ac  = pvlib.inverter.pvwatts(dc, pdc0=PDC0, eta_inv_nom=ETA_INV)
df["pvlib_ac_W"] = np.array(ac).clip(0)

# Time cyclicals
df["hour_sin"]  = np.sin(2*np.pi*idx_tz.hour/24)
df["hour_cos"]  = np.cos(2*np.pi*idx_tz.hour/24)
df["month_sin"] = np.sin(2*np.pi*idx_tz.month/12)
df["month_cos"] = np.cos(2*np.pi*idx_tz.month/12)
df["doy_sin"]   = np.sin(2*np.pi*idx_tz.dayofyear/365)
df["doy_cos"]   = np.cos(2*np.pi*idx_tz.dayofyear/365)

# Monsoon regime (Sri Lanka)
mo = idx_tz.month
df["monsoon_sw"]     = np.where((mo>=5) & (mo<=9), 1, 0).astype(float)
df["monsoon_ne"]     = np.where((mo==12) | (mo<=2), 1, 0).astype(float)
df["monsoon_inter1"] = np.where((mo==3) | (mo==4), 1, 0).astype(float)
df["monsoon_inter2"] = np.where(mo==10, 1, 0).astype(float)

# ── Step 3: NWP forecast columns (shift current obs forward) ──────────────────
print("Building NWP forecast columns …")
n   = len(df)
ghi = df["ghi"].values
cld = df["cloud_opacity"].values

for h in range(1, 25):
    fg = np.zeros(n); fc = np.zeros(n)
    fg[:n-h] = ghi[h:]
    fc[:n-h] = cld[h:]
    df[f"ghi_fcast_h{h}"]           = fg
    df[f"cloud_opacity_fcast_h{h}"] = fc

gcols = [f"ghi_fcast_h{h}" for h in range(1, 25)]
df["ghi_fcast_mean_24h"]     = df[gcols].mean(axis=1).values
df["ghi_fcast_max_24h"]      = df[gcols].max(axis=1).values
df["total_irradiance_ahead"] = df[gcols].sum(axis=1).values
df["daylight_hours_ahead"]   = (df[gcols] > 10).sum(axis=1).values.astype(float)

# h+24 anchor features
df["clearness_nwp_h24"]  = (df["ghi_fcast_h24"] / (df["clearsky_ghi"] + 1e-6)).clip(0, 1.5)
pvh24 = np.zeros(n); pvh24[:n-24] = df["pvlib_ac_W"].values[24:]
df["pvlib_clearsky_h24"] = pvh24
ath24 = np.zeros(n); ath24[:n-24] = df["air_temp"].values[24:]
df["air_temp_fcast_h24"] = ath24

# ── Step 4: Lag features ──────────────────────────────────────────────────────
print("Building lag features …")

# Use PV_Logger actuals where available, else use pvlib_ac_W as proxy
pv_actual = pd.Series(dtype=float)
try:
    act = pd.read_csv("data/raw/PV_Logger - samples.csv",
                      usecols=["sample_time_local", "site_power_w"])
    act["ts"]   = pd.to_datetime(act["sample_time_local"]).dt.tz_localize(None)
    act["hour"] = act["ts"].dt.floor("h").dt.tz_localize(TZ)
    act["kw"]   = act["site_power_w"] / 1000.0
    pv_actual   = act.groupby("hour")["kw"].mean()
    print(f"  PV_Logger actuals loaded: {len(pv_actual)} hours")
except Exception as e:
    print(f"  PV_Logger actuals not available ({e}), using pvlib proxy for lags")

# pv_lag24 / pv_lag48 — use actual where available, pvlib otherwise
pv_series = df["pvlib_ac_W"] / 1000.0   # proxy
for ts, val in pv_actual.items():
    if ts in pv_series.index:
        pv_series.at[ts] = val

df["pv_lag24"] = pv_series.shift(24).fillna(0).values
df["pv_lag48"] = pv_series.shift(48).fillna(0).values
df["ghi_lag24"] = df["ghi"].shift(24).fillna(0).values
df["clearness_lag24"] = (df["pv_lag24"] / (df["pvlib_ac_W"]/1000.0 + 1e-6)).clip(0, 1.5)

# ── Step 5: Save ──────────────────────────────────────────────────────────────
# Get _CORE_FEATURES by executing the module in an isolated namespace
_ns = {}
src = open("cnn_lstm_solcast.py").read().split("if __name__")[0]
exec(src, _ns)
core_feats = _ns["_CORE_FEATURES"]

# Also keep pv_ac_W equivalent for the target
df["pv_ac_W"] = pv_series.values * 1000.0   # W — actual where available, pvlib proxy elsewhere

for c in core_feats:
    if c not in df.columns:
        df[c] = 0.0

save_cols = [c for c in core_feats if c in df.columns] + ["pv_ac_W", "pvlib_ac_W"]
save_cols = list(dict.fromkeys(save_cols))  # deduplicate

df_out = df[save_cols].copy()
df_out.index = df_out.index.tz_localize(None)   # strip tz for parquet compat
df_out.to_parquet(OUT_PATH)

print(f"\nSaved {len(df_out)} rows × {len(df_out.columns)} cols → {OUT_PATH}")
print(f"Date range: {df_out.index.min().date()} → {df_out.index.max().date()}")
print(f"GHI range:  {df_out['ghi'].min():.1f} – {df_out['ghi'].max():.1f} W/m²")
print(f"pvlib_ac_W: {df_out['pvlib_ac_W'].min():.1f} – {df_out['pvlib_ac_W'].max():.1f} W")
print(f"pv_ac_W (actual where avail): non-zero hours = {(df_out['pv_ac_W']>0).sum()}")
