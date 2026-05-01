"""
serve_forecast.py
-----------------
FastAPI inference server for the CNN-LSTM PV forecasting pipeline.

At every /measurement POST the server:
  1. Appends the measurement to a rolling 24-hour buffer
  2. Fetches the next 24-hour Open-Meteo forecast (free, no API key)
  3. Builds the full 83-feature vector (same as training)
  4. Runs the CNN-LSTM model
  5. Caches the 24-horizon forecast

GET /forecast returns the most recent cached forecast.
GET /health   returns model status and buffer fill level.

RSCAD integration
-----------------
Option A (file exchange):  call POST /measurement every 15 min from a script,
                           read GET /forecast and write results to a CSV that
                           RSCAD's SCRIPT component polls.

Option B (GTNET-SKT):      wrap the GET /forecast call in a UDP bridge that
                           sends the 24 float values to the RTDS GTNET card.

Run
---
    python serve_forecast.py            # default: http://localhost:8000
    python serve_forecast.py --port 8080

Dependencies
------------
    pip install fastapi uvicorn[standard] httpx
"""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pvlib
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# ── Load model + scalers at startup ──────────────────────────────────────────
MODEL_PATH   = Path("solcast_lstm_openmeteo_model.keras")
FEAT_SCALER  = Path("scaler_features_openmeteo.pkl")
TGT_SCALER   = Path("scaler_target_openmeteo.pkl")

print("Loading model …")
import tensorflow as tf
from tensorflow import keras

model        = keras.models.load_model(str(MODEL_PATH), compile=False)
with open(FEAT_SCALER, "rb") as f: scaler_feat = pickle.load(f)
with open(TGT_SCALER,  "rb") as f: scaler_tgt  = pickle.load(f)
print(f"Model loaded  ({MODEL_PATH.name})")

# Load _CORE_FEATURES from training script
ns = {}
exec(open("cnn_lstm_solcast.py").read().split("if __name__")[0], ns)
_CORE_FEATURES = ns["_CORE_FEATURES"]

# ── Site config ───────────────────────────────────────────────────────────────
LAT, LON, ALT, TZ = 6.77, 79.88, 20, "Asia/Colombo"
PDC0, GAMMA, ETA  = 6000.0, -0.0037, 0.96
TILT, AZIMUTH     = 10, 180
SEQ_LEN           = 24

loc = pvlib.location.Location(LAT, LON, TZ, ALT)
tp  = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]

# NWP column indices to zero when not available
NWP_COLS = (
    [f"ghi_fcast_h{h}" for h in range(1, 25)] +
    [f"cloud_opacity_fcast_h{h}" for h in range(1, 25)] +
    ["ghi_fcast_mean_24h", "ghi_fcast_max_24h", "total_irradiance_ahead",
     "daylight_hours_ahead", "clearness_nwp_h24", "pvlib_clearsky_h24",
     "air_temp_fcast_h24"]
)
NWP_IDX = [i for i, c in enumerate(_CORE_FEATURES) if c in NWP_COLS]

# ── Rolling measurement buffer ────────────────────────────────────────────────
_buffer: deque[dict] = deque(maxlen=48)
_last_forecast: Optional[dict] = None

# ── Forecast history log (JSONL) ──────────────────────────────────────────────
HISTORY_LOG = Path("results/forecast_history.jsonl")
HISTORY_LOG.parent.mkdir(parents=True, exist_ok=True)

def _append_history(fc: dict) -> None:
    with open(HISTORY_LOG, "a") as f:
        f.write(json.dumps(fc) + "\n")

def _load_history() -> list[dict]:
    if not HISTORY_LOG.exists():
        return []
    records = []
    with open(HISTORY_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PV Forecast Server",
    description="CNN-LSTM 24-hour ahead solar PV forecasting — Open-Meteo powered",
    version="1.0",
)

# ── Request/response models ───────────────────────────────────────────────────
class Measurement(BaseModel):
    timestamp:         str            # ISO-8601  e.g. "2026-04-29T08:00:00+05:30"
    ghi:               Optional[float] = None   # W/m²  (if available from pyranometer)
    cloud_pct:         Optional[float] = None   # % cloud cover  (from OWM)
    air_temp:          Optional[float] = None   # °C
    relative_humidity: Optional[float] = None   # %
    surface_pressure:  Optional[float] = None   # hPa
    wind_speed:        Optional[float] = None   # m/s
    pv_ac_kw:          Optional[float] = None   # actual PV output (kW) — for lag features


# ── Open-Meteo NWP fetch ──────────────────────────────────────────────────────
def fetch_om_forecast(hours: int = 26) -> pd.DataFrame:
    """Fetch next `hours` hours of OM forecast for the site."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=shortwave_radiation,direct_normal_irradiance,diffuse_radiation,"
        "cloud_cover,temperature_2m,relative_humidity_2m,surface_pressure,"
        "dew_point_2m,wind_speed_10m"
        f"&timezone={TZ.replace('/', '%2F')}&forecast_days=2"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = r.json()["hourly"]
        fc  = pd.DataFrame(raw)
        fc["time"] = pd.to_datetime(fc["time"]).dt.tz_localize(TZ)
        fc = fc.set_index("time").rename(columns={
            "shortwave_radiation":      "ghi",
            "direct_normal_irradiance": "dni",
            "diffuse_radiation":        "dhi",
            "cloud_cover":              "cloud_opacity",
            "temperature_2m":           "air_temp",
            "relative_humidity_2m":     "relative_humidity",
            "surface_pressure":         "surface_pressure",
            "dew_point_2m":             "dewpoint_temp",
            "wind_speed_10m":           "wind_speed",
        })
        return fc
    except Exception as e:
        print(f"OM forecast fetch failed: {e}")
        return pd.DataFrame()


def _build_row_features(ts: pd.Timestamp, row: dict, fc_df: pd.DataFrame) -> pd.Series:
    """Build one hour's worth of model features from observation + OM forecast."""
    ts_tz   = ts.tz_localize(TZ) if ts.tzinfo is None else ts.tz_convert(TZ)
    cs      = loc.get_clearsky(pd.DatetimeIndex([ts_tz]), model="ineichen")
    solp    = loc.get_solarposition(pd.DatetimeIndex([ts_tz]))

    ghi     = float(row.get("ghi")    or cs["ghi"].iloc[0])
    dni     = float(row.get("dni")    or 0.0)
    dhi     = float(row.get("dhi")    or cs["dhi"].iloc[0])
    cloud   = float(row.get("cloud_opacity", 50))
    if ghi == cs["ghi"].iloc[0]:   # no measured GHI → attenuate clearsky
        cf  = cloud / 100.0
        ghi = float(cs["ghi"].iloc[0] * (1 - 0.75 * cf**3.4))
        dhi = float(cs["dhi"].iloc[0] * (1 - 0.75 * cf**3.4))
        cos_z = float(np.cos(np.radians(solp["zenith"].iloc[0])).clip(0.087))
        dni   = max(float((ghi - dhi) / cos_z), 0.0) if float(solp["zenith"].iloc[0]) < 87 else 0.0

    aoi_deg = float(pvlib.irradiance.aoi(TILT, AZIMUTH,
                    float(solp["apparent_zenith"].iloc[0]),
                    float(solp["azimuth"].iloc[0])))
    poa_d = float(max(dni * np.cos(np.radians(aoi_deg)), 0))
    poa_g = poa_d + dhi*(1+np.cos(np.radians(TILT)))/2 + ghi*0.25*(1-np.cos(np.radians(TILT)))/2
    t_air = float(row.get("air_temp", 30))
    w_spd = float(row.get("wind_speed", 3))
    # Use numpy arrays of length 1 so pvlib returns pandas-compatible types
    poa_arr    = np.array([poa_g])
    tair_arr   = np.array([t_air])
    wspd_arr   = np.array([w_spd])
    cell_t_arr = pvlib.temperature.sapm_cell(poa_arr, tair_arr, wspd_arr, **tp)
    dc_arr     = pvlib.pvsystem.pvwatts_dc(poa_arr, cell_t_arr, pdc0=PDC0, gamma_pdc=GAMMA)
    ac_arr     = pvlib.inverter.pvwatts(dc_arr, pdc0=PDC0, eta_inv_nom=ETA)
    ac         = float(np.clip(ac_arr[0], 0, None))

    feats = {
        "ghi": ghi, "dni": dni, "dhi": dhi,
        "cloud_opacity": cloud,
        "ghi_clearsky_ratio": float(ghi / (cs["ghi"].iloc[0] + 1e-6)),
        "clearness_index_hourly": float(ghi / (cs["ghi"].iloc[0] + 1e-6)),
        "clearsky_ghi": float(cs["ghi"].iloc[0]),
        "air_temp": t_air,
        "relative_humidity": float(row.get("relative_humidity", 70)),
        "surface_pressure":  float(row.get("surface_pressure", 1010)),
        "dewpoint_temp":     t_air - ((100 - float(row.get("relative_humidity", 70))) / 5),
        "pvlib_ac_W": ac,
        "cos_solar_zenith":    max(float(np.cos(np.radians(float(solp["zenith"].iloc[0])))), 0.0),
        "solar_elevation_deg": float(solp["elevation"].iloc[0]),
        "hour_sin":  float(np.sin(2*np.pi*ts_tz.hour/24)),
        "hour_cos":  float(np.cos(2*np.pi*ts_tz.hour/24)),
        "month_sin": float(np.sin(2*np.pi*ts_tz.month/12)),
        "month_cos": float(np.cos(2*np.pi*ts_tz.month/12)),
        "doy_sin":   float(np.sin(2*np.pi*ts_tz.dayofyear/365)),
        "doy_cos":   float(np.cos(2*np.pi*ts_tz.dayofyear/365)),
        "monsoon_sw":     1.0 if 5 <= ts_tz.month <= 9  else 0.0,
        "monsoon_ne":     1.0 if ts_tz.month in (12,1,2) else 0.0,
        "monsoon_inter1": 1.0 if ts_tz.month in (3,4)   else 0.0,
        "monsoon_inter2": 1.0 if ts_tz.month == 10      else 0.0,
    }

    # NWP forecast columns from OM (h+1..h+24)
    if not fc_df.empty:
        future_idx = pd.date_range(ts_tz + pd.Timedelta(hours=1),
                                   periods=24, freq="h", tz=TZ)
        for h, fts in enumerate(future_idx, 1):
            if fts in fc_df.index:
                feats[f"ghi_fcast_h{h}"]           = float(fc_df.at[fts, "ghi"])
                feats[f"cloud_opacity_fcast_h{h}"] = float(fc_df.at[fts, "cloud_opacity"])
            else:
                feats[f"ghi_fcast_h{h}"]           = 0.0
                feats[f"cloud_opacity_fcast_h{h}"] = 50.0
        gc = [feats[f"ghi_fcast_h{h}"] for h in range(1,25)]
        feats["ghi_fcast_mean_24h"]     = float(np.mean(gc))
        feats["ghi_fcast_max_24h"]      = float(np.max(gc))
        feats["total_irradiance_ahead"] = float(np.sum(gc))
        feats["daylight_hours_ahead"]   = float(sum(1 for v in gc if v > 10))
        feats["clearness_nwp_h24"]      = float(feats["ghi_fcast_h24"] /
                                                 (feats["clearsky_ghi"] + 1e-6))
        h24_ts = ts_tz + pd.Timedelta(hours=24)
        feats["pvlib_clearsky_h24"]  = feats["pvlib_ac_W"]  # proxy
        feats["air_temp_fcast_h24"]  = float(fc_df.at[h24_ts,"air_temp"]) \
                                       if h24_ts in fc_df.index else t_air
    else:
        for c in NWP_COLS: feats[c] = 0.0

    return pd.Series(feats)


def _run_inference(fc_df: pd.DataFrame) -> dict:
    """Build 24h lookback from buffer + OM forecast, run model, return predictions."""
    if len(_buffer) < SEQ_LEN:
        raise ValueError(f"Buffer has {len(_buffer)}/{SEQ_LEN} hours — need more data")

    rows = list(_buffer)[-SEQ_LEN:]
    feature_rows = []
    for r in rows:
        ts   = pd.Timestamp(r["timestamp"])
        feat = _build_row_features(ts, r, fc_df)
        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)
    for c in _CORE_FEATURES:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    X_raw    = feat_df[_CORE_FEATURES].fillna(0).values.astype(np.float32)
    X_scaled = scaler_feat.transform(X_raw)
    X_input  = X_scaled[np.newaxis, :, :]

    y_norm   = model.predict(X_input, verbose=0)
    y_kw     = np.clip(
        scaler_tgt.inverse_transform(y_norm.reshape(-1,1)).reshape(24), 0, None)

    last_ts  = pd.Timestamp(rows[-1]["timestamp"])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(TZ)
    else:
        last_ts = last_ts.tz_convert(TZ)

    horizons = {}
    for h in range(24):
        target_ts = last_ts + pd.Timedelta(hours=h+1)
        horizons[f"h{h+1:02d}"] = {
            "datetime": target_ts.isoformat(),
            "predicted_kW": round(float(y_kw[h]), 3),
        }

    return {
        "generated_at":  datetime.now().isoformat(),
        "lookback_from": pd.Timestamp(rows[0]["timestamp"]).isoformat(),
        "lookback_to":   last_ts.isoformat(),
        "buffer_hours":  len(_buffer),
        "horizon_kw":    {k: v["predicted_kW"] for k, v in horizons.items()},
        "horizons":      horizons,
        "daily_total_kWh": round(float(y_kw.sum()), 2),
        "peak_kW":         round(float(y_kw.max()), 2),
        "peak_hour":       f"h{int(y_kw.argmax())+1:02d}",
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/measurement", summary="Submit a new hourly measurement")
async def post_measurement(m: Measurement):
    """
    Append one hourly measurement to the rolling buffer.
    Triggers a fresh 24-hour forecast if the buffer has ≥ 24 hours.
    """
    global _last_forecast
    record = m.model_dump()
    _buffer.append(record)

    if len(_buffer) >= SEQ_LEN:
        fc_df = fetch_om_forecast()
        try:
            _last_forecast = _run_inference(fc_df)
            _append_history(_last_forecast)
            return {"status": "ok", "forecast_updated": True,
                    "buffer_hours": len(_buffer),
                    "peak_kW": _last_forecast["peak_kW"],
                    "daily_total_kWh": _last_forecast["daily_total_kWh"]}
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"status": "error", "message": str(e), "forecast_updated": False}

    return {"status": "ok", "forecast_updated": False,
            "buffer_hours": len(_buffer),
            "message": f"Need {SEQ_LEN - len(_buffer)} more hours to forecast"}


@app.get("/forecast", summary="Get the latest 24-hour forecast")
async def get_forecast():
    """Returns the most recent 24-horizon PV forecast (kW per hour)."""
    if _last_forecast is None:
        raise HTTPException(status_code=503,
                            detail="No forecast available yet — submit ≥ 24 measurements first")
    return JSONResponse(_last_forecast)


@app.get("/forecast/csv", summary="Get forecast as CSV text")
async def get_forecast_csv():
    """Returns the forecast as plain CSV (for RSCAD SCRIPT component file polling)."""
    if _last_forecast is None:
        raise HTTPException(status_code=503, detail="No forecast available yet")
    lines = ["horizon,datetime,predicted_kW"]
    for h, info in _last_forecast["horizons"].items():
        lines.append(f"{h},{info['datetime']},{info['predicted_kW']}")
    return JSONResponse({"csv": "\n".join(lines)})


@app.get("/health", summary="Server health and model status")
async def health():
    return {
        "status":       "ok",
        "model":        MODEL_PATH.name,
        "buffer_hours": len(_buffer),
        "buffer_full":  len(_buffer) >= SEQ_LEN,
        "forecast_ready": _last_forecast is not None,
        "last_forecast":  _last_forecast["generated_at"] if _last_forecast else None,
        "site":         {"lat": LAT, "lon": LON, "capacity_kW": PDC0/1000},
    }


@app.get("/history", summary="Get all stored forecast history")
async def get_history():
    """Returns all stored forecast records for dashboard charting."""
    records = _load_history()
    # Also include current forecast if not yet persisted
    if _last_forecast and (not records or
            records[-1].get("generated_at") != _last_forecast.get("generated_at")):
        records.append(_last_forecast)
    return JSONResponse({"count": len(records), "records": records})


@app.get("/history/daily", summary="Daily aggregated forecast history")
async def get_daily_history():
    """Returns day-level aggregates: total kWh, peak kW, mean kW per day."""
    records = _load_history()
    if _last_forecast:
        records.append(_last_forecast)

    days: dict[str, dict] = {}
    for rec in records:
        # Use the lookback_to date as the forecast date
        try:
            date_str = rec["lookback_to"][:10]
        except Exception:
            continue
        kw_values = list(rec["horizon_kw"].values())
        entry = {
            "date":        date_str,
            "total_kWh":   round(sum(kw_values), 2),
            "peak_kW":     round(max(kw_values), 2),
            "mean_kW":     round(sum(kw_values) / len(kw_values), 2),
            "hourly_kw":   kw_values,
            "generated_at": rec["generated_at"],
        }
        # Keep most recent forecast per day
        if date_str not in days or rec["generated_at"] > days[date_str]["generated_at"]:
            days[date_str] = entry

    sorted_days = sorted(days.values(), key=lambda x: x["date"])
    return JSONResponse({"days": sorted_days})


@app.get("/", include_in_schema=False)
async def root():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>PV Forecast Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --navy:#1a3a5c; --blue:#2e5fa3; --orange:#e65100;
    --green:#2e7d32; --bg:#f0f4f8; --card:#ffffff;
    --text:#1a1a2e; --muted:#6b7280; --border:#e2e8f0;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text)}
  header{background:var(--navy);color:#fff;padding:1rem 2rem;display:flex;
         justify-content:space-between;align-items:center;box-shadow:0 2px 8px rgba(0,0,0,.3)}
  header h1{font-size:1.25rem;font-weight:700;letter-spacing:.5px}
  header .meta{font-size:.8rem;opacity:.8;text-align:right}
  .status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;
              background:#4ade80;margin-right:6px;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .tabs{display:flex;gap:0;background:var(--navy);padding:0 2rem}
  .tab{padding:.6rem 1.4rem;cursor:pointer;color:rgba(255,255,255,.65);
       border-bottom:3px solid transparent;font-size:.9rem;font-weight:500;
       transition:all .2s}
  .tab.active,.tab:hover{color:#fff;border-bottom-color:#60a5fa}
  .panel{display:none;padding:1.5rem 2rem;animation:fadeIn .2s}
  .panel.active{display:block}
  @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
  .kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:1.5rem}
  .kpi{background:var(--card);border-radius:10px;padding:1rem 1.2rem;
       box-shadow:0 1px 4px rgba(0,0,0,.08);border-left:4px solid var(--blue)}
  .kpi .val{font-size:1.6rem;font-weight:700;color:var(--navy)}
  .kpi .lbl{font-size:.75rem;color:var(--muted);margin-top:.2rem;text-transform:uppercase;letter-spacing:.5px}
  .card{background:var(--card);border-radius:12px;padding:1.2rem;
        box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:1.5rem}
  .card h3{font-size:.95rem;font-weight:600;color:var(--navy);margin-bottom:.8rem}
  .chart-wrap{position:relative;height:280px}
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  th{background:var(--navy);color:#fff;padding:.55rem .8rem;text-align:left;font-weight:600}
  td{padding:.45rem .8rem;border-bottom:1px solid var(--border)}
  tr:nth-child(even) td{background:#f8fafc}
  tr:hover td{background:#eff6ff}
  .badge{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.75rem;font-weight:600}
  .badge-green{background:#dcfce7;color:#166534}
  .badge-orange{background:#fff7ed;color:#9a3412}
  .badge-blue{background:#dbeafe;color:#1e40af}
  #nodata{text-align:center;padding:3rem;color:var(--muted);font-size:.95rem}
  .loading{text-align:center;padding:2rem;color:var(--muted)}
</style>
</head>
<body>

<header>
  <div>
    <h1><span class="status-dot"></span>PV Forecast Dashboard</h1>
    <div style="font-size:.8rem;opacity:.7;margin-top:.2rem">University of Moratuwa · 6 kWp · Sri Lanka</div>
  </div>
  <div class="meta" id="header-meta">Loading…</div>
</header>

<div class="tabs">
  <div class="tab active" onclick="switchTab('daily')">Daily</div>
  <div class="tab" onclick="switchTab('weekly')">Weekly</div>
  <div class="tab" onclick="switchTab('monthly')">Monthly</div>
</div>

<!-- ── DAILY ── -->
<div id="panel-daily" class="panel active">
  <div class="kpi-row">
    <div class="kpi"><div class="val" id="kpi-total">—</div><div class="lbl">Daily Total (kWh)</div></div>
    <div class="kpi" style="border-color:var(--orange)"><div class="val" id="kpi-peak">—</div><div class="lbl">Peak Output (kW)</div></div>
    <div class="kpi" style="border-color:var(--green)"><div class="val" id="kpi-peak-hour">—</div><div class="lbl">Peak Hour</div></div>
    <div class="kpi" style="border-color:#7c3aed"><div class="val" id="kpi-updated">—</div><div class="lbl">Last Updated</div></div>
  </div>
  <div class="card">
    <h3>24-Hour Ahead Forecast</h3>
    <div class="chart-wrap"><canvas id="chart-daily"></canvas></div>
  </div>
  <div class="card">
    <h3>Hourly Breakdown</h3>
    <table id="table-daily">
      <thead><tr><th>Horizon</th><th>Local Time</th><th>Predicted (kW)</th><th>% of Capacity</th><th>Level</th></tr></thead>
      <tbody id="tbody-daily"></tbody>
    </table>
  </div>
</div>

<!-- ── WEEKLY ── -->
<div id="panel-weekly" class="panel">
  <div class="kpi-row">
    <div class="kpi"><div class="val" id="kpi-w-total">—</div><div class="lbl">7-Day Total (kWh)</div></div>
    <div class="kpi" style="border-color:var(--orange)"><div class="val" id="kpi-w-avg">—</div><div class="lbl">Avg Daily (kWh)</div></div>
    <div class="kpi" style="border-color:var(--green)"><div class="val" id="kpi-w-best">—</div><div class="lbl">Best Day (kWh)</div></div>
    <div class="kpi" style="border-color:#7c3aed"><div class="val" id="kpi-w-days">—</div><div class="lbl">Days with Data</div></div>
  </div>
  <div class="card">
    <h3>Daily Generation — Last 7 Days</h3>
    <div class="chart-wrap"><canvas id="chart-weekly"></canvas></div>
  </div>
  <div class="card">
    <h3>Daily Summary</h3>
    <table>
      <thead><tr><th>Date</th><th>Total (kWh)</th><th>Peak (kW)</th><th>Avg (kW)</th></tr></thead>
      <tbody id="tbody-weekly"></tbody>
    </table>
  </div>
</div>

<!-- ── MONTHLY ── -->
<div id="panel-monthly" class="panel">
  <div class="kpi-row">
    <div class="kpi"><div class="val" id="kpi-m-total">—</div><div class="lbl">30-Day Total (kWh)</div></div>
    <div class="kpi" style="border-color:var(--orange)"><div class="val" id="kpi-m-avg">—</div><div class="lbl">Avg Daily (kWh)</div></div>
    <div class="kpi" style="border-color:var(--green)"><div class="val" id="kpi-m-best-day">—</div><div class="lbl">Best Day</div></div>
    <div class="kpi" style="border-color:#7c3aed"><div class="val" id="kpi-m-days">—</div><div class="lbl">Days with Data</div></div>
  </div>
  <div class="card">
    <h3>Daily Generation — Last 30 Days</h3>
    <div class="chart-wrap" style="height:320px"><canvas id="chart-monthly"></canvas></div>
  </div>
</div>

<script>
let charts = {};

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ['daily','weekly','monthly'][i]===tab));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-'+tab).classList.add('active');
}

function makeChart(id, type, labels, datasets, opts={}) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id).getContext('2d');
  charts[id] = new Chart(ctx, {
    type, data: {labels, datasets},
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: datasets.length > 1 },
                 tooltip: { callbacks: { label: c => ` ${c.parsed.y.toFixed(2)} ${opts.unit||''}` }}},
      scales: {
        x: { grid: { color:'#f0f4f8' }, ticks: { font:{size:11} } },
        y: { beginAtZero:true, grid:{ color:'#e2e8f0' },
             ticks:{ font:{size:11}, callback: v => v.toFixed(1) },
             title:{ display:!!opts.yLabel, text:opts.yLabel, font:{size:11} }}
      },
      ...opts.extra
    }
  });
}

function levelBadge(kw) {
  const pct = kw / 6 * 100;
  if (pct >= 70) return '<span class="badge badge-green">High</span>';
  if (pct >= 30) return '<span class="badge badge-blue">Medium</span>';
  if (kw > 0.05) return '<span class="badge badge-orange">Low</span>';
  return '<span class="badge" style="background:#f1f5f9;color:#64748b">Night</span>';
}

async function loadDaily() {
  const resp = await fetch('/forecast').catch(() => null);
  if (!resp || !resp.ok) {
    document.getElementById('panel-daily').innerHTML =
      '<div id="nodata">No forecast available yet.<br>Submit at least 24 measurements to generate a forecast.</div>';
    return;
  }
  const fc = await resp.json();
  const kws = Object.values(fc.horizon_kw);
  const labels = Object.keys(fc.horizon_kw).map(h => 'h+'+parseInt(h.slice(1)));

  document.getElementById('kpi-total').textContent = fc.daily_total_kWh.toFixed(1);
  document.getElementById('kpi-peak').textContent  = fc.peak_kW.toFixed(2) + ' kW';
  document.getElementById('kpi-peak-hour').textContent = fc.peak_hour.replace('h','h+');
  document.getElementById('kpi-updated').textContent = fc.generated_at.slice(11,16);
  document.getElementById('header-meta').innerHTML =
    'Last forecast: '+fc.generated_at.slice(0,19).replace('T',' ')+'<br>'+
    'Daily total: '+fc.daily_total_kWh+' kWh · Peak: '+fc.peak_kW+' kW';

  const gradColors = kws.map(v => `rgba(46,95,163,${Math.max(0.15, v/6)})`);
  makeChart('chart-daily','bar',labels,[{
    label:'Predicted kW', data:kws,
    backgroundColor: gradColors,
    borderColor:'rgba(46,95,163,0.8)', borderWidth:1, borderRadius:4,
  }],{unit:'kW', yLabel:'Power (kW)'});

  const tbody = document.getElementById('tbody-daily');
  tbody.innerHTML = '';
  const base = new Date(fc.lookback_to);
  kws.forEach((kw,i) => {
    const t = new Date(base.getTime() + (i+1)*3600000);
    const hh = String(t.getHours()).padStart(2,'0')+':00';
    tbody.innerHTML += `<tr>
      <td><b>h+${i+1}</b></td>
      <td>${hh}</td>
      <td><b>${kw.toFixed(3)}</b></td>
      <td>${(kw/6*100).toFixed(1)}%</td>
      <td>${levelBadge(kw)}</td>
    </tr>`;
  });
}

async function loadWeeklyMonthly() {
  const resp = await fetch('/history/daily').catch(() => null);
  if (!resp || !resp.ok) return;
  const data = await resp.json();
  const days = data.days || [];

  // Weekly: last 7 days
  const w7 = days.slice(-7);
  const wTotal = w7.reduce((s,d) => s+d.total_kWh,0);
  document.getElementById('kpi-w-total').textContent = wTotal.toFixed(1);
  document.getElementById('kpi-w-avg').textContent   = w7.length ? (wTotal/w7.length).toFixed(1) : '—';
  const wBest = w7.reduce((b,d) => d.total_kWh>b?d.total_kWh:b, 0);
  document.getElementById('kpi-w-best').textContent = wBest.toFixed(1);
  document.getElementById('kpi-w-days').textContent = w7.length;

  const wLabels = w7.map(d => d.date.slice(5));
  const wData   = w7.map(d => d.total_kWh);
  makeChart('chart-weekly','bar',wLabels,[{
    label:'kWh', data:wData,
    backgroundColor:'rgba(46,95,163,0.75)',
    borderColor:'rgba(26,58,92,0.9)', borderWidth:1, borderRadius:5,
  }],{unit:'kWh', yLabel:'Daily Generation (kWh)'});

  const tbody = document.getElementById('tbody-weekly');
  tbody.innerHTML = '';
  [...w7].reverse().forEach(d => {
    tbody.innerHTML += `<tr>
      <td>${d.date}</td>
      <td><b>${d.total_kWh.toFixed(1)}</b></td>
      <td>${d.peak_kW.toFixed(2)}</td>
      <td>${d.mean_kW.toFixed(2)}</td>
    </tr>`;
  });

  // Monthly: last 30 days
  const m30 = days.slice(-30);
  const mTotal = m30.reduce((s,d) => s+d.total_kWh,0);
  document.getElementById('kpi-m-total').textContent = mTotal.toFixed(1);
  document.getElementById('kpi-m-avg').textContent   = m30.length ? (mTotal/m30.length).toFixed(1) : '—';
  document.getElementById('kpi-m-days').textContent  = m30.length;
  const mBest = m30.reduce((b,d) => d.total_kWh>b.total_kWh?d:b, {total_kWh:0,date:'-'});
  document.getElementById('kpi-m-best-day').textContent = mBest.date.slice(5)+' ('+mBest.total_kWh.toFixed(1)+')';

  const mLabels = m30.map(d => d.date.slice(5));
  const mData   = m30.map(d => d.total_kWh);
  const mColors = mData.map(v => v >= 40 ? 'rgba(46,125,50,0.75)' :
                                  v >= 25 ? 'rgba(46,95,163,0.75)' :
                                            'rgba(230,81,0,0.65)');
  makeChart('chart-monthly','bar',mLabels,[{
    label:'kWh', data:mData,
    backgroundColor:mColors,
    borderColor:'rgba(0,0,0,0.1)', borderWidth:1, borderRadius:4,
  }],{unit:'kWh', yLabel:'Daily Generation (kWh)',
      extra:{plugins:{legend:{display:false}}}});
}

// Load all on start
loadDaily();
loadWeeklyMonthly();
// Refresh daily forecast every 5 minutes
setInterval(loadDaily, 300000);
setInterval(loadWeeklyMonthly, 300000);
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
