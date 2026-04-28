# Solcast-Based Solar PV Forecasting for Microgrid Control

A hybrid solar PV forecasting system combining site-level irradiance inputs,
physics-based PV modelling, time-series feature engineering, and machine
learning — designed to support cost-optimised microgrid energy management.

---

## Overview

This repository builds a production-quality solar PV forecasting pipeline for
the **University of Moratuwa Smartgrid Lab** microgrid.

In a microgrid, accurate forecasts of solar generation allow the energy
management system to schedule dispatchable resources — battery storage, diesel
generation, and grid import/export — at minimum cost.  This pipeline supplies
those forecasts.

**Primary external data source: Solcast**

Unlike coarse reanalysis sources (e.g. NASA POWER), Solcast provides
satellite-derived irradiance at 5-minute resolution with high site-level
accuracy, making it well-suited for:
- capturing cloud-driven generation ramps
- wet-season variability analysis
- high-resolution sub-hourly feature engineering

See also the companion repository for a NASA POWER-based pipeline:
[solar-generation-forecasting](https://github.com/NethminiRathnayake/solar-generation-forecasting)
(shared evaluation metrics and compatible feature naming).

---

## System

| Property | Value |
|---|---|
| Site | University of Moratuwa, Sri Lanka |
| Coordinates | 6.7912°N, 79.9005°E |
| Timezone | Asia/Colombo (UTC+5:30) |
| Installed capacity | ~350 kWp (estimated; 3 sub-arrays, 8 inverters) |
| Peak observed AC output | ~295 kW |
| Local data resolution | 5-minute |
| Local data coverage | April 2022 – April 2023 |
| Solcast data resolution | 5-minute |
| Solcast data coverage | January 2020 – February 2024 |

---

## Forecasting Approach

```
                    ┌──────────────────────────┐
                    │   Local PV Plant Data     │
                    │   (5-min, 2022–2023)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Solcast Weather Data    │
                    │   (5-min, 2020–2024)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Timestamp Alignment      │
                    │  (inner join, 5-min)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Data Cleaning            │
                    │  (overflow, bounds,       │
                    │   interpolation)          │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  pvlib Simulation         │
                    │  (PVWatts, Faiman temp,   │
                    │   POA via Perez)          │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  5-min → Hourly           │
                    │  Aggregation              │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Feature Engineering      │
                    │  · Time / solar position  │
                    │  · Solcast weather ratios │
                    │  · pvlib residual         │
                    │  · Lag features (h-1…168) │
                    │  · Rolling statistics     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  CNN-LSTM Forecaster       │
                    │  + Fine-tuning + Ensemble │
                    │  (MIMO, h+1…h+24)         │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  PV Generation Forecast   │
                    │  (hourly, 24h ahead)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Microgrid Optimisation   │
                    │  (cost scheduling)        │
                    └──────────────────────────┘
```

### Physics-first design

A pvlib PVWatts simulation is run on Solcast irradiance inputs to produce a
deterministic physics baseline.  The ML model then learns to correct the
gap between the physics simulation and real measurements — the **pvlib
residual** — which encodes systematic losses from soiling, shading, inverter
clipping, and wet-season cloud variability.

This approach is more physically grounded and generalisable than a black-box
regression, and is easier to interpret in a research or viva context.

---

## Repository Structure

```
solcast-solar-power-generation/
├── configs/
│   ├── site.yaml          # Site coordinates, pvlib parameters
│   ├── model.yaml         # XGBoost hyperparameters, horizon settings
│   └── pipeline.yaml      # Data paths, cleaning thresholds, feature switches
│
├── data/
│   ├── raw/               # Local PV plant CSV (not committed to Git)
│   ├── external/          # Solcast CSV files (not committed to Git)
│   ├── interim/           # Cleaned 5-min parquet (generated)
│   └── processed/         # Hourly feature matrix parquet (generated)
│
├── src/
│   ├── utils/             # config.py, logger.py
│   ├── data/              # local_pv.py, solcast.py, alignment.py
│   ├── preprocessing/     # cleaning.py
│   ├── physics/           # pvlib_model.py
│   ├── features/          # aggregation.py, time_features.py,
│   │                      # weather_features.py, physics_features.py,
│   │                      # lag_features.py
│   ├── models/            # baseline.py, gradient_boost.py, train.py
│   └── evaluation/        # metrics.py, plots.py
│
├── scripts/
│   ├── 01_prepare_data.py      # Load → clean → align → pvlib → save
│   ├── 02_build_features.py    # Aggregate → engineer features → save
│   ├── 03_train.py             # Split → train XGBoost DMS → save
│   └── 04_evaluate.py          # Predict → metrics → plots
│
├── notebooks/             # Exploratory analysis (Jupyter)
├── tests/                 # pytest unit tests
└── results/
    ├── figures/           # Generated plots
    └── metrics/           # CSV metric tables
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place data files

```
data/raw/Smartgrid lab solar PV data.csv
data/external/solcast_weather_data_2020.csv
data/external/solcast_weather_data_2021.csv
data/external/solcast_weather_data_2022.csv
data/external/solcast_weather_data_2023_end.csv
```

### 3. Run the pipeline

```bash
# Step 1: Load, align, clean, simulate
python scripts/01_prepare_data.py

# Step 2: Build the hourly feature matrix
python scripts/02_build_features.py

# Step 3: Train XGBoost DMS models
python scripts/03_train.py

# Step 4: Evaluate and generate plots
python scripts/04_evaluate.py
```

Results are written to `results/figures/` and `results/metrics/`.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary irradiance source | Solcast (satellite) | Higher spatial/temporal resolution than NASA POWER; site-specific |
| Physics model | pvlib PVWatts | Interpretable, no module database required, appropriate for system-level forecasting |
| ML strategy | Direct Multi-Step (DMS) | 24 independent models; no error propagation vs recursive; well-suited to periodic solar data |
| Feature resolution | 5-min features → hourly targets | Preserves sub-hourly cloud information in features while keeping the forecast output practical for dispatch |
| Target variable | `pv_ac_W` (AC bus reading) | Most reliable aggregate; includes all 3 sub-arrays; cross-repo compatible |
| Cross-repo naming | Identical to `solar-generation-forecasting` | Allows shared evaluation notebooks and metric comparison |

---

## Evaluation Metrics

Computed per horizon (h+1 to h+24) and as a mean:

| Metric | Unit | Description |
|---|---|---|
| RMSE | W | Root mean squared error |
| MAE | W | Mean absolute error |
| MBE | W | Mean bias error (positive = over-prediction) |
| MAPE | % | Mean absolute percentage error (daytime only) |
| nRMSE | % | RMSE normalised by mean observed power |
| R² | — | Coefficient of determination |

Baselines: persistence (ŷ = y(t)) and same-day-yesterday (ŷ = y(t−24+h)).

---

## Model Selection — Why CNN-LSTM

Multiple forecasting approaches were evaluated on the same held-out test set
(last 15% of real PV data, approximately January–March 2023):

| Model | R² (mean) | MAE | Notes |
|---|---|---|---|
| **CNN-LSTM + Fine-tuning + Ensemble** | **0.9568** | **8.49 kW** | **→ used in microgrid controller** |
| XGBoost DMS | 0.8837 | 14.43 kW | Best tabular model |
| Hybrid (XGB + LSTM) | 0.8837 | 14.43 kW | No gain over XGBoost alone |
| Same-day baseline | 0.7963 | 16.34 kW | Strong diurnal reference |
| LSTM pretrain + finetune | 0.7695 | 24.64 kW | Synthetic→real domain gap |
| LSTM direct (synthetic) | 0.7429 | 23.96 kW | Fast but lower accuracy |
| Persistence baseline | −0.968 | 77.23 kW | Lower bound |

The **CNN-LSTM** was selected for production use in the microgrid controller.
Its architecture — Conv1D feature extraction followed by a Bidirectional LSTM
and a MIMO Dense output — predicts all 24 horizons in a single forward pass and
generalises best to the site's cloud-transition months.

### What made the difference

Three improvements over the initial CNN-LSTM baseline drove the final accuracy:

1. **h+24 anchor features** — `clearness_nwp_h24 = ghi_fcast_h24 / clearsky_ghi_h24`
   and `pvlib_clearsky_h24` give the model a direct normalised signal for
   conditions at the target horizon, lifting h+24 R² from 0.87 → 0.958.

2. **Progressive fine-tuning on real validation data** — three fine-tuning passes
   with month-aware oversampling resolved November (R²: −0.28 → 0.976),
   October (0.70 → 0.961), and April (0.76 → 0.907), which were degraded by
   Sri Lanka's cloud-transition season patterns.

3. **Month-aware ensemble routing** — March 2023 falls entirely in the test set
   with no fine-tuning signal available. A hard router sends March predictions
   through the Phase 1 model (R²=0.934) and all other months through the
   fine-tuned model, preserving accuracy across all 12 months.

### Final performance

| Metric | Value |
|---|---|
| Overall R² | 0.9568  (95% CI: 0.9541 – 0.9591) |
| RMSE | 16.16 kW |
| MAE | 8.49 kW (15.1% of mean observed) |
| Daytime R² | 0.9180 |
| Horizons ≥ R² 0.955 | All 24 |
| Months ≥ R² 0.90 | 11 / 12 (May limited by 1-year data) |
| vs same-day baseline | +0.1476 R² |

### Key scripts

| Script | Purpose |
|---|---|
| `solcast_cnn_lstm.py` | Phase 1 — build and train the CNN-LSTM on synthetic data |
| `finetune_cnn_lstm.py` | Phase 2/3 — progressive fine-tuning on real val set |
| `validate_cnn_lstm.py` | Ensemble evaluation and per-horizon / monthly metrics |
| `visualize_cnn_lstm.py` | Generate all analysis figures |
| `draw_methodology.py` | Produce the methodology pipeline diagram |

---

## Extending This Repository

### Direction A: Hourly microgrid optimisation (implemented)
The current pipeline produces 24 hourly forecasts for day-ahead scheduling.

### Direction B: High-resolution 5-min forecasting
The 5-minute aligned dataset in `data/interim/` is preserved for sub-hourly
modelling.  To extend:
1. Skip the aggregation step in `02_build_features.py`.
2. Adjust lag depths to 5-min steps (e.g. lag_5min, lag_60min).
3. Train a model for 288 steps ahead (5-min × 24 hours).
4. Optionally add Himawari cloud-motion velocity (CMV) features when available.

---

## Related Repositories

- **[solar-generation-forecasting](https://github.com/NethminiRathnayake/solar-generation-forecasting)**
  — NASA POWER-based pipeline for the same site.  Compatible evaluation
  metrics and feature naming allow direct comparison between data sources.

---

## License

MIT — see [LICENSE](LICENSE).
