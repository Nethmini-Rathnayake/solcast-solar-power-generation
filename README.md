# Solcast-Based Solar PV Forecasting for Microgrid Control

## Overview

This project develops a high-resolution solar PV forecasting pipeline for a cost-optimized microgrid controller. The forecasting module is designed to predict future PV generation so that a microgrid energy management system can schedule available resources such as battery storage, diesel generation, and grid power at minimum operating cost.

Unlike coarse satellite/reanalysis sources, this version of the project uses **Solcast** as the primary weather and irradiance data source. Solcast provides higher spatial and temporal resolution, making it more suitable for capturing site-level irradiance variability, cloud-driven ramps, and wet-season fluctuations.

The forecasting pipeline combines:

- **Local measured PV plant data**
- **Solcast irradiance and meteorological data**
- **Physics-based PV simulation using pvlib**
- **Feature engineering for time series forecasting**
- **Machine learning models for multi-horizon prediction**

---

## Motivation

In a microgrid, solar generation is highly variable and depends strongly on weather conditions. To optimize cost in real time, the controller requires accurate forecasts of PV output.

The original version of the project used NASA POWER as the primary external data source due to its long historical coverage. However, NASA POWER has relatively coarse spatial resolution and is less effective at capturing local cloud motion and rapid irradiance changes.

Since this project now has access to more than three years of Solcast data, Solcast becomes a stronger primary source because it offers:

- better site-level relevance
- higher temporal resolution
- improved cloud-sensitive irradiance estimation
- better representation of local wet-season variability

---

## Objectives

The main objectives of this repository are:

1. Build a Solcast-driven PV forecasting pipeline
2. Align Solcast weather data with local measured PV output
3. Generate realistic PV power estimates using physics-based modeling
4. Engineer forecasting features from weather, time, and historical PV behavior
5. Train machine learning models for short-term and day-ahead PV forecasting
6. Support a cost-optimized microgrid controller with accurate solar predictions

---

## Pipeline

```text
Local PV Data
        +
Solcast Weather/Irradiance Data
        ↓
Data Cleaning and Time Alignment
        ↓
Calibration / Validation
        ↓
Physics-Based PV Simulation
        ↓
Feature Engineering
        ↓
Forecast Model Training
        ↓
PV Generation Forecast
        ↓
Microgrid Optimization Input
