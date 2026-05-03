"""
spatiotemporal_grid_features.py
================================
Spatiotemporal 11×11 grid feature extraction from Open-Meteo for the
Solcast-based solar PV forecasting pipeline (Nethmini-Rathnayake/solcast-solar-power-generation).

Implements the approach from Li et al. (2025):
  - Extract an 11×11 neighbourhood of weather grid points around your target site
  - Flatten each timestep into a spatial feature vector
  - Merge with existing point-based features for XGBoost residual correction

Architecture overview
---------------------
              ┌──────────────────────────────────┐
              │  Open-Meteo Grid API              │
              │  11×11 = 121 lat/lon points       │
              └──────────────┬───────────────────┘
                             │
              ┌──────────────▼───────────────────┐
              │  SpatiotemporalGridExtractor       │
              │  · build_grid_coords()             │
              │  · fetch_grid_data()               │
              │  · flatten_to_feature_vector()     │
              └──────────────┬───────────────────┘
                             │
              ┌──────────────▼───────────────────┐
              │  SpatialFeatureMatrix             │
              │  shape: (T, 121 × n_vars)         │
              │  column names: ghi_i05_j05, …     │
              └──────────────┬───────────────────┘
                             │
              ┌──────────────▼───────────────────┐
              │  Merge with point-based features  │
              │  (your existing feature matrix)   │
              └──────────────┬───────────────────┘
                             │
              ┌──────────────▼───────────────────┐
              │  XGBoost DMS residual corrector   │
              │  (24 models, h+1 … h+24)          │
              └──────────────────────────────────┘

Dependencies
------------
    pip install openmeteo-requests requests-cache retry-requests numpy pandas tqdm

Open-Meteo is free for non-commercial use and requires no API key.
Grid resolution is ~11 km (0.1°), so an 11×11 patch covers ≈110 km × 110 km.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# University of Moratuwa Smartgrid Lab (from repo README)
TARGET_LAT: float = 6.7912
TARGET_LON: float = 79.9005

# Grid parameters (Li et al. 2025 style)
GRID_SIZE: int = 11          # 11×11 neighbourhood
GRID_STEP: float = 0.1       # ~11 km spacing in degrees (Open-Meteo native resolution)

# Weather variables to extract at each grid point.
# Chosen to match existing Solcast/Open-Meteo features in the pipeline.
SPATIAL_VARIABLES: List[str] = [
    "shortwave_radiation",          # GHI proxy  (W/m²)
    "direct_normal_irradiance",     # DNI        (W/m²)
    "diffuse_radiation",            # DHI        (W/m²)
    "cloud_cover",                  # %
    "temperature_2m",               # °C
    "wind_speed_10m",               # m/s
    "wind_direction_10m",           # °
    "relative_humidity_2m",         # %
    "precipitation",                # mm
]

# Open-Meteo API hourly endpoint
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Dataclass for one grid point's fetched data
# ---------------------------------------------------------------------------

@dataclass
class GridPointData:
    lat: float
    lon: float
    i: int          # row index in the 11×11 grid (0-based)
    j: int          # col index in the 11×11 grid (0-based)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Core extractor class
# ---------------------------------------------------------------------------

class SpatiotemporalGridExtractor:
    """
    Extracts an GRID_SIZE × GRID_SIZE patch of Open-Meteo weather data
    centred on the target coordinates and flattens it into a feature
    vector suitable for XGBoost.

    Parameters
    ----------
    center_lat, center_lon : float
        Target site coordinates.
    grid_size : int
        Side length of the square grid patch (default 11).
    grid_step : float
        Lat/lon spacing in degrees (default 0.1 ≈ 11 km at the equator).
    variables : list[str]
        Open-Meteo hourly variable names to fetch at each grid point.
    cache_dir : str
        Directory for HTTP response caching.  Set to None to disable.
    """

    def __init__(
        self,
        center_lat: float = TARGET_LAT,
        center_lon: float = TARGET_LON,
        grid_size: int = GRID_SIZE,
        grid_step: float = GRID_STEP,
        variables: List[str] = SPATIAL_VARIABLES,
        cache_dir: str = ".openmeteo_cache",
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.grid_size = grid_size
        self.grid_step = grid_step
        self.variables = variables

        # Set up a cached + retry-enabled requests session
        cache_session = requests_cache.CachedSession(cache_dir, expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.om = openmeteo_requests.Client(session=retry_session)

    # ------------------------------------------------------------------
    # Step 1: Build the 11×11 coordinate grid
    # ------------------------------------------------------------------

    def build_grid_coords(self) -> List[tuple[int, int, float, float]]:
        """
        Returns a list of (i, j, lat, lon) tuples for every grid point.

        The centre of the grid corresponds to (i=5, j=5) for an 11×11 grid.
        Rows (i) increase southward; columns (j) increase eastward.
        """
        half = self.grid_size // 2  # = 5 for grid_size=11
        coords = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                lat = self.center_lat + (half - i) * self.grid_step
                lon = self.center_lon + (j - half) * self.grid_step
                coords.append((i, j, round(lat, 4), round(lon, 4)))
        return coords

    # ------------------------------------------------------------------
    # Step 2: Fetch weather data for all grid points
    # ------------------------------------------------------------------

    def fetch_grid_data(
        self,
        start_date: str,
        end_date: str,
        historical: bool = True,
        batch_size: int = 10,
        sleep_between_batches: float = 1.0,
    ) -> List[GridPointData]:
        """
        Fetch hourly weather data for all 121 grid points.

        Open-Meteo supports up to 20 locations per request via its
        ensemble/batch parameter; we batch to stay well within rate limits.

        Parameters
        ----------
        start_date, end_date : str
            ISO-8601 date strings, e.g. "2022-04-01", "2023-04-30".
        historical : bool
            If True, use the historical archive endpoint (recommended for
            training data).  If False, use the forecast endpoint.
        batch_size : int
            Number of grid points per API request.
        sleep_between_batches : float
            Seconds to sleep between batches (be polite to the free API).

        Returns
        -------
        list of GridPointData, one per grid point.
        """
        coords = self.build_grid_coords()
        url = OPEN_METEO_HISTORICAL_URL if historical else OPEN_METEO_URL

        all_data: List[GridPointData] = []
        batches = [coords[k:k + batch_size] for k in range(0, len(coords), batch_size)]

        print(f"Fetching {len(coords)} grid points in {len(batches)} batches...")

        for batch in tqdm(batches, desc="Open-Meteo batches"):
            lats = [c[2] for c in batch]
            lons = [c[3] for c in batch]

            params = {
                "latitude": lats,
                "longitude": lons,
                "hourly": self.variables,
                "start_date": start_date,
                "end_date": end_date,
                "timezone": "Asia/Colombo",
            }

            try:
                responses = self.om.weather_api(url, params=params)
            except Exception as exc:
                print(f"  [WARNING] Batch failed: {exc}.  Skipping.")
                time.sleep(sleep_between_batches * 3)
                continue

            for (i, j, lat, lon), response in zip(batch, responses):
                hourly = response.Hourly()
                timestamps = pd.date_range(
                    start=pd.Timestamp(hourly.Time(), unit="s", tz="Asia/Colombo"),
                    end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="Asia/Colombo"),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                )
                records = {"timestamp": timestamps}
                for k, var in enumerate(self.variables):
                    records[var] = hourly.Variables(k).ValuesAsNumpy()

                gp = GridPointData(
                    lat=lat, lon=lon, i=i, j=j, df=pd.DataFrame(records)
                )
                all_data.append(gp)

            time.sleep(sleep_between_batches)

        print(f"  Done — {len(all_data)} grid points fetched successfully.")
        return all_data

    # ------------------------------------------------------------------
    # Step 3: Flatten to feature matrix  (T × [121 × n_vars])
    # ------------------------------------------------------------------

    def flatten_to_feature_matrix(
        self,
        grid_data: List[GridPointData],
        normalize: bool = True,
    ) -> pd.DataFrame:
        """
        Pivot all grid-point time series into one wide feature matrix.

        Column naming convention:
            <variable>_i<row>_j<col>
        e.g.  shortwave_radiation_i05_j05  is the centre point.

        Parameters
        ----------
        grid_data : list of GridPointData
        normalize : bool
            If True, apply per-variable z-score normalisation across the
            spatial dimension (i.e., subtract the mean and divide by the
            std computed over all 121 points for that variable at each
            timestep).  This matches the preprocessing in Li et al. (2025).

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex (hourly, Asia/Colombo timezone)
            Columns: one column per (variable, grid_point) pair
        """
        if not grid_data:
            raise ValueError("grid_data is empty — fetch data first.")

        # Sort to ensure consistent column ordering (row-major: i then j)
        grid_data_sorted = sorted(grid_data, key=lambda g: (g.i, g.j))

        frames = []
        for gp in grid_data_sorted:
            df = gp.df.set_index("timestamp")
            # Rename columns: ghi → ghi_i05_j05
            rename_map = {
                var: f"{var}_i{gp.i:02d}_j{gp.j:02d}"
                for var in self.variables
            }
            frames.append(df[self.variables].rename(columns=rename_map))

        wide = pd.concat(frames, axis=1)

        if normalize:
            wide = self._spatial_normalize(wide)

        return wide

    def _spatial_normalize(self, wide: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score normalise each variable across its 121 spatial columns,
        per timestep, to produce a spatially standardised feature vector.
        """
        normalized_parts = []
        for var in self.variables:
            var_cols = [c for c in wide.columns if c.startswith(var + "_")]
            sub = wide[var_cols]
            row_mean = sub.mean(axis=1)
            row_std = sub.std(axis=1).replace(0, 1)  # avoid /0
            normalized = sub.sub(row_mean, axis=0).div(row_std, axis=0)
            normalized_parts.append(normalized)
        return pd.concat(normalized_parts, axis=1)

    # ------------------------------------------------------------------
    # Step 4: Add spatial summary statistics (optional enrichment)
    # ------------------------------------------------------------------

    def compute_spatial_statistics(
        self, wide: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute per-variable spatial summary stats (mean, std, min, max,
        gradient_magnitude) from the 11×11 grid.

        These compact descriptors complement the full flattened vector and
        can be used as a lower-dimensional alternative (or in addition).

        Returns
        -------
        pd.DataFrame with columns like:
            shortwave_radiation_spatial_mean,
            shortwave_radiation_spatial_std, …
        """
        stats = {}
        for var in self.variables:
            var_cols = [c for c in wide.columns if c.startswith(var + "_")]
            sub = wide[var_cols].values  # (T, 121)

            stats[f"{var}_spatial_mean"] = sub.mean(axis=1)
            stats[f"{var}_spatial_std"] = sub.std(axis=1)
            stats[f"{var}_spatial_min"] = sub.min(axis=1)
            stats[f"{var}_spatial_max"] = sub.max(axis=1)
            stats[f"{var}_spatial_range"] = sub.max(axis=1) - sub.min(axis=1)

            # Spatial gradient magnitude: reshape to (T, 11, 11) and use Sobel
            grid_3d = sub.reshape(-1, self.grid_size, self.grid_size)
            gy = np.gradient(grid_3d, axis=1)  # N-S gradient
            gx = np.gradient(grid_3d, axis=2)  # E-W gradient
            grad_mag = np.sqrt(gx**2 + gy**2).mean(axis=(1, 2))
            stats[f"{var}_spatial_grad_mag"] = grad_mag

        return pd.DataFrame(stats, index=wide.index)

    # ------------------------------------------------------------------
    # Step 5: Merge with the existing point-based feature matrix
    # ------------------------------------------------------------------

    def merge_with_existing_features(
        self,
        existing_features: pd.DataFrame,
        spatial_wide: pd.DataFrame,
        spatial_stats: Optional[pd.DataFrame] = None,
        use_full_grid: bool = True,
        use_stats: bool = True,
    ) -> pd.DataFrame:
        """
        Merge the spatial grid features with your existing hourly feature
        matrix (output of src/features/ in the pipeline).

        Parameters
        ----------
        existing_features : pd.DataFrame
            Your current feature matrix with a DatetimeIndex.
        spatial_wide : pd.DataFrame
            Full 11×11 flattened feature matrix from flatten_to_feature_matrix().
        spatial_stats : pd.DataFrame or None
            Spatial summary statistics from compute_spatial_statistics().
        use_full_grid : bool
            Include all 121×n_vars columns.  Set False to use only stats.
        use_stats : bool
            Include spatial summary statistics.

        Returns
        -------
        pd.DataFrame
            Merged feature matrix ready for XGBoost training.
        """
        parts = [existing_features]

        if use_full_grid:
            parts.append(spatial_wide)
        if use_stats and spatial_stats is not None:
            parts.append(spatial_stats)

        merged = pd.concat(parts, axis=1, join="inner")
        n_dropped = len(existing_features) - len(merged)
        if n_dropped > 0:
            print(
                f"  [INFO] {n_dropped} timestamps dropped on inner join "
                f"(Open-Meteo grid vs existing features)."
            )
        return merged


# ---------------------------------------------------------------------------
# Convenience wrapper for the full pipeline
# ---------------------------------------------------------------------------

def build_spatial_feature_matrix(
    start_date: str,
    end_date: str,
    existing_features: Optional[pd.DataFrame] = None,
    center_lat: float = TARGET_LAT,
    center_lon: float = TARGET_LON,
    grid_size: int = GRID_SIZE,
    grid_step: float = GRID_STEP,
    variables: List[str] = SPATIAL_VARIABLES,
    normalize: bool = True,
    add_stats: bool = True,
    use_full_grid: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    End-to-end convenience function: fetch → flatten → (optionally merge).

    Usage
    -----
    >>> feat_matrix = build_spatial_feature_matrix(
    ...     start_date="2022-04-01",
    ...     end_date="2023-04-30",
    ...     existing_features=your_existing_df,   # from 02_build_features.py
    ...     save_path="data/processed/spatial_features.parquet",
    ... )

    Parameters
    ----------
    start_date, end_date : str
        Training period.
    existing_features : pd.DataFrame or None
        Pass your current feature matrix to merge spatial features in.
        If None, returns only the spatial feature matrix.
    save_path : str or None
        If provided, saves the result as a Parquet file.

    Returns
    -------
    pd.DataFrame
    """
    extractor = SpatiotemporalGridExtractor(
        center_lat=center_lat,
        center_lon=center_lon,
        grid_size=grid_size,
        grid_step=grid_step,
        variables=variables,
    )

    print("─" * 60)
    print(f"Grid: {grid_size}×{grid_size} = {grid_size**2} points")
    print(f"Centre: ({center_lat}°N, {center_lon}°E)")
    print(f"Coverage: ±{grid_size//2 * grid_step:.1f}° lat/lon "
          f"≈ ±{grid_size//2 * grid_step * 111:.0f} km")
    print(f"Variables: {len(variables)} — {variables}")
    print(f"Period: {start_date} → {end_date}")
    print(f"Spatial columns (full grid): {grid_size**2 * len(variables):,}")
    print("─" * 60)

    # 1. Fetch
    grid_data = extractor.fetch_grid_data(start_date, end_date, historical=True)

    # 2. Flatten
    spatial_wide = extractor.flatten_to_feature_matrix(grid_data, normalize=normalize)
    print(f"Spatial feature matrix shape: {spatial_wide.shape}")

    # 3. Stats
    spatial_stats = None
    if add_stats:
        spatial_stats = extractor.compute_spatial_statistics(spatial_wide)
        print(f"Spatial statistics shape: {spatial_stats.shape}")

    # 4. Merge
    if existing_features is not None:
        result = extractor.merge_with_existing_features(
            existing_features=existing_features,
            spatial_wide=spatial_wide,
            spatial_stats=spatial_stats,
            use_full_grid=use_full_grid,
            use_stats=add_stats,
        )
        print(f"Final merged feature matrix shape: {result.shape}")
    else:
        parts = [spatial_wide]
        if add_stats and spatial_stats is not None:
            parts.append(spatial_stats)
        result = pd.concat(parts, axis=1)

    # 5. Save
    if save_path:
        result.to_parquet(save_path)
        print(f"Saved → {save_path}")

    return result


# ---------------------------------------------------------------------------
# XGBoost integration helpers
# ---------------------------------------------------------------------------

def prepare_xgboost_spatial_splits(
    spatial_features: pd.DataFrame,
    residuals: pd.Series,
    test_ratio: float = 0.2,
    horizon: int = 24,
) -> dict:
    """
    Prepare train/test splits aligned to how the existing DMS XGBoost
    models are trained in scripts/03_train.py.

    The spatial feature matrix is used directly as X; the pvlib residual
    (measured_pv_W − pvlib_ac_W) is the y target.

    Parameters
    ----------
    spatial_features : pd.DataFrame
        Output of build_spatial_feature_matrix() (merged or standalone).
    residuals : pd.Series
        Time-aligned pvlib residual series (your target variable).
    test_ratio : float
        Fraction of data held out for evaluation (chronological split).
    horizon : int
        Forecast horizon in hours (used to label the split dict).

    Returns
    -------
    dict with keys X_train, X_test, y_train, y_test (each a DataFrame/Series).
    """
    # Align on common timestamps
    common_idx = spatial_features.index.intersection(residuals.index)
    X = spatial_features.loc[common_idx]
    y = residuals.loc[common_idx]

    # Drop nighttime rows (where pvlib simulation is ~0) to match pipeline
    daytime_mask = y.abs() > 1.0  # 1 W threshold
    X = X[daytime_mask]
    y = y[daytime_mask]

    # Chronological split (no shuffling — temporal integrity)
    split_idx = int(len(X) * (1 - test_ratio))
    return {
        "X_train": X.iloc[:split_idx],
        "X_test": X.iloc[split_idx:],
        "y_train": y.iloc[:split_idx],
        "y_test": y.iloc[split_idx:],
        "n_spatial_features": spatial_features.shape[1],
        "horizon": horizon,
    }


def get_feature_importance_by_group(
    booster,
    feature_names: List[str],
    variables: List[str] = SPATIAL_VARIABLES,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Aggregate XGBoost feature importances by variable group (e.g., all
    columns starting with 'shortwave_radiation_') so you can see which
    spatial variable contributes most to residual correction.

    Parameters
    ----------
    booster : xgboost.Booster or xgboost.XGBRegressor
    feature_names : list[str]
        Ordered list of feature names passed to XGBoost.
    importance_type : str
        "gain", "weight", or "cover".

    Returns
    -------
    pd.DataFrame sorted by importance descending.
    """
    try:
        import xgboost as xgb
        if hasattr(booster, "get_booster"):
            booster = booster.get_booster()
        scores = booster.get_score(importance_type=importance_type)
    except ImportError:
        raise ImportError("xgboost is required: pip install xgboost")

    rows = []
    for var in variables:
        group_cols = [f for f in feature_names if f.startswith(var + "_")]
        group_score = sum(scores.get(c, 0.0) for c in group_cols)
        rows.append({"variable": var, "importance": group_score,
                     "n_cols": len(group_cols)})

    # Also include non-spatial columns
    spatial_cols = set(
        c for var in variables for c in feature_names if c.startswith(var + "_")
    )
    other_cols = [f for f in feature_names if f not in spatial_cols]
    other_score = sum(scores.get(c, 0.0) for c in other_cols)
    rows.append({"variable": "_point_features", "importance": other_score,
                 "n_cols": len(other_cols)})

    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    df["importance_pct"] = 100 * df["importance"] / df["importance"].sum()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optional: lightweight CNN spatial encoder (PyTorch)
# For the hybrid CNN-LSTM + spatial path described in Li et al. (2025).
# ---------------------------------------------------------------------------

def build_cnn_spatial_encoder(
    n_variables: int = len(SPATIAL_VARIABLES),
    grid_size: int = GRID_SIZE,
    out_dim: int = 64,
):
    """
    Builds a small 2-D CNN that encodes one 11×11 spatial snapshot into a
    compact latent vector.  The output can be concatenated with your CNN-LSTM
    temporal features before the final dense layer.

    Requires PyTorch.  The output tensor has shape (batch, out_dim).

    Architecture
    ------------
        Input:  (batch, n_variables, 11, 11)
        Conv2d(n_variables → 32, kernel=3, padding=1) → ReLU
        Conv2d(32 → 64, kernel=3, padding=1) → ReLU
        AdaptiveAvgPool2d(4, 4)                      → (batch, 64, 4, 4)
        Flatten                                       → (batch, 1024)
        Linear(1024 → out_dim)                       → (batch, out_dim)

    Usage
    -----
    >>> import torch
    >>> encoder = build_cnn_spatial_encoder()
    >>> x = torch.randn(32, 9, 11, 11)   # batch=32, 9 variables, 11×11
    >>> latent = encoder(x)              # (32, 64)
    """
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required: pip install torch")

    import torch.nn as nn

    class SpatialCNNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(n_variables, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, out_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.head(self.cnn(x))

    return SpatialCNNEncoder()


def reshape_wide_to_tensor(
    wide: pd.DataFrame,
    variables: List[str] = SPATIAL_VARIABLES,
    grid_size: int = GRID_SIZE,
):
    """
    Reshape the flattened (T × [121 × n_vars]) feature matrix into a
    4-D NumPy array of shape (T, n_vars, grid_size, grid_size) suitable
    for the CNN encoder above.

    Parameters
    ----------
    wide : pd.DataFrame
        Output of flatten_to_feature_matrix() (before or after normalisation).

    Returns
    -------
    np.ndarray of shape (T, n_vars, 11, 11)
    """
    T = len(wide)
    n_vars = len(variables)
    arr = np.zeros((T, n_vars, grid_size, grid_size), dtype=np.float32)

    for v_idx, var in enumerate(variables):
        for i in range(grid_size):
            for j in range(grid_size):
                col = f"{var}_i{i:02d}_j{j:02d}"
                if col in wide.columns:
                    arr[:, v_idx, i, j] = wide[col].values.astype(np.float32)

    return arr


# ---------------------------------------------------------------------------
# Integration with scripts/03_train.py  (drop-in example)
# ---------------------------------------------------------------------------

EXAMPLE_INTEGRATION = '''
# ── scripts/03_train.py  (modified excerpt) ──────────────────────────────

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from spatiotemporal_grid_features import (
    build_spatial_feature_matrix,
    prepare_xgboost_spatial_splits,
    get_feature_importance_by_group,
)

# 1. Load your existing hourly feature matrix (from 02_build_features.py)
existing = pd.read_parquet("data/processed/hourly_features.parquet")

# 2. Build / load the spatial feature matrix
spatial_feat = build_spatial_feature_matrix(
    start_date="2022-04-01",
    end_date="2023-04-30",
    existing_features=existing,          # merges automatically
    save_path="data/processed/spatial_features.parquet",
)

# 3. Define target: pvlib residual (already in your existing features)
residuals = spatial_feat["pvlib_residual_W"]   # adjust column name as needed
X = spatial_feat.drop(columns=["pvlib_residual_W", "pv_ac_W"])

# 4. Prepare DMS splits (one per horizon — here shown for h+1)
splits = prepare_xgboost_spatial_splits(
    spatial_features=X,
    residuals=residuals,
    test_ratio=0.2,
)

# 5. Train XGBoost residual corrector with spatial features
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.6,        # important: subsample spatial columns
    tree_method="hist",
    early_stopping_rounds=30,
    eval_metric="rmse",
)
model.fit(
    splits["X_train"], splits["y_train"],
    eval_set=[(splits["X_test"], splits["y_test"])],
    verbose=100,
)

# 6. Evaluate
y_pred = model.predict(splits["X_test"])
rmse = mean_squared_error(splits["y_test"], y_pred, squared=False)
print(f"Residual RMSE: {rmse:.1f} W")

# 7. Inspect which spatial variables matter most
importance_df = get_feature_importance_by_group(model, list(X.columns))
print(importance_df)
'''


# ---------------------------------------------------------------------------
# CLI entry point (quick test / first run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch 11×11 Open-Meteo spatial grid")
    parser.add_argument("--start", default="2022-04-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2022-04-07", help="End date   YYYY-MM-DD")
    parser.add_argument("--save",  default="data/processed/spatial_features_test.parquet")
    parser.add_argument("--no-full-grid", action="store_true",
                        help="Save only spatial summary stats (smaller file)")
    args = parser.parse_args()

    result = build_spatial_feature_matrix(
        start_date=args.start,
        end_date=args.end,
        normalize=True,
        add_stats=True,
        use_full_grid=not args.no_full_grid,
        save_path=args.save,
    )

    print("\nSample columns (first 10):", list(result.columns[:10]))
    print("Sample statistics columns:", [c for c in result.columns if "spatial_" in c][:5])
    print("\nFirst 3 rows:")
    print(result.head(3).to_string())
    print(f"\nTotal features: {result.shape[1]:,}")
    print("\nIntegration example:")
    print(EXAMPLE_INTEGRATION)