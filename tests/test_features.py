"""
tests/test_features.py
-----------------------
Unit tests for the feature engineering modules.

Run with:
    pytest tests/test_features.py -v
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_hourly_df(n=200, tz="Asia/Colombo") -> pd.DataFrame:
    """Synthetic hourly DataFrame with typical columns."""
    idx = pd.date_range("2022-06-01 00:00", periods=n, freq="1h", tz=tz)
    rng = np.random.default_rng(42)
    hour = np.array([t.hour for t in idx])
    is_day = (hour >= 6) & (hour <= 18)

    return pd.DataFrame(
        {
            "pv_ac_W": np.where(is_day, rng.uniform(10000, 200000, n), 0.0),
            "pvlib_ac_W": np.where(is_day, rng.uniform(8000, 180000, n), 0.0),
            "ghi": np.where(is_day, rng.uniform(50, 900, n), 0.0),
            "dhi": np.where(is_day, rng.uniform(30, 200, n), 0.0),
            "dni": np.where(is_day, rng.uniform(0, 700, n), 0.0),
            "clearsky_ghi": np.where(is_day, rng.uniform(100, 1000, n), 0.0),
            "clearsky_dni": np.where(is_day, rng.uniform(100, 900, n), 0.0),
            "gti": np.where(is_day, rng.uniform(50, 950, n), 0.0),
            "clearsky_gti": np.where(is_day, rng.uniform(50, 950, n), 0.0),
            "clearsky_dhi": np.where(is_day, rng.uniform(30, 200, n), 0.0),
            "cloud_opacity": rng.uniform(0, 80, n),
            "air_temp": rng.uniform(25, 35, n),
            "dewpoint_temp": rng.uniform(18, 28, n),
            "relative_humidity": rng.uniform(60, 95, n),
            "surface_pressure": rng.uniform(1010, 1025, n),
            "clearness_index": np.where(is_day, rng.uniform(0.3, 1.0, n), np.nan),
            "data_ok": np.ones(n, dtype=bool),
        },
        index=idx,
    )


def _make_site_cfg() -> SimpleNamespace:
    site = SimpleNamespace(
        name="Test",
        latitude=6.7912,
        longitude=79.9005,
        timezone="Asia/Colombo",
        elevation_m=20,
    )
    return SimpleNamespace(site=site)


def _make_pipeline_features_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        solcast_irradiance_cols=["ghi", "dni", "dhi", "clearsky_ghi", "cloud_opacity"],
        solcast_met_cols=["air_temp", "relative_humidity"],
        lag_hours=[1, 2, 24],
        rolling_windows=[6, 24],
    )


# ── Aggregation tests ─────────────────────────────────────────────────────────

class TestAggregation:
    def test_5min_to_hourly(self):
        from src.features.aggregation import aggregate_to_hourly
        idx = pd.date_range("2022-06-01", periods=120, freq="5min", tz="Asia/Colombo")
        df = pd.DataFrame({"pv_ac_W": np.ones(120) * 1000, "data_ok": True}, index=idx)
        result = aggregate_to_hourly(df)
        assert len(result) == 10  # 120 rows / 12 per hour = 10 hours

    def test_output_has_same_columns(self):
        from src.features.aggregation import aggregate_to_hourly
        idx = pd.date_range("2022-06-01", periods=24, freq="5min", tz="Asia/Colombo")
        df = pd.DataFrame({"pv_ac_W": 1.0, "ghi": 500.0, "data_ok": True}, index=idx)
        result = aggregate_to_hourly(df)
        for col in ("pv_ac_W", "ghi", "data_ok"):
            assert col in result.columns

    def test_naive_index_raises(self):
        from src.features.aggregation import aggregate_to_hourly
        idx = pd.date_range("2022-06-01", periods=12, freq="5min")  # no tz
        df = pd.DataFrame({"pv_ac_W": 1.0}, index=idx)
        with pytest.raises(ValueError):
            aggregate_to_hourly(df)


# ── Time features tests ───────────────────────────────────────────────────────

class TestTimeFeatures:
    def test_sin_cos_columns_present(self):
        from src.features.time_features import add_time_features
        df = _make_hourly_df(48)
        cfg = _make_site_cfg()
        result = add_time_features(df, cfg)
        for col in ("hour_sin", "hour_cos", "month_sin", "month_cos",
                    "doy_sin", "doy_cos", "solar_elevation_deg",
                    "cos_solar_zenith", "is_daytime"):
            assert col in result.columns, f"Missing: {col}"

    def test_hour_sin_bounded(self):
        from src.features.time_features import add_time_features
        df = _make_hourly_df(100)
        result = add_time_features(df, _make_site_cfg())
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()


# ── Physics features tests ────────────────────────────────────────────────────

class TestPhysicsFeatures:
    def test_physics_columns_present(self):
        from src.features.physics_features import add_physics_features
        df = _make_hourly_df(50)
        result = add_physics_features(df)
        for col in ("pvlib_ac_kW", "pvlib_residual", "clearness_index_hourly"):
            assert col in result.columns

    def test_pvlib_ac_kw_is_w_div_1000(self):
        from src.features.physics_features import add_physics_features
        df = _make_hourly_df(10)
        df["pvlib_ac_W"] = 100_000.0
        result = add_physics_features(df)
        assert abs(result["pvlib_ac_kW"].iloc[0] - 100.0) < 1e-6


# ── Lag features tests ────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag_columns_present(self):
        from src.features.lag_features import add_lag_features
        df = _make_hourly_df(200)
        cfg = _make_pipeline_features_cfg()
        result = add_lag_features(df, cfg)
        assert "lag_1h" in result.columns
        assert "lag_24h" in result.columns
        assert "rolling_mean_6h" in result.columns

    def test_target_matrix_columns_present(self):
        from src.features.lag_features import add_lag_features, build_target_matrix
        df = _make_hourly_df(400)
        cfg = _make_pipeline_features_cfg()
        df = add_lag_features(df, cfg)
        result = build_target_matrix(df, n_horizons=24)
        assert "target_h1" in result.columns
        assert "target_h24" in result.columns

    def test_no_nan_targets_after_build(self):
        from src.features.lag_features import add_lag_features, build_target_matrix
        df = _make_hourly_df(500)
        df = add_lag_features(df)
        result = build_target_matrix(df, n_horizons=24)
        target_cols = [f"target_h{h}" for h in range(1, 25)]
        assert result[target_cols].notna().all().all()
