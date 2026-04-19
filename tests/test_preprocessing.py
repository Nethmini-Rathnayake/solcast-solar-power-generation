"""
tests/test_preprocessing.py
-----------------------------
Unit tests for src/preprocessing/cleaning.py.

Run with:
    pytest tests/test_preprocessing.py -v
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _make_cfg(**kwargs) -> SimpleNamespace:
    defaults = {
        "overflow_sentinel_w": 32_767_000.0,
        "pv_min_w": 0.0,
        "pv_max_w": 400_000.0,
        "clearsky_ghi_night_threshold_wm2": 5.0,
        "max_interpolation_steps": 6,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_df(n=20, tz="Asia/Colombo") -> pd.DataFrame:
    idx = pd.date_range("2022-06-01 06:00", periods=n, freq="5min", tz=tz)
    return pd.DataFrame(
        {
            "pv_ac_W": np.linspace(1000, 50000, n),
            "clearsky_ghi": np.linspace(100, 600, n),
            "inv_3_1_W": np.ones(n) * 5000,
        },
        index=idx,
    )


class TestOverflowRemoval:
    def test_overflow_set_to_nan(self):
        from src.preprocessing.cleaning import _remove_inverter_overflows
        df = _make_df()
        df.loc[df.index[5], "inv_3_1_W"] = 32_767_000.0
        cfg = _make_cfg()
        result = _remove_inverter_overflows(df, cfg)
        assert np.isnan(result.loc[result.index[5], "inv_3_1_W"])

    def test_normal_values_unchanged(self):
        from src.preprocessing.cleaning import _remove_inverter_overflows
        df = _make_df()
        cfg = _make_cfg()
        result = _remove_inverter_overflows(df, cfg)
        assert result["inv_3_1_W"].notna().all()


class TestPhysicalBounds:
    def test_above_max_set_to_nan(self):
        from src.preprocessing.cleaning import _apply_physical_bounds
        df = _make_df()
        df.loc[df.index[3], "pv_ac_W"] = 500_000.0  # above 400 kW
        result = _apply_physical_bounds(df, _make_cfg())
        assert np.isnan(result.loc[result.index[3], "pv_ac_W"])

    def test_below_zero_set_to_nan(self):
        from src.preprocessing.cleaning import _apply_physical_bounds
        df = _make_df()
        df.loc[df.index[2], "pv_ac_W"] = -100.0
        result = _apply_physical_bounds(df, _make_cfg())
        assert np.isnan(result.loc[result.index[2], "pv_ac_W"])


class TestInterpolation:
    def test_short_gap_filled(self):
        from src.preprocessing.cleaning import _interpolate_short_gaps
        df = _make_df(20)
        df.loc[df.index[5:8], "pv_ac_W"] = np.nan  # 3-step gap
        result = _interpolate_short_gaps(df, _make_cfg())
        assert result["pv_ac_W"].iloc[6] > 0  # interpolated

    def test_long_gap_not_filled(self):
        from src.preprocessing.cleaning import _interpolate_short_gaps
        df = _make_df(30)
        df.loc[df.index[5:15], "pv_ac_W"] = np.nan  # 10-step gap > limit 6
        result = _interpolate_short_gaps(df, _make_cfg())
        # Middle of the gap should still be NaN
        assert np.isnan(result["pv_ac_W"].iloc[10])


class TestQualityFlag:
    def test_data_ok_column_added(self):
        from src.preprocessing.cleaning import _add_quality_flag
        df = _make_df()
        df.loc[df.index[0], "pv_ac_W"] = np.nan
        result = _add_quality_flag(df)
        assert "data_ok" in result.columns
        assert not result.loc[result.index[0], "data_ok"]
        assert result.loc[result.index[1], "data_ok"]
