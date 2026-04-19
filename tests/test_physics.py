"""
tests/test_physics.py
----------------------
Unit tests for src/physics/pvlib_model.py.

Run with:
    pytest tests/test_physics.py -v
"""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_site_cfg() -> SimpleNamespace:
    """Minimal site config matching configs/site.yaml structure."""
    temp_params = SimpleNamespace(u0=25.0, u1=6.84)
    pv_system = SimpleNamespace(
        tilt_deg=10,
        azimuth_deg=180,
        pdc0_kw=350.0,
        gamma_pdc=-0.0037,
        eta_inv_nom=0.96,
        temp_model_params=temp_params,
    )
    site = SimpleNamespace(
        name="Test Site",
        latitude=6.7912,
        longitude=79.9005,
        timezone="Asia/Colombo",
        elevation_m=20,
    )
    return SimpleNamespace(site=site, pv_system=pv_system)


def _make_solcast_df(n=48, tz="Asia/Colombo") -> pd.DataFrame:
    """Synthetic daytime-like DataFrame with required Solcast columns."""
    idx = pd.date_range("2022-06-14 06:00", periods=n, freq="5min", tz=tz)
    return pd.DataFrame(
        {
            "ghi": np.linspace(50, 800, n),
            "dni": np.linspace(0, 700, n),
            "dhi": np.linspace(50, 150, n),
            "clearsky_ghi": np.linspace(100, 900, n),
            "air_temp": np.full(n, 30.0),
            "pv_ac_W": np.linspace(0, 250_000, n),   # local measured
            "data_ok": np.ones(n, dtype=bool),
        },
        index=idx,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPvlibSimulation:
    def test_output_columns_present(self):
        from src.physics.pvlib_model import run_pvlib_simulation
        df = _make_solcast_df()
        cfg = _make_site_cfg()
        result = run_pvlib_simulation(df, cfg)
        for col in ("pvlib_ac_W", "pvlib_dc_W", "poa_global_Wm2",
                    "cell_temp_C", "clearness_index"):
            assert col in result.columns, f"Missing column: {col}"

    def test_pvlib_ac_non_negative(self):
        from src.physics.pvlib_model import run_pvlib_simulation
        df = _make_solcast_df()
        result = run_pvlib_simulation(df, _make_site_cfg())
        assert (result["pvlib_ac_W"] >= 0).all()

    def test_pvlib_ac_bounded_by_capacity(self):
        """Peak simulated AC should not far exceed nameplate capacity."""
        from src.physics.pvlib_model import run_pvlib_simulation
        df = _make_solcast_df()
        cfg = _make_site_cfg()
        result = run_pvlib_simulation(df, cfg)
        capacity_w = cfg.pv_system.pdc0_kw * 1_000
        # Allow small overage from PVWatts model details
        assert result["pvlib_ac_W"].max() <= capacity_w * 1.05

    def test_clearness_index_bounded(self):
        from src.physics.pvlib_model import run_pvlib_simulation
        df = _make_solcast_df()
        result = run_pvlib_simulation(df, _make_site_cfg())
        valid = result["clearness_index"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.1).all()

    def test_night_ghi_gives_zero_ac(self):
        """Zero irradiance should produce zero (or near-zero) AC output."""
        from src.physics.pvlib_model import run_pvlib_simulation
        idx = pd.date_range("2022-06-14 00:00", periods=12, freq="5min",
                            tz="Asia/Colombo")
        df = pd.DataFrame(
            {
                "ghi": np.zeros(12),
                "dni": np.zeros(12),
                "dhi": np.zeros(12),
                "clearsky_ghi": np.zeros(12),
                "air_temp": np.full(12, 25.0),
                "pv_ac_W": np.zeros(12),
                "data_ok": np.ones(12, dtype=bool),
            },
            index=idx,
        )
        result = run_pvlib_simulation(df, _make_site_cfg())
        assert result["pvlib_ac_W"].max() < 100.0  # negligible
