"""
tests/test_data_loaders.py
---------------------------
Unit tests for src/data/local_pv.py, src/data/solcast.py, and
src/data/alignment.py.

Run with:
    pytest tests/test_data_loaders.py -v
"""

import pandas as pd
import pytest


# ── local_pv ─────────────────────────────────────────────────────────────────

class TestLoadLocalPv:
    def test_missing_file_raises(self, tmp_path):
        from src.data.local_pv import load_local_pv
        with pytest.raises(FileNotFoundError):
            load_local_pv(tmp_path / "nonexistent.csv")

    def test_output_has_expected_columns(self):
        """Smoke test against real data file — skip if not available."""
        pytest.importorskip("pandas")
        from pathlib import Path
        data_path = Path("data/raw/Smartgrid lab solar PV data.csv")
        if not data_path.exists():
            pytest.skip("Raw data not available.")
        from src.data.local_pv import load_local_pv
        df = load_local_pv(data_path)
        assert "pv_ac_W" in df.columns
        assert df.index.tz is not None
        assert df.index.name == "datetime_local"

    def test_datetime_is_tz_aware(self):
        from pathlib import Path
        data_path = Path("data/raw/Smartgrid lab solar PV data.csv")
        if not data_path.exists():
            pytest.skip("Raw data not available.")
        from src.data.local_pv import load_local_pv
        df = load_local_pv(data_path)
        assert str(df.index.tz) == "Asia/Colombo"


# ── solcast ───────────────────────────────────────────────────────────────────

class TestLoadSolcast:
    def test_missing_dir_raises(self, tmp_path):
        from src.data.solcast import load_solcast
        with pytest.raises(FileNotFoundError):
            load_solcast(tmp_path / "nonexistent_dir")

    def test_empty_dir_raises(self, tmp_path):
        from src.data.solcast import load_solcast
        with pytest.raises(FileNotFoundError):
            load_solcast(tmp_path)

    def test_output_tz_is_colombo(self):
        from pathlib import Path
        data_dir = Path("data/external/")
        if not data_dir.exists():
            pytest.skip("Solcast data not available.")
        from src.data.solcast import load_solcast
        df = load_solcast(data_dir)
        assert str(df.index.tz) == "Asia/Colombo"
        assert "ghi" in df.columns
        assert "cloud_opacity" in df.columns


# ── alignment ─────────────────────────────────────────────────────────────────

class TestAlign:
    def _make_df(self, start, periods, tz="Asia/Colombo", cols=None):
        idx = pd.date_range(start, periods=periods, freq="5min", tz=tz)
        data = {c: 1.0 for c in (cols or ["pv_ac_W"])}
        return pd.DataFrame(data, index=idx)

    def test_inner_join_retains_overlap(self):
        from src.data.alignment import align
        pv = self._make_df("2022-06-01", 100, cols=["pv_ac_W"])
        sc = self._make_df("2022-06-01", 80, cols=["ghi"])
        result = align(pv, sc)
        assert len(result) == 80
        assert "pv_ac_W" in result.columns
        assert "ghi" in result.columns

    def test_no_overlap_raises(self):
        from src.data.alignment import align
        pv = self._make_df("2022-06-01", 100, cols=["pv_ac_W"])
        sc = self._make_df("2023-06-01", 100, cols=["ghi"])
        with pytest.raises(ValueError, match="empty"):
            align(pv, sc)

    def test_naive_index_raises(self):
        from src.data.alignment import align
        idx = pd.date_range("2022-06-01", periods=10, freq="5min")  # no tz
        pv = pd.DataFrame({"pv_ac_W": 1.0}, index=idx)
        sc = self._make_df("2022-06-01", 10, cols=["ghi"])
        with pytest.raises(ValueError, match="timezone-naive"):
            align(pv, sc)
