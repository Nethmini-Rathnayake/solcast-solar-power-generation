"""
scripts/01_prepare_data.py
---------------------------
Pipeline Step 1: Load → clean → align → physics simulation → save interim.

What this script does
---------------------
1. Loads the raw local PV plant CSV (5-min, 1 year).
2. Loads all four Solcast historical CSV files (5-min, 4+ years) and combines.
3. Aligns both datasets on the 5-minute timestamp (inner join).
4. Cleans the aligned data (overflow removal, bounds check, interpolation).
5. Runs the pvlib simulation on the cleaned 5-min data.
6. Saves the result to data/interim/aligned_5min.parquet.

Run from the project root:
    python scripts/01_prepare_data.py

Output
------
    data/interim/aligned_5min.parquet
    ~104,000 rows × ~35 columns, tz-aware (Asia/Colombo) DatetimeIndex
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.local_pv import load_local_pv
from src.data.solcast import load_solcast
from src.data.alignment import align
from src.preprocessing.cleaning import clean
from src.physics.pvlib_model import run_pvlib_simulation

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Step 1: Data Preparation ===")

    cfg = load_config()
    pipe = cfg.pipeline

    # ── 1. Load local PV data ─────────────────────────────────────────────────
    local_pv = load_local_pv(pipe.paths.raw_local)

    # ── 2. Load Solcast data ──────────────────────────────────────────────────
    solcast = load_solcast(pipe.paths.external_solcast_dir)

    # ── 3. Align ──────────────────────────────────────────────────────────────
    aligned = align(local_pv, solcast)

    # ── 4. Clean ──────────────────────────────────────────────────────────────
    cleaned = clean(aligned, pipe.cleaning)

    # ── 5. pvlib simulation ───────────────────────────────────────────────────
    simulated = run_pvlib_simulation(cleaned, cfg.site)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    out_path = Path(pipe.paths.interim_5min)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    simulated.to_parquet(out_path)

    logger.info(
        "Interim data saved: %s | %d rows | %d columns",
        out_path,
        len(simulated),
        len(simulated.columns),
    )
    logger.info("=== Step 1 complete ===")


if __name__ == "__main__":
    main()
