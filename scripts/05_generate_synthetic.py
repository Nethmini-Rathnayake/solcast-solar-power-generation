"""
scripts/05_generate_synthetic.py
---------------------------------
Pipeline Step 5: Generate residual-corrected 4-year synthetic PV.

What this script does
---------------------
1. Loads the 1-year real PV data from the hourly feature matrix
   (produced by scripts/02_build_features.py).
2. Calls ``build_corrected_synthetic_pv()`` which:
     a. Loads full 4-year Solcast (5-min).
     b. Runs pvlib PVWatts simulation on the full period.
     c. Aggregates to hourly and adds all feature columns.
     d. Extracts the 1-year overlap window (Apr 2022 – Apr 2023) and
        joins real PV measurements.
     e. Trains XGBoost + LSTM residual correction models on the overlap.
     f. Applies corrections to the full 4-year period.
3. Saves:
     data/processed/synthetic_corrected_4yr.parquet
       — Full 4-year corrected synthetic PV (hourly).
     models/residual/xgb_residual.pkl
       — XGBoost residual model.
     models/residual/lstm_residual.pt
       — LSTM residual model.

Prerequisites
-------------
    scripts/02_build_features.py  (produces data/processed/feature_matrix_hourly.parquet)

Run from the project root:
    python scripts/05_generate_synthetic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.synthetic.corrected_pv import build_corrected_synthetic_pv

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Step 5: Generate Corrected Synthetic PV ===")

    cfg  = load_config()
    pipe = cfg.pipeline

    # ── 1. Load hourly real PV feature matrix ────────────────────────────────
    feat_path = Path(pipe.paths.processed_hourly)
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Hourly feature matrix not found: {feat_path}. "
            "Run scripts/02_build_features.py first."
        )

    df_hourly = pd.read_parquet(feat_path)
    logger.info(
        "Hourly feature matrix loaded: %d rows × %d columns.",
        len(df_hourly),
        len(df_hourly.columns),
    )

    if "pv_ac_W" not in df_hourly.columns:
        raise KeyError(
            "'pv_ac_W' column not found in feature matrix. "
            "Ensure scripts/01_prepare_data.py and 02_build_features.py have run."
        )

    # ── 2. Build corrected synthetic PV ──────────────────────────────────────
    logger.info("Building corrected synthetic PV (this may take several minutes) …")
    corrected_df, xgb_model, lstm_model = build_corrected_synthetic_pv(
        cfg,
        real_pv_hourly=df_hourly[["pv_ac_W"]],
    )

    # ── 3. Save corrected synthetic parquet ──────────────────────────────────
    synth_dir = Path("data/processed")
    synth_dir.mkdir(parents=True, exist_ok=True)
    synth_path = synth_dir / "synthetic_corrected_4yr.parquet"
    corrected_df.to_parquet(synth_path)
    logger.info(
        "Corrected synthetic PV saved: %s | %d rows | %d columns",
        synth_path,
        len(corrected_df),
        len(corrected_df.columns),
    )

    # ── 4. Save residual models ───────────────────────────────────────────────
    residual_dir = Path("models/residual")
    residual_dir.mkdir(parents=True, exist_ok=True)

    xgb_path  = residual_dir / "xgb_residual.pkl"
    lstm_path = residual_dir / "lstm_residual.pt"

    xgb_model.save(xgb_path)
    lstm_model.save(lstm_path)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    n_daytime = (corrected_df["pv_corrected_W"] > 0).sum()
    pv_max    = corrected_df["pv_corrected_W"].max()
    pv_mean_daytime = corrected_df.loc[
        corrected_df["pv_corrected_W"] > 0, "pv_corrected_W"
    ].mean()

    logger.info(
        "Synthetic PV summary:\n"
        "  Total hourly rows   : %d\n"
        "  Daytime rows (>0 W) : %d\n"
        "  Peak pv_corrected_W : %.0f W\n"
        "  Mean daytime power  : %.0f W",
        len(corrected_df),
        n_daytime,
        pv_max,
        pv_mean_daytime,
    )
    logger.info("=== Step 5 complete ===")


if __name__ == "__main__":
    main()
