"""
scripts/03_train.py
--------------------
Pipeline Step 3: Train all forecast models (baselines + XGBoost DMS).

What this script does
---------------------
1. Loads the hourly feature matrix parquet.
2. Splits into train / validation / test (chronological).
3. Trains persistence and same-day baseline models.
4. Trains the 24-model XGBoost Direct Multi-Step forecaster.
5. Saves all models to models/.
6. Saves the test set to data/processed/test.parquet for step 4.

Run from the project root:
    python scripts/03_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.train import build_feature_matrix, run_training_pipeline

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Step 3: Model Training ===")

    cfg = load_config()
    pipe = cfg.pipeline

    # ── Load feature matrix ───────────────────────────────────────────────────
    feat_path = Path(pipe.paths.processed_hourly)
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {feat_path}. "
            "Run scripts/02_build_features.py first."
        )
    df = pd.read_parquet(feat_path)
    logger.info("Feature matrix loaded: %d rows × %d columns.", len(df), len(df.columns))

    # ── Train XGBoost DMS model (includes split + save) ───────────────────────
    model = run_training_pipeline(df, cfg)

    # ── Log top feature importances ───────────────────────────────────────────
    top_features = model.feature_importances().head(10)
    logger.info("Top 10 features by mean importance:")
    for _, row in top_features.iterrows():
        logger.info("  %-35s %.4f", row["feature"], row["mean_importance"])

    logger.info("=== Step 3 complete ===")


if __name__ == "__main__":
    main()
