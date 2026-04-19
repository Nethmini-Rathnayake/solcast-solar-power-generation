"""
src/synthetic/xgb_residual.py
-------------------------------
XGBoost residual correction model.

Trained on the 1-year overlap window to predict:
    residual_W = real_pv_W - pvlib_synthetic_W

Applied to the full 4-year Solcast period to convert uncorrected synthetic
PV into corrected synthetic PV that "behaves" like the real plant.

Why XGBoost for residuals?
--------------------------
The residual is driven by non-linear feature interactions:
  - Soiling loss is higher in the dry season (NE monsoon) and lower just
    after SW monsoon rain — this is a (monsoon_category × GHI) interaction.
  - Inverter clipping creates a hard threshold: residual spikes negative
    near peak GHI, disappearing when cloud reduces irradiance.
  - These threshold effects are exactly what gradient-boosted trees
    handle better than linear or kernel methods.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from src.synthetic.residual_features import TABULAR_FEATURE_COLS, RESIDUAL_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBResidualModel:
    """XGBoost model that learns and predicts the pvlib → real PV residual.

    Parameters
    ----------
    model_cfg:
        The ``xgboost`` namespace from ``model.yaml``.  Uses the same
        hyperparameters as the main XGBoost DMS forecaster.
    """

    def __init__(self, model_cfg: SimpleNamespace) -> None:
        self._params = _build_params(model_cfg)
        self._model: xgb.XGBRegressor | None = None
        self._feature_cols: list[str] = TABULAR_FEATURE_COLS

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df_overlap: pd.DataFrame) -> "XGBResidualModel":
        """Train on the 1-year overlap DataFrame.

        Parameters
        ----------
        df_overlap:
            Output of ``build_residual_features`` — contains feature columns
            and ``residual_W`` target.

        Returns
        -------
        self
        """
        available = [c for c in self._feature_cols if c in df_overlap.columns]
        missing   = [c for c in self._feature_cols if c not in df_overlap.columns]
        if missing:
            logger.warning("Missing feature columns (filled with 0): %s", missing)
            for c in missing:
                df_overlap = df_overlap.copy()
                df_overlap[c] = 0.0

        X = df_overlap[available].fillna(0.0)
        y = df_overlap[RESIDUAL_COL].fillna(0.0)

        # Chronological 80/20 validation split (no shuffle)
        split_idx = int(len(X) * 0.8)
        X_tr, X_vl = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_vl = y.iloc[:split_idx], y.iloc[split_idx:]

        self._model = xgb.XGBRegressor(**self._params)
        self._model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False,
        )
        self._feature_cols = available

        val_pred = self._model.predict(X_vl)
        val_rmse = float(np.sqrt(np.mean((y_vl.values - val_pred) ** 2)))
        logger.info(
            "XGBoost residual model trained: val_RMSE=%.1f W  "
            "(best_iteration=%s)",
            val_rmse,
            getattr(self._model, "best_iteration", "N/A"),
        )
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict residual for every row in ``df``.

        Parameters
        ----------
        df:
            Feature DataFrame (must contain the columns used at training time).

        Returns
        -------
        pd.Series
            Predicted residual in Watts, indexed by ``df.index``.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X = df[self._feature_cols].fillna(0.0)
        pred = self._model.predict(X)
        return pd.Series(pred, index=df.index, name="residual_pred_W")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"model": self._model, "features": self._feature_cols}, fh)
        logger.info("XGBoost residual model saved: %s", path)

    def load(self, path: str | Path) -> "XGBResidualModel":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        self._model = state["model"]
        self._feature_cols = state["features"]
        logger.info("XGBoost residual model loaded: %s", path)
        return self


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_params(cfg: SimpleNamespace) -> dict:
    return {
        "n_estimators":        cfg.n_estimators,
        "max_depth":           cfg.max_depth,
        "learning_rate":       cfg.learning_rate,
        "subsample":           cfg.subsample,
        "colsample_bytree":    cfg.colsample_bytree,
        "min_child_weight":    cfg.min_child_weight,
        "reg_alpha":           cfg.reg_alpha,
        "reg_lambda":          cfg.reg_lambda,
        "objective":           cfg.objective,
        "eval_metric":         cfg.eval_metric,
        "early_stopping_rounds": cfg.early_stopping_rounds,
        "random_state":        cfg.random_state,
        "n_jobs":              cfg.n_jobs,
    }
