"""
src/evaluation/metrics.py
--------------------------
Forecast evaluation metrics for 24h-ahead PV power prediction.

Cross-repo compatibility
------------------------
Metric names, function signatures, and DataFrame output format are kept
identical to ``solar-generation-forecasting`` (NASA POWER repository) so
that evaluation notebooks and comparison tables can be shared across both
repos without modification.

Metrics computed
----------------
  RMSE   — root mean squared error            [W]
  MAE    — mean absolute error                [W]
  MBE    — mean bias error                    [W]  (positive = over-prediction)
  MAPE   — mean absolute % error             [%]   (daytime only)
  nRMSE  — RMSE / mean(observed)             [%]   (scale-free)
  R²     — coefficient of determination              (Pearson r²)

All metrics are computed per-horizon (h=1…24) and returned as a DataFrame.
A summary row ("mean") is also included.

Usage
-----
    from src.evaluation.metrics import compute_metrics, summarise_metrics

    metrics_df = compute_metrics(y_true_df, y_pred_df)
    print(summarise_metrics(metrics_df))
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum observed power to include in MAPE (avoids division by near-zero)
_MAPE_MIN_W: float = 1_000.0


def compute_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    eval_cfg: SimpleNamespace | None = None,
) -> pd.DataFrame:
    """Compute per-horizon forecast metrics.

    Parameters
    ----------
    y_true:
        Observed targets — columns named ``target_h1`` … ``target_hN``
        (output of ``lag_features.build_target_matrix``).
    y_pred:
        Model predictions — columns named ``pred_h1`` … ``pred_hN``
        (output of any forecaster's ``predict`` method).
    eval_cfg:
        Optional ``evaluation`` namespace from ``pipeline.yaml``.
        Overrides ``_MAPE_MIN_W`` if ``mape_min_w`` is present.

    Returns
    -------
    pd.DataFrame
        Index: horizon labels (``h1`` … ``hN``, plus ``mean``).
        Columns: RMSE, MAE, MBE, MAPE, nRMSE, R2.
    """
    mape_min = _MAPE_MIN_W
    if eval_cfg is not None:
        mape_min = getattr(eval_cfg, "mape_min_w", mape_min)

    # Infer number of horizons from y_true
    target_cols = sorted(
        [c for c in y_true.columns if c.startswith("target_h")],
        key=lambda c: int(c.replace("target_h", "")),
    )
    n_horizons = len(target_cols)

    if n_horizons == 0:
        raise ValueError(
            "No 'target_h*' columns found in y_true.  "
            "Expected columns named target_h1 … target_hN."
        )

    rows: list[dict] = []
    for h_idx, t_col in enumerate(target_cols):
        h = h_idx + 1
        p_col = f"pred_h{h}"

        if p_col not in y_pred.columns:
            logger.warning("Prediction column '%s' not found — skipping.", p_col)
            continue

        # Align on index
        common_idx = y_true.index.intersection(y_pred.index)
        obs = y_true.loc[common_idx, t_col].values.astype(float)
        prd = y_pred.loc[common_idx, p_col].values.astype(float)

        # Remove NaN pairs
        valid = np.isfinite(obs) & np.isfinite(prd)
        obs, prd = obs[valid], prd[valid]

        if len(obs) == 0:
            logger.warning("No valid pairs for horizon h=%d.", h)
            continue

        rows.append(
            {
                "horizon": f"h{h}",
                **_compute_row(obs, prd, mape_min),
            }
        )

    if not rows:
        raise ValueError("No metrics computed — check column names and alignment.")

    metrics_df = pd.DataFrame(rows).set_index("horizon")

    # ── Summary row (mean across all horizons) ────────────────────────────────
    mean_row = metrics_df.mean(numeric_only=True).rename("mean")
    metrics_df = pd.concat([metrics_df, mean_row.to_frame().T])

    logger.info(
        "Metrics computed for %d horizons. "
        "Mean RMSE=%.1f W, MAE=%.1f W, R²=%.3f.",
        n_horizons,
        metrics_df.loc["mean", "RMSE"],
        metrics_df.loc["mean", "MAE"],
        metrics_df.loc["mean", "R2"],
    )
    return metrics_df


def compute_daytime_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    daytime_mask: pd.Series,
    eval_cfg: SimpleNamespace | None = None,
) -> pd.DataFrame:
    """Compute metrics restricted to daytime rows only.

    Daytime is defined by ``daytime_mask`` (boolean Series aligned to y_true
    index), typically ``pvlib_ac_W > 0`` or ``solar_elevation > 5°``.

    Night hours drag down RMSE artificially (easy zeros) — daytime-only metrics
    better reflect the model's real forecasting skill during generation hours.
    """
    mask = daytime_mask.reindex(y_true.index).fillna(False)
    if mask.sum() == 0:
        logger.warning("daytime_mask has no True rows — returning all-hour metrics.")
        return compute_metrics(y_true, y_pred, eval_cfg)
    return compute_metrics(y_true[mask], y_pred.loc[y_pred.index.intersection(y_true[mask].index)], eval_cfg)


def summarise_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted summary suitable for printing or saving.

    Rounds values and adds units to column names.
    """
    display = metrics_df.copy()
    for col in ("RMSE", "MAE", "MBE"):
        if col in display.columns:
            display[col] = display[col].round(1)
    for col in ("MAPE", "nRMSE"):
        if col in display.columns:
            display[col] = display[col].round(2)
    if "R2" in display.columns:
        display["R2"] = display["R2"].round(4)
    return display


# ── Internal helper ───────────────────────────────────────────────────────────

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _compute_row(
    obs: np.ndarray,
    prd: np.ndarray,
    mape_min: float,
) -> dict[str, float]:
    """Compute all metrics for a single horizon."""
    residuals = prd - obs

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    mbe = float(np.mean(residuals))

    # MAPE: daytime only (avoid division by near-zero nighttime values)
    daytime = obs >= mape_min
    if daytime.sum() > 0:
        mape = float(
            np.mean(np.abs(residuals[daytime]) / obs[daytime]) * 100.0
        )
    else:
        mape = float("nan")

    obs_mean = obs.mean()
    nrmse = (rmse / obs_mean * 100.0) if obs_mean > 0 else float("nan")
    r2 = _safe_r2(obs, prd)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MBE": mbe,
        "MAPE": mape,
        "nRMSE": nrmse,
        "R2": r2,
    }
