"""
src/evaluation/plots.py
------------------------
Forecast evaluation plots for the Solcast PV forecasting pipeline.

Plots produced
--------------
1. ``plot_forecast_vs_actual``   : Time-series overlay of predicted vs observed
                                    PV output for a chosen horizon.
2. ``plot_metrics_by_horizon``   : RMSE, MAE, nRMSE across h=1…24 as bar charts.
3. ``plot_scatter``              : Predicted vs observed scatter plot per horizon.
4. ``plot_error_distribution``   : Residual histograms per horizon.
5. ``plot_feature_importance``   : Top-N feature importances from the XGBoost model.

All plots are saved to ``results/figures/`` and optionally returned as
``matplotlib.figure.Figure`` objects for inline notebook display.

Usage
-----
    from src.evaluation.plots import plot_metrics_by_horizon, plot_forecast_vs_actual

    plot_metrics_by_horizon(metrics_df, save_dir="results/figures/")
    plot_forecast_vs_actual(y_true, y_pred, horizon=6, save_dir="results/figures/")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Consistent style across all plots
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)

_FIGURE_SIZE_WIDE = (12, 4)
_FIGURE_SIZE_SQUARE = (6, 6)
_FIGURE_SIZE_TALL = (12, 6)


def plot_metrics_by_horizon(
    metrics_df: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """Bar charts of RMSE, MAE, nRMSE, and R² across forecast horizons.

    Parameters
    ----------
    metrics_df:
        Output of ``compute_metrics`` — indexed by ``h1`` … ``hN`` + ``mean``.
    save_dir:
        If provided, saves the figure as ``metrics_by_horizon.png``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Drop the summary "mean" row for horizon-level plot
    df = metrics_df.drop(index="mean", errors="ignore")
    horizons = [int(idx.replace("h", "")) for idx in df.index]

    fig, axes = plt.subplots(2, 2, figsize=_FIGURE_SIZE_TALL)
    fig.suptitle("Forecast Metrics by Horizon (XGBoost DMS)", fontsize=13)

    _bar(axes[0, 0], horizons, df["RMSE"], "RMSE (W)", color="#2196F3")
    _bar(axes[0, 1], horizons, df["MAE"], "MAE (W)", color="#4CAF50")
    _bar(axes[1, 0], horizons, df["nRMSE"], "nRMSE (%)", color="#FF9800")
    _bar(axes[1, 1], horizons, df["R2"], "R²", color="#9C27B0")

    plt.tight_layout()
    _maybe_save(fig, save_dir, "metrics_by_horizon.png")
    return fig


def plot_forecast_vs_actual(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    horizon: int = 1,
    n_days: int = 7,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """Time-series overlay of predicted vs observed PV output.

    Parameters
    ----------
    y_true:
        Observed target DataFrame (``target_h*`` columns).
    y_pred:
        Predicted DataFrame (``pred_h*`` columns).
    horizon:
        Horizon to plot (1–24).
    n_days:
        Number of days to display (plotted from the end of the test set).
    save_dir:
        Directory to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t_col = f"target_h{horizon}"
    p_col = f"pred_h{horizon}"

    if t_col not in y_true.columns or p_col not in y_pred.columns:
        raise KeyError(
            f"Columns '{t_col}' / '{p_col}' not found.  "
            f"Valid horizons: 1–{len([c for c in y_true.columns if c.startswith('target_h')])}."
        )

    common = y_true.index.intersection(y_pred.index)
    obs = y_true.loc[common, t_col] / 1_000   # W → kW
    prd = y_pred.loc[common, p_col] / 1_000

    # Show last n_days
    n_show = min(n_days * 24, len(obs))
    obs = obs.iloc[-n_show:]
    prd = prd.iloc[-n_show:]

    fig, ax = plt.subplots(figsize=_FIGURE_SIZE_WIDE)
    ax.plot(obs.index, obs.values, label="Observed", color="#2196F3", lw=1.5)
    ax.plot(prd.index, prd.values, label=f"Predicted (h+{horizon})",
            color="#FF5722", lw=1.5, alpha=0.85)
    ax.set_ylabel("PV Output (kW)")
    ax.set_title(f"Forecast vs Actual — Horizon h+{horizon}")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    plt.tight_layout()

    _maybe_save(fig, save_dir, f"forecast_vs_actual_h{horizon}.png")
    return fig


def plot_scatter(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    horizon: int = 1,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """Predicted vs observed scatter plot for a single horizon.

    Parameters
    ----------
    y_true, y_pred:
        Target and prediction DataFrames.
    horizon:
        Horizon to plot.
    save_dir:
        Directory to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t_col = f"target_h{horizon}"
    p_col = f"pred_h{horizon}"

    common = y_true.index.intersection(y_pred.index)
    obs = y_true.loc[common, t_col].values / 1_000
    prd = y_pred.loc[common, p_col].values / 1_000

    valid = np.isfinite(obs) & np.isfinite(prd)
    obs, prd = obs[valid], prd[valid]

    max_val = max(obs.max(), prd.max()) * 1.05

    fig, ax = plt.subplots(figsize=_FIGURE_SIZE_SQUARE)
    ax.scatter(obs, prd, alpha=0.3, s=10, color="#2196F3", rasterized=True)
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="1:1 line")
    ax.set_xlabel("Observed PV (kW)")
    ax.set_ylabel("Predicted PV (kW)")
    ax.set_title(f"Scatter: Predicted vs Observed — h+{horizon}")
    ax.legend()
    plt.tight_layout()

    _maybe_save(fig, save_dir, f"scatter_h{horizon}.png")
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of top-N mean feature importances.

    Parameters
    ----------
    importance_df:
        Output of ``XGBoostDMSForecaster.feature_importances()``.
        Expected columns: ``feature``, ``mean_importance``.
    top_n:
        Number of features to display.
    save_dir:
        Directory to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    top = importance_df.head(top_n).sort_values("mean_importance")

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    ax.barh(top["feature"], top["mean_importance"], color="#2196F3")
    ax.set_xlabel("Mean Importance (across 24 horizon models)")
    ax.set_title(f"Top-{top_n} Feature Importances — XGBoost DMS")
    plt.tight_layout()

    _maybe_save(fig, save_dir, "feature_importance.png")
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(
    ax: plt.Axes,
    horizons: list[int],
    values: pd.Series,
    ylabel: str,
    color: str,
) -> None:
    """Internal helper: styled bar chart on a given Axes."""
    ax.bar(horizons, values, color=color, alpha=0.8, width=0.7)
    ax.set_xlabel("Horizon (h+)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(horizons[::3])


def _maybe_save(
    fig: plt.Figure,
    save_dir: str | Path | None,
    filename: str,
) -> None:
    """Save figure to ``save_dir/filename`` if ``save_dir`` is provided."""
    if save_dir is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    logger.info("Figure saved: %s", out_path)
