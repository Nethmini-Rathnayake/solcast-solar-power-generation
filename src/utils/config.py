"""
src/utils/config.py
--------------------
Config loader for the Solcast PV forecasting pipeline.

Loads three YAML files (site, model, pipeline) and merges them into a single
``PipelineConfig`` namespace so that all modules share a consistent,
type-hinted handle to configuration values.

Usage
-----
    from src.utils.config import load_config
    cfg = load_config()                          # uses default paths
    cfg = load_config(config_dir="my_configs/")  # override directory

    # Access nested values naturally:
    lat = cfg.site.latitude
    n_horizons = cfg.model.forecasting.n_horizons
    interim_path = cfg.pipeline.paths.interim_5min
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG_DIR = Path("configs")


def _dict_to_namespace(d: dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a nested dict to a SimpleNamespace tree."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class PipelineConfig:
    """Container for all pipeline configuration namespaces.

    Attributes
    ----------
    site:
        Site parameters from ``site.yaml`` (location, pvlib params).
    model:
        Model parameters from ``model.yaml`` (XGBoost, horizons, splits).
    pipeline:
        Operational parameters from ``pipeline.yaml`` (paths, cleaning,
        feature switches, evaluation thresholds).
    """

    def __init__(
        self,
        site: SimpleNamespace,
        model: SimpleNamespace,
        pipeline: SimpleNamespace,
    ) -> None:
        self.site = site
        self.model = model
        self.pipeline = pipeline

    def __repr__(self) -> str:
        return (
            f"PipelineConfig("
            f"site={self.site.name!r}, "
            f"horizons={self.model.forecasting.n_horizons}, "
            f"resolution={self.model.forecasting.temporal_resolution!r}"
            f")"
        )


def load_config(config_dir: str | Path = _DEFAULT_CONFIG_DIR) -> PipelineConfig:
    """Load and merge all three config files into a ``PipelineConfig``.

    Parameters
    ----------
    config_dir:
        Directory containing ``site.yaml``, ``model.yaml``, and
        ``pipeline.yaml``.  Defaults to ``configs/`` relative to the
        working directory (i.e. the project root).

    Returns
    -------
    PipelineConfig
    """
    config_dir = Path(config_dir)

    site_raw = _load_yaml(config_dir / "site.yaml")
    model_raw = _load_yaml(config_dir / "model.yaml")
    pipeline_raw = _load_yaml(config_dir / "pipeline.yaml")

    cfg = PipelineConfig(
        site=_dict_to_namespace(site_raw.get("site", site_raw)),
        model=_dict_to_namespace(model_raw.get("model", model_raw)),
        pipeline=_dict_to_namespace(pipeline_raw.get("pipeline", pipeline_raw)),
    )

    logger.info("Config loaded: %r", cfg)
    return cfg
