"""
src/utils/logger.py
--------------------
Centralised logging setup for the Solcast PV forecasting pipeline.

All modules obtain a logger via ``get_logger(__name__)``.  On the first call,
a console handler (INFO) and an optional rotating file handler (DEBUG) are
attached to the root logger so that every child logger inherits them.

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loading Solcast data …")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_INITIALIZED = False
_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "pipeline.log"
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 3


def _initialise_root_logger(log_to_file: bool = True) -> None:
    """Configure the root logger once.  Subsequent calls are no-ops."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — DEBUG and above (optional)
    if log_to_file:
        _LOG_DIR.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    _INITIALIZED = True


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """Return a named logger, initialising the root logger if needed.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    log_to_file:
        If True (default), also write DEBUG-level output to ``logs/pipeline.log``.

    Returns
    -------
    logging.Logger
    """
    _initialise_root_logger(log_to_file=log_to_file)
    return logging.getLogger(name)
