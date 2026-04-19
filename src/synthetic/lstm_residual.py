"""
src/synthetic/lstm_residual.py
--------------------------------
LSTM residual correction model (PyTorch).

Architecture
------------
Input:  (batch, seq_len=24, 8 features)
        Features per timestep: ghi_norm, air_temp_norm, relative_humidity_norm,
        cloud_opacity_norm, clearness_index_norm, pvlib_ac_norm, hour_sin, hour_cos

2-layer LSTM → dropout → fully-connected → scalar residual output

Why LSTM for residuals?
-----------------------
The XGBoost residual model handles feature interactions at each hour
independently.  The LSTM complements it by capturing sequential context:
  - A sequence of gradually brightening skies before the current hour
    predicts lower soiling loss than a sudden clear burst.
  - Recent high-temperature hours predict elevated cell temperature and
    higher negative residual (thermal losses).
These are patterns that a single-timestep tabular model cannot see.

The two residual models (XGBoost + LSTM) are combined with equal weight
by default (configurable in corrected_pv.py).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.synthetic.residual_features import SEQUENCE_FEATURE_COLS, RESIDUAL_COL
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class ResidualSequenceDataset(Dataset):
    """Sliding-window dataset for LSTM residual training.

    Each sample: (X_seq, y_scalar)
      X_seq  : float32 tensor (seq_len, n_features) — past context window
      y_scalar: float32 tensor (1,)                 — residual at current hour

    When the DataFrame contains all hours (daytime_only=False), only samples
    whose TARGET timestep is daytime are yielded.  The full hourly sequence
    (including nighttime) is used as context so the LSTM sees continuous time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: list[str],
    ) -> None:
        self.seq_len = seq_len
        # NaN-safe: fillna(0) before converting to numpy
        self.X = df[feature_cols].fillna(0.0).values.astype(np.float32)
        self.y = df[RESIDUAL_COL].fillna(0.0).values.astype(np.float32)

        # Determine which target timesteps are valid for training
        # (daytime rows only — residual is meaningful only when pvlib > 0)
        if "is_daytime" in df.columns:
            daytime_arr = df["is_daytime"].values.astype(bool)
        else:
            # Fallback: treat all rows as daytime (old daytime-filtered data)
            daytime_arr = np.ones(len(df), dtype=bool)

        # Pre-compute valid sample indices to avoid checking in __getitem__
        self._valid = [
            i for i in range(len(self.X) - seq_len)
            if daytime_arr[i + seq_len]
        ]

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int):
        i = self._valid[idx]
        x = self.X[i : i + self.seq_len]          # (seq_len, n_feat)
        y = self.y[i + self.seq_len]               # scalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ── PyTorch Model ─────────────────────────────────────────────────────────────

class _LSTMResidualNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last = out[:, -1, :]           # (batch, hidden_size) — last timestep
        return self.fc(self.dropout(last)).squeeze(-1)  # (batch,)


# ── Wrapper ───────────────────────────────────────────────────────────────────

class LSTMResidualModel:
    """LSTM wrapper: training, prediction, save/load.

    Parameters
    ----------
    lstm_residual_cfg:
        The ``lstm_residual`` namespace from ``model.yaml``.
    """

    def __init__(self, lstm_residual_cfg: SimpleNamespace) -> None:
        self._cfg = lstm_residual_cfg
        self._feature_cols = SEQUENCE_FEATURE_COLS
        self._net: _LSTMResidualNet | None = None
        self._residual_scale: float = 1.0   # set in fit(); used to normalise target
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df_overlap: pd.DataFrame) -> "LSTMResidualModel":
        """Train on the 1-year overlap DataFrame.

        Parameters
        ----------
        df_overlap:
            Output of ``build_residual_features``.

        Returns
        -------
        self
        """
        cfg = self._cfg
        torch.manual_seed(cfg.random_state)

        available = [c for c in self._feature_cols if c in df_overlap.columns]
        for c in self._feature_cols:
            if c not in df_overlap.columns:
                df_overlap = df_overlap.copy()
                df_overlap[c] = 0.0
        self._feature_cols = available if available else self._feature_cols

        # ── Target normalisation ──────────────────────────────────────────────
        # Residuals in raw Watts (±50 kW) make MSE ~10⁹, causing very slow
        # gradient updates.  Normalise to [-1, 1] by the max absolute residual
        # so the network predicts a dimensionless fraction and MSE stays ~O(1).
        abs_max = df_overlap[RESIDUAL_COL].abs().max()
        self._residual_scale = float(abs_max) if abs_max > 0 else 1.0
        df_overlap = df_overlap.copy()
        df_overlap[RESIDUAL_COL] = df_overlap[RESIDUAL_COL] / self._residual_scale
        logger.info(
            "LSTM residual: target normalised by %.1f W → range [%.3f, %.3f].",
            self._residual_scale,
            df_overlap[RESIDUAL_COL].min(),
            df_overlap[RESIDUAL_COL].max(),
        )

        # Chronological 80/20 split
        split_idx = int(len(df_overlap) * 0.8)
        df_tr = df_overlap.iloc[:split_idx]
        df_vl = df_overlap.iloc[split_idx:]

        train_ds = ResidualSequenceDataset(df_tr, cfg.seq_len, self._feature_cols)
        val_ds   = ResidualSequenceDataset(df_vl, cfg.seq_len, self._feature_cols)

        if len(train_ds) == 0:
            raise ValueError(
                f"Not enough rows ({len(df_tr)}) for seq_len={cfg.seq_len}."
            )

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

        n_features = len(self._feature_cols)
        self._net = _LSTMResidualNet(
            input_size=n_features,
            hidden_size=cfg.hidden_size,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_count = 0
        best_state: dict | None = None

        for epoch in range(1, cfg.n_epochs + 1):
            train_loss = _run_epoch(self._net, train_loader, criterion,
                                     optimizer, self._device, train=True)
            val_loss   = _run_epoch(self._net, val_loader, criterion,
                                     None, self._device, train=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                patience_count += 1

            if epoch % 20 == 0 or epoch == cfg.n_epochs:
                logger.info(
                    "LSTM residual — epoch %3d/%d  train_loss=%.2f  val_loss=%.2f",
                    epoch, cfg.n_epochs, train_loss, val_loss,
                )
            if patience_count >= cfg.early_stopping_patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        best_val_rmse = np.sqrt(best_val_loss) * self._residual_scale
        logger.info(
            "LSTM residual training complete. "
            "Best val_loss=%.6f (normalised) → val_RMSE=%.1f W",
            best_val_loss, best_val_rmse,
        )
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict residual for every eligible row in ``df``.

        Rows within the first ``seq_len`` steps will have NaN predictions
        (no sufficient history yet).  Output is de-normalised back to Watts.

        Parameters
        ----------
        df:
            Feature DataFrame with sequence columns present.

        Returns
        -------
        pd.Series indexed by ``df.index``, NaN for the warm-up period.
        """
        if self._net is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        cfg = self._cfg
        for c in self._feature_cols:
            if c not in df.columns:
                df = df.copy()
                df[c] = 0.0

        X = df[self._feature_cols].fillna(0.0).values.astype(np.float32)
        preds = np.full(len(df), np.nan, dtype=np.float32)

        self._net.eval()
        with torch.no_grad():
            for i in range(cfg.seq_len, len(X)):
                seq = torch.from_numpy(X[i - cfg.seq_len : i]).unsqueeze(0).to(self._device)
                # De-normalise: network output is in [-1, 1]; scale back to Watts
                preds[i] = self._net(seq).item() * self._residual_scale

        return pd.Series(preds, index=df.index, name="lstm_residual_pred_W")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict":      self._net.state_dict() if self._net else None,
                "feature_cols":    self._feature_cols,
                "cfg":             self._cfg,
                "residual_scale":  self._residual_scale,
            },
            path,
        )
        logger.info("LSTM residual model saved: %s", path)

    def load(self, path: str | Path) -> "LSTMResidualModel":
        checkpoint = torch.load(path, map_location=self._device)
        self._cfg             = checkpoint["cfg"]
        self._feature_cols    = checkpoint["feature_cols"]
        self._residual_scale  = checkpoint.get("residual_scale", 1.0)  # backward compat
        cfg = self._cfg
        self._net = _LSTMResidualNet(
            input_size=len(self._feature_cols),
            hidden_size=cfg.hidden_size,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(self._device)
        self._net.load_state_dict(checkpoint["state_dict"])
        logger.info(
            "LSTM residual model loaded: %s  (residual_scale=%.1f W)",
            path, self._residual_scale,
        )
        return self


# ── Training loop helper ─────────────────────────────────────────────────────

def _run_epoch(
    net: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> float:
    net.train(train)
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = net(X_batch)
        loss = criterion(pred, y_batch)
        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)
