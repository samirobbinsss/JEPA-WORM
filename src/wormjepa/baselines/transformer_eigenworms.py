"""Transformer-on-eigenworms baseline — the kill-criterion comparator (Story 3.5).

The PRD's kill-criterion (NFR / measurable-outcome row 1) requires JEPA to beat
this baseline at the 1 s future-pose horizon. The baseline is an autoregressive
Transformer over eigenworm coefficients of the worm pose:

1. **Eigenworm basis** — fit via PCA on flattened pose during ``fit()``. In
   Phase 0 v0 we PCA on the synthetic dataset's pose; Story 4.7 replaces this
   with the Stephens 2008 basis pinned in pre-registration.
2. **Autoregressive Transformer** — learns to predict the next eigen coefficient
   vector given the history. Causal self-attention; teacher forcing during fit.
3. **Future-pose prediction** — for each clip, autoregressively rolls out the
   transformer for ``lookback_frames`` steps starting from ``pose[T-1-lookback]``,
   then inverse-projects through the eigen basis back to ``(K, D)`` pose.

The baseline is parameter-light by design — Phase 0 trains on a single 12 GB GPU
and the architectural focus is on JEPA, not on tuning this comparator.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import torch
from torch import nn

from wormjepa.baselines.base import Baseline, BaselinePredictions, FuturePoseHorizon
from wormjepa.data import DatasetSample

_DEFAULT_HORIZONS_SECONDS: tuple[float, ...] = (0.1, 1.0, 5.0)
_DEFAULT_FRAME_DT_SECONDS = 0.1


class _AutoregressiveTransformer(nn.Module):
    """Tiny causal Transformer over eigen-coefficient sequences."""

    def __init__(self, n_eigen: int, d_model: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_eigen, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, n_eigen)

    def forward(self, eigen_seq: torch.Tensor) -> torch.Tensor:
        # eigen_seq: (B, T, n_eigen). Causal mask so position t sees only <= t.
        _b, t, _ = eigen_seq.shape
        h = self.input_proj(eigen_seq)
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=eigen_seq.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.output_proj(h)  # next-step predictions per position


class TransformerEigenwormsBaseline(Baseline):
    """Autoregressive Transformer over eigenworm coefficients."""

    def __init__(
        self,
        horizons_seconds: tuple[float, ...] = _DEFAULT_HORIZONS_SECONDS,
        frame_dt_seconds: float = _DEFAULT_FRAME_DT_SECONDS,
        n_eigen: int = 4,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        n_epochs: int = 3,
        learning_rate: float = 1.0e-3,
    ) -> None:
        self._horizons = horizons_seconds
        self._frame_dt = frame_dt_seconds
        self._n_eigen = n_eigen
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._n_epochs = n_epochs
        self._lr = learning_rate
        self._basis: torch.Tensor | None = None  # (K*D, n_eigen)
        self._pose_mean: torch.Tensor | None = None  # (K*D,)
        self._model: _AutoregressiveTransformer | None = None

    @classmethod
    def from_config(cls, section: object) -> Self:
        from wormjepa.configs.baseline_config import BaselineSection

        if not isinstance(section, BaselineSection):
            msg = f"Expected BaselineSection, got {type(section).__name__}"
            raise TypeError(msg)
        return cls(
            horizons_seconds=tuple(section.horizons_seconds),
            frame_dt_seconds=section.frame_dt_seconds,
        )

    @property
    def name(self) -> str:
        return "transformer_eigenworms"

    def _fit_basis(self, flat_poses: torch.Tensor) -> None:
        """Fit a top-``n_eigen`` PCA basis on stacked, mean-centered flat poses.

        ``flat_poses`` has shape ``(N_total_frames, K*D)``. We center, SVD, and
        store the right singular vectors as the basis matrix ``(K*D, n_eigen)``.
        """
        mean = flat_poses.mean(dim=0)
        centered = flat_poses - mean
        # torch.linalg.svd returns U, S, Vh — Vh is (K*D, K*D).
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        self._basis = vh[: self._n_eigen].t().contiguous()  # (K*D, n_eigen)
        self._pose_mean = mean

    def _project(self, flat_pose: torch.Tensor) -> torch.Tensor:
        """Project ``(T, K*D)`` flat pose into ``(T, n_eigen)`` eigen space."""
        assert self._basis is not None and self._pose_mean is not None
        return (flat_pose - self._pose_mean) @ self._basis

    def _unproject(self, eigen: torch.Tensor) -> torch.Tensor:
        """Inverse projection from ``(..., n_eigen)`` back to ``(..., K*D)``."""
        assert self._basis is not None and self._pose_mean is not None
        return eigen @ self._basis.t() + self._pose_mean

    def fit(self, dataset: Iterable[DatasetSample]) -> Self:
        # 1) Collect all pose sequences as flat (T, K*D) tensors.
        sequences: list[torch.Tensor] = []
        for sample in dataset:
            if sample.pose is None:
                continue
            t, k, d = sample.pose.shape
            sequences.append(sample.pose.reshape(t, k * d))
        if not sequences:
            msg = "No pose data in dataset; TransformerEigenwormsBaseline.fit cannot proceed."
            raise ValueError(msg)

        all_frames = torch.cat(sequences, dim=0)
        self._fit_basis(all_frames)

        # 2) Project each sequence to eigen space.
        eigen_seqs = [self._project(seq) for seq in sequences]

        # 3) Train a tiny autoregressive Transformer on next-step prediction.
        self._model = _AutoregressiveTransformer(
            n_eigen=self._n_eigen,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
        )
        opt = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _epoch in range(self._n_epochs):
            for seq in eigen_seqs:
                if seq.shape[0] < 2:
                    continue
                inp = seq[:-1].unsqueeze(0)  # (1, T-1, n_eigen)
                tgt = seq[1:].unsqueeze(0)  # (1, T-1, n_eigen)
                pred = self._model(inp)
                loss = loss_fn(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model.eval()
        return self

    @torch.no_grad()
    def _rollout(self, history: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Roll out ``n_steps`` predictions starting from ``history`` (T_hist, n_eigen)."""
        assert self._model is not None
        seq = history.unsqueeze(0)  # (1, T_hist, n_eigen)
        for _ in range(n_steps):
            pred = self._model(seq)  # (1, T_cur, n_eigen)
            next_step = pred[:, -1:, :]  # (1, 1, n_eigen) — last position's prediction
            seq = torch.cat([seq, next_step], dim=1)
        return seq[0, -1, :]  # (n_eigen,)

    def predict(self, dataset: Iterable[DatasetSample]) -> BaselinePredictions:
        if self._model is None or self._basis is None:
            msg = "TransformerEigenwormsBaseline.predict requires .fit() first."
            raise RuntimeError(msg)

        future_pose: list[FuturePoseHorizon] = []
        for sample in dataset:
            if sample.pose is None:
                continue
            t, k, d = sample.pose.shape
            flat = sample.pose.reshape(t, k * d)
            eigen_seq = self._project(flat)  # (T, n_eigen)

            for horizon in self._horizons:
                lookback_frames = max(1, round(horizon / self._frame_dt))
                source_idx = max(0, t - 1 - lookback_frames)
                history = eigen_seq[: source_idx + 1]  # (T_hist, n_eigen)
                eigen_pred = self._rollout(history, lookback_frames)  # (n_eigen,)
                flat_pred = self._unproject(eigen_pred.unsqueeze(0))[0]  # (K*D,)
                predicted = flat_pred.reshape(k, d)
                ground_truth = sample.pose[t - 1].clone()
                future_pose.append(
                    FuturePoseHorizon(
                        worm_id=sample.worm_id,
                        session_id=sample.session_id,
                        horizon_seconds=horizon,
                        predicted=predicted,
                        ground_truth=ground_truth,
                    )
                )
        return BaselinePredictions(future_pose=future_pose, latent_by_worm=None)
