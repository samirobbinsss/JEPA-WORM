"""Pose-only TCN baseline — headline neural-decoding comparator (Story 3.6).

This is the load-bearing comparator for the PRD's neural-decoding partial-R²
headline metric: the JEPA latent must clear that bar against this baseline's
latent (and the Tierpsy feature set, added in Epic 6 Story 6.1).

Architecture: a small temporal convolutional network with dilated convolutions
along the time axis. ``fit()`` trains the encoder + a future-pose regression
head jointly on the dataset. ``predict()`` exposes both:

- per-clip future-pose forecasts (the future-pose metric);
- per-worm latent matrices ``(N_frames, D_latent)`` for downstream residualization.

Phase 0 v0 simplification: full JEPA-objective training (online encoder, EMA
target, predictor, masked future prediction) is documented as the architectural
intent but reduced here to a supervised future-pose regression objective. The
substantive role — "pose-only TCN with a non-trivial learned latent" — is
preserved. Epic 6 may refine the training objective if the comparator proves
too weak on real data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import torch
from torch import nn

from wormjepa.baselines._components import CausalTCN
from wormjepa.baselines.base import Baseline, BaselinePredictions, FuturePoseHorizon
from wormjepa.data import DatasetSample, WormID

_DEFAULT_HORIZONS_SECONDS: tuple[float, ...] = (0.1, 1.0, 5.0)
_DEFAULT_FRAME_DT_SECONDS = 0.1


class PoseOnlyTCNBaseline(Baseline):
    """Pose-only TCN baseline — headline neural-decoding comparator."""

    def __init__(
        self,
        horizons_seconds: tuple[float, ...] = _DEFAULT_HORIZONS_SECONDS,
        frame_dt_seconds: float = _DEFAULT_FRAME_DT_SECONDS,
        latent_dim: int = 32,
        n_layers: int = 3,
        kernel_size: int = 3,
        n_epochs: int = 3,
        learning_rate: float = 1.0e-3,
    ) -> None:
        self._horizons = horizons_seconds
        self._frame_dt = frame_dt_seconds
        self._latent_dim = latent_dim
        self._n_layers = n_layers
        self._kernel_size = kernel_size
        self._n_epochs = n_epochs
        self._lr = learning_rate
        self._encoder: CausalTCN | None = None
        self._pose_head: nn.Linear | None = None
        self._pose_shape: tuple[int, int] | None = None  # (K, D) once known

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
        return "pose_tcn"

    def fit(self, dataset: Iterable[DatasetSample]) -> Self:
        sequences: list[torch.Tensor] = []
        pose_shape: tuple[int, int] | None = None
        for sample in dataset:
            if sample.pose is None:
                continue
            t, k, d = sample.pose.shape
            pose_shape = (k, d)
            sequences.append(sample.pose.reshape(t, k * d))
        if not sequences or pose_shape is None:
            msg = "No pose data in dataset; PoseOnlyTCNBaseline.fit cannot proceed."
            raise ValueError(msg)
        self._pose_shape = pose_shape

        input_dim = pose_shape[0] * pose_shape[1]
        self._encoder = CausalTCN(
            input_dim=input_dim,
            latent_dim=self._latent_dim,
            n_layers=self._n_layers,
            kernel_size=self._kernel_size,
        )
        self._pose_head = nn.Linear(self._latent_dim, input_dim)
        opt = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._pose_head.parameters()),
            lr=self._lr,
        )
        loss_fn = nn.MSELoss()

        self._encoder.train()
        self._pose_head.train()
        for _epoch in range(self._n_epochs):
            for seq in sequences:
                if seq.shape[0] < 2:
                    continue
                inp = seq[:-1].unsqueeze(0)
                tgt = seq[1:].unsqueeze(0)
                latent = self._encoder(inp)
                pred = self._pose_head(latent)
                loss = loss_fn(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._encoder.eval()
        self._pose_head.eval()
        return self

    @torch.no_grad()
    def _rollout_pose(self, history_flat: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressively roll out flat pose ``(T_hist, input_dim) → (input_dim,)``."""
        assert self._encoder is not None and self._pose_head is not None
        seq = history_flat.unsqueeze(0)  # (1, T_hist, input_dim)
        last_pred = seq[:, -1, :]
        for _ in range(n_steps):
            latent = self._encoder(seq)
            last_pred = self._pose_head(latent[:, -1, :])  # (1, input_dim)
            seq = torch.cat([seq, last_pred.unsqueeze(1)], dim=1)
        return last_pred[0]

    @torch.no_grad()
    def _encode_clip(self, flat_seq: torch.Tensor) -> torch.Tensor:
        """Return per-frame latent ``(T, latent_dim)`` for one clip."""
        assert self._encoder is not None
        return self._encoder(flat_seq.unsqueeze(0))[0]

    def predict(self, dataset: Iterable[DatasetSample]) -> BaselinePredictions:
        if self._encoder is None or self._pose_head is None or self._pose_shape is None:
            msg = "PoseOnlyTCNBaseline.predict requires .fit() first."
            raise RuntimeError(msg)
        k, d = self._pose_shape

        future_pose: list[FuturePoseHorizon] = []
        latent_by_worm: dict[WormID, list[torch.Tensor]] = {}

        for sample in dataset:
            if sample.pose is None:
                continue
            t, _k, _d = sample.pose.shape
            flat = sample.pose.reshape(t, k * d)

            # Encode the full clip; accumulate per-worm latents.
            clip_latent = self._encode_clip(flat)
            latent_by_worm.setdefault(sample.worm_id, []).append(clip_latent)

            for horizon in self._horizons:
                lookback_frames = max(1, round(horizon / self._frame_dt))
                source_idx = max(0, t - 1 - lookback_frames)
                history = flat[: source_idx + 1]
                flat_pred = self._rollout_pose(history, lookback_frames)
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

        latents_concat = {wid: torch.cat(parts, dim=0) for wid, parts in latent_by_worm.items()}
        return BaselinePredictions(future_pose=future_pose, latent_by_worm=latents_concat)
