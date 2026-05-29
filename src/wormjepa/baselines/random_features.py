"""Random-features baseline (Story 3.7).

A JEPA-literature-standard sanity check: the headline result must beat
random untrained features by a non-trivial margin. The encoder is a causal
TCN with frozen random weights; a trainable linear head produces pose
forecasts on top of the random latent.

The latent is exposed for downstream neural-decoding (Epic 6) — a random
latent should *not* decode neurons above the kinematic baseline; if it does,
something is wrong with the probe.

Matched-parameter-count is a Phase 0 goal but operationalized loosely in v0:
the encoder's layer width and depth match the pose-only TCN defaults. The
exact ``param_count_target`` from the architecture is a Phase 6.x refinement
when the comparator is wired into the full eval pipeline.
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


class RandomFeaturesBaseline(Baseline):
    """Frozen random TCN encoder + trainable linear pose head."""

    def __init__(
        self,
        horizons_seconds: tuple[float, ...] = _DEFAULT_HORIZONS_SECONDS,
        frame_dt_seconds: float = _DEFAULT_FRAME_DT_SECONDS,
        latent_dim: int = 32,
        n_layers: int = 3,
        kernel_size: int = 3,
        n_epochs: int = 3,
        learning_rate: float = 1.0e-3,
        seed: int = 0,
    ) -> None:
        self._horizons = horizons_seconds
        self._frame_dt = frame_dt_seconds
        self._latent_dim = latent_dim
        self._n_layers = n_layers
        self._kernel_size = kernel_size
        self._n_epochs = n_epochs
        self._lr = learning_rate
        self._seed = seed
        self._encoder: CausalTCN | None = None
        self._pose_head: nn.Linear | None = None
        self._pose_shape: tuple[int, int] | None = None

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
        return "random_features"

    def _build_frozen_encoder(self, input_dim: int) -> CausalTCN:
        # Build encoder under a seeded RNG so the random features are deterministic
        # given the seed.
        gen = torch.Generator().manual_seed(self._seed)
        encoder = CausalTCN(
            input_dim=input_dim,
            latent_dim=self._latent_dim,
            n_layers=self._n_layers,
            kernel_size=self._kernel_size,
        )
        # Re-initialize all parameters from the seeded RNG, then freeze.
        for p in encoder.parameters():
            with torch.no_grad():
                p.copy_(torch.empty_like(p).normal_(generator=gen))
            p.requires_grad_(False)
        encoder.eval()
        return encoder

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
            msg = "No pose data in dataset; RandomFeaturesBaseline.fit cannot proceed."
            raise ValueError(msg)
        self._pose_shape = pose_shape

        input_dim = pose_shape[0] * pose_shape[1]
        self._encoder = self._build_frozen_encoder(input_dim)
        self._pose_head = nn.Linear(self._latent_dim, input_dim)

        opt = torch.optim.Adam(self._pose_head.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        self._pose_head.train()
        for _epoch in range(self._n_epochs):
            for seq in sequences:
                if seq.shape[0] < 2:
                    continue
                inp = seq[:-1].unsqueeze(0)
                tgt = seq[1:].unsqueeze(0)
                with torch.no_grad():
                    latent = self._encoder(inp)
                pred = self._pose_head(latent)
                loss = loss_fn(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._pose_head.eval()
        return self

    @torch.no_grad()
    def _rollout_pose(self, history_flat: torch.Tensor, n_steps: int) -> torch.Tensor:
        assert self._encoder is not None and self._pose_head is not None
        seq = history_flat.unsqueeze(0)
        last_pred = seq[:, -1, :]
        for _ in range(n_steps):
            latent = self._encoder(seq)
            last_pred = self._pose_head(latent[:, -1, :])
            seq = torch.cat([seq, last_pred.unsqueeze(1)], dim=1)
        return last_pred[0]

    @torch.no_grad()
    def _encode_clip(self, flat_seq: torch.Tensor) -> torch.Tensor:
        assert self._encoder is not None
        return self._encoder(flat_seq.unsqueeze(0))[0]

    def predict(self, dataset: Iterable[DatasetSample]) -> BaselinePredictions:
        if self._encoder is None or self._pose_head is None or self._pose_shape is None:
            msg = "RandomFeaturesBaseline.predict requires .fit() first."
            raise RuntimeError(msg)
        k, d = self._pose_shape

        future_pose: list[FuturePoseHorizon] = []
        latent_by_worm: dict[WormID, list[torch.Tensor]] = {}

        for sample in dataset:
            if sample.pose is None:
                continue
            t, _k, _d = sample.pose.shape
            flat = sample.pose.reshape(t, k * d)
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
