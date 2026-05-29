"""EMA target encoder for JEPA training (Story 5.2).

The EMA target is a frozen copy of the online encoder whose weights are
updated via an exponential moving average of the online weights. The target's
forward pass produces the *targets* the predictor learns to forecast; the
target's gradients are stopped (``torch.no_grad()`` + ``requires_grad=False``).

This is the standard JEPA / BYOL / DINO pattern: an asymmetric setup where
the predictor sees the online encoder's outputs and predicts the (slower-moving)
target encoder's outputs.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMATarget(nn.Module):
    """Exponential-moving-average copy of an online encoder.

    Wraps any encoder that honours the ``forward(video) -> latents`` contract
    — :class:`~wormjepa.models.encoder.WormJEPAEncoder` (legacy path) or
    :class:`~wormjepa.models.vjepa_loader.TrainableVJEPAEncoder` (the V-JEPA
    headline path). EMA-tracking that encoder is the standard JEPA
    anti-collapse mechanism: the target moves slower than the online encoder
    and stop-grad breaks the trivial constant solution.

    Args:
        online: The trainable encoder being EMA-tracked.
        decay: EMA decay rate. Default ``0.996`` (V-JEPA 2 default).
    """

    def __init__(self, online: nn.Module, decay: float = 0.996) -> None:
        super().__init__()
        if not (0.0 < decay < 1.0):
            msg = f"EMA decay must be in (0, 1); got {decay}"
            raise ValueError(msg)
        self._decay = decay
        # Deep-copy so the target has its own parameters.
        self._target: nn.Module = copy.deepcopy(online)
        for p in self._target.parameters():
            p.requires_grad_(False)
        self._target.eval()

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        """Update the target's parameters via ``θ_t = decay·θ_t + (1-decay)·θ_o``."""
        for p_t, p_o in zip(self._target.parameters(), online.parameters(), strict=True):
            p_t.data.mul_(self._decay).add_(p_o.data, alpha=1.0 - self._decay)

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Run the target encoder under stop-grad and return its latents."""
        return self._target(video)

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def target_encoder(self) -> nn.Module:
        return self._target
