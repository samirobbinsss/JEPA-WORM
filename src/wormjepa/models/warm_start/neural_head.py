"""Neural auxiliary head: predict head-neuron activity from latent (Story 5.6).

This head is the load-bearing element of FR17's "neurally grounded latent
without test-time neural data": it consumes neural activity *only* during
training (as a loss target), and the deployed encoder's forward pass never
invokes it. Test-time inference goes through the encoder alone.
"""

from __future__ import annotations

import torch
from torch import nn


class NeuralAuxiliaryHead(nn.Module):
    """MLP predicting per-frame head-neuron activity from the latent."""

    def __init__(
        self,
        latent_dim: int,
        n_neurons: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(64, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_neurons),
        )
        self._n_neurons = n_neurons

    def forward(self, online_latent: torch.Tensor, neural_target: torch.Tensor) -> torch.Tensor:
        """Return MSE between predicted and target neural activity.

        Args:
            online_latent: ``(B, T, D)``.
            neural_target: ``(B, T, n_neurons)``.
        """
        if neural_target.shape[-1] != self._n_neurons:
            msg = (
                f"neural_target last dim {neural_target.shape[-1]} != "
                f"head's n_neurons {self._n_neurons}"
            )
            raise ValueError(msg)
        pred = self.net(online_latent)
        return torch.mean((pred - neural_target) ** 2)
