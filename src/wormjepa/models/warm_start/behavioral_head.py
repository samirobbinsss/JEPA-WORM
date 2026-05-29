"""Behavioral-state classifier head (Story 5.7).

Predicts the worm's behavioral state (forward / reversal / Omega-turn / pause /
quiescence) from the latent. Uses cross-entropy on a small subset of latent
dimensions — the architectural intent is to bias a few axes toward
behavioral discriminability, not to use the full latent.
"""

from __future__ import annotations

import torch
from torch import nn


class BehavioralHead(nn.Module):
    """Classifier predicting Flavell-style behavioral states."""

    def __init__(self, latent_dim: int, n_states: int = 5) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, n_states)
        self._n_states = n_states

    def forward(self, online_latent: torch.Tensor, state_labels: torch.Tensor) -> torch.Tensor:
        """Return cross-entropy loss against integer state labels.

        Args:
            online_latent: ``(B, T, D)``.
            state_labels: ``(B, T)`` integer tensor of state indices.
        """
        logits = self.proj(online_latent)  # (B, T, n_states)
        b, t, c = logits.shape
        return nn.functional.cross_entropy(logits.reshape(b * t, c), state_labels.reshape(b * t))
