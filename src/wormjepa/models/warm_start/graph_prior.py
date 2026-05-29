"""Connectome graph-prior side-task (Story 5.5).

A side-task that predicts (a subset of) Cook 2019 connectome edge weights
from a designated neural-aligned subspace of the latent. The architectural
intent is to *bias* the latent toward connectome structure, not force exact
reconstruction. Phase 0 v0 uses a synthetic placeholder graph (random
adjacency) until the real Cook 2019 data is wired in via a pre-registration
artifact; the interface is identical either way.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphPriorHead(nn.Module):
    """Predicts edge weights from a slice of the online latent."""

    def __init__(
        self,
        latent_dim: int,
        n_neural_aligned_dims: int = 16,
        n_edges: int = 32,
    ) -> None:
        super().__init__()
        if n_neural_aligned_dims > latent_dim:
            msg = f"n_neural_aligned_dims={n_neural_aligned_dims} exceeds latent_dim={latent_dim}"
            raise ValueError(msg)
        self._n_dims = n_neural_aligned_dims
        self._n_edges = n_edges
        self.proj = nn.Linear(n_neural_aligned_dims, n_edges)
        # Targets are buffers — set via set_target_edges() before training.
        self.register_buffer("target_edges", torch.zeros((n_edges,)), persistent=True)
        self._target_set = False

    def set_target_edges(self, edges: torch.Tensor) -> None:
        """Set the ground-truth edge weights ``(n_edges,)`` from a connectome source.

        Co-locates the tensor with the module's existing buffer device so a
        CPU-originating connectome can be supplied even after ``self`` has been
        moved to a non-CPU accelerator.
        """
        if edges.shape != (self._n_edges,):
            msg = f"Expected edges shape ({self._n_edges},); got {tuple(edges.shape)}"
            raise ValueError(msg)
        self.target_edges = edges.to(self.target_edges.device)
        self._target_set = True

    def forward(self, online_latent: torch.Tensor) -> torch.Tensor:
        """Return MSE between predicted edges (mean across positions) and target edges.

        Args:
            online_latent: ``(B, T, D)``.
        """
        if not self._target_set:
            msg = "GraphPriorHead.set_target_edges(...) must be called before forward()."
            raise RuntimeError(msg)
        # Average over batch+time, then project. This is intentionally
        # coarse — the prior is "bias the latent toward connectome
        # structure," not "memorize edges per frame."
        latent_slice = online_latent[..., : self._n_dims]
        pooled = latent_slice.mean(dim=(0, 1))  # (n_dims,)
        pred = self.proj(pooled)
        return torch.mean((pred - self.target_edges) ** 2)
