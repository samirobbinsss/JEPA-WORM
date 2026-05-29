"""Eigenworm warm-start regularizer (Story 5.4).

Pushes the first ``n_eigen`` dimensions of the latent toward the corresponding
eigenworm coefficients of the input pose. Encodes the Stephens 2008 inductive
bias that worm posture lives on a ~4D manifold.

Phase 0 v0: fits a PCA basis on pose during ``fit_basis()``; Story 4.8 will
swap in the Stephens 2008 published basis pinned in pre-registration. Both
shapes are compatible — only the source of the basis matrix changes.
"""

from __future__ import annotations

import torch
from torch import nn


class EigenwormHead(nn.Module):
    """Regularizer aligning latent[..., :n_eigen] with pose-derived eigen coefficients."""

    def __init__(self, latent_dim: int, n_eigen: int = 4) -> None:
        super().__init__()
        if n_eigen > latent_dim:
            msg = f"n_eigen={n_eigen} exceeds latent_dim={latent_dim}"
            raise ValueError(msg)
        self._n_eigen = n_eigen
        # Buffers (not parameters): basis is fit on data, not learned by SGD.
        self.register_buffer("basis", torch.zeros((1, n_eigen)), persistent=True)
        self.register_buffer("pose_mean", torch.zeros((1,)), persistent=True)
        self._fitted = False

    @property
    def n_eigen(self) -> int:
        return self._n_eigen

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit_basis(self, flat_poses: torch.Tensor) -> None:
        """Fit PCA basis on stacked flat poses ``(N, K*D)``."""
        mean = flat_poses.mean(dim=0, keepdim=True)
        centered = flat_poses - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        # basis shape (K*D, n_eigen)
        basis = vh[: self._n_eigen].t().contiguous()
        self.basis = basis
        self.pose_mean = mean.squeeze(0)
        self._fitted = True

    def forward(self, online_latent: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Return MSE between ``online_latent[..., :n_eigen]`` and pose's eigen coefs.

        Args:
            online_latent: ``(B, T, D)``.
            pose: ``(B, T, K, Dpose)`` — pose for the same frames.
        """
        if not self._fitted:
            msg = "EigenwormHead.fit_basis(...) must be called before forward()."
            raise RuntimeError(msg)
        b, t, k, dp = pose.shape
        flat = pose.reshape(b, t, k * dp)
        # Project to eigen space: (B, T, n_eigen)
        eigen_coefs = (flat - self.pose_mean) @ self.basis
        latent_slice = online_latent[..., : self._n_eigen]
        return torch.mean((latent_slice - eigen_coefs) ** 2)
