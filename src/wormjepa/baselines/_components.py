"""Shared neural-network components used by multiple Phase 0 baselines."""

from __future__ import annotations

import torch
from torch import nn


class CausalTCN(nn.Module):
    """Causal dilated 1D convolution stack over the time axis.

    Used by :class:`wormjepa.baselines.pose_tcn.PoseOnlyTCNBaseline` and
    :class:`wormjepa.baselines.random_features.RandomFeaturesBaseline`. Right-pads
    each layer by ``(K-1) * dilation`` and crops the trailing positions afterward
    so a forward pass at position ``t`` sees only inputs at ``<= t``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_layers: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_dim
        for layer_idx in range(n_layers):
            dilation = 2**layer_idx
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=latent_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            layers.append(nn.GELU())
            in_channels = latent_dim
        self.net = nn.Sequential(*layers)
        self.kernel_size = kernel_size
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim) -> (B, input_dim, T) for Conv1d.
        h = x.transpose(1, 2)
        h = self.net(h)
        total_pad = sum((self.kernel_size - 1) * (2**i) for i in range(self.n_layers))
        if total_pad > 0:
            h = h[..., :-total_pad]
        return h.transpose(1, 2)
