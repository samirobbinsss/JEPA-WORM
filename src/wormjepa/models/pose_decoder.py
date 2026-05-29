"""Pose-decoder head for dev-loop visualisation.

**Not** part of the pre-registered warm-start head set (FR16/17). This head
exists solely to give the GUI's ``ClipViewer`` something visual to render:
the encoder's spatial-token grid at each step is decoded back to ``(K, 2)``
keypoint coordinates and the GUI overlays them on the frame strip in red,
next to the green ground-truth dots. Watching the red dots converge onto
the green dots as training progresses is the intended dev-loop signal.

Architecture: keypoint coordinates are inherently *spatial*, so decoding
them from a spatially-mean-pooled latent is near-impossible (verified
empirically — predicted keypoints scatter randomly and never converge).
This head therefore consumes the encoder's *un-pooled* spatial-token grid
``(B, T, S, D)``: ``K`` learned keypoint-query embeddings cross-attend (via
``nn.MultiheadAttention``) to the ``S`` spatial tokens of each frame, and a
small MLP maps each attended query to its ``(x, y)`` coordinate. With
``S = 1`` (the legacy non-V-JEPA encoder path) the cross-attention
degenerates gracefully to attending over a single token.

The deployed encoder (test-time path) never invokes this head. Use only
during training when ``clips_dir`` is set on :func:`wormjepa.training.loop.train_jepa`.
"""

from __future__ import annotations

import torch
from torch import nn


class PoseDecoderHead(nn.Module):
    """Cross-attention head mapping a spatial-token grid to ``(K, 2)`` keypoints.

    ``K`` learned keypoint-query embeddings cross-attend to the ``S`` spatial
    tokens of each ``(B, T)`` frame; a small MLP then maps each attended
    query to its ``(x, y)`` coordinate.

    Args:
        latent_dim: Dimensionality of the encoder's spatial tokens (``D``).
        n_keypoints: Number of pose keypoints (``K``).
        n_heads: Number of cross-attention heads. Default 4; clamped so it
            always divides ``latent_dim`` (a hard ``nn.MultiheadAttention``
            requirement) — falls back to 1 when ``latent_dim`` is not
            divisible by the requested head count.
        hidden_dim: Width of the coordinate-MLP hidden layer. Defaults to
            ``max(64, latent_dim)``.
    """

    def __init__(
        self,
        latent_dim: int,
        n_keypoints: int,
        n_heads: int = 4,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(64, latent_dim)
        self._latent_dim = latent_dim
        self._n_keypoints = n_keypoints
        # nn.MultiheadAttention requires embed_dim % num_heads == 0.
        heads = n_heads if latent_dim % n_heads == 0 else 1
        self._n_heads = heads
        # One learned query embedding per keypoint.
        self.keypoint_queries = nn.Parameter(torch.zeros(n_keypoints, latent_dim))
        nn.init.trunc_normal_(self.keypoint_queries, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def predict(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """Forward-only inference path.

        Args:
            spatial_tokens: ``(B, T, S, D)`` encoder spatial-token grid.

        Returns:
            ``(B, T, K, 2)`` predicted keypoint coordinates.
        """
        if spatial_tokens.ndim != 4:
            msg = (
                f"PoseDecoderHead expects spatial tokens (B, T, S, D); "
                f"got shape {tuple(spatial_tokens.shape)}"
            )
            raise ValueError(msg)
        b, t, s, d = spatial_tokens.shape
        if d != self._latent_dim:
            msg = f"spatial-token D dim {d} != head's latent_dim {self._latent_dim}"
            raise ValueError(msg)
        # Fold (B, T) into the attention batch dim.
        tokens = spatial_tokens.reshape(b * t, s, d)  # (B*T, S, D)
        queries = self.keypoint_queries.unsqueeze(0).expand(b * t, -1, -1)  # (B*T, K, D)
        attended, _ = self.attn(query=queries, key=tokens, value=tokens)  # (B*T, K, D)
        coords = self.coord_head(attended)  # (B*T, K, 2)
        return coords.reshape(b, t, self._n_keypoints, 2)

    def forward(self, spatial_tokens: torch.Tensor, pose_target: torch.Tensor) -> torch.Tensor:
        """Return MSE between predicted and ground-truth pose keypoints.

        Args:
            spatial_tokens: ``(B, T, S, D)`` encoder spatial-token grid.
            pose_target: ``(B, T, K, 2)`` ground-truth keypoint coordinates.
        """
        if pose_target.shape[-2] != self._n_keypoints:
            msg = (
                f"pose_target K dim {pose_target.shape[-2]} != "
                f"head's n_keypoints {self._n_keypoints}"
            )
            raise ValueError(msg)
        pred = self.predict(spatial_tokens)
        return torch.mean((pred - pose_target) ** 2)
