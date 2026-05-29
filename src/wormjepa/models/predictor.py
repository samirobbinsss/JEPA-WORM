"""JEPA predictor: forecast target-encoder latents from online latents (Story 5.2).

Cross-attention design. Position queries (one per frame) cross-attend to the
*visible* (un-masked) online-encoder latents and predict the target-encoder
latents at every position; the loss is taken only at the masked positions.

Why cross-attention, not a plain encoder over the full sequence: the
predictor's only *content* input is the visible online tokens. Masked
positions are excluded from cross-attention via ``memory_key_padding_mask``,
so (a) the predictor never sees a masked frame's own online latent — no
leakage of the answer — and (b) if the online encoder collapses, the visible
context is constant and the predictor genuinely cannot forecast varied
targets. Collapse is therefore penalised by the JEPA loss itself rather than
left for the variance regularizer to fight alone. An earlier design (a plain
TransformerEncoder over mask-token-substituted latents) let the predictor
forecast position-conditionally from the position embedding alone, ignoring
the online encoder entirely — which let the online encoder collapse freely.
"""

from __future__ import annotations

import torch
from torch import nn


class JEPAPredictor(nn.Module):
    """Cross-attention Transformer mapping online latents to target latents.

    For each of the ``T`` positions a learned position-embedding query
    cross-attends to the visible online-encoder latents and emits a
    predicted target latent. The loss is then MSE between predictor outputs
    and the target encoder's (stop-grad) latents, at the masked positions.

    Args:
        latent_dim: Online and target encoder latent dim.
        n_layers: Number of Transformer decoder layers.
        n_heads: Number of attention heads.
        max_frames: Sequence-length capacity of the learned position
            embedding. Clips longer than this raise at forward time.
    """

    def __init__(
        self,
        latent_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=4 * latent_dim,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        # Learned position embedding — both the cross-attention queries and
        # the additive positional signal on the visible-context memory.
        self.max_frames = max_frames
        self.pos_embed = nn.Parameter(torch.zeros((1, max_frames, latent_dim)))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, online_latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            online_latents: ``(B, T, D)`` — online encoder output.
            mask: ``(B, T)`` boolean — True = masked (to predict).

        Returns:
            ``(B, T, D)`` — predicted target-encoder latents. Loss is taken
            only at masked positions; the unmasked positions are still emitted
            (the caller selects with the mask).
        """
        b, t, d = online_latents.shape
        if t > self.max_frames:
            msg = f"JEPAPredictor: clip has {t} frames, max_frames={self.max_frames}."
            raise ValueError(msg)
        pos = self.pos_embed[:, :t, :]  # (1, T, D)
        # Memory = online latents + position signal. masked positions are
        # excluded from cross-attention by `memory_key_padding_mask`, so the
        # predictor attends only to the visible online context.
        memory = online_latents + pos  # (B, T, D)
        queries = pos.expand(b, t, d)  # (B, T, D) — one position query per frame
        return self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=mask,  # True = masked = ignored as a key
        )
