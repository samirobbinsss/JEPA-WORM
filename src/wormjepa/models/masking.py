"""Masked spatiotemporal masking strategy for JEPA training (Story 5.3).

V-JEPA 2 masks contiguous spatiotemporal blocks; the predictor forecasts the
target encoder's latents at the masked positions from the visible ones.
Phase 0 v0 uses a simpler random per-frame mask: each (B, T) position is
independently masked with probability ``masking_ratio``. Real V-JEPA 2.1-style
block masking is an Epic 6 refinement if needed.
"""

from __future__ import annotations

import torch


def random_temporal_mask(
    n_frames: int,
    n_batches: int,
    masking_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a boolean mask of shape ``(B, T)`` with ``masking_ratio`` of
    positions marked ``True`` (i.e., masked / to-be-predicted).

    Each row contains at least one masked position and at least one visible
    position so the predictor and the visible-context encoder both have work.
    """
    if not (0.0 < masking_ratio < 1.0):
        msg = f"masking_ratio must be in (0, 1); got {masking_ratio}"
        raise ValueError(msg)
    if n_frames < 2:
        msg = "n_frames must be >= 2 to support at least one masked and one visible position"
        raise ValueError(msg)

    if generator is None:
        random_values = torch.rand((n_batches, n_frames))
    else:
        random_values = torch.rand((n_batches, n_frames), generator=generator)

    mask = random_values < masking_ratio

    # Guarantee at least one masked and one visible per row.
    for b in range(n_batches):
        if not mask[b].any():
            mask[b, 0] = True
        if mask[b].all():
            mask[b, -1] = False
    return mask
