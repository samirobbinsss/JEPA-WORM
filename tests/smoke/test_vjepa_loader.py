"""Loader-only smoke for the frozen V-JEPA 2.1 target encoder (Story 8.11a).

Opt-in via ``WORMJEPA_TEST_VJEPA21=1`` because the test triggers a
~300 MB checkpoint download from ``dl.fbaipublicfiles.com`` on first run
(cached afterwards under ``~/.cache/wormjepa/checkpoints/``).

Asserts:

1. ``build_frozen_vjepa_target('vjepa2_1_vit_base_384')`` returns a
   :class:`FrozenVJEPATarget` whose every parameter has
   ``requires_grad=False`` and which sits in ``eval()`` mode.
2. ``forward`` on a synthetic ``(B=1, T=16, C=3, H=384, W=384)`` tensor
   returns shape ``(1, 16, 768)`` — i.e. the per-frame contract the rest of
   the training loop expects, with ViT-B's 768-dim latent.

Runner / loop integration (cfg.jepa.frozen_target branch in
``training/runner.py``) is intentionally deferred to Story 8.11b, which
populates ``configs/headline.yaml`` and exercises the path end-to-end.
"""

from __future__ import annotations

import os

import pytest
import torch

_ENV_FLAG = "WORMJEPA_TEST_VJEPA21"

pytestmark = pytest.mark.skipif(
    os.environ.get(_ENV_FLAG) != "1",
    reason=(
        f"Set {_ENV_FLAG}=1 to run; pulls a ~300 MB V-JEPA 2.1 checkpoint "
        f"from dl.fbaipublicfiles.com on first run."
    ),
)


def test_frozen_vjepa_target_loads_and_runs_forward() -> None:
    from wormjepa.models.vjepa_loader import (
        FrozenVJEPATarget,
        build_frozen_vjepa_target,
    )

    target = build_frozen_vjepa_target("vjepa2_1_vit_base_384")
    assert isinstance(target, FrozenVJEPATarget)
    assert target.embed_dim == 768
    assert not target.training, "Frozen target should be in eval() mode."
    for name, p in target.named_parameters():
        assert not p.requires_grad, (
            f"Parameter {name} has requires_grad=True; frozen target must have all params frozen."
        )

    video = torch.randn(1, 16, 3, 384, 384)
    with torch.no_grad():
        out = target(video)
    assert out.shape == (1, 16, 768), (
        f"Expected (1, 16, 768) per-frame latents from ViT-B; got {tuple(out.shape)}."
    )


def test_trainable_vjepa_encoder_forward_tokens_grid_and_pool_identity() -> None:
    """`forward_tokens` returns the un-pooled (B, T, S, D) spatial grid and
    `forward` equals `forward_tokens().mean(dim=2)`.

    Uses ``random_init=True`` so no checkpoint download is needed — the
    architecture (and thus the token-grid plumbing) is identical. Still
    gated by the env flag because constructing the V-JEPA ViT is heavy.
    """
    from wormjepa.models.vjepa_loader import build_trainable_vjepa_encoder

    encoder = build_trainable_vjepa_encoder("vjepa2_1_vit_large_384", random_init=True)
    video = torch.randn(1, 16, 3, 384, 384)
    tokens = encoder.forward_tokens(video)
    assert tokens.ndim == 4, f"forward_tokens must return (B, T, S, D); got {tuple(tokens.shape)}"
    b, t, s, d = tokens.shape
    assert (b, t, d) == (1, 16, encoder.embed_dim)
    assert s > 1, f"V-JEPA spatial grid should have S>1 tokens; got S={s}"
    pooled = encoder(video)
    assert pooled.shape == (1, 16, encoder.embed_dim)
    assert torch.allclose(pooled, tokens.mean(dim=2), atol=1e-5), (
        "forward() must equal forward_tokens().mean(dim=2)."
    )
