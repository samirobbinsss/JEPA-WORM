"""Unit tests for the exploratory eval probes (Exp A pose-decodability, Exp E
held-out baseline split). These are env-gated, fire no pre-registered gate, and
exist to test whether the kill_criterion firing is a readout/baseline artifact.
"""

# Deliberately exercises private orchestrator internals with duck-typed test
# doubles (SimpleNamespace encoder, object() cfg); relax the two strict checks
# those inevitably trip.
# pyright: reportPrivateUsage=false, reportArgumentType=false

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wormjepa.data import WormID
from wormjepa.data.contract import DatasetSample, SourceDataset
from wormjepa.eval import orchestrator
from wormjepa.eval.residualization import partial_r2


def test_pose_decodability_partial_r2_detects_pooling_info_loss() -> None:
    """Exp A core: partial_r2(spatial, pooled, pose) is positive when pose is
    decodable from the spatial feature but washed out of the pooled feature -
    i.e. the metric detects mean-pooling discarding pose-relevant structure."""
    rng = np.random.default_rng(0)
    n_worms, n_per = 4, 50
    n = n_worms * n_per
    pose = rng.normal(size=(n, 8))  # K*2 pose coords
    # spatial carries pose (decodes cross-worm); pooled has it destroyed.
    spatial = np.concatenate(
        [pose + 0.05 * rng.normal(size=(n, 8)), rng.normal(size=(n, 4))], axis=1
    )
    pooled = rng.normal(size=(n, 12))
    worms = [f"w{i}" for i in range(n_worms) for _ in range(n_per)]
    result = partial_r2(
        jepa_latent=spatial, kinematic_features=pooled, neural_target=pose, worm_ids=worms
    )
    # r2_jepa == spatial-pose-R^2; r2_kinematic == pooled-pose-R^2; partial == spatial - pooled.
    assert result.r2_jepa > result.r2_kinematic
    assert result.partial_r2 > 0.0
    assert result.r2_jepa > 0.5  # spatial decodes pose well
    assert result.r2_kinematic < 0.1  # pooled does not


class _TinyTokenEncoder(nn.Module):
    """Minimal encoder exposing ``forward_tokens`` -> (B, T, S, D) with tokens
    that vary across S, so the [mean, std, max] summary is non-degenerate."""

    def __init__(self, s: int = 3, d: int = 4) -> None:
        super().__init__()
        self.s, self.d = s, d
        self._p = nn.Parameter(torch.zeros(1))  # gives the module a device

    def forward_tokens(self, video: torch.Tensor) -> torch.Tensor:
        b, t = video.shape[0], video.shape[1]
        base = video.reshape(b, t, -1).mean(-1)  # (B, T)
        grid = torch.arange(self.s * self.d, dtype=torch.float32).reshape(self.s, self.d)
        return base[:, :, None, None] + grid  # (B, T, S, D)


def test_spatial_token_features_shape_and_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exp A plumbing: [mean,std,max] over S -> (Nframes, 3D), with rows aligned
    one-to-one to per-frame worm ids in loader order."""
    state = SimpleNamespace(online_encoder=_TinyTokenEncoder(s=3, d=4))

    def _fake_loader(_cfg: object):
        for w in range(2):
            yield DatasetSample(
                video_clip=torch.rand(5, 3, 8, 8),
                pose=torch.rand(5, 4, 2),
                neural=torch.rand(5, 8),
                worm_id=WormID(f"w{w}"),
                session_id="s",
                source_dataset=SourceDataset("baaiworm"),
            )

    monkeypatch.setattr(orchestrator, "_build_eval_loader", _fake_loader)
    feats, worms = orchestrator._spatial_token_features(state, object(), max_clips=2)
    assert feats.shape == (10, 12)  # 2 worms x 5 frames; 3 (mean/std/max) x D=4
    assert worms == ["w0"] * 5 + ["w1"] * 5


def test_spatial_token_features_skips_encoder_without_forward_tokens() -> None:
    """A legacy encoder lacking forward_tokens -> empty so the probe skips."""
    state = SimpleNamespace(online_encoder=nn.Linear(2, 2))
    feats, worms = orchestrator._spatial_token_features(state, object(), max_clips=4)
    assert feats.size == 0
    assert worms == []


def test_exp_e_baseline_split_is_disjoint_and_default_unchanged() -> None:
    """Exp E: the held-out baseline offset (20000) differs from the canonical
    eval offset (10000) so the baseline never sees the eval cohort. The default
    offset must stay 10000 (cache-key stability)."""
    assert orchestrator._BASELINE_TRAIN_SEED_OFFSET == 20000
    # Default keyword is 10000 - verified via the function signature default.
    import inspect

    sig = inspect.signature(orchestrator._build_eval_loader_spec)
    assert sig.parameters["seed_offset"].default == 10000
