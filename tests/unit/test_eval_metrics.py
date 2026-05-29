"""Unit tests for residualization + future-pose + motif ARI (Stories 6.1-6.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wormjepa.baselines.base import FuturePoseHorizon
from wormjepa.data import SessionID, WormID
from wormjepa.eval.future_pose import future_pose_metric
from wormjepa.eval.motif_ari import motif_ari
from wormjepa.eval.residualization import partial_r2

# --- partial_r2 ---


def test_partial_r2_zero_when_jepa_equals_kinematic() -> None:
    rng = np.random.default_rng(0)
    n_per_worm = 30
    n_worms = 4
    jepa = rng.normal(size=(n_per_worm * n_worms, 8))
    kin = jepa.copy()
    target = rng.normal(size=(n_per_worm * n_worms, 4))
    worms = [f"w{i}" for i in range(n_worms) for _ in range(n_per_worm)]
    result = partial_r2(jepa, kin, target, worms)
    assert abs(result.partial_r2) < 0.1


def test_partial_r2_positive_when_jepa_strictly_better() -> None:
    """Construct a case where JEPA latent perfectly predicts target and kinematic doesn't."""
    rng = np.random.default_rng(1)
    n_worms = 4
    n_per_worm = 40
    target = rng.normal(size=(n_per_worm * n_worms, 3))
    jepa = target.copy()  # perfect signal in JEPA
    kin = rng.normal(size=(n_per_worm * n_worms, 8))  # noise in kinematic
    worms = [f"w{i}" for i in range(n_worms) for _ in range(n_per_worm)]
    result = partial_r2(jepa, kin, target, worms)
    assert result.partial_r2 > 0.5


def test_partial_r2_requires_two_worms() -> None:
    j = np.zeros((10, 2))
    k = np.zeros((10, 2))
    t = np.zeros((10, 2))
    with pytest.raises(ValueError, match=">=2"):
        partial_r2(j, k, t, ["w"] * 10)


def test_partial_r2_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="first dim"):
        partial_r2(np.zeros((10, 2)), np.zeros((5, 2)), np.zeros((10, 2)), ["w"] * 10)


# --- future_pose_metric ---


def _fp(worm: str, horizon: float, k: int = 4) -> FuturePoseHorizon:
    return FuturePoseHorizon(
        worm_id=WormID(worm),
        session_id=SessionID(worm + "_s0"),
        horizon_seconds=horizon,
        predicted=torch.zeros((k, 2)),
        ground_truth=torch.ones((k, 2)),  # error = sqrt(2) per keypoint
    )


def test_future_pose_metric_three_horizons() -> None:
    preds = [_fp(f"w{i}", h) for i in range(4) for h in (0.1, 1.0, 5.0)]
    entry = future_pose_metric(preds, producer="kalman", n_bootstrap=50, method="percentile")
    assert entry.name == "future_pose"
    assert entry.producer == "kalman"
    assert {se.key for se in entry.sub_entries} == {"0.1s", "1s", "5s"}
    for sub in entry.sub_entries:
        assert sub.ci.grouping == "worm"
        # Per-clip error is sqrt(2) ≈ 1.414 for every entry; CI should bracket that.
        assert abs(sub.ci.point - float(np.sqrt(2))) < 0.05


# --- motif_ari ---


def test_motif_ari_perfectly_clusterable_data() -> None:
    """If latent perfectly encodes labels, Hungarian-matched ARI ≈ 1."""
    rng = np.random.default_rng(0)
    n_worms = 4
    n_per_worm = 50
    n_states = 3
    worms: list[str] = []
    labels: list[int] = []
    latents: list[np.ndarray] = []
    centroids = rng.normal(size=(n_states, 8)) * 5.0
    for i in range(n_worms):
        for _ in range(n_per_worm):
            s = int(rng.integers(0, n_states))
            latents.append(centroids[s] + rng.normal(size=(8,)) * 0.1)
            labels.append(s)
            worms.append(f"w{i}")
    result = motif_ari(np.asarray(latents), np.asarray(labels), worms, n_states=n_states)
    assert result.ari > 0.9


def test_motif_ari_requires_two_worms() -> None:
    with pytest.raises(ValueError, match=">=2"):
        motif_ari(np.zeros((10, 4)), np.zeros(10, dtype=np.int64), ["w"] * 10)


def test_motif_ari_random_labels_ari_near_zero() -> None:
    """Random labels independent of latent → ARI near zero."""
    rng = np.random.default_rng(1)
    n_worms = 4
    n_per_worm = 100
    latents = rng.normal(size=(n_worms * n_per_worm, 8))
    labels = rng.integers(0, 5, size=n_worms * n_per_worm)
    worms = [f"w{i}" for i in range(n_worms) for _ in range(n_per_worm)]
    result = motif_ari(latents, labels, worms)
    assert abs(result.ari) < 0.1
