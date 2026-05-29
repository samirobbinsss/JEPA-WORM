"""Unit tests for ``wormjepa.baselines.random_features``."""

from __future__ import annotations

import torch

from wormjepa.baselines.random_features import RandomFeaturesBaseline
from wormjepa.data.loaders.synthetic import SyntheticLoader


def _small_baseline(seed: int = 0) -> RandomFeaturesBaseline:
    return RandomFeaturesBaseline(
        latent_dim=8,
        n_layers=2,
        kernel_size=3,
        n_epochs=1,
        seed=seed,
    )


def test_name() -> None:
    assert RandomFeaturesBaseline().name == "random_features"


def test_fit_then_predict_returns_three_horizons_per_clip() -> None:
    loader = SyntheticLoader(n_worms=2, clips_per_worm=2, clip_frames=8, seed=0)
    baseline = _small_baseline().fit(loader)
    predictions = baseline.predict(
        SyntheticLoader(n_worms=2, clips_per_worm=2, clip_frames=8, seed=0)
    )
    assert len(predictions.future_pose) == 12


def test_encoder_is_frozen_after_fit() -> None:
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=6, seed=0)
    baseline = _small_baseline().fit(loader)
    assert baseline._encoder is not None
    for p in baseline._encoder.parameters():
        assert p.requires_grad is False


def test_same_seed_yields_same_initial_features() -> None:
    """Two baselines built with the same seed produce identical pre-training features."""
    flat = torch.randn(6, 20)
    a = _small_baseline(seed=42)
    b = _small_baseline(seed=42)
    a._encoder = a._build_frozen_encoder(input_dim=20)
    b._encoder = b._build_frozen_encoder(input_dim=20)
    with torch.no_grad():
        feat_a = a._encoder(flat.unsqueeze(0))
        feat_b = b._encoder(flat.unsqueeze(0))
    assert torch.allclose(feat_a, feat_b)


def test_different_seeds_yield_different_features() -> None:
    flat = torch.randn(6, 20)
    a = _small_baseline(seed=1)
    b = _small_baseline(seed=2)
    a._encoder = a._build_frozen_encoder(input_dim=20)
    b._encoder = b._build_frozen_encoder(input_dim=20)
    with torch.no_grad():
        feat_a = a._encoder(flat.unsqueeze(0))
        feat_b = b._encoder(flat.unsqueeze(0))
    assert not torch.allclose(feat_a, feat_b)


def test_latent_is_exposed() -> None:
    loader = SyntheticLoader(n_worms=2, clips_per_worm=1, clip_frames=6, seed=0)
    baseline = _small_baseline().fit(loader)
    predictions = baseline.predict(
        SyntheticLoader(n_worms=2, clips_per_worm=1, clip_frames=6, seed=0)
    )
    assert predictions.latent_by_worm is not None
    assert len(predictions.latent_by_worm) == 2


def test_predict_before_fit_raises() -> None:
    baseline = RandomFeaturesBaseline()
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4)
    try:
        baseline.predict(loader)
    except RuntimeError as exc:
        assert "fit" in str(exc)
        return
    raise AssertionError("predict() before fit() should raise RuntimeError")
