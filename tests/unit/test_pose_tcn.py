"""Unit tests for ``wormjepa.baselines.pose_tcn``."""

from __future__ import annotations

import torch

from wormjepa.baselines.pose_tcn import PoseOnlyTCNBaseline
from wormjepa.data.loaders.synthetic import SyntheticLoader


def _small_baseline() -> PoseOnlyTCNBaseline:
    return PoseOnlyTCNBaseline(
        latent_dim=8,
        n_layers=2,
        kernel_size=3,
        n_epochs=1,
    )


def test_name() -> None:
    assert PoseOnlyTCNBaseline().name == "pose_tcn"


def test_fit_then_predict_returns_three_horizons_per_clip() -> None:
    loader = SyntheticLoader(n_worms=2, clips_per_worm=2, clip_frames=8, seed=0)
    baseline = _small_baseline().fit(loader)
    predictions = baseline.predict(
        SyntheticLoader(n_worms=2, clips_per_worm=2, clip_frames=8, seed=0)
    )
    assert len(predictions.future_pose) == 12
    assert {fp.horizon_seconds for fp in predictions.future_pose} == {0.1, 1.0, 5.0}


def test_predicted_shape_matches_ground_truth() -> None:
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=8, seed=0)
    baseline = _small_baseline().fit(loader)
    predictions = baseline.predict(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=8, seed=0)
    )
    for fp in predictions.future_pose:
        assert fp.predicted.shape == fp.ground_truth.shape


def test_latent_is_exposed_per_worm() -> None:
    loader = SyntheticLoader(n_worms=3, clips_per_worm=2, clip_frames=6, seed=0)
    baseline = _small_baseline().fit(loader)
    predictions = baseline.predict(
        SyntheticLoader(n_worms=3, clips_per_worm=2, clip_frames=6, seed=0)
    )
    assert predictions.latent_by_worm is not None
    # Three unique worms; each contributes 2 clips * 6 frames = 12 frames.
    assert len(predictions.latent_by_worm) == 3
    for latent in predictions.latent_by_worm.values():
        assert latent.shape[0] == 12  # frames per worm
        assert latent.shape[1] == 8  # latent_dim


def test_predict_before_fit_raises() -> None:
    baseline = PoseOnlyTCNBaseline()
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4)
    try:
        baseline.predict(loader)
    except RuntimeError as exc:
        assert "fit" in str(exc)
        return
    raise AssertionError("predict() before fit() should raise RuntimeError")


def test_causal_tcn_no_future_leakage() -> None:
    """Modifying a future frame should not change the latent at earlier positions."""
    baseline = _small_baseline().fit(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=8, seed=0)
    )
    assert baseline._encoder is not None
    flat_a = torch.randn(8, 20)
    flat_b = flat_a.clone()
    flat_b[6:] = 99.0  # mutate the last two positions
    lat_a = baseline._encode_clip(flat_a)
    lat_b = baseline._encode_clip(flat_b)
    # Positions 0..5 must be identical under causal convolution.
    assert torch.allclose(lat_a[:6], lat_b[:6], atol=1e-5)
