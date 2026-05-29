"""Unit tests for ``wormjepa.baselines.kalman``."""

from __future__ import annotations

import torch

from wormjepa.baselines.kalman import KalmanBaseline
from wormjepa.data.loaders.synthetic import SyntheticLoader


def test_kalman_baseline_name() -> None:
    assert KalmanBaseline().name == "kalman"


def test_kalman_baseline_fit_returns_self() -> None:
    baseline = KalmanBaseline()
    loader = SyntheticLoader(n_worms=2, clips_per_worm=1, clip_frames=4, seed=0)
    assert baseline.fit(loader) is baseline


def test_kalman_baseline_predict_returns_three_horizons_per_clip() -> None:
    baseline = KalmanBaseline().fit(
        SyntheticLoader(n_worms=2, clips_per_worm=2, clip_frames=8, seed=1)
    )
    predictions = baseline.predict(
        SyntheticLoader(n_worms=2, clips_per_worm=2, clip_frames=8, seed=1)
    )
    # 2 worms x 2 clips x 3 horizons = 12 entries.
    assert len(predictions.future_pose) == 12
    assert {h.horizon_seconds for h in predictions.future_pose} == {0.1, 1.0, 5.0}


def test_kalman_baseline_predict_persistence_for_small_horizon() -> None:
    """At the smallest horizon (0.1s = 1 frame at 0.1s/frame), pred = pose[T-2]."""
    baseline = KalmanBaseline().fit(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4, seed=42)
    )
    predictions = baseline.predict(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4, seed=42)
    )
    one_horizon = next(p for p in predictions.future_pose if p.horizon_seconds == 0.1)
    # Reconstruct the source frame ourselves from the same loader.
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4, seed=42)
    sample = next(iter(loader))
    assert sample.pose is not None
    expected = sample.pose[sample.pose.shape[0] - 2]  # 1-frame lookback
    assert torch.allclose(one_horizon.predicted, expected)


def test_kalman_baseline_predict_clips_to_zero_for_long_horizon() -> None:
    """At a horizon longer than the clip, source index clips to frame 0."""
    baseline = KalmanBaseline().fit(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4, seed=0)
    )
    predictions = baseline.predict(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4, seed=0)
    )
    five_s = next(p for p in predictions.future_pose if p.horizon_seconds == 5.0)
    loader = SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4, seed=0)
    sample = next(iter(loader))
    assert sample.pose is not None
    assert torch.allclose(five_s.predicted, sample.pose[0])


def test_kalman_baseline_no_latent() -> None:
    baseline = KalmanBaseline().fit(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4, seed=0)
    )
    predictions = baseline.predict(
        SyntheticLoader(n_worms=1, clips_per_worm=1, clip_frames=4, seed=0)
    )
    assert predictions.latent_by_worm is None


def test_kalman_baseline_skips_samples_without_pose() -> None:
    """Samples whose ``pose`` is None (e.g., WormBehavior DB) are silently skipped.

    Future-pose prediction is undefined without pose; the baseline can't
    invent ground truth.
    """

    from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID

    def _no_pose_samples() -> object:
        yield DatasetSample(
            video_clip=torch.zeros((4, 3, 8, 8)),
            pose=None,
            neural=None,
            worm_id=WormID("w"),
            session_id=SessionID("s"),
            source_dataset=SourceDataset("synthetic"),
        )

    baseline = KalmanBaseline()
    predictions = baseline.predict(_no_pose_samples())  # type: ignore[arg-type]
    assert predictions.future_pose == []
