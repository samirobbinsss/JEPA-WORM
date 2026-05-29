"""Unit tests for the Phase-A two-view hardware-pilot scaffolding."""

from __future__ import annotations

import pytest
import torch

from wormjepa.data.contract import DatasetSample, WormID
from wormjepa.hardware import CameraPair, MockCameraPair, TwoViewSample


def test_mock_camera_pair_yields_n_clips_with_correct_shape():
    rig = MockCameraPair(n_worms=2, height=32, width=32, channels=3)
    clips = list(rig.iter_samples(n_clips=4, clip_frames=8))
    assert len(clips) == 4
    for sample in clips:
        assert isinstance(sample, TwoViewSample)
        assert sample.video_primary.shape == (8, 3, 32, 32)
        assert sample.video_secondary.shape == (8, 3, 32, 32)
        assert sample.timestamps.shape == (8,)
        assert sample.rig_id == "mock_rig_0"
        assert sample.session_id.startswith("mock_rig_0_session_")


def test_mock_camera_pair_is_deterministic_across_runs():
    rig_a = MockCameraPair(n_worms=2, height=16, width=16, seed=42)
    rig_b = MockCameraPair(n_worms=2, height=16, width=16, seed=42)
    a = list(rig_a.iter_samples(n_clips=2, clip_frames=4))
    b = list(rig_b.iter_samples(n_clips=2, clip_frames=4))
    for sa, sb in zip(a, b, strict=True):
        assert torch.equal(sa.video_primary, sb.video_primary)
        assert torch.equal(sa.video_secondary, sb.video_secondary)


def test_mock_camera_pair_different_seeds_produce_different_clips():
    rig_a = MockCameraPair(n_worms=1, height=16, width=16, seed=42)
    rig_b = MockCameraPair(n_worms=1, height=16, width=16, seed=43)
    sample_a = next(rig_a.iter_samples(n_clips=1, clip_frames=4))
    sample_b = next(rig_b.iter_samples(n_clips=1, clip_frames=4))
    assert not torch.equal(sample_a.video_primary, sample_b.video_primary)


def test_two_view_sample_validates_time_alignment():
    bad_primary = torch.zeros((8, 3, 16, 16))
    bad_secondary = torch.zeros((7, 3, 16, 16))  # T mismatch
    with pytest.raises(ValueError, match="not time-aligned"):
        TwoViewSample(
            video_primary=bad_primary,
            video_secondary=bad_secondary,
            timestamps=torch.arange(8, dtype=torch.float32),
            worm_id=WormID("w0"),
            session_id="s0",
            rig_id="r0",
        )


def test_two_view_sample_validates_rank():
    bad = torch.zeros((3, 16, 16))  # missing T dim
    with pytest.raises(ValueError, match="expected"):
        TwoViewSample(
            video_primary=bad,
            video_secondary=bad,
            timestamps=torch.arange(3, dtype=torch.float32),
            worm_id=WormID("w0"),
            session_id="s0",
            rig_id="r0",
        )


def test_two_view_sample_collapses_to_dataset_sample():
    rig = MockCameraPair(n_worms=1, height=8, width=8)
    two_view = next(rig.iter_samples(n_clips=1, clip_frames=4))
    ds = two_view.to_dataset_sample()
    assert isinstance(ds, DatasetSample)
    assert torch.equal(ds.video_clip, two_view.video_primary)
    # Secondary view is dropped at this layer; future multi-view encoder
    # will read it off the TwoViewSample directly.
    assert ds.pose is None
    assert ds.neural is None
    assert ds.worm_id == two_view.worm_id


def test_camera_pair_is_abstract():
    with pytest.raises(TypeError):
        CameraPair()  # type: ignore[abstract]


def test_mock_camera_pair_refuses_after_close():
    rig = MockCameraPair(n_worms=1, height=8, width=8)
    rig.close()
    with pytest.raises(RuntimeError, match="after close"):
        list(rig.iter_samples(n_clips=1, clip_frames=2))
