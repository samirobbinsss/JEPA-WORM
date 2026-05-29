"""Unit tests for the file-backed two-camera driver."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
import torch

from wormjepa.data.contract import DatasetSample
from wormjepa.hardware import FileBackedCameraPair, TwoViewSample

_N_FRAMES = 32
_HEIGHT = 8
_WIDTH = 8
_FPS = 8


def _write_mp4(path: Path, *, fill_value: int, n_frames: int = _N_FRAMES) -> None:
    """Encode an ``n_frames``-frame MP4 of solid-coloured frames.

    The frames are not all identical — a per-frame ramp on the green
    channel makes them distinguishable, which is enough to verify the
    decoder is producing real per-frame data (not the same array
    repeated). The red channel carries ``fill_value`` so primary and
    secondary streams can be told apart by content.
    """
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("h264", rate=_FPS)
        stream.width = _WIDTH
        stream.height = _HEIGHT
        stream.pix_fmt = "yuv420p"
        for t in range(n_frames):
            arr = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
            arr[..., 0] = fill_value  # red — distinguishes primary vs secondary
            arr[..., 1] = (t * 8) % 256  # green ramp — distinguishes frames
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


@pytest.fixture
def video_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Return paths to a (primary, secondary) MP4 pair on disk."""
    primary = tmp_path / "primary.mp4"
    secondary = tmp_path / "secondary.mp4"
    _write_mp4(primary, fill_value=50)
    _write_mp4(secondary, fill_value=200)
    return primary, secondary


def test_iter_samples_yields_expected_clips_and_shape(video_pair: tuple[Path, Path]) -> None:
    primary, secondary = video_pair
    rig = FileBackedCameraPair(primary, secondary, rig_id="testrig")
    try:
        clips = list(rig.iter_samples(n_clips=3, clip_frames=8))
    finally:
        rig.close()

    assert len(clips) == 3
    for clip_idx, sample in enumerate(clips):
        assert isinstance(sample, TwoViewSample)
        assert sample.video_primary.shape == (8, 3, _HEIGHT, _WIDTH)
        assert sample.video_secondary.shape == (8, 3, _HEIGHT, _WIDTH)
        assert sample.video_primary.dtype == torch.float32
        assert sample.video_primary.min() >= 0.0
        assert sample.video_primary.max() <= 1.0
        assert sample.timestamps.shape == (8,)
        assert sample.rig_id == "testrig"
        assert sample.session_id == f"testrig_clip_{clip_idx:04d}"
        assert sample.worm_id == f"file_worm_{clip_idx:04d}"


def test_iter_samples_stops_when_files_run_out(video_pair: tuple[Path, Path]) -> None:
    """32 source frames at clip_frames=10 ⇒ 3 full clips, 4th is truncated."""
    primary, secondary = video_pair
    rig = FileBackedCameraPair(primary, secondary)
    try:
        clips = list(rig.iter_samples(n_clips=10, clip_frames=10))
    finally:
        rig.close()
    assert len(clips) == 3


def test_time_alignment_post_init_accepts_synthesised_timestamps(
    video_pair: tuple[Path, Path],
) -> None:
    """The TwoViewSample contract validates time-alignment in __post_init__.

    A successfully-yielded sample is implicit proof that primary[i] and
    secondary[i] share the same length and that ``timestamps`` matches
    that length. We additionally check the explicit timestamp values:
    frame 0 is at t=0, the cadence equals 1/fps.
    """
    primary, secondary = video_pair
    rig = FileBackedCameraPair(primary, secondary)
    try:
        clip = next(rig.iter_samples(n_clips=1, clip_frames=4))
    finally:
        rig.close()
    assert clip.timestamps[0].item() == pytest.approx(0.0)
    expected_step = 1.0 / _FPS
    diffs = clip.timestamps[1:] - clip.timestamps[:-1]
    for d in diffs.tolist():
        assert d == pytest.approx(expected_step)


def test_close_releases_handles_and_blocks_further_iteration(
    video_pair: tuple[Path, Path],
) -> None:
    primary, secondary = video_pair
    rig = FileBackedCameraPair(primary, secondary)
    list(rig.iter_samples(n_clips=1, clip_frames=4))
    rig.close()
    # Re-closing is idempotent.
    rig.close()
    with pytest.raises(RuntimeError, match="after close"):
        list(rig.iter_samples(n_clips=1, clip_frames=4))


def test_to_dataset_sample_preserves_primary_view(video_pair: tuple[Path, Path]) -> None:
    primary, secondary = video_pair
    rig = FileBackedCameraPair(primary, secondary, rig_id="rt")
    try:
        clip = next(rig.iter_samples(n_clips=1, clip_frames=4))
    finally:
        rig.close()
    ds = clip.to_dataset_sample()
    assert isinstance(ds, DatasetSample)
    assert torch.equal(ds.video_clip, clip.video_primary)
    assert ds.pose is None
    assert ds.neural is None
    assert ds.worm_id == clip.worm_id
    assert str(ds.session_id) == clip.session_id


def test_start_frame_seeks_past_preamble(video_pair: tuple[Path, Path]) -> None:
    """start_frame=8 should skip 8 frames before clip 0, so timestamps start at 8/fps."""
    primary, secondary = video_pair
    rig = FileBackedCameraPair(primary, secondary, start_frame=8)
    try:
        clip = next(rig.iter_samples(n_clips=1, clip_frames=4))
    finally:
        rig.close()
    assert clip.timestamps[0].item() == pytest.approx(8.0 / _FPS)


def test_missing_primary_file_raises(tmp_path: Path) -> None:
    primary = tmp_path / "does_not_exist.mp4"
    secondary = tmp_path / "also_missing.mp4"
    rig = FileBackedCameraPair(primary, secondary)
    try:
        with pytest.raises(FileNotFoundError, match="primary video not found"):
            list(rig.iter_samples(n_clips=1, clip_frames=2))
    finally:
        rig.close()


def test_negative_start_frame_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_frame must be >= 0"):
        FileBackedCameraPair(tmp_path / "a.mp4", tmp_path / "b.mp4", start_frame=-1)
