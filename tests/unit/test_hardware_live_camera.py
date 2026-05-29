"""Unit tests for the OpenCV-backed live two-camera driver.

Real cameras are unavailable in CI, so these tests patch
``cv2.VideoCapture`` and assert the driver's contract behaviour:
shapes, dtypes, range, BGR→RGB conversion, clean shutdown, and
graceful abort when a capture stops producing frames.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from wormjepa.data.contract import DatasetSample
from wormjepa.hardware import LiveCameraPair, TwoViewSample
from wormjepa.hardware.two_view_contract import IN_HOUSE_LIVE_SOURCE

_HEIGHT = 4
_WIDTH = 6


def _pop_factory(captures: list[Any]):
    """Return a callable that pops a queued mock VideoCapture per call.

    Wraps the queue in a named function so pyright can infer the
    parameter type — bare lambdas trip ``reportUnknownLambdaType``.
    """

    def _factory(_src: int | str) -> Any:
        return captures.pop(0)

    return _factory


def _make_capture(
    frames: list[np.ndarray] | None = None,
    *,
    is_opened: bool = True,
    grab_fails_at: int | None = None,
    retrieve_fails_at: int | None = None,
) -> MagicMock:
    """Build a mock that emulates :class:`cv2.VideoCapture`.

    Each call to ``retrieve()`` returns the next frame in ``frames``
    (looping if exhausted). ``grab_fails_at`` / ``retrieve_fails_at``
    let a single call return ``False`` on the Nth invocation so we can
    exercise the abort-mid-clip branch.
    """
    cap = MagicMock()
    cap.isOpened.return_value = is_opened

    frames = frames if frames is not None else [np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)]
    grab_counter = {"n": 0}
    retrieve_counter = {"n": 0}

    def _grab() -> bool:
        grab_counter["n"] += 1
        return not (grab_fails_at is not None and grab_counter["n"] == grab_fails_at)

    def _retrieve() -> tuple[bool, np.ndarray]:
        idx = retrieve_counter["n"]
        retrieve_counter["n"] += 1
        if retrieve_fails_at is not None and retrieve_counter["n"] == retrieve_fails_at:
            return False, np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        return True, frames[idx % len(frames)]

    cap.grab.side_effect = _grab
    cap.retrieve.side_effect = _retrieve
    return cap


def test_constructor_opens_both_captures_with_given_indices() -> None:
    """``iter_samples`` must instantiate VideoCapture for both sources."""
    cap_p = _make_capture()
    cap_s = _make_capture()
    captures: list[Any] = [cap_p, cap_s]
    seen_args: list[int | str] = []

    def _factory(src: int | str) -> Any:
        seen_args.append(src)
        return captures.pop(0)

    with patch("wormjepa.hardware.live_camera.cv2.VideoCapture", side_effect=_factory):
        rig = LiveCameraPair(0, 1, rig_id="testlive")
        try:
            # Consume one clip to force open of both captures.
            list(rig.iter_samples(n_clips=1, clip_frames=2))
        finally:
            rig.close()

    assert seen_args == [0, 1]
    cap_p.isOpened.assert_called()
    cap_s.isOpened.assert_called()


def test_iter_samples_yields_two_view_sample_with_correct_shape_dtype_range() -> None:
    frame = np.full((_HEIGHT, _WIDTH, 3), 128, dtype=np.uint8)
    cap_p = _make_capture(frames=[frame])
    cap_s = _make_capture(frames=[frame])
    captures: list[Any] = [cap_p, cap_s]

    with patch(
        "wormjepa.hardware.live_camera.cv2.VideoCapture",
        side_effect=_pop_factory(captures),
    ):
        rig = LiveCameraPair(0, 1, rig_id="testlive")
        try:
            clips = list(rig.iter_samples(n_clips=2, clip_frames=3))
        finally:
            rig.close()

    assert len(clips) == 2
    for clip_idx, sample in enumerate(clips):
        assert isinstance(sample, TwoViewSample)
        assert sample.video_primary.shape == (3, 3, _HEIGHT, _WIDTH)
        assert sample.video_secondary.shape == (3, 3, _HEIGHT, _WIDTH)
        assert sample.video_primary.dtype == torch.float32
        assert sample.video_secondary.dtype == torch.float32
        assert sample.video_primary.min() >= 0.0
        assert sample.video_primary.max() <= 1.0
        assert sample.timestamps.shape == (3,)
        assert sample.timestamps[0].item() >= 0.0
        assert sample.rig_id == "testlive"
        assert sample.session_id == f"testlive_clip_{clip_idx:04d}"
        assert sample.worm_id == f"live_worm_{clip_idx:04d}"
        assert sample.source_dataset == IN_HOUSE_LIVE_SOURCE


def test_bgr_to_rgb_conversion_applied() -> None:
    """A BGR frame of (B=10, G=20, R=200) must surface as RGB (200, 20, 10)."""
    bgr = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 200  # R

    cap_p = _make_capture(frames=[bgr])
    cap_s = _make_capture(frames=[bgr])
    captures: list[Any] = [cap_p, cap_s]

    with patch(
        "wormjepa.hardware.live_camera.cv2.VideoCapture",
        side_effect=_pop_factory(captures),
    ):
        rig = LiveCameraPair(0, 1)
        try:
            sample = next(rig.iter_samples(n_clips=1, clip_frames=1))
        finally:
            rig.close()

    # Tensor layout is (T, C, H, W). Channel 0 = R, 1 = G, 2 = B post-conversion.
    # Pixel values normalised by /255.
    primary = sample.video_primary[0]  # (C, H, W)
    assert primary[0].mean().item() == pytest.approx(200 / 255.0)
    assert primary[1].mean().item() == pytest.approx(20 / 255.0)
    assert primary[2].mean().item() == pytest.approx(10 / 255.0)


def test_close_calls_release_on_both_captures() -> None:
    cap_p = _make_capture()
    cap_s = _make_capture()
    captures: list[Any] = [cap_p, cap_s]

    with patch(
        "wormjepa.hardware.live_camera.cv2.VideoCapture",
        side_effect=_pop_factory(captures),
    ):
        rig = LiveCameraPair(0, 1)
        list(rig.iter_samples(n_clips=1, clip_frames=2))
        rig.close()

    cap_p.release.assert_called_once()
    cap_s.release.assert_called_once()

    # Re-closing is idempotent and further iteration is blocked.
    rig.close()
    with pytest.raises(RuntimeError, match="after close"):
        list(rig.iter_samples(n_clips=1, clip_frames=1))


def test_grab_failure_mid_clip_aborts_without_partial_emission() -> None:
    """If grab() returns False on frame 2 of a 3-frame clip, the clip is dropped."""
    cap_p = _make_capture(grab_fails_at=2)
    cap_s = _make_capture()
    captures: list[Any] = [cap_p, cap_s]

    with patch(
        "wormjepa.hardware.live_camera.cv2.VideoCapture",
        side_effect=_pop_factory(captures),
    ):
        rig = LiveCameraPair(0, 1)
        try:
            clips = list(rig.iter_samples(n_clips=1, clip_frames=3))
        finally:
            rig.close()

    assert clips == []


def test_retrieve_failure_mid_clip_aborts_without_partial_emission() -> None:
    """retrieve() returning False also aborts cleanly, no partial TwoViewSample."""
    cap_p = _make_capture()
    cap_s = _make_capture(retrieve_fails_at=2)
    captures: list[Any] = [cap_p, cap_s]

    with patch(
        "wormjepa.hardware.live_camera.cv2.VideoCapture",
        side_effect=_pop_factory(captures),
    ):
        rig = LiveCameraPair(0, 1)
        try:
            clips = list(rig.iter_samples(n_clips=2, clip_frames=3))
        finally:
            rig.close()

    # First clip should fail at frame 2; second clip never starts.
    assert clips == []


def test_grab_failure_after_first_full_clip_keeps_first_clip() -> None:
    """If clip 0 completes cleanly and grab() fails inside clip 1, clip 0 still ships."""
    # 3 frames * 1 successful clip = 3 grabs. Fail on grab 4 (first grab of clip 1).
    cap_p = _make_capture(grab_fails_at=4)
    cap_s = _make_capture()
    captures: list[Any] = [cap_p, cap_s]

    with patch(
        "wormjepa.hardware.live_camera.cv2.VideoCapture",
        side_effect=_pop_factory(captures),
    ):
        rig = LiveCameraPair(0, 1)
        try:
            clips = list(rig.iter_samples(n_clips=5, clip_frames=3))
        finally:
            rig.close()

    assert len(clips) == 1
    assert clips[0].video_primary.shape == (3, 3, _HEIGHT, _WIDTH)


def test_isopened_false_raises() -> None:
    cap_p = _make_capture(is_opened=False)
    cap_s = _make_capture()
    captures: list[Any] = [cap_p, cap_s]

    with patch(
        "wormjepa.hardware.live_camera.cv2.VideoCapture",
        side_effect=_pop_factory(captures),
    ):
        rig = LiveCameraPair(0, 1)
        try:
            with pytest.raises(RuntimeError, match="primary capture failed to open"):
                list(rig.iter_samples(n_clips=1, clip_frames=1))
        finally:
            rig.close()


def test_string_device_path_passed_through() -> None:
    """``/dev/video0``-style device paths must reach VideoCapture unchanged."""
    cap_p = _make_capture()
    cap_s = _make_capture()
    captures: list[Any] = [cap_p, cap_s]
    seen_args: list[int | str] = []

    def _factory(src: int | str) -> Any:
        seen_args.append(src)
        return captures.pop(0)

    with patch("wormjepa.hardware.live_camera.cv2.VideoCapture", side_effect=_factory):
        rig = LiveCameraPair("/dev/video0", "/dev/video1")
        try:
            list(rig.iter_samples(n_clips=1, clip_frames=1))
        finally:
            rig.close()

    assert seen_args == ["/dev/video0", "/dev/video1"]


def test_invalid_channels_rejected() -> None:
    with pytest.raises(ValueError, match="channels must be 3"):
        LiveCameraPair(0, 1, channels=1)


def test_invalid_fps_rejected() -> None:
    with pytest.raises(ValueError, match="fps must be > 0"):
        LiveCameraPair(0, 1, fps=0)


def test_negative_n_clips_rejected() -> None:
    rig = LiveCameraPair(0, 1)
    try:
        with pytest.raises(ValueError, match="n_clips must be >= 0"):
            list(rig.iter_samples(n_clips=-1, clip_frames=1))
    finally:
        rig.close()


def test_zero_clip_frames_rejected() -> None:
    rig = LiveCameraPair(0, 1)
    try:
        with pytest.raises(ValueError, match="clip_frames must be >= 1"):
            list(rig.iter_samples(n_clips=1, clip_frames=0))
    finally:
        rig.close()


def test_to_dataset_sample_preserves_primary_view() -> None:
    frame = np.full((_HEIGHT, _WIDTH, 3), 64, dtype=np.uint8)
    cap_p = _make_capture(frames=[frame])
    cap_s = _make_capture(frames=[frame])
    captures: list[Any] = [cap_p, cap_s]

    with patch(
        "wormjepa.hardware.live_camera.cv2.VideoCapture",
        side_effect=_pop_factory(captures),
    ):
        rig = LiveCameraPair(0, 1, rig_id="rl")
        try:
            clip = next(rig.iter_samples(n_clips=1, clip_frames=2))
        finally:
            rig.close()

    ds = clip.to_dataset_sample()
    assert isinstance(ds, DatasetSample)
    assert torch.equal(ds.video_clip, clip.video_primary)
    assert ds.pose is None
    assert ds.neural is None
    assert ds.worm_id == clip.worm_id
    assert str(ds.session_id) == clip.session_id
    assert ds.source_dataset == IN_HOUSE_LIVE_SOURCE
