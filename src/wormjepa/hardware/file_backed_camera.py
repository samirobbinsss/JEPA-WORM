"""File-backed two-camera driver (Phase A prep).

Decodes two pre-recorded MP4 (or any PyAV-decodable) files synchronously
and yields :class:`TwoViewSample` clips. Useful for:

- Replaying captures from a previous rig session against the live
  training pipeline (no camera attached).
- Exercising the :class:`CameraPair` contract path with real video data
  before any physical rig exists.
- Fixture-driven testing where deterministic byte-identical input
  matters (mock would do, but a real container round-trip is closer to
  the production code path).

The driver assumes the two files were captured frame-locked — frame N
of primary is taken to correspond to frame N of secondary. If your rig
emits separately-timed streams, normalise them upstream before pointing
this driver at them.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import av
import torch

from wormjepa.data.contract import WormID
from wormjepa.hardware.camera_pair import CameraPair
from wormjepa.hardware.two_view_contract import SYNTHETIC_SOURCE, TwoViewSample

if TYPE_CHECKING:
    from av.container.input import InputContainer
    from av.video.stream import VideoStream

_DEFAULT_RIG_ID = "file_rig_0"
_DEFAULT_FPS = 30.0


class FileBackedCameraPair(CameraPair):
    """Two-camera driver that reads frames from a pair of video files.

    Args:
        primary_path: Path to the primary-view video container.
        secondary_path: Path to the secondary-view video container.
        rig_id: Identifier stamped on each yielded :class:`TwoViewSample`.
        start_frame: Number of leading frames to skip on both streams
            before any clip is emitted. Lets callers seek past a sync
            preamble that the rig records before the experiment starts.

    The PyAV containers are opened lazily on the first
    :meth:`iter_samples` call and re-opened on every subsequent call
    (each iteration starts from ``start_frame``). They are closed by
    :meth:`close`; calling :meth:`iter_samples` after :meth:`close`
    raises :class:`RuntimeError`.
    """

    def __init__(
        self,
        primary_path: Path,
        secondary_path: Path,
        *,
        rig_id: str = _DEFAULT_RIG_ID,
        start_frame: int = 0,
    ) -> None:
        if start_frame < 0:
            msg = f"FileBackedCameraPair: start_frame must be >= 0, got {start_frame}"
            raise ValueError(msg)
        self._primary_path = Path(primary_path)
        self._secondary_path = Path(secondary_path)
        self._rig_id = rig_id
        self._start_frame = start_frame
        self._primary_container: InputContainer | None = None
        self._secondary_container: InputContainer | None = None
        self._closed = False

    def iter_samples(self, n_clips: int, *, clip_frames: int) -> Iterator[TwoViewSample]:
        if self._closed:
            msg = "FileBackedCameraPair.iter_samples called after close()"
            raise RuntimeError(msg)
        if n_clips < 0:
            msg = f"n_clips must be >= 0, got {n_clips}"
            raise ValueError(msg)
        if clip_frames < 1:
            msg = f"clip_frames must be >= 1, got {clip_frames}"
            raise ValueError(msg)
        if not self._primary_path.is_file():
            msg = f"FileBackedCameraPair: primary video not found: {self._primary_path}"
            raise FileNotFoundError(msg)
        if not self._secondary_path.is_file():
            msg = f"FileBackedCameraPair: secondary video not found: {self._secondary_path}"
            raise FileNotFoundError(msg)

        # Close any previously-opened containers so iter_samples is restartable.
        self._close_containers()
        self._primary_container = av.open(str(self._primary_path))
        self._secondary_container = av.open(str(self._secondary_path))

        primary_stream = self._primary_container.streams.video[0]
        secondary_stream = self._secondary_container.streams.video[0]
        primary_stream.thread_type = "AUTO"
        secondary_stream.thread_type = "AUTO"

        fps = _average_fps(primary_stream)

        primary_iter = self._primary_container.decode(primary_stream)
        secondary_iter = self._secondary_container.decode(secondary_stream)

        # Honour start_frame by discarding leading decoded frames.
        for _ in range(self._start_frame):
            if next(primary_iter, None) is None or next(secondary_iter, None) is None:
                # Ran out before we even got to start_frame — yield nothing.
                return

        global_frame_idx = self._start_frame
        for clip_idx in range(n_clips):
            primary_buf: list[torch.Tensor] = []
            secondary_buf: list[torch.Tensor] = []
            timestamp_buf: list[float] = []
            for _ in range(clip_frames):
                primary_frame = next(primary_iter, None)
                secondary_frame = next(secondary_iter, None)
                if primary_frame is None or secondary_frame is None:
                    # Either stream ran dry — stop without emitting a partial clip.
                    return
                primary_buf.append(_frame_to_tensor(primary_frame))
                secondary_buf.append(_frame_to_tensor(secondary_frame))
                timestamp_buf.append(global_frame_idx / fps)
                global_frame_idx += 1

            worm_id = WormID(f"file_worm_{clip_idx:04d}")
            session_id = f"{self._rig_id}_clip_{clip_idx:04d}"
            yield TwoViewSample(
                video_primary=torch.stack(primary_buf, dim=0),
                video_secondary=torch.stack(secondary_buf, dim=0),
                timestamps=torch.tensor(timestamp_buf, dtype=torch.float32),
                worm_id=worm_id,
                session_id=session_id,
                rig_id=self._rig_id,
                source_dataset=SYNTHETIC_SOURCE,
            )

    def close(self) -> None:
        self._close_containers()
        self._closed = True

    def _close_containers(self) -> None:
        if self._primary_container is not None:
            self._primary_container.close()
            self._primary_container = None
        if self._secondary_container is not None:
            self._secondary_container.close()
            self._secondary_container = None


def _average_fps(stream: VideoStream) -> float:
    """Return the stream's average fps, falling back to a sensible default.

    PyAV exposes ``average_rate`` as a :class:`fractions.Fraction` or
    ``None`` for streams that don't advertise it. We coerce to float
    when present and default to 30 fps otherwise — the timestamp axis
    is informational here, not a metrology-grade timebase.
    """
    rate: Fraction | None = stream.average_rate
    if rate is None or rate == 0:
        return _DEFAULT_FPS
    return float(rate)


def _frame_to_tensor(frame: av.VideoFrame) -> torch.Tensor:
    """Convert a PyAV ``VideoFrame`` to ``(3, H, W)`` float32 in [0, 1]."""
    rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
