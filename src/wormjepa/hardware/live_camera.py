"""OpenCV-backed live two-camera driver (Phase A prep).

Drives two ``cv2.VideoCapture`` handles synchronously and yields
:class:`TwoViewSample` clips at the requested ``clip_frames`` cadence.
This is the first driver in :mod:`wormjepa.hardware` that targets a
physical rig — the mock and file-backed siblings cover the no-camera
case.

Synchronisation strategy
------------------------

The driver uses OpenCV's two-step ``grab()`` / ``retrieve()`` pattern
to keep the two cameras as time-aligned as possible **in software**:

1. Call ``grab()`` on both captures back-to-back. ``grab()`` returns
   immediately once the next frame is latched in the driver; doing the
   two ``grab()`` calls one after the other minimises the inter-camera
   shutter offset to ~one bus / USB transfer.
2. Call ``retrieve()`` on each capture to actually pull the latched
   frame out as a numpy array. ``retrieve()`` does the heavy decoding
   work and can be slower than ``grab()`` without harming sync.

Important caveats — this is **software sync, not hardware sync**:

- Genuine frame-locked acquisition requires a shared trigger line
  (e.g. a GPIO master pulse fan-out into both cameras' trigger inputs)
  and the two captures emitting frames on the rising edge of that
  pulse. OpenCV's ``VideoCapture`` cannot drive a trigger line; the
  rig-master-trigger approach is Phase A hardware-pilot work.
- The timestamps emitted here come from :func:`time.monotonic` taken
  **after** both retrieves on each frame. They are wall-clock
  references useful for spotting dropped frames, **not**
  metrology-grade frame timestamps.
- If your rig already supports hardware trigger and you want
  trigger-accurate timestamps, replace this driver with one that
  reads the camera's on-board frame timestamp register (e.g. Basler
  pylon / FLIR Spinnaker SDKs).

The driver assumes both cameras emit the same resolution. Frames are
returned in BGR by OpenCV; we convert to RGB and normalise to
``float32`` in ``[0, 1]`` so the tensor shape and dtype match the
file-backed driver's contract exactly.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import cv2  # pyright: ignore[reportMissingTypeStubs]  # opencv ships no stubs
import numpy as np
import torch

from wormjepa.data.contract import WormID
from wormjepa.hardware.camera_pair import CameraPair
from wormjepa.hardware.two_view_contract import IN_HOUSE_LIVE_SOURCE, TwoViewSample

if TYPE_CHECKING:
    from typing import Any

_DEFAULT_RIG_ID = "live_rig_0"


class LiveCameraPair(CameraPair):
    """Two-camera driver backed by ``cv2.VideoCapture`` for live capture.

    Args:
        primary_index_or_device: OpenCV camera index (``int``) or
            device path (``str``, e.g. ``"/dev/video0"``) for the
            primary camera.
        secondary_index_or_device: Same, for the secondary camera.
        rig_id: Identifier stamped on each yielded
            :class:`TwoViewSample`.
        fps: Nominal capture rate in Hz. Informational only — the
            actual emitted ``timestamps`` come from
            :func:`time.monotonic`, not from this value. Stored so a
            future hardware-triggered subclass can compare expected
            cadence against measured.
        channels: Per-frame channel count emitted. Must be 3 — the
            BGR→RGB conversion path only handles 3-channel frames.
            Surface area kept symmetric with :class:`MockCameraPair`
            for orchestration code that swaps drivers at runtime.

    Captures are opened lazily on the first :meth:`iter_samples` call
    and re-opened on every subsequent call (each iteration starts a
    fresh capture session). They are released by :meth:`close`;
    calling :meth:`iter_samples` after :meth:`close` raises
    :class:`RuntimeError`.
    """

    def __init__(
        self,
        primary_index_or_device: int | str,
        secondary_index_or_device: int | str,
        *,
        rig_id: str = _DEFAULT_RIG_ID,
        fps: float = 30.0,
        channels: int = 3,
    ) -> None:
        if channels != 3:
            msg = f"LiveCameraPair: channels must be 3 (RGB), got {channels}"
            raise ValueError(msg)
        if fps <= 0:
            msg = f"LiveCameraPair: fps must be > 0, got {fps}"
            raise ValueError(msg)
        self._primary_src = primary_index_or_device
        self._secondary_src = secondary_index_or_device
        self._rig_id = rig_id
        self._fps = fps
        self._channels = channels
        self._primary_cap: Any | None = None
        self._secondary_cap: Any | None = None
        self._closed = False

    def iter_samples(self, n_clips: int, *, clip_frames: int) -> Iterator[TwoViewSample]:
        if self._closed:
            msg = "LiveCameraPair.iter_samples called after close()"
            raise RuntimeError(msg)
        if n_clips < 0:
            msg = f"n_clips must be >= 0, got {n_clips}"
            raise ValueError(msg)
        if clip_frames < 1:
            msg = f"clip_frames must be >= 1, got {clip_frames}"
            raise ValueError(msg)

        # Close any previously-opened captures so iter_samples is restartable.
        self._release_captures()
        self._primary_cap = cv2.VideoCapture(self._primary_src)
        self._secondary_cap = cv2.VideoCapture(self._secondary_src)

        if not self._primary_cap.isOpened():
            self._release_captures()
            msg = f"LiveCameraPair: primary capture failed to open: {self._primary_src!r}"
            raise RuntimeError(msg)
        if not self._secondary_cap.isOpened():
            self._release_captures()
            msg = f"LiveCameraPair: secondary capture failed to open: {self._secondary_src!r}"
            raise RuntimeError(msg)

        for clip_idx in range(n_clips):
            clip_start = time.monotonic()
            primary_buf: list[torch.Tensor] = []
            secondary_buf: list[torch.Tensor] = []
            timestamp_buf: list[float] = []

            clip_aborted = False
            for _ in range(clip_frames):
                # grab() both captures back-to-back to minimise the
                # inter-camera shutter offset; retrieve() afterwards.
                if not self._primary_cap.grab() or not self._secondary_cap.grab():
                    clip_aborted = True
                    break
                ok_p, frame_p = self._primary_cap.retrieve()
                ok_s, frame_s = self._secondary_cap.retrieve()
                if not ok_p or not ok_s:
                    clip_aborted = True
                    break
                primary_buf.append(_bgr_frame_to_tensor(frame_p))
                secondary_buf.append(_bgr_frame_to_tensor(frame_s))
                timestamp_buf.append(time.monotonic() - clip_start)

            if clip_aborted:
                # A capture ran dry mid-clip — stop without emitting partial data.
                return

            worm_id = WormID(f"live_worm_{clip_idx:04d}")
            session_id = f"{self._rig_id}_clip_{clip_idx:04d}"
            yield TwoViewSample(
                video_primary=torch.stack(primary_buf, dim=0),
                video_secondary=torch.stack(secondary_buf, dim=0),
                timestamps=torch.tensor(timestamp_buf, dtype=torch.float32),
                worm_id=worm_id,
                session_id=session_id,
                rig_id=self._rig_id,
                source_dataset=IN_HOUSE_LIVE_SOURCE,
            )

    def close(self) -> None:
        self._release_captures()
        self._closed = True

    def _release_captures(self) -> None:
        if self._primary_cap is not None:
            self._primary_cap.release()
            self._primary_cap = None
        if self._secondary_cap is not None:
            self._secondary_cap.release()
            self._secondary_cap = None


def _bgr_frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert a BGR ``(H, W, 3)`` uint8 frame to ``(3, H, W)`` float32 RGB in [0, 1].

    OpenCV emits BGR by convention; the rest of the pipeline (and the
    file-backed driver) speaks RGB, so we explicitly convert via
    :func:`cv2.cvtColor` rather than the cheaper ``[..., ::-1]`` slice
    — ``cvtColor`` is the canonical contract OpenCV documents, and a
    future colour-space-aware subclass (e.g. raw Bayer cameras) can
    swap the constant without touching the surrounding code.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
