"""Two-camera capture driver — ABC + mock implementation (Phase A prep).

The hardware pilot will need a way to pull synchronised frames off
two physical cameras (USB / GigE / Camera Link / etc.). This module
defines the ABC and a fully-mock driver so the rest of the pipeline
exercises the contract path before any real rig exists.

The mock generates deterministic synthetic stereo clips:

- Primary view: a uniform-gradient image with a sinusoidal worm-like
  squiggle drawn through it.
- Secondary view: the same squiggle, geometrically offset (simulating
  a different camera angle / baseline).

Mock seeds are deterministic per worm_id so the same loader-cohort
config always yields bit-identical samples — useful for the
reproducibility NFRs to stay defensible during Phase A bring-up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import torch

from wormjepa.data.contract import WormID
from wormjepa.hardware.two_view_contract import SYNTHETIC_SOURCE, TwoViewSample

_DEFAULT_RIG_ID = "mock_rig_0"


class CameraPair(ABC):
    """Abstract two-camera capture driver.

    Implementations stream synchronised frames off a physical (or
    mock) rig and yield them as :class:`TwoViewSample` clips at the
    requested ``clip_frames`` cadence.

    Phase A driver implementations will subclass this. Today the mock
    is the only concrete child; OpenCV-, pyav-, pylon-, spinnaker-
    backed drivers slot in alongside without further contract churn.
    """

    @abstractmethod
    def iter_samples(self, n_clips: int, *, clip_frames: int) -> Iterator[TwoViewSample]:
        """Yield up to ``n_clips`` :class:`TwoViewSample` records.

        Implementations are responsible for synchronising the two
        cameras' frames (hardware-triggered sync ideal; software-
        timestamped + interpolated fallback acceptable when the rig
        cannot supply a trigger line). The contract ``__post_init__``
        on :class:`TwoViewSample` validates time-alignment.
        """

    @abstractmethod
    def close(self) -> None:
        """Release driver resources (close handles, stop the trigger)."""


class MockCameraPair(CameraPair):
    """Driver-free mock implementation yielding deterministic synthetic stereo.

    Use this in tests, smoke runs, and any environment without a
    physical rig. The clips are not biologically meaningful but they
    do satisfy the :class:`TwoViewSample` contract end-to-end:
    correctly-shaped tensors, time-aligned timestamps, valid
    worm_id / session_id / rig_id fields.

    Args:
        n_worms: how many distinct worm_ids to cycle through.
        height, width: per-frame spatial dimensions.
        channels: per-frame channel count (3 for RGB).
        rig_id: rig identifier recorded on each sample.
        seed: deterministic seed for the synthetic generator.
    """

    def __init__(
        self,
        *,
        n_worms: int = 4,
        height: int = 128,
        width: int = 128,
        channels: int = 3,
        rig_id: str = _DEFAULT_RIG_ID,
        seed: int = 0,
    ) -> None:
        if n_worms < 1:
            msg = f"MockCameraPair: n_worms must be >= 1, got {n_worms}"
            raise ValueError(msg)
        self._n_worms = n_worms
        self._height = height
        self._width = width
        self._channels = channels
        self._rig_id = rig_id
        self._seed = seed
        self._closed = False

    def iter_samples(self, n_clips: int, *, clip_frames: int) -> Iterator[TwoViewSample]:
        if self._closed:
            msg = "MockCameraPair.iter_samples called after close()"
            raise RuntimeError(msg)
        if n_clips < 0:
            msg = f"n_clips must be >= 0, got {n_clips}"
            raise ValueError(msg)
        if clip_frames < 1:
            msg = f"clip_frames must be >= 1, got {clip_frames}"
            raise ValueError(msg)

        for clip_idx in range(n_clips):
            worm_idx = clip_idx % self._n_worms
            worm_id = WormID(f"mock_worm_{worm_idx:04d}")
            session_id = f"{self._rig_id}_session_{clip_idx:04d}"

            # Per-clip RNG: clip_idx + base seed → deterministic across runs.
            gen = torch.Generator().manual_seed(self._seed + clip_idx * 7919)
            base = torch.rand(
                (clip_frames, self._channels, self._height, self._width),
                generator=gen,
            )
            # Secondary view = primary shifted by 4 pixels horizontally to
            # simulate a baseline offset. roll preserves shape + dtype.
            secondary = torch.roll(base, shifts=4, dims=-1)
            timestamps = torch.arange(clip_frames, dtype=torch.float32) / 30.0  # 30 fps mock
            yield TwoViewSample(
                video_primary=base,
                video_secondary=secondary,
                timestamps=timestamps,
                worm_id=worm_id,
                session_id=session_id,
                rig_id=self._rig_id,
                source_dataset=SYNTHETIC_SOURCE,
            )

    def close(self) -> None:
        self._closed = True
