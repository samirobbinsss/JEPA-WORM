"""Phase A-prep loader: wraps :class:`MockCameraPair` into the
:class:`DatasetSample` contract used by the rest of Phase 0.

Today the loader collapses each :class:`TwoViewSample` to its primary
view via :meth:`TwoViewSample.to_dataset_sample`. The secondary view
is dropped at this layer; a Phase A multi-view encoder consumes it by
calling :class:`MockCameraPair` (or a real driver) directly and
holding the full :class:`TwoViewSample`.

This loader exists so headline-style YAML configs can include a
``two_camera_mock`` entry as a hardware-pilot stand-in — i.e. so the
**dataset composition** path is exercised before any real rig ships.
Pre-reg locks live in the SPEC modules (per-dataset
``data/sources/<name>.py``); the hardware module is intentionally not
pre-registered until Phase A's biological cohort is.
"""

from __future__ import annotations

from collections.abc import Iterator

from wormjepa.data.contract import DatasetSample
from wormjepa.hardware import MockCameraPair


class TwoCameraMockLoader:
    """:class:`MockCameraPair`-backed loader yielding single-view
    :class:`DatasetSample` records.

    Args:
        n_worms: distinct worm ids cycled through the mock.
        clips_per_worm: clips emitted per worm before exhausting.
        clip_frames: frames per clip.
        image_size: ``(H, W)``.
        rig_id: stamped on each TwoViewSample.session_id.
        seed: deterministic seed for the mock generator.
    """

    def __init__(
        self,
        *,
        n_worms: int = 4,
        clips_per_worm: int = 3,
        clip_frames: int = 16,
        image_size: tuple[int, int] = (64, 64),
        rig_id: str = "mock_rig_0",
        seed: int = 0,
    ) -> None:
        if n_worms < 1:
            msg = f"TwoCameraMockLoader: n_worms must be >= 1, got {n_worms}"
            raise ValueError(msg)
        if clips_per_worm < 1:
            msg = f"TwoCameraMockLoader: clips_per_worm must be >= 1, got {clips_per_worm}"
            raise ValueError(msg)
        if clip_frames < 1:
            msg = f"TwoCameraMockLoader: clip_frames must be >= 1, got {clip_frames}"
            raise ValueError(msg)
        h, w = image_size
        self._rig = MockCameraPair(
            n_worms=n_worms,
            height=h,
            width=w,
            channels=3,
            rig_id=rig_id,
            seed=seed,
        )
        self._n_clips = n_worms * clips_per_worm
        self._clip_frames = clip_frames

    def __iter__(self) -> Iterator[DatasetSample]:
        for two_view in self._rig.iter_samples(
            n_clips=self._n_clips, clip_frames=self._clip_frames
        ):
            yield two_view.to_dataset_sample()
