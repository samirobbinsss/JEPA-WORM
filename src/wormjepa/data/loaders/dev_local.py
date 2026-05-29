"""Single-MP4 dev loader for "does the model work?" smoke tests.

This loader decodes a local video file via PyAV and yields ``DatasetSample``
instances under the unified iterator contract. It exists for dev-loop
validation only — to confirm the encoder, training loop, and eval pipeline
behave sensibly on real worm video before the production loaders
(Stories 2.3-2.7) wire DOI-pinned datasets.

**Dev-only — not for reportable runs.** No DOI. No SHA verification. No
MANIFEST.lock entry. ``source_dataset`` is set to ``"dev_local"`` so any
downstream code that mistakes one of these samples for production data is
visible at audit time.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import av
import torch
from torch.nn import functional as F  # noqa: N812

from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID

if TYPE_CHECKING:
    from collections.abc import Iterable


class DevLocalLoader:
    """Yields fixed-length clips from a single local video file.

    Args:
        video_path: Path to an MP4 (or any PyAV-decodable container).
        clip_frames: Frames per emitted clip (T dimension).
        image_size: ``(H, W)`` to bilinearly resize each frame to. ``None``
            keeps the source resolution (which may not match what the encoder
            config expects — pass an explicit size to be safe).
        worm_id: WormID stamp for every emitted sample. Caller's responsibility
            to pick something that doesn't collide with production worm-ids.
            Default ``"dev_local_w00"`` is collision-free with the
            ``synth_w*`` / ``wormid-*`` / ``flavell_*`` patterns used elsewhere.
        session_id: SessionID stamp; defaults to ``worm_id + "_s00"``.
        max_clips: Cap the number of clips emitted. ``None`` means iterate to
            EOF. Small caps are sensible for smoke tests.
    """

    def __init__(
        self,
        video_path: Path | str,
        clip_frames: int = 8,
        image_size: tuple[int, int] | None = (64, 64),
        *,
        worm_id: str = "dev_local_w00",
        session_id: str | None = None,
        max_clips: int | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.clip_frames = clip_frames
        self.image_size = image_size
        self.worm_id = WormID(worm_id)
        self.session_id = SessionID(session_id if session_id is not None else f"{worm_id}_s00")
        self.max_clips = max_clips

    def __iter__(self) -> Iterator[DatasetSample]:
        if not self.video_path.is_file():
            msg = f"DevLocalLoader: video file not found: {self.video_path}"
            raise FileNotFoundError(msg)

        emitted = 0
        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            buf: list[torch.Tensor] = []
            for frame in container.decode(stream):
                buf.append(_frame_to_tensor(frame, self.image_size))
                if len(buf) == self.clip_frames:
                    yield DatasetSample(
                        video_clip=torch.stack(buf, dim=0),
                        pose=None,
                        neural=None,
                        worm_id=self.worm_id,
                        session_id=self.session_id,
                        source_dataset=SourceDataset("dev_local"),
                    )
                    buf = []
                    emitted += 1
                    if self.max_clips is not None and emitted >= self.max_clips:
                        return


def _frame_to_tensor(
    frame: av.VideoFrame,
    image_size: tuple[int, int] | None,
) -> torch.Tensor:
    """Convert a PyAV VideoFrame to a ``(C, H, W)`` float32 tensor in [0, 1].

    The source frame is decoded as RGB24 regardless of the input pixel format.
    Bilinear resize is applied when ``image_size`` is given.
    """
    rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
    if image_size is None:
        return tensor
    h, w = image_size
    return F.interpolate(tensor.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]


def iter_samples(
    video_path: Path | str,
    *,
    clip_frames: int = 8,
    image_size: tuple[int, int] | None = (64, 64),
    max_clips: int | None = 4,
) -> Iterable[DatasetSample]:
    """Convenience wrapper that returns an iterator with sensible defaults.

    Equivalent to ``iter(DevLocalLoader(...))`` with smoke-friendly defaults
    (``max_clips=4`` to keep dev-loop iteration fast).
    """
    return iter(
        DevLocalLoader(
            video_path,
            clip_frames=clip_frames,
            image_size=image_size,
            max_clips=max_clips,
        )
    )
