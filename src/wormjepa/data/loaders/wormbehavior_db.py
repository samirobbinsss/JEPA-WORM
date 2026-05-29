"""WormBehavior Database loader — Story 8.5 real implementation.

The WormBehavior Database (Yemini/Brown 2013) is a per-experiment Zenodo
archive — thousands of individual records. The locked subset at v2
contains two anchor records (one N2 wild-type, one mutant); the full
100-record stratified subset lands via a CHANGELOG ``Frozen-artifact
changes`` entry when this loader is exercised at scale.

Format: Schafer-lab HDF5 (mask/full_data + skeleton coordinates).
Shared reader at :mod:`wormjepa.data.loaders._schafer_hdf5`.
"""

from __future__ import annotations

from pathlib import Path

from wormjepa.data import SourceDataset
from wormjepa.data.loaders._schafer_hdf5 import SchaferZenodoSubsetLoader
from wormjepa.data.sources.wormbehavior_db import SPEC


class WormBehaviorDBLoader(SchaferZenodoSubsetLoader):
    """Iterates over WormBehavior DB Zenodo records under a local root.

    Args:
        local_root: Directory containing the layout
            ``<local_root>/<zenodo_record_id>/*.hdf5``. Record IDs come from
            :data:`wormjepa.data.sources.wormbehavior_db.SPEC`.
        clip_frames: Frames per emitted clip (T dimension). Default 8.
        image_size: ``(H, W)`` to bilinearly resize each frame to. ``None``
            keeps the source resolution. Defaults to ``(64, 64)``.

    Emits :class:`wormjepa.data.DatasetSample` with:
      - ``video_clip``: ``(T, 3, H, W)`` float32 in [0, 1]
      - ``pose``: ``(T, K, 2)`` float32 when the source file has a skeleton
        dataset; ``None`` otherwise.
      - ``neural``: always ``None`` (WormBehaviorDB has no neural data).
      - ``worm_id``: ``"wormbehavior_<record_id>_<filename_stem>"``.
      - ``source_dataset``: ``"wormbehavior_db"``.
    """

    def __init__(
        self,
        local_root: Path | str,
        clip_frames: int = 8,
        image_size: tuple[int, int] | None = (64, 64),
    ) -> None:
        super().__init__(
            local_root=local_root,
            spec=SPEC,
            source_dataset=SourceDataset("wormbehavior_db"),
            worm_id_prefix="wormbehavior_",
            clip_frames=clip_frames,
            image_size=image_size,
        )
