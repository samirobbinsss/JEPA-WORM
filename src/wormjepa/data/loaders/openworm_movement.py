"""Open Worm Movement Database loader — Story 8.6 real implementation.

The Open Worm Movement Database (Javer 2018) overlaps in source material
with the WormBehavior Database but the PRD lists them as separate
datasets. The locked subset at v2 contains two anchor records; the full
100-record stratified subset lands via a CHANGELOG ``Frozen-artifact
changes`` entry when this loader is exercised at scale.

Format: Schafer-lab HDF5 (mask/full_data + skeleton coordinates).
Shared reader at :mod:`wormjepa.data.loaders._schafer_hdf5`.
"""

from __future__ import annotations

from pathlib import Path

from wormjepa.data import SourceDataset
from wormjepa.data.loaders._schafer_hdf5 import SchaferZenodoSubsetLoader
from wormjepa.data.sources.openworm_movement import SPEC


class OpenWormMovementLoader(SchaferZenodoSubsetLoader):
    """Iterates over Open Worm Movement DB Zenodo records under a local root.

    Args:
        local_root: Directory containing
            ``<local_root>/<zenodo_record_id>/*.hdf5``. Record IDs come from
            :data:`wormjepa.data.sources.openworm_movement.SPEC`.
        clip_frames: Frames per emitted clip (T dimension). Default 8.
        image_size: ``(H, W)`` to bilinearly resize each frame to. Default
            ``(64, 64)``.

    Emits :class:`wormjepa.data.DatasetSample` with:
      - ``video_clip``: ``(T, 3, H, W)`` float32 in [0, 1]
      - ``pose``: ``(T, K, 2)`` float32 when the source file has a skeleton
        dataset; ``None`` otherwise.
      - ``neural``: always ``None``.
      - ``worm_id``: ``"openworm_<record_id>_<filename_stem>"``.
      - ``source_dataset``: ``"openworm_movement"``.
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
            source_dataset=SourceDataset("openworm_movement"),
            worm_id_prefix="openworm_",
            clip_frames=clip_frames,
            image_size=image_size,
        )
