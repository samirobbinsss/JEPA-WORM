"""Unified iterator contract for every JEPA-WORM dataset loader.

All five public dataset loaders (WormID, Atanas/Flavell-2023, WormBehavior DB,
Open Worm Movement DB, BAAIWorm) yield :class:`DatasetSample` instances. The
tuple shape is deliberately consistent across loaders so the training code is
dataset-agnostic and worm-level grouping survives end-to-end.

NewType aliases (:data:`WormID`, :data:`SessionID`, :data:`SourceDataset`)
distinguish identifier strings from arbitrary strings at the type-checker
level — passing a raw ``str`` where a ``WormID`` is required is a pyright
error in strict mode. This is the first link in the chain that ends at
:class:`wormjepa.eval.bootstrap.WormGrouping` (Epic 3 Story 3.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, NewType

if TYPE_CHECKING:
    import torch

WormID = NewType("WormID", str)
"""Unique identifier for an individual worm across the project.

Format is dataset-specific (e.g., ``"WormID-2023-014"`` for the WormID corpus,
``"flavell_w42"`` for Atanas/Flavell-2023). The loader is responsible for
producing a stable, deterministic worm-id per real-world worm so that
leave-one-worm-out CV groups frames correctly.
"""

SessionID = NewType("SessionID", str)
"""Unique identifier for a recording session of a particular worm.

A worm may appear in multiple sessions (e.g., different days, different
imaging rigs). Session identifiers preserve provenance so the
``session-ID classifier`` diagnostic (FR31) can detect rig shortcuts.
"""

SourceDataset = NewType("SourceDataset", str)
"""Identifier for the originating public dataset.

Conventional values: ``"wormid"``, ``"flavell_2023"``, ``"wormbehavior_db"``,
``"openworm_movement"``, ``"baaiworm"``. Kept as :class:`NewType` rather than
:class:`typing.Literal` so adding a future dataset does not require a code
change in the contract module itself.
"""


class DatasetSample(NamedTuple):
    """A single sample yielded by any JEPA-WORM dataset loader.

    Attributes:
        video_clip: ``(T, C, H, W)`` tensor of one clip from the worm's recording.
            Required for every loader; the encoder operates on video alone (FR17).
        pose: ``(T, K, D)`` tensor of 2D or 3D keypoint coordinates, or ``None``
            when the source dataset does not include pose. ``K`` is the number
            of keypoints; ``D`` is 2 or 3 depending on the rig.
        neural: ``(T, N)`` tensor of head-neuron activity, or ``None`` when the
            source dataset does not include neural recordings. Used only by
            training-time warm-start auxiliary heads (FR16); the deployed
            encoder never sees this field.
        worm_id: The :class:`WormID` of the individual this clip belongs to.
            Propagated through the pipeline so worm-level grouping survives.
        session_id: The :class:`SessionID` of the recording session.
        source_dataset: The :class:`SourceDataset` this sample originated from.
            Used by the session-ID-classifier diagnostic to detect shortcuts.
        behavioral_state: ``(T,)`` long tensor of per-frame behavioral-state class
            indices, or ``None`` when the source dataset has no behavioral
            annotation. Consumed by Story 6.3's motif-ARI metric
            (``wormjepa.eval.motif_ari``). Kept separate from ``neural`` so the
            integer label type is preserved end-to-end.
        frame_rate: Sampling rate of ``video_clip`` in frames per second, or
            ``None`` when the source loader does not know its rate. Propagated
            so cross-dataset clip timing is comparable — a clip of ``T`` frames
            spans ``T / frame_rate`` seconds, and the Schafer/Flavell corpora
            sit at very different rates (~30 Hz tracker vs ~3 Hz volumetric).
            Strictly additive and defaulted to ``None``: loaders that cannot
            determine their rate leave it unset rather than guessing.
    """

    video_clip: torch.Tensor
    pose: torch.Tensor | None
    neural: torch.Tensor | None
    worm_id: WormID
    session_id: SessionID
    source_dataset: SourceDataset
    behavioral_state: torch.Tensor | None = None
    frame_rate: float | None = None
