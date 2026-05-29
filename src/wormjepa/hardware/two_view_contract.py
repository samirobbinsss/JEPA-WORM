"""Two-view sample contract (Phase A prep, hardware-pilot scaffolding).

The pre-registered :class:`wormjepa.data.contract.DatasetSample` has a
single ``video_clip`` field — the FR17 vision-only encoder consumes
exactly one ``(T, C, H, W)`` tensor per worm-clip. The eventual
hardware pilot will capture **two synchronised camera views per
worm** (top + side, or stereo pair); this module's
:class:`TwoViewSample` is the contract those captures will land in,
and :meth:`TwoViewSample.to_dataset_sample` collapses to the
single-view DatasetSample the rest of Phase 0 expects.

The secondary view is preserved on :class:`TwoViewSample` itself so a
future Phase A multi-view encoder (e.g. seq-JEPA stereo fusion, late-
fusion attention across views) can consume it without further
contract churn. Today the secondary view is observation-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wormjepa.data.contract import DatasetSample, SessionID, SourceDataset, WormID

#: Canonical `SourceDataset` tags the hardware module emits. Re-exported
#: as module-level constants (not underscore-prefixed) so downstream
#: drivers can import without pyright reportPrivateUsage warnings.
SYNTHETIC_SOURCE: SourceDataset = SourceDataset("synthetic")
IN_HOUSE_LIVE_SOURCE: SourceDataset = SourceDataset("in_house_live")


@dataclass(frozen=True, slots=True)
class TwoViewSample:
    """One time-aligned two-camera capture of a single worm.

    Attributes:
        video_primary: ``(T, C, H, W)`` primary-camera clip — the view
            the FR17 single-view encoder consumes.
        video_secondary: ``(T, C, H, W)`` secondary-camera clip,
            captured synchronously with ``video_primary`` (same ``T``,
            same frame timing). Phase A decides whether to fuse,
            ignore, or sub-select; today the field is informational.
        timestamps: ``(T,)`` per-frame wall-clock seconds since the
            recording start. Allows downstream code to detect
            primary/secondary desynchronisation if the hardware drops
            a frame.
        worm_id: :class:`WormID` of the recorded worm. Pre-registered
            cohort assignment uses this.
        session_id: free-form session identifier (rig + date + worm).
        rig_id: identifier of the physical capture rig — useful when
            the eventual Phase A scales beyond one apparatus and the
            session-classifier diagnostic needs to disambiguate.
        primary_intrinsics_sha256: optional canonical SHA of the
            primary camera's intrinsic-calibration record. The Phase A
            pre-reg ought to lock this; the field is here so the
            contract is ready.
        secondary_intrinsics_sha256: same, secondary camera.
        source_dataset: always
            :data:`wormjepa.data.contract.SourceDataset.SYNTHETIC` for
            mock data; a future ``IN_HOUSE_LIVE`` variant adds when
            real-rig data lands (out of Phase 0 scope).
    """

    video_primary: torch.Tensor
    video_secondary: torch.Tensor
    timestamps: torch.Tensor
    worm_id: WormID
    session_id: str
    rig_id: str
    primary_intrinsics_sha256: str | None = None
    secondary_intrinsics_sha256: str | None = None
    source_dataset: SourceDataset = SYNTHETIC_SOURCE

    def __post_init__(self) -> None:
        if self.video_primary.ndim != 4:
            msg = (
                f"TwoViewSample.video_primary expected (T, C, H, W); "
                f"got shape {tuple(self.video_primary.shape)}"
            )
            raise ValueError(msg)
        if self.video_secondary.ndim != 4:
            msg = (
                f"TwoViewSample.video_secondary expected (T, C, H, W); "
                f"got shape {tuple(self.video_secondary.shape)}"
            )
            raise ValueError(msg)
        if self.video_primary.shape[0] != self.video_secondary.shape[0]:
            msg = (
                f"TwoViewSample: primary T={self.video_primary.shape[0]} != "
                f"secondary T={self.video_secondary.shape[0]} — cameras are not "
                f"time-aligned. Check rig sync trigger."
            )
            raise ValueError(msg)
        if self.timestamps.shape[0] != self.video_primary.shape[0]:
            msg = (
                f"TwoViewSample: timestamps length {self.timestamps.shape[0]} "
                f"!= clip frames {self.video_primary.shape[0]}"
            )
            raise ValueError(msg)

    def to_dataset_sample(self) -> DatasetSample:
        """Collapse to the single-view :class:`DatasetSample` Phase 0 consumes.

        Today the secondary view is dropped. A Phase A multi-view
        encoder would either:

        - Replace the collapse with a stacked-channel
          ``torch.cat([primary, secondary], dim=1)`` (FR17 vision-only
          stays satisfied — channels grow, but neural data is still
          absent from the input).
        - Or land a sibling :class:`MultiViewSample` field on the
          downstream contract (frozen-artifact change).

        Pose + neural fields are populated as ``None`` because the
        hardware pilot's first lap is video-only; pose tracking +
        calcium imaging integration is a separate Phase A milestone.
        """
        return DatasetSample(
            video_clip=self.video_primary,
            pose=None,
            neural=None,
            worm_id=self.worm_id,
            session_id=SessionID(self.session_id),
            source_dataset=self.source_dataset,
        )
