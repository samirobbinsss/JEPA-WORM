"""Shared ``dataset:`` config block — used by both JEPA and baseline runs.

Story 8.9 introduced ``DatasetSection`` on :class:`JEPARunConfig` so the
runner could choose between synthetic and the real loaders via config.
Story 8.10 extends the same selection to baseline runs, which means the
schema needs to live in one place that both ``jepa_config`` and
``baseline_config`` can import without a circular dependency.

This module is the canonical home. :class:`DatasetLoaderSpec`,
:class:`DatasetSection`, and the :data:`LoaderName` literal moved here
from ``jepa_config`` — the old import paths still work via re-exports in
:mod:`wormjepa.configs.jepa_config`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LoaderName = Literal[
    "synthetic",
    "flavell_2023",
    "wormbehavior_db",
    "openworm_movement",
    "wormid",
    "baaiworm",
    "two_camera_mock",
]
"""Loader keys accepted in the ``dataset.loaders[].name`` field.

Each value maps to a concrete loader class via
:func:`wormjepa.data.composition.build_loader`.
"""


class DatasetLoaderSpec(BaseModel):
    """One entry in ``dataset.loaders`` — a single loader instantiation.

    Either ``synthetic`` / ``baaiworm`` (knobs default; no on-disk path)
    or one of the real-data loaders, in which case ``local_root`` (or
    ``local_dandi_root`` for WormID) must point at a directory of files
    in the expected layout.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: LoaderName
    """Which loader implementation to instantiate."""

    local_root: str | None = None
    """On-disk root for real loaders. Required when ``name`` is one of the
    real-data loaders. Ignored by ``synthetic`` and ``baaiworm`` (which
    synthesise locally)."""

    cohort: Literal["train", "eval", "all"] = "all"
    """WormID-only cohort selector. Ignored by other loaders."""

    clip_frames: int = 8
    """Frames per emitted clip (T dim). Default 8 matches the encoder."""

    image_size: int = 64
    """Square image size (H == W). Set to 0 to keep the source resolution."""

    # Synthetic / BAAIWorm knobs (ignored by file-based loaders).
    n_worms: int = 4
    clips_per_worm: int = 3
    n_keypoints: int = 4
    n_neurons: int = 8


class DatasetSection(BaseModel):
    """The ``dataset:`` block of a JEPA-run or baseline-run config.

    Defaults to a single ``synthetic`` loader so existing configs without
    a ``dataset:`` section still validate. When ``loaders`` lists more
    than one entry, the runner chains them in order via
    :class:`wormjepa.data.composition.ChainedLoader`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    loaders: list[DatasetLoaderSpec] = Field(
        default_factory=lambda: [DatasetLoaderSpec(name="synthetic")]
    )
