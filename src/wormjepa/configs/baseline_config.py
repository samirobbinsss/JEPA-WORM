"""Pydantic schema for ``configs/baselines/<name>.yaml``.

Story 8.10 added an optional ``dataset:`` block that mirrors the JEPA
config's selection — baselines and JEPA now share the same loader
composition path via :func:`wormjepa.data.composition.build_loader`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from wormjepa.configs.dataset import DatasetSection
from wormjepa.configs.models import WormJEPAConfig


class BaselineSection(BaseModel):
    """The ``baseline:`` block of a baseline-run config."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    horizons_seconds: list[float] = Field(default_factory=lambda: [0.1, 1.0, 5.0])
    frame_dt_seconds: float = 0.1
    seed: int = 0
    """RNG seed for synthetic / BAAIWorm loaders + any fit-time randomness.

    Story 8.10 surfaces this so baselines can be re-run deterministically
    against the same dataset spec; the three-seed headline sweep (Story
    8.11) varies it across {42, 1337, 8675309}.
    """


class BaselineRunConfig(WormJEPAConfig):
    """Top-level config for ``wormjepa run --config configs/baselines/<name>.yaml``."""

    baseline: BaselineSection
    dataset: DatasetSection = Field(default_factory=DatasetSection)
