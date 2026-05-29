"""Common ``Baseline`` interface for every Phase 0 baseline.

The four MVP baselines (Kalman, Transformer-on-eigenworms, pose-only TCN,
random-features) implement this interface. All baselines produce future-pose
predictions at the three pre-registered horizons; latent-exposing baselines
(pose-TCN, random-features) additionally populate the ``latent_by_worm`` field
for the neural-decoding probe's residualization (Epic 6 Story 6.1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Self

import torch
from pydantic import BaseModel, ConfigDict

from wormjepa.data import WormID

if TYPE_CHECKING:
    from wormjepa.data import DatasetSample


class FuturePoseHorizon(BaseModel):
    """A single future-pose prediction at one horizon, for one worm-clip pair.

    Carries the worm_id (mandatory for downstream worm-level grouping) so the
    eval module can attach predictions to the correct cohort element.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    worm_id: WormID
    session_id: str
    horizon_seconds: float
    predicted: torch.Tensor  # shape (K, D) — K keypoints, D dims (2 or 3)
    ground_truth: torch.Tensor  # same shape


class BaselinePredictions(BaseModel):
    """The full set of predictions a baseline produces over an eval dataset.

    Future-pose entries are required (every Phase 0 baseline reports them).
    Latent-by-worm is optional and populated only by latent-exposing baselines.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    future_pose: list[FuturePoseHorizon]
    latent_by_worm: dict[WormID, torch.Tensor] | None = None
    """Per-worm latent matrices of shape ``(N_frames, D_latent)``, ``None`` if
    this baseline does not expose a latent (Kalman, Transformer-eigenworms)."""


class Baseline(ABC):
    """Abstract base class for every Phase 0 baseline.

    Subclasses implement :meth:`fit`, :meth:`predict`, and the :attr:`name`
    property. The orchestrator (Epic 5 Story 5.8) iterates baselines via this
    interface; the metrics module (Epic 6) consumes :class:`BaselinePredictions`
    via the worm-level bootstrap API.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this baseline.

        Must match ``configs/baselines/<name>.yaml`` filename stem so that
        ``wormjepa run --config configs/baselines/<name>.yaml`` and the
        results-table row for this baseline align.
        """

    @classmethod
    @abstractmethod
    def from_config(cls, section: object) -> Self:
        """Construct an instance from the ``baseline:`` section of a config.

        ``section`` is the ``BaselineSection`` pydantic model from
        :mod:`wormjepa.configs.baseline_config`; typed as ``object`` here to
        keep the base class free of the config-package import (cycle).
        """

    @abstractmethod
    def fit(self, dataset: Iterable[DatasetSample]) -> Self:
        """Train the baseline on ``dataset``. Returns ``self`` for chaining."""

    @abstractmethod
    def predict(self, dataset: Iterable[DatasetSample]) -> BaselinePredictions:
        """Produce predictions over the held-out dataset.

        The dataset iterator yields :class:`DatasetSample` instances; the
        baseline returns a :class:`BaselinePredictions` aggregating every
        horizon prediction (and optionally a latent table) for the worms
        encountered.
        """
