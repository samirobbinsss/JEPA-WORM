"""Kalman / persistence baseline for future-pose prediction (Story 3.4).

Phase 0 v0 uses pure persistence: the prediction at time ``t + Δt`` is the
pose at time ``t``. This is the documented future-pose floor — the "easy bar"
the PRD calls out. A proper constant-velocity Kalman filter is an optional
upgrade if persistence proves too easy a comparator on real data.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

from wormjepa.baselines.base import Baseline, BaselinePredictions, FuturePoseHorizon
from wormjepa.data import DatasetSample

# Pre-registered future-pose horizons. Real values must match
# pre-registration/configs/headline.yaml when Story 4.8 locks the manifest.
_HORIZONS_SECONDS: tuple[float, ...] = (0.1, 1.0, 5.0)

# Synthetic data assumption used by tests/fixtures: frames are spaced by this
# many seconds. The persistence prediction at horizon Δt looks back Δt seconds,
# i.e. ⌊Δt / dt⌋ frames.
_DEFAULT_FRAME_DT_SECONDS = 0.1


class KalmanBaseline(Baseline):
    """Persistence baseline for future-pose: predicts ``pose(t + Δt) = pose(t)``.

    Future-pose floor per PRD; primary purpose is to confirm that JEPA and the
    other baselines clear an "easy bar." Implements :class:`Baseline` so the
    orchestrator can swap it in alongside Transformer-on-eigenworms and the
    pose-only TCN.
    """

    def __init__(
        self,
        horizons_seconds: tuple[float, ...] = _HORIZONS_SECONDS,
        frame_dt_seconds: float = _DEFAULT_FRAME_DT_SECONDS,
    ) -> None:
        self._horizons = horizons_seconds
        self._frame_dt = frame_dt_seconds
        # Persistence is parameter-free; nothing to learn at fit time.
        self._fitted = False

    @classmethod
    def from_config(cls, section: object) -> Self:
        """Construct a ``KalmanBaseline`` from a ``BaselineSection``."""
        from wormjepa.configs.baseline_config import BaselineSection

        if not isinstance(section, BaselineSection):
            msg = f"Expected BaselineSection, got {type(section).__name__}"
            raise TypeError(msg)
        return cls(
            horizons_seconds=tuple(section.horizons_seconds),
            frame_dt_seconds=section.frame_dt_seconds,
        )

    @property
    def name(self) -> str:
        return "kalman"

    def fit(self, dataset: Iterable[DatasetSample]) -> Self:
        """No-op (persistence has no parameters to learn).

        Iterates ``dataset`` once to honor the contract that ``fit`` consumes
        the training set, but does not retain anything. The unused iteration
        is intentional: it lets future Kalman-filter upgrades fit a
        constant-velocity model here without changing the call site.
        """
        for _ in dataset:
            pass
        self._fitted = True
        return self

    def predict(self, dataset: Iterable[DatasetSample]) -> BaselinePredictions:
        """For each clip x horizon, predict ``pose(t + Δt) = pose(t)``."""
        future_pose: list[FuturePoseHorizon] = []
        for sample in dataset:
            if sample.pose is None:
                continue
            pose = sample.pose  # (T, K, D)
            t_total = pose.shape[0]
            for horizon in self._horizons:
                lookback_frames = max(1, round(horizon / self._frame_dt))
                # Predict frame index t_total - 1 using frame
                # (t_total - 1 - lookback_frames) — clipped to >= 0.
                source_idx = max(0, t_total - 1 - lookback_frames)
                predicted = pose[source_idx].clone()
                ground_truth = pose[t_total - 1].clone()
                future_pose.append(
                    FuturePoseHorizon(
                        worm_id=sample.worm_id,
                        session_id=sample.session_id,
                        horizon_seconds=horizon,
                        predicted=predicted,
                        ground_truth=ground_truth,
                    )
                )
        return BaselinePredictions(future_pose=future_pose, latent_by_worm=None)
