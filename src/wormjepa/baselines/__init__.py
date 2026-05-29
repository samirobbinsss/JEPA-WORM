"""Baseline models for JEPA-WORM Phase 0.

Four MVP baselines (FR20-FR24):

- :class:`wormjepa.baselines.kalman.KalmanBaseline` — future-pose floor (Story 3.4)
- :class:`wormjepa.baselines.transformer_eigenworms.TransformerEigenwormsBaseline` —
  kill-criterion comparator (Story 3.5)
- :class:`wormjepa.baselines.pose_tcn.PoseOnlyTCNBaseline` — headline neural-decoding
  comparator (Story 3.6)
- :class:`wormjepa.baselines.random_features.RandomFeaturesBaseline` — matched
  parameter-count sanity check (Story 3.7)

All four implement :class:`wormjepa.baselines.base.Baseline` and write into the
shared metrics-output schema defined in :mod:`wormjepa.eval.metrics_schema`.
"""

from wormjepa.baselines.base import Baseline, BaselinePredictions, FuturePoseHorizon

__all__ = ["Baseline", "BaselinePredictions", "FuturePoseHorizon"]
