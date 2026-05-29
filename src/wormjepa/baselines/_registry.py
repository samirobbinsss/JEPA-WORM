"""Registry mapping baseline names to their implementing classes.

Populated lazily on first import of the baselines package. The CLI dispatcher
(:func:`wormjepa.cli.run.run_command`) consults this registry to find the
right class given a ``baseline.name`` config value.
"""

from __future__ import annotations

from wormjepa.baselines.base import Baseline
from wormjepa.baselines.kalman import KalmanBaseline
from wormjepa.baselines.pose_tcn import PoseOnlyTCNBaseline
from wormjepa.baselines.random_features import RandomFeaturesBaseline
from wormjepa.baselines.transformer_eigenworms import TransformerEigenwormsBaseline

REGISTRY: dict[str, type[Baseline]] = {
    "kalman": KalmanBaseline,
    "transformer_eigenworms": TransformerEigenwormsBaseline,
    "pose_tcn": PoseOnlyTCNBaseline,
    "random_features": RandomFeaturesBaseline,
}
