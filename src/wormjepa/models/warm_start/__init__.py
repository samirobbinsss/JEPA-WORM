"""Warm-start auxiliary heads for JEPA training (Stories 5.4-5.7).

Each head consumes the **online encoder's latents** during training and
contributes an auxiliary loss term. None of the heads is invoked at test
time — the deployed encoder (FR17) sees only video. Heads are toggleable
per-config via the training loop (Story 5.8) so the neural-prior ablation
(FR29) and BAAIWorm-augmentation ablation (FR30) can disable specific arms.
"""

from wormjepa.models.warm_start.behavioral_head import BehavioralHead
from wormjepa.models.warm_start.eigenworm import EigenwormHead
from wormjepa.models.warm_start.graph_prior import GraphPriorHead
from wormjepa.models.warm_start.neural_head import NeuralAuxiliaryHead

__all__ = [
    "BehavioralHead",
    "EigenwormHead",
    "GraphPriorHead",
    "NeuralAuxiliaryHead",
]
