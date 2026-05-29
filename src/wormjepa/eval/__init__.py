"""Evaluation, metrics, and diagnostic gates for JEPA-WORM.

The eval module covers:

- Bootstrap CI computation (worm-level grouping mandatory at the API level —
  see :class:`wormjepa.eval.bootstrap.WormGrouping`).
- Three pre-registered metrics: future-pose, motif ARI, neural-decoding partial R².
- Four ablation/diagnostic gates: neural-prior ablation, BAAIWorm augmentation
  ablation, session-ID classifier, within-state stratified decoding.
- Multiple-comparison correction.
- Gate evaluation (CI-lower-bound thresholds).
- Shared metrics schema written into ``results/<run-id>/metrics.json``.
"""

from wormjepa.eval.metrics_schema import (
    BootstrapCI,
    MetricEntry,
    MetricsOutput,
    SubEntry,
)

__all__ = [
    "BootstrapCI",
    "MetricEntry",
    "MetricsOutput",
    "SubEntry",
]
