"""Ablation runners (Story 6.8 / FR29 + FR30 / rows 4-5).

Two arms:

- **Neural-prior ablation**: ΔR² between the full model and a
  kinematic-warm-start-only model. Threshold ≥ 0.02 (NFR18 / Row 4).
- **BAAIWorm-augmentation ablation**: ΔR² between training with vs without
  BAAIWorm synthetic augmentation. No fixed threshold (the value is the
  answer; row 5).

The ablation API takes two pre-computed CIs (one per arm) and produces a
ΔR² CI by subtracting the arms' point estimates and bootstrapping the
difference. Phase 0 v0 uses a paired-bootstrap simplification: subtract
worm-level means; the standalone ablation runner runs the full bootstrap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wormjepa.data import WormID
from wormjepa.eval.bootstrap import WormGrouping, bootstrap_ci
from wormjepa.eval.metrics_schema import BootstrapCI, MetricEntry


@dataclass(frozen=True, slots=True)
class AblationResult:
    delta_r2: BootstrapCI
    full_arm_mean: float
    ablated_arm_mean: float


def ablation_delta(
    full_arm_per_worm_r2: Sequence[float],
    ablated_arm_per_worm_r2: Sequence[float],
    worm_ids: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    method: str = "bca",
) -> AblationResult:
    """Bootstrap the worm-paired difference ``full_arm - ablated_arm``.

    Both per-arm sequences must align element-wise with ``worm_ids``.
    """
    if not (len(full_arm_per_worm_r2) == len(ablated_arm_per_worm_r2) == len(worm_ids)):
        msg = "ablation arms and worm_ids must share length"
        raise ValueError(msg)
    diffs = np.asarray(full_arm_per_worm_r2) - np.asarray(ablated_arm_per_worm_r2)
    grouping = WormGrouping(worm_ids=tuple(WormID(w) for w in worm_ids))
    ci = bootstrap_ci(diffs, grouping, n_samples=n_bootstrap, method=method)  # type: ignore[arg-type]
    return AblationResult(
        delta_r2=ci,
        full_arm_mean=float(np.mean(full_arm_per_worm_r2)),
        ablated_arm_mean=float(np.mean(ablated_arm_per_worm_r2)),
    )


def neural_prior_ablation_entry(
    full_per_worm: Sequence[float],
    kinematic_only_per_worm: Sequence[float],
    worm_ids: Sequence[str],
    *,
    n_bootstrap: int = 1000,
) -> MetricEntry:
    """Produce the row-4 MetricEntry: ΔR² (full - kinematic-only)."""
    result = ablation_delta(
        full_per_worm, kinematic_only_per_worm, worm_ids, n_bootstrap=n_bootstrap
    )
    return MetricEntry(
        name="neural_prior_ablation_delta_r2",
        producer="jepa",
        ci=result.delta_r2,
        notes=(f"full={result.full_arm_mean:.3f}, kin_only={result.ablated_arm_mean:.3f}"),
    )


def baaiworm_ablation_entry(
    with_per_worm: Sequence[float],
    without_per_worm: Sequence[float],
    worm_ids: Sequence[str],
    *,
    n_bootstrap: int = 1000,
) -> MetricEntry:
    """Produce the row-5 MetricEntry: ΔR² (with BAAIWorm - without).

    No fixed threshold — the value itself is the answer (reported-only).
    The entry name matches the pre-registration / STATUS.md row-5 identity
    ``baaiworm_augmentation_ablation_delta_r2``.
    """
    result = ablation_delta(with_per_worm, without_per_worm, worm_ids, n_bootstrap=n_bootstrap)
    return MetricEntry(
        name="baaiworm_augmentation_ablation_delta_r2",
        producer="jepa",
        ci=result.delta_r2,
        notes=(f"with_baaiworm={result.full_arm_mean:.3f}, without={result.ablated_arm_mean:.3f}"),
    )
