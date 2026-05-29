"""Future-pose prediction error metric (Story 6.2 / FR25 / row 1).

The PRD's kill-criterion comparator: future-pose error at Δt ∈ {0.1s, 1s, 5s},
mean per-keypoint Euclidean distance, leave-one-worm-out CV, worm-level
bootstrap CIs.

This module operates on already-computed predictions (one ``FuturePoseHorizon``
per worm-clip-horizon, produced by :class:`wormjepa.baselines.base.Baseline`
or the JEPA decoder). It aggregates across horizons, applies bootstrap CIs,
and returns one :class:`wormjepa.eval.MetricEntry` with per-horizon sub-rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from wormjepa.baselines.base import FuturePoseHorizon
from wormjepa.data import WormID
from wormjepa.eval.bootstrap import WormGrouping, bootstrap_ci
from wormjepa.eval.metrics_schema import BootstrapCI, MetricEntry, SubEntry


def _per_clip_error(entry: FuturePoseHorizon) -> float:
    diff = entry.predicted - entry.ground_truth
    return float(torch.linalg.vector_norm(diff, dim=-1).mean())


def future_pose_metric(
    predictions: Sequence[FuturePoseHorizon],
    *,
    producer: str,
    n_bootstrap: int = 1000,
    method: str = "bca",
) -> MetricEntry:
    """Aggregate per-horizon future-pose error with worm-level bootstrap CIs.

    Args:
        predictions: Iterable of ``FuturePoseHorizon`` entries (one per
            clip x horizon). Different horizons within the same clip share
            the same worm_id.
        producer: Identifier of the model/baseline (recorded on the entry).
        n_bootstrap: Bootstrap samples per CI.
        method: BCa or percentile.

    Returns:
        A :class:`MetricEntry` with ``name="future_pose"``, one ``SubEntry``
        per horizon. The top-level CI is left as NaN (the per-horizon rows
        are the meaningful values).
    """
    horizons = sorted({p.horizon_seconds for p in predictions})
    sub_entries: list[SubEntry] = []
    for h in horizons:
        bucket = [p for p in predictions if p.horizon_seconds == h]
        errors = np.asarray([_per_clip_error(p) for p in bucket])
        grouping = WormGrouping(worm_ids=tuple(WormID(str(p.worm_id)) for p in bucket))
        ci = bootstrap_ci(
            errors,
            grouping,
            n_samples=n_bootstrap,
            method=method,  # type: ignore[arg-type]
        )
        sub_entries.append(SubEntry(key=f"{h:g}s", ci=ci))

    nan_ci = BootstrapCI(
        point=float("nan"),
        lower=float("nan"),
        upper=float("nan"),
        n_samples=n_bootstrap,
        method=method,  # type: ignore[arg-type]
    )
    return MetricEntry(
        name="future_pose",
        producer=producer,
        ci=nan_ci,
        sub_entries=sub_entries,
        notes="per-horizon mean Euclidean keypoint error; worm-level bootstrap",
    )
