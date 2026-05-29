"""Baseline-run orchestration.

Loads the named baseline, fits + predicts on the configured dataset, computes
the future-pose metric with worm-level bootstrap CIs, and assembles a
:class:`MetricsOutput` for the metrics writer.

Story 8.10 wired the loader selection through ``cfg.dataset`` (was hardcoded
to ``SyntheticLoader(seed=0)``). Same composition path as the JEPA runner
via :func:`wormjepa.data.composition.build_loader`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from wormjepa import WormJEPAError
from wormjepa.baselines._registry import REGISTRY
from wormjepa.baselines.base import BaselinePredictions, FuturePoseHorizon
from wormjepa.data.composition import build_loader
from wormjepa.eval.bootstrap import WormGrouping, bootstrap_ci
from wormjepa.eval.metrics_schema import (
    BootstrapCI,
    MetricEntry,
    MetricsOutput,
    SubEntry,
)

if TYPE_CHECKING:
    from wormjepa.configs.baseline_config import BaselineRunConfig


def _per_clip_horizon_error(entry: FuturePoseHorizon) -> float:
    """Mean Euclidean distance between predicted and ground-truth pose.

    ``predicted`` and ``ground_truth`` both have shape ``(K, D)`` for a single
    frame. The metric is the mean over keypoints of the per-keypoint
    Euclidean distance.
    """
    diff = entry.predicted - entry.ground_truth
    return float(torch.linalg.vector_norm(diff, dim=-1).mean())


def _future_pose_metric_entry(
    predictions: BaselinePredictions, producer: str, n_bootstrap: int
) -> MetricEntry:
    """Aggregate future-pose entries into one ``MetricEntry`` with per-horizon sub-rows."""
    horizons = sorted({fp.horizon_seconds for fp in predictions.future_pose})
    sub_entries: list[SubEntry] = []
    for h in horizons:
        per_h = [fp for fp in predictions.future_pose if fp.horizon_seconds == h]
        errors = np.asarray([_per_clip_horizon_error(fp) for fp in per_h])
        grouping = WormGrouping(worm_ids=tuple(fp.worm_id for fp in per_h))
        ci = bootstrap_ci(
            errors,
            grouping,
            n_samples=n_bootstrap,
            method="percentile",
        )
        sub_entries.append(SubEntry(key=f"{h:g}s", ci=ci))

    # Top-level CI is the union across horizons (informational; reporting uses
    # the per-horizon sub-rows). We use a NaN top-level CI here since there
    # is no single canonical scalar.
    nan_ci = BootstrapCI(
        point=float("nan"),
        lower=float("nan"),
        upper=float("nan"),
        n_samples=n_bootstrap,
        method="percentile",
    )
    return MetricEntry(
        name="future_pose",
        producer=producer,
        ci=nan_ci,
        sub_entries=sub_entries,
        notes="per-horizon mean Euclidean keypoint error",
    )


def run_baseline(config: BaselineRunConfig, run_id: str, n_bootstrap: int = 200) -> MetricsOutput:
    """Fit + predict the configured baseline on the configured dataset.

    Builds the dataset iterable from ``config.dataset.loaders`` (default: a
    single synthetic loader at ``config.baseline.seed``). Calls the baseline's
    ``fit()`` then ``predict()`` on freshly-iterated copies of the same
    spec — keeps fit/predict deterministic without forcing baselines to
    handle re-iteration themselves.

    Args:
        config: Parsed :class:`BaselineRunConfig`.
        run_id: The run-id produced by :func:`wormjepa.cli.run_ids.generate_run_id`.
        n_bootstrap: Bootstrap sample count for the per-horizon CIs.

    Returns:
        A :class:`MetricsOutput` populated with the baseline's future-pose row.
    """
    cls = REGISTRY.get(config.baseline.name)
    if cls is None:
        msg = f"Unknown baseline {config.baseline.name!r}. Registered: {sorted(REGISTRY)}."
        raise WormJEPAError(msg)

    baseline = cls.from_config(config.baseline)

    seed = config.baseline.seed
    fit_loader = build_loader(config.dataset.loaders, seed=seed)
    baseline.fit(fit_loader)
    # Re-build the loader so fit and predict see the same sequence from the top.
    # build_loader returns a fresh iterable on each call; iter() is one-shot for
    # the chained loader, so this re-construction matters.
    predict_loader = build_loader(config.dataset.loaders, seed=seed)
    predictions = baseline.predict(predict_loader)

    entry = _future_pose_metric_entry(predictions, producer=baseline.name, n_bootstrap=n_bootstrap)
    return MetricsOutput(run_id=run_id, entries=[entry])
