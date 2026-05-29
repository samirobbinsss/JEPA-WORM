"""Neural-decoding probe wrapper (Story 6.4 / FR27 / row 3).

Wraps the frozen probe code in ``pre-registration/probes/neural_decoding.py``
and decorates the result with a worm-level bootstrap CI. Phase 0 v0 calls
the residualization helper directly (the frozen probe stub returns
``NotImplementedError``); the wrapper is structured so a real probe
implementation can drop in without touching call sites.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wormjepa.data import WormID
from wormjepa.eval.bootstrap import WormGrouping, bootstrap_ci
from wormjepa.eval.metrics_schema import BootstrapCI, MetricEntry
from wormjepa.eval.residualization import partial_r2


def neural_decoding_partial_r2(
    jepa_latent: np.ndarray,
    kinematic_features: np.ndarray,
    neural_target: np.ndarray,
    worm_ids: Sequence[str],
    *,
    producer: str = "jepa",
    n_bootstrap: int = 1000,
) -> tuple[MetricEntry, BootstrapCI]:
    """Compute partial-R² with a worm-level bootstrap CI.

    Returns:
        Pair of ``(MetricEntry, BootstrapCI)``. The entry carries the
        bootstrapped CI on the per-worm partial-R² values; the standalone
        :class:`BootstrapCI` is also returned for callers that need it
        before assembling a full ``MetricsOutput``.
    """
    result = partial_r2(jepa_latent, kinematic_features, neural_target, worm_ids)
    values = np.asarray(result.per_worm_partial_r2)
    grouping = WormGrouping(worm_ids=tuple(WormID(w) for w in result.worm_ids))
    ci = bootstrap_ci(values, grouping, n_samples=n_bootstrap, method="bca")
    entry = MetricEntry(
        name="neural_probe_partial_r2",
        producer=producer,
        ci=ci,
        notes=(
            f"r2_jepa={result.r2_jepa:.3f}, r2_kin={result.r2_kinematic:.3f}, "
            f"partial={result.partial_r2:.3f}"
        ),
    )
    return entry, ci
