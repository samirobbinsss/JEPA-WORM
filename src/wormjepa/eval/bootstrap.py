"""Worm-level bootstrap confidence intervals.

**Load-bearing API** (FR28 / NFR16): the bootstrap-CI entry point requires a
:class:`WormGrouping` parameter. Calling ``bootstrap_ci(values, np.array(...))``
without constructing a ``WormGrouping`` is a pyright type error. This is the
structural defense against accidentally running frame-level bootstrap (which
would be anti-conservative by orders of magnitude on serially-correlated
within-worm data).

Both percentile and BCa (bias-corrected and accelerated) methods are supported.
Default is ``"bca"``; both are exposed for the pre-registered commitment in
``pre-registration/PRE-REGISTRATION.md`` (Story 4.7).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.stats import norm

from wormjepa import BootstrapGroupingError
from wormjepa.data import WormID
from wormjepa.eval.metrics_schema import BootstrapCI


class WormGrouping(BaseModel):
    """Maps each observation in a value array to the worm it came from.

    Required parameter for :func:`bootstrap_ci`. The bootstrap resamples
    *worms* (with replacement) on each iteration, then aggregates all
    observations belonging to the resampled worms — never resampling
    observations directly. This preserves the within-worm autocorrelation
    structure that frame-level bootstrap would erase.

    Attributes:
        worm_ids: Sequence of length ``N`` (where ``N`` is the number of
            observations in the value array). Position ``i`` of ``worm_ids``
            is the :class:`WormID` of the ``i``-th observation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    worm_ids: tuple[WormID, ...]

    @field_validator("worm_ids", mode="before")
    @classmethod
    def _coerce_to_tuple(cls, v: object) -> tuple[WormID, ...]:
        if isinstance(v, tuple):
            return v
        if isinstance(v, list):
            return tuple(v)  # type: ignore[arg-type]
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            return tuple(v)  # type: ignore[misc]
        msg = f"worm_ids must be a Sequence of WormID; got {type(v).__name__}"
        raise BootstrapGroupingError(msg)

    def __len__(self) -> int:
        return len(self.worm_ids)


def _unique_preserving_order(items: Sequence[WormID]) -> list[WormID]:
    """Return the unique entries in ``items`` in first-appearance order."""
    seen: dict[WormID, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def _group_indices(grouping: WormGrouping) -> dict[WormID, np.ndarray]:
    """Index buckets keyed by worm_id. Each bucket is the positions of that worm."""
    buckets: dict[WormID, list[int]] = {}
    for idx, wid in enumerate(grouping.worm_ids):
        buckets.setdefault(wid, []).append(idx)
    return {wid: np.asarray(idxs, dtype=np.int64) for wid, idxs in buckets.items()}


def _percentile_ci(boot_stats: np.ndarray, level: float) -> tuple[float, float]:
    alpha = 1.0 - level
    lo = float(np.quantile(boot_stats, alpha / 2.0))
    hi = float(np.quantile(boot_stats, 1.0 - alpha / 2.0))
    return lo, hi


def _bca_ci(
    boot_stats: np.ndarray,
    point: float,
    jackknife_stats: np.ndarray,
    level: float,
) -> tuple[float, float]:
    """BCa interval using the jackknife-on-worms acceleration estimate."""
    # Bias correction z0
    prop_below = float(np.mean(boot_stats < point))
    # Guard against degenerate (constant) distributions.
    if prop_below in {0.0, 1.0}:
        return _percentile_ci(boot_stats, level)
    z0 = float(norm.ppf(prop_below))

    # Acceleration via jackknife
    jack_mean = float(np.mean(jackknife_stats))
    diffs = jack_mean - jackknife_stats
    num = float(np.sum(diffs**3))
    den = 6.0 * (float(np.sum(diffs**2)) ** 1.5)
    if den == 0.0:
        return _percentile_ci(boot_stats, level)
    a = num / den

    alpha = 1.0 - level
    z_alpha_lo = float(norm.ppf(alpha / 2.0))
    z_alpha_hi = float(norm.ppf(1.0 - alpha / 2.0))

    def _adjusted(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if denom == 0.0:
            return 0.5
        return float(norm.cdf(z0 + (z0 + z) / denom))

    a1 = _adjusted(z_alpha_lo)
    a2 = _adjusted(z_alpha_hi)
    lo = float(np.quantile(boot_stats, a1))
    hi = float(np.quantile(boot_stats, a2))
    return lo, hi


def bootstrap_ci(
    values: np.ndarray,
    grouping: WormGrouping,
    n_samples: int = 1000,
    method: Literal["bca", "percentile"] = "bca",
    level: float = 0.95,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Compute a worm-level bootstrap confidence interval for ``statistic(values)``.

    On each bootstrap iteration, the unique worms in ``grouping`` are resampled
    with replacement to a cohort of the same size; observations belonging to
    the resampled worms are then aggregated and the ``statistic`` is applied.

    Args:
        values: ``(N,)`` array of per-observation scalar values (e.g., per-frame
            errors, per-clip metric values).
        grouping: Worm assignment for each of the ``N`` observations — see
            :class:`WormGrouping`. **Mandatory positional / keyword argument.**
        n_samples: Number of bootstrap iterations. Must be ≥ 1 (typical: 1000+,
            up to 10000 per NFR16).
        method: ``"bca"`` (default) or ``"percentile"``. BCa uses jackknife
            acceleration over worms; percentile is the simpler quantile method.
        level: Confidence level, default 0.95.
        statistic: Function applied to the values of each bootstrap cohort.
            Default is the mean.
        rng: Optional :class:`numpy.random.Generator` for reproducibility.

    Returns:
        A :class:`BootstrapCI` with worm-level grouping recorded.

    Raises:
        BootstrapGroupingError: If ``len(values) != len(grouping)`` or if the
            grouping has zero unique worms.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.shape[0] != len(grouping):
        msg = (
            f"values has {values.shape[0]} observations but grouping has "
            f"{len(grouping)} worm-ids; lengths must match."
        )
        raise BootstrapGroupingError(msg)
    if n_samples < 1:
        msg = f"n_samples must be ≥ 1; got {n_samples}"
        raise BootstrapGroupingError(msg)

    unique_worms = _unique_preserving_order(grouping.worm_ids)
    if not unique_worms:
        msg = "WormGrouping has zero worms; cannot bootstrap."
        raise BootstrapGroupingError(msg)

    buckets = _group_indices(grouping)
    n_worms = len(unique_worms)
    worm_to_idx = {wid: i for i, wid in enumerate(unique_worms)}
    bucket_arrays = [buckets[wid] for wid in unique_worms]

    rng = rng or np.random.default_rng()

    # Bootstrap distribution
    boot_stats = np.empty(n_samples, dtype=np.float64)
    for b in range(n_samples):
        resampled_worm_positions = rng.integers(0, n_worms, size=n_worms)
        # Gather indices from each resampled worm's bucket
        idx_list = [bucket_arrays[p] for p in resampled_worm_positions]
        idxs = np.concatenate(idx_list)
        boot_stats[b] = statistic(values[idxs])

    point = statistic(values)

    if method == "percentile":
        lo, hi = _percentile_ci(boot_stats, level)
    else:
        # Jackknife over worms (leave-one-worm-out) for BCa acceleration.
        jack_stats = np.empty(n_worms, dtype=np.float64)
        for k in range(n_worms):
            remaining = [bucket_arrays[i] for i in range(n_worms) if i != k]
            if not remaining:
                jack_stats[k] = point
                continue
            idxs = np.concatenate(remaining)
            jack_stats[k] = statistic(values[idxs])
        lo, hi = _bca_ci(boot_stats, point, jack_stats, level)
        _ = worm_to_idx  # silence "unused" — kept for clarity

    return BootstrapCI(
        point=point,
        lower=lo,
        upper=hi,
        level=level,
        method=method,
        n_samples=n_samples,
    )
