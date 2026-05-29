"""Unit tests for ``wormjepa.eval.bootstrap``."""

from __future__ import annotations

import numpy as np
import pytest

from wormjepa import BootstrapGroupingError
from wormjepa.data import WormID
from wormjepa.eval.bootstrap import WormGrouping, bootstrap_ci


def _grouping(worm_ids: list[str]) -> WormGrouping:
    return WormGrouping(worm_ids=tuple(WormID(w) for w in worm_ids))


def test_worm_grouping_frozen() -> None:
    g = _grouping(["a", "a", "b"])
    with pytest.raises(ValueError, match="frozen"):
        g.worm_ids = ()  # type: ignore[misc]


def test_worm_grouping_length() -> None:
    g = _grouping(["a", "b", "c", "a"])
    assert len(g) == 4


def test_bootstrap_ci_constant_array_yields_degenerate_ci() -> None:
    """A constant-valued sample has zero variance; the CI collapses to a point."""
    values = np.full(20, 3.14)
    grouping = _grouping(["w1"] * 5 + ["w2"] * 5 + ["w3"] * 5 + ["w4"] * 5)
    ci = bootstrap_ci(
        values, grouping, n_samples=200, method="percentile", rng=np.random.default_rng(0)
    )
    assert ci.point == pytest.approx(3.14)
    assert ci.lower == pytest.approx(3.14)
    assert ci.upper == pytest.approx(3.14)


def test_bootstrap_ci_length_mismatch_raises() -> None:
    values = np.zeros(5)
    grouping = _grouping(["w1", "w2"])
    with pytest.raises(BootstrapGroupingError, match="lengths must match"):
        bootstrap_ci(values, grouping)


def test_bootstrap_ci_empty_grouping_raises() -> None:
    values = np.array([], dtype=np.float64)
    with pytest.raises(BootstrapGroupingError):
        bootstrap_ci(values, WormGrouping(worm_ids=()))


def test_bootstrap_ci_zero_samples_raises() -> None:
    values = np.zeros(5)
    grouping = _grouping(["w"] * 5)
    with pytest.raises(BootstrapGroupingError, match="n_samples"):
        bootstrap_ci(values, grouping, n_samples=0)


def test_bootstrap_ci_records_method_and_grouping() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    grouping = _grouping(["a", "a", "b", "b", "c", "c"])
    ci = bootstrap_ci(
        values, grouping, n_samples=100, method="percentile", rng=np.random.default_rng(0)
    )
    assert ci.method == "percentile"
    assert ci.grouping == "worm"
    assert ci.n_samples == 100


def test_bootstrap_ci_resamples_worms_not_frames() -> None:
    """Within-worm autocorrelation: a single worm contributes correlated samples.

    If the bootstrap erroneously resampled frames, the CI on the mean of a
    two-worm dataset with very different per-worm means would be narrow.
    Resampling worms with replacement keeps the CI wide enough to reflect
    the worm-level uncertainty.
    """
    # Two worms, very different means.
    w1_values = np.full(100, 10.0)
    w2_values = np.full(100, 20.0)
    values = np.concatenate([w1_values, w2_values])
    grouping = _grouping(["w1"] * 100 + ["w2"] * 100)

    ci = bootstrap_ci(
        values,
        grouping,
        n_samples=500,
        method="percentile",
        rng=np.random.default_rng(42),
    )
    # The mean is 15.0. The CI should span between the per-worm means
    # because resampling 2 worms with replacement produces cohorts of {w1,w1},
    # {w1,w2}, {w2,w1}, {w2,w2} — means 10, 15, 15, 20.
    # A frame-level bootstrap would collapse the CI to ~15 because frames
    # are iid copies of 10 and 20.
    assert ci.point == pytest.approx(15.0)
    assert ci.lower <= 10.5  # picked up the {w1, w1} cohort
    assert ci.upper >= 19.5  # picked up the {w2, w2} cohort


def test_bootstrap_ci_default_method_is_bca() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    grouping = _grouping(["a", "a", "b", "b", "c", "c"])
    ci = bootstrap_ci(values, grouping, n_samples=200, rng=np.random.default_rng(0))
    assert ci.method == "bca"


def test_bootstrap_ci_nominal_coverage_with_iid_worm_means() -> None:
    """Coverage check: 95% percentile CI covers the true mean ≥ 85% of the time.

    Loose bound (85%) to avoid Monte Carlo flake while still flagging gross bugs.
    """
    rng = np.random.default_rng(2026_05_12)
    true_mean = 0.0
    n_worms = 30
    n_frames_per_worm = 50
    n_trials = 60
    n_covered = 0

    for _ in range(n_trials):
        worm_means = rng.normal(loc=true_mean, scale=1.0, size=n_worms)
        per_worm_values = [rng.normal(loc=m, scale=0.5, size=n_frames_per_worm) for m in worm_means]
        values = np.concatenate(per_worm_values)
        worm_ids: list[str] = []
        for i in range(n_worms):
            worm_ids.extend([f"w{i}"] * n_frames_per_worm)
        grouping = _grouping(worm_ids)
        ci = bootstrap_ci(
            values,
            grouping,
            n_samples=400,
            method="percentile",
            rng=rng,
        )
        if ci.lower <= true_mean <= ci.upper:
            n_covered += 1

    coverage = n_covered / n_trials
    assert coverage >= 0.85, f"Coverage {coverage:.2f} below loose bound 0.85"


def test_bootstrap_ci_custom_statistic() -> None:
    """The statistic parameter accepts any (np.ndarray) -> float function."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    grouping = _grouping(["a", "a", "b", "b", "c", "c"])
    ci_median = bootstrap_ci(
        values,
        grouping,
        n_samples=100,
        method="percentile",
        statistic=lambda x: float(np.median(x)),
        rng=np.random.default_rng(0),
    )
    # Population median = 3.5
    assert ci_median.point == pytest.approx(3.5)


def test_bootstrap_ci_bca_method_runs_successfully() -> None:
    """BCa with non-degenerate data produces a finite CI bracketing the point."""
    rng = np.random.default_rng(0)
    n_worms = 20
    worm_means = rng.normal(loc=2.0, scale=0.5, size=n_worms)
    per_worm_values = [rng.normal(loc=m, scale=0.2, size=30) for m in worm_means]
    values = np.concatenate(per_worm_values)
    worm_ids: list[str] = []
    for i in range(n_worms):
        worm_ids.extend([f"w{i}"] * 30)
    grouping = _grouping(worm_ids)
    ci = bootstrap_ci(values, grouping, n_samples=500, method="bca", rng=rng)
    assert ci.method == "bca"
    assert ci.lower <= ci.point <= ci.upper
    assert np.isfinite(ci.lower) and np.isfinite(ci.upper)
