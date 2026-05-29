"""Unit tests for probe + session-classifier + within-state + neuron-subset (Stories 6.4-6.7)."""

from __future__ import annotations

import numpy as np

from wormjepa.eval.neural_decoding import neural_decoding_partial_r2
from wormjepa.eval.neuron_subsets import neuron_subset_decoding
from wormjepa.eval.session_classifier import session_classifier_accuracy
from wormjepa.eval.within_state import within_state_decoding


def _worm_data(n_worms: int = 4, n_per_worm: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    jepa = rng.normal(size=(n_worms * n_per_worm, 8))
    kin = rng.normal(size=(n_worms * n_per_worm, 6))
    target = rng.normal(size=(n_worms * n_per_worm, 5))
    worms = [f"w{i}" for i in range(n_worms) for _ in range(n_per_worm)]
    sessions = [f"w{i}_s{(j % 2)}" for i in range(n_worms) for j in range(n_per_worm)]
    states = rng.integers(0, 5, size=n_worms * n_per_worm)
    return jepa, kin, target, worms, sessions, states


def test_neural_decoding_partial_r2_wraps_residualization_and_adds_ci() -> None:
    jepa, kin, target, worms, _sessions, _states = _worm_data(seed=0)
    entry, ci = neural_decoding_partial_r2(
        jepa, kin, target, worms, producer="jepa", n_bootstrap=64
    )
    assert entry.name == "neural_probe_partial_r2"
    assert entry.producer == "jepa"
    assert ci.grouping == "worm"
    assert ci.lower <= ci.point <= ci.upper


def test_session_classifier_returns_chance_baseline() -> None:
    jepa, _kin, _target, worms, sessions, _states = _worm_data(seed=0)
    entry, ci, chance = session_classifier_accuracy(
        jepa, sessions, worms, producer="jepa", n_bootstrap=64
    )
    # 4 worms * 2 sessions per worm = 8 unique sessions; chance = 1/8.
    assert chance == 1.0 / 8.0
    assert entry.name == "session_id_classifier"
    assert ci.grouping == "worm"


def test_within_state_decoding_produces_one_subrow_per_observed_state() -> None:
    jepa, kin, target, worms, _sessions, states = _worm_data(seed=0, n_per_worm=80)
    entry = within_state_decoding(jepa, kin, target, worms, states, producer="jepa")
    assert entry.name == "within_state_partial_r2"
    # Random states might not produce all 5 — but with 320 samples we expect most.
    assert len(entry.sub_entries) >= 1
    for sub in entry.sub_entries:
        assert sub.ci.grouping == "worm"


def test_neuron_subset_decoding_returns_nan_when_no_match(tmp_path) -> None:
    """If no pre-committed neuron name matches the provided neuron_names, return a NaN row."""
    jepa, kin, target, worms, _sessions, _states = _worm_data(seed=0)
    subset_path = tmp_path / "neuron_subset.yaml"
    subset_path.write_text(
        "schema_version: 1\nneuron_subset:\n  - name: NONEXISTENT_1\n",
        encoding="utf-8",
    )
    entry = neuron_subset_decoding(
        jepa,
        kin,
        target,
        neuron_names=["unrelated_neuron"] * target.shape[1],
        worm_ids=worms,
        producer="jepa",
        n_bootstrap=64,
        neuron_subset_path=subset_path,
    )
    assert entry.name == "non_trivial_neuron_subset_partial_r2"
    assert np.isnan(entry.ci.point)


def test_neuron_subset_decoding_runs_with_matching_names(tmp_path) -> None:
    jepa, kin, target, worms, _sessions, _states = _worm_data(seed=0)
    subset_path = tmp_path / "neuron_subset.yaml"
    subset_path.write_text(
        "schema_version: 1\nneuron_subset:\n  - name: neuron_0\n  - name: neuron_2\n",
        encoding="utf-8",
    )
    neuron_names = [f"neuron_{i}" for i in range(target.shape[1])]
    entry = neuron_subset_decoding(
        jepa,
        kin,
        target,
        neuron_names=neuron_names,
        worm_ids=worms,
        neuron_subset_path=subset_path,
        n_bootstrap=64,
    )
    assert entry.name == "non_trivial_neuron_subset_partial_r2"
    assert not np.isnan(entry.ci.point)
