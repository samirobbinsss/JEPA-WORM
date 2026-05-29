"""Non-trivial neuron subset decoding (Story 6.7 / FR33 / row 8).

Restricts the neural-decoding probe to the pre-committed subset of neurons
in ``pre-registration/neuron_subset.yaml`` — neurons whose decoding from the
kinematic baseline alone is non-trivial. Reported as a separate row so
"we decoded easy-from-Tierpsy neurons" cannot inflate the headline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from wormjepa.eval.metrics_schema import MetricEntry
from wormjepa.eval.neural_decoding import neural_decoding_partial_r2
from wormjepa.paths import project_root


def _load_neuron_subset(neuron_subset_path: Path | None = None) -> list[str]:
    """Read ``pre-registration/neuron_subset.yaml`` and return neuron names."""
    path = neuron_subset_path or (project_root() / "pre-registration" / "neuron_subset.yaml")
    if not path.is_file():
        msg = f"neuron_subset.yaml not found at {path}"
        raise FileNotFoundError(msg)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    subset = data.get("neuron_subset", [])
    return [str(entry["name"]) for entry in subset if "name" in entry]


def neuron_subset_decoding(
    jepa_latent: np.ndarray,
    kinematic_features: np.ndarray,
    neural_target: np.ndarray,
    neuron_names: Sequence[str],
    worm_ids: Sequence[str],
    *,
    producer: str = "jepa",
    n_bootstrap: int = 1000,
    neuron_subset_path: Path | None = None,
) -> MetricEntry:
    """Partial-R² decoding restricted to the pre-committed non-trivial neurons.

    Args:
        jepa_latent: ``(N, D_jepa)``.
        kinematic_features: ``(N, D_kin)``.
        neural_target: ``(N, n_neurons)``.
        neuron_names: Names aligning with the columns of ``neural_target``.
        worm_ids: ``(N,)`` worm-id strings.
        producer: Identifier for ``MetricEntry.producer``.
        n_bootstrap: Bootstrap-CI sample count.
        neuron_subset_path: Override path to the subset file (test hook).

    Returns:
        :class:`MetricEntry` with the partial-R² on the subset, or a NaN
        entry if no committed neurons match the available columns.
    """
    subset = _load_neuron_subset(neuron_subset_path)
    name_to_idx = {n: i for i, n in enumerate(neuron_names)}
    cols = [name_to_idx[n] for n in subset if n in name_to_idx]
    if not cols:
        from wormjepa.eval.metrics_schema import BootstrapCI

        nan_ci = BootstrapCI(
            point=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n_samples=1,
            method="percentile",
        )
        return MetricEntry(
            name="non_trivial_neuron_subset_partial_r2",
            producer=producer,
            ci=nan_ci,
            notes=(
                "none of the pre-committed neurons matched provided column names; "
                "Story 6.7 entry-point populates real names."
            ),
        )
    restricted_target = neural_target[:, cols]
    entry, _ci = neural_decoding_partial_r2(
        jepa_latent,
        kinematic_features,
        restricted_target,
        worm_ids,
        producer=producer,
        n_bootstrap=n_bootstrap,
    )
    return entry.model_copy(update={"name": "non_trivial_neuron_subset_partial_r2"})
