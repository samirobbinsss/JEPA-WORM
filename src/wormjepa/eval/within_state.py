"""Within-state stratified decoding (Story 6.6 / FR32 / row 7).

For each Flavell behavioral state, compute partial-R² of JEPA vs kinematic
on that state's frames only. Rules out the trivial "latent merely encodes
behavioral state" explanation: if JEPA's headline R² disappears after
within-state stratification, the latent isn't capturing within-state neural
structure beyond what state identity already implies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from wormjepa.eval.metrics_schema import BootstrapCI, MetricEntry, SubEntry
from wormjepa.eval.residualization import partial_r2

_STATE_NAMES: tuple[str, ...] = ("forward", "reversal", "omega_turn", "pause", "quiescence")


def within_state_decoding(
    jepa_latent: np.ndarray,
    kinematic_features: np.ndarray,
    neural_target: np.ndarray,
    worm_ids: Sequence[str],
    behavioral_states: np.ndarray,
    *,
    producer: str = "jepa",
    state_names: tuple[str, ...] = _STATE_NAMES,
) -> MetricEntry:
    """Partial-R² stratified by behavioral state.

    Args:
        jepa_latent: ``(N, D_jepa)``.
        kinematic_features: ``(N, D_kin)``.
        neural_target: ``(N, D_neural)``.
        worm_ids: ``(N,)`` strings.
        behavioral_states: ``(N,)`` integer state labels (0..n_states-1).
        state_names: Names for each state index (used in sub-row keys).
        producer: Identifier for ``MetricEntry.producer``.

    Returns:
        :class:`MetricEntry` with one sub-row per state (states with too
        few samples or worms are skipped with a warning in ``notes``).
    """
    sub_entries: list[SubEntry] = []
    skipped: list[str] = []
    for state_idx, name in enumerate(state_names):
        mask = behavioral_states == state_idx
        if not mask.any():
            skipped.append(name)
            continue
        sub_worms = [w for w, m in zip(worm_ids, mask.tolist(), strict=True) if m]
        if len(set(sub_worms)) < 2:
            skipped.append(name)
            continue
        try:
            result = partial_r2(
                jepa_latent[mask],
                kinematic_features[mask],
                neural_target[mask],
                sub_worms,
            )
        except ValueError:
            skipped.append(name)
            continue
        # Degenerate CI on the per-worm partial-R² (one point per worm).
        values = np.asarray(result.per_worm_partial_r2)
        point = float(values.mean())
        lower = float(np.min(values))
        upper = float(np.max(values))
        ci = BootstrapCI(
            point=point,
            lower=lower,
            upper=upper,
            n_samples=len(values),
            method="percentile",
        )
        sub_entries.append(SubEntry(key=name, ci=ci))

    nan_ci = BootstrapCI(
        point=float("nan"),
        lower=float("nan"),
        upper=float("nan"),
        n_samples=1,
        method="percentile",
    )
    notes = "within-state partial-R²; states skipped: " + (",".join(skipped) or "none")
    return MetricEntry(
        name="within_state_partial_r2",
        producer=producer,
        ci=nan_ci,
        sub_entries=sub_entries,
        notes=notes,
    )
