"""Session-ID shortcut diagnostic (Story 6.5 / FR31 / row 6).

A linear classifier predicting session-ID from the latent. The PRD's
session-ID diagnostic gate: 95% CI of accuracy must contain the chance
baseline (NFR19); above-chance decoding means the latent has spuriously
captured session-level artifacts and the headline result is held until the
shortcut is removed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from wormjepa.data import WormID
from wormjepa.eval.bootstrap import WormGrouping, bootstrap_ci
from wormjepa.eval.metrics_schema import BootstrapCI, MetricEntry


def session_classifier_accuracy(
    latent: np.ndarray,
    session_ids: Sequence[str],
    worm_ids: Sequence[str],
    *,
    producer: str = "jepa",
    n_bootstrap: int = 1000,
) -> tuple[MetricEntry, BootstrapCI, float]:
    """Leave-one-worm-out balanced accuracy of a session-ID classifier.

    Returns:
        Tuple ``(entry, ci, chance_baseline)`` where:
        - ``entry`` is the populated :class:`MetricEntry` for results.json,
        - ``ci`` is the bootstrap CI on per-worm accuracy,
        - ``chance_baseline`` = 1 / n_sessions (uniform-prior chance).
    """
    n = latent.shape[0]
    if len(session_ids) != n or len(worm_ids) != n:
        msg = "latent, session_ids, worm_ids must share first dim"
        raise ValueError(msg)
    worm_arr = np.asarray(worm_ids)
    unique_sessions = list(dict.fromkeys(session_ids))
    unique_worms = list(dict.fromkeys(worm_ids))
    if len(unique_worms) < 2:
        msg = "session_classifier_accuracy requires >=2 distinct worms"
        raise ValueError(msg)

    chance = 1.0 / max(len(unique_sessions), 1)
    session_to_idx = {s: i for i, s in enumerate(unique_sessions)}
    y = np.asarray([session_to_idx[s] for s in session_ids])

    per_worm: list[float] = []
    for held_out in unique_worms:
        train_mask = worm_arr != held_out
        test_mask = worm_arr == held_out
        if not train_mask.any() or not test_mask.any():
            continue
        if len(np.unique(y[train_mask])) < 2:
            # Held-out worm leaves the train set with only one class — skip.
            continue
        clf = LogisticRegression(max_iter=200)
        clf.fit(latent[train_mask], y[train_mask])
        pred = clf.predict(latent[test_mask])
        per_worm.append(float(np.mean(pred == y[test_mask])))

    if not per_worm:
        msg = "session_classifier_accuracy could not produce any per-worm folds"
        raise ValueError(msg)

    values = np.asarray(per_worm)
    grouping = WormGrouping(worm_ids=tuple(WormID(w) for w in unique_worms[: len(per_worm)]))
    ci = bootstrap_ci(values, grouping, n_samples=n_bootstrap, method="bca")
    entry = MetricEntry(
        name="session_id_classifier",
        producer=producer,
        ci=ci,
        notes=(f"chance={chance:.3f}; at-chance gate requires CI to contain chance"),
    )
    return entry, ci, chance
