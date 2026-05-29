"""Partial-R² residualization for the neural-decoding probe (Story 6.1).

Computes ``partial_R² = R²(JEPA-latent → target) - R²(kinematic-baseline → target)``
using two leave-one-worm-out ridge regressions: one fit on JEPA latents, one
on a kinematic-baseline feature matrix. Both predict the same neural target.

This is the PRD's headline neural-decoding metric (FR27 / measurable-outcome
row 3): partial R² ≥ 0.05 over the Tierpsy-256 + temporal-derivatives +
pose-only TCN baseline. The headline claim only matters if JEPA beats the
*strongest* pose-only baseline, not the weakest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True, slots=True)
class PartialR2Result:
    """Decomposition of the partial-R² metric.

    Attributes:
        partial_r2: ``r2_jepa - r2_kinematic`` averaged across leave-one-worm folds.
        r2_jepa: Mean R² when the linear probe is fit on JEPA latents.
        r2_kinematic: Mean R² when fit on the kinematic baseline features.
        per_worm_partial_r2: Held-out partial-R² per worm (length = n_worms).
        worm_ids: Worm identifier per element of ``per_worm_partial_r2``.
    """

    partial_r2: float
    r2_jepa: float
    r2_kinematic: float
    per_worm_partial_r2: list[float]
    worm_ids: list[str]


def _fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float
) -> np.ndarray:
    """Fit Ridge on ``(train_x, train_y)``, predict on ``test_x``."""
    model = Ridge(alpha=alpha)
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-output R²: mean across output dimensions of per-dim R²."""
    ss_res = float(((y_true - y_pred) ** 2).sum(axis=0).mean())
    ss_tot = float(((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0).mean())
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def partial_r2(
    jepa_latent: np.ndarray,
    kinematic_features: np.ndarray,
    neural_target: np.ndarray,
    worm_ids: Sequence[str],
    *,
    alpha: float = 1.0,
) -> PartialR2Result:
    """Compute leave-one-worm-out partial-R² of JEPA over a kinematic baseline.

    Args:
        jepa_latent: ``(N, D_jepa)`` per-frame JEPA latents.
        kinematic_features: ``(N, D_kin)`` per-frame kinematic features (e.g.,
            Tierpsy-256 + temporal derivatives + pose-only TCN latent).
        neural_target: ``(N, D_neural)`` per-frame neural activity.
        worm_ids: ``(N,)`` sequence of worm-id strings aligning the rows.
        alpha: Ridge regularization strength.

    Returns:
        :class:`PartialR2Result`.

    Raises:
        ValueError: If shapes are inconsistent or fewer than 2 distinct worms.
    """
    n = jepa_latent.shape[0]
    if kinematic_features.shape[0] != n or neural_target.shape[0] != n:
        msg = "jepa_latent, kinematic_features, neural_target must share first dim"
        raise ValueError(msg)
    if len(worm_ids) != n:
        msg = "worm_ids length must match first dim of features"
        raise ValueError(msg)
    unique = list(dict.fromkeys(worm_ids))
    if len(unique) < 2:
        msg = "partial_r2 requires >=2 distinct worms for leave-one-worm-out"
        raise ValueError(msg)

    worm_array = np.asarray(worm_ids)
    per_worm_jepa: list[float] = []
    per_worm_kin: list[float] = []
    for held_out in unique:
        train_mask = worm_array != held_out
        test_mask = worm_array == held_out
        if not train_mask.any() or not test_mask.any():
            continue
        y_test = neural_target[test_mask]
        y_pred_jepa = _fit_predict(
            jepa_latent[train_mask], neural_target[train_mask], jepa_latent[test_mask], alpha
        )
        y_pred_kin = _fit_predict(
            kinematic_features[train_mask],
            neural_target[train_mask],
            kinematic_features[test_mask],
            alpha,
        )
        per_worm_jepa.append(_r2_score(y_test, y_pred_jepa))
        per_worm_kin.append(_r2_score(y_test, y_pred_kin))

    r2_jepa = float(np.mean(per_worm_jepa))
    r2_kin = float(np.mean(per_worm_kin))
    per_worm_partial = [j - k for j, k in zip(per_worm_jepa, per_worm_kin, strict=True)]
    return PartialR2Result(
        partial_r2=r2_jepa - r2_kin,
        r2_jepa=r2_jepa,
        r2_kinematic=r2_kin,
        per_worm_partial_r2=per_worm_partial,
        worm_ids=unique,
    )
