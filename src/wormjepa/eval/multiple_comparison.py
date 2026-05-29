"""Multiple-comparison correction (Story 6.9 / FR34 / NFR17).

Wraps :func:`statsmodels.stats.multitest.multipletests` with the Phase 0
pre-registered defaults: Holm method (per PRE-REGISTRATION.md), alpha = 0.05.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from statsmodels.stats.multitest import multipletests

CorrectionMethod = Literal["holm", "bh"]


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """Outcome of applying multiple-comparison correction to a family of p-values."""

    corrected_pvalues: list[float]
    reject: list[bool]
    method: CorrectionMethod
    alpha: float


def apply_correction(
    p_values: Sequence[float],
    method: CorrectionMethod = "holm",
    alpha: float = 0.05,
) -> CorrectionResult:
    """Apply Holm or Benjamini-Hochberg correction.

    Args:
        p_values: Family of raw p-values.
        method: ``"holm"`` (family-wise error rate) or ``"bh"`` (FDR).
        alpha: Significance level.

    Returns:
        :class:`CorrectionResult` with corrected p-values + reject decisions.
    """
    if not (0.0 < alpha < 1.0):
        msg = f"alpha must be in (0, 1); got {alpha}"
        raise ValueError(msg)
    arr = np.asarray(p_values, dtype=np.float64)
    if arr.size == 0:
        return CorrectionResult(corrected_pvalues=[], reject=[], method=method, alpha=alpha)
    sm_method = "holm" if method == "holm" else "fdr_bh"
    result = multipletests(arr, alpha=alpha, method=sm_method)
    reject = np.asarray(result[0]).tolist()
    corrected = np.asarray(result[1]).tolist()
    return CorrectionResult(
        corrected_pvalues=[float(p) for p in corrected],
        reject=[bool(r) for r in reject],
        method=method,
        alpha=alpha,
    )
