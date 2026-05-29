"""Frozen neural-decoding probe code (Story 4.8).

This module is the load-bearing frozen artifact for the PRD's neural-decoding
headline metric (FR27 / measurable-outcomes row 3). Its SHA is recorded in
MANIFEST.lock and verified before every reportable run.

Phase 0 v0 ships a minimal probe interface; the real implementation lands in
Epic 6 Story 6.4. Both versions live here so the frozen interface is locked
from day one. Any downstream change to this file's AST-canonical bytes
requires a CHANGELOG ``### Frozen-artifact changes`` entry.
"""

from __future__ import annotations


def neural_decoding_partial_r2(
    jepa_latent: object,
    kinematic_features: object,
    neural_target: object,
    grouping: object,
) -> object:
    """Compute the partial R² of ``jepa_latent`` against ``neural_target``,
    residualizing against ``kinematic_features``.

    The argument types are deliberately ``object`` in this frozen interface
    so that AST canonicalization stays stable across Phase 0 if the real
    implementation in Epic 6 Story 6.4 adds typed wrapper helpers around
    this call. The wrapper (``wormjepa.eval.neural_decoding``) is responsible
    for narrowing the types at the call site.

    Implementation is filled in by Story 6.4. Until then, this function
    raises ``NotImplementedError`` so accidental use surfaces visibly.
    """
    msg = "neural_decoding_partial_r2 is a frozen-interface stub. See Epic 6 Story 6.4."
    raise NotImplementedError(msg)
