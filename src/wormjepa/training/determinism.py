"""Deterministic-execution config (NFR6).

Enables PyTorch's deterministic algorithm mode where feasible without an
unacceptable performance penalty. Operations that don't have deterministic
implementations are documented in :data:`NON_DETERMINISTIC_OPS`.
"""

from __future__ import annotations

import os

import torch

NON_DETERMINISTIC_OPS: tuple[str, ...] = (
    # torch.nn.functional.scaled_dot_product_attention: backend selection is
    # non-deterministic on CUDA at fp16/bf16 without TORCH_USE_DETERMINISTIC_ALGORITHMS=1.
    "torch.nn.functional.scaled_dot_product_attention (bf16/fp16 on CUDA)",
    # torch.cuda.atomic_add: GPU reductions over non-deterministic index sets.
    "torch.scatter_add_ with non-deterministic index ordering",
)
"""Operations known to be non-deterministic; documented for the audit trail."""


def enable_determinism(*, warn_only: bool = True) -> None:
    """Enable PyTorch deterministic algorithms.

    Args:
        warn_only: When True, non-deterministic operations log a warning
            instead of raising. Phase 0 default is True so the training loop
            can proceed even if a layer hits a non-deterministic kernel.
    """
    # CUBLAS workspace config: required for deterministic matmul on CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
