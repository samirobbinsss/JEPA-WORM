"""Seed management for reproducibility (Story 5.9 / NFR9).

Every reportable run calls :func:`set_seeds` once at startup. The function
sets seeds for Python ``random``, NumPy (both legacy + ``default_rng``-style),
and PyTorch (CPU + CUDA). Returns a structured record suitable for serializing
to ``results/<run-id>/seed.txt``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """Snapshot of seeds set by :func:`set_seeds`."""

    seed: int
    python_random: int
    numpy_legacy: int
    numpy_generator_state_hash: str
    torch_cpu: int
    torch_cuda: int | None

    def to_text(self) -> str:
        cuda_line = "" if self.torch_cuda is None else f"torch_cuda={self.torch_cuda}\n"
        return (
            f"seed={self.seed}\n"
            f"python_random={self.python_random}\n"
            f"numpy_legacy={self.numpy_legacy}\n"
            f"numpy_generator_state_hash={self.numpy_generator_state_hash}\n"
            f"torch_cpu={self.torch_cpu}\n"
            f"{cuda_line}"
        )


def set_seeds(seed: int) -> SeedRecord:
    """Seed every relevant RNG. Returns a :class:`SeedRecord` documenting what was set."""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    state_hash = hash(rng.bit_generator.state["state"]["state"])  # for audit only

    torch.manual_seed(seed)
    cuda_seed = seed if torch.cuda.is_available() else None
    if cuda_seed is not None:
        torch.cuda.manual_seed_all(cuda_seed)

    return SeedRecord(
        seed=seed,
        python_random=seed,
        numpy_legacy=seed,
        numpy_generator_state_hash=hex(state_hash & 0xFFFFFFFFFFFFFFFF),
        torch_cpu=seed,
        torch_cuda=cuda_seed,
    )
