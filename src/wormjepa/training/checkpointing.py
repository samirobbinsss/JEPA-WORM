"""Checkpoint save/resume for JEPA training (Story 5.10).

Each checkpoint serializes: model state dicts (online encoder, predictor,
EMA target, every active warm-start head), optimizer state, training step
counter, RNG states, and any other metadata the training loop needs to
resume bit-identically (modulo documented non-determinism).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from wormjepa.training.loop import JEPATrainingState


def save_checkpoint(state: JEPATrainingState, path: Path) -> None:
    """Save a complete training-state checkpoint to ``path``."""
    payload: dict[str, Any] = {
        "step": state.step,
        "online_encoder": state.online_encoder.state_dict(),
        "predictor": state.predictor.state_dict(),
        "ema_target": state.ema_target.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "rng": {
            "python_random": random.getstate(),
            "numpy_legacy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        },
        "warm_start": {name: head.state_dict() for name, head in state.warm_start_heads.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(state: JEPATrainingState, path: Path) -> None:
    """Restore a training state from a checkpoint at ``path``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state.step = int(payload["step"])
    state.online_encoder.load_state_dict(payload["online_encoder"])
    state.predictor.load_state_dict(payload["predictor"])
    state.ema_target.load_state_dict(payload["ema_target"])
    state.optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["rng"]["python_random"])
    np.random.set_state(payload["rng"]["numpy_legacy"])
    torch.set_rng_state(payload["rng"]["torch_cpu"])
    if payload["rng"]["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
    for name, head_state in payload["warm_start"].items():
        if name in state.warm_start_heads:
            state.warm_start_heads[name].load_state_dict(head_state)
