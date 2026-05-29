"""Integration smoke for frozen V-JEPA 2.1 target in the training loop (Story 8.11b P2).

Builds a tiny ``JEPARunConfig`` with ``frozen_target=true`` +
``vjepa_variant='vjepa2_1_vit_base_384'`` against a single-worm synthetic
loader at 384x384 / T=16, runs ``run_jepa`` end-to-end for 2 gradient
steps, and asserts:

1. ``state.online_encoder`` is a :class:`TrainableVJEPAEncoder` (V-JEPA 2.1
   weights, trainable).
2. ``state.ema_target`` is a :class:`FrozenVJEPATarget` (V-JEPA 2.1 weights,
   frozen) and every one of its parameters has ``requires_grad=False`` even
   after the optimiser steps.
3. The frozen target stays in ``eval()`` mode after the loop's
   ``model.train()`` / ``.eval()`` discipline.
4. The training loop produces at least a ``jepa`` loss entry — the EMA
   update path is correctly skipped for the frozen target (which has no
   ``.update()`` method).

Opt-in via ``WORMJEPA_TEST_VJEPA21=1`` (same flag as the loader smoke):
this exercises the real V-JEPA 2.1 weight load + a forward/backward pass
on the actual ViT-B at 384x384, which is heavy enough that we do not
want it firing in default CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ENV_FLAG = "WORMJEPA_TEST_VJEPA21"

pytestmark = pytest.mark.skipif(
    os.environ.get(_ENV_FLAG) != "1",
    reason=(
        f"Set {_ENV_FLAG}=1 to run; reuses the ~300 MB V-JEPA 2.1 checkpoint "
        f"cached by the loader smoke and runs a real forward/backward at 384x384."
    ),
)


def test_frozen_vjepa_target_runs_training_loop(tmp_path: Path) -> None:
    from wormjepa.cli.run_ids import generate_run_id
    from wormjepa.configs.dataset import DatasetLoaderSpec, DatasetSection
    from wormjepa.configs.jepa_config import JEPARunConfig, JEPASection
    from wormjepa.models.vjepa_loader import FrozenVJEPATarget, TrainableVJEPAEncoder
    from wormjepa.paths import project_root
    from wormjepa.training.runner import run_jepa

    cfg = JEPARunConfig(
        schema_version=1,
        jepa=JEPASection(
            model_name="vjepa2_1_vit_base_384",
            img_size=384,
            latent_dim=768,
            masking_ratio=0.5,
            n_steps=2,
            learning_rate=1.0e-4,
            ema_decay=0.996,
            seed=42,
            frozen_target=True,
            vjepa_variant="vjepa2_1_vit_base_384",
        ),
        dataset=DatasetSection(
            loaders=[
                DatasetLoaderSpec(
                    name="synthetic",
                    clip_frames=16,
                    n_worms=1,
                    clips_per_worm=2,
                    image_size=384,
                ),
            ]
        ),
    )

    run_id = generate_run_id(config_path=tmp_path / "jepa_frozen_smoke.yaml")
    metrics, state = run_jepa(cfg, run_id)

    # Type assertions
    assert isinstance(state.online_encoder, TrainableVJEPAEncoder), (
        f"online_encoder should be TrainableVJEPAEncoder; got {type(state.online_encoder).__name__}"
    )
    assert isinstance(state.ema_target, FrozenVJEPATarget), (
        f"ema_target should be FrozenVJEPATarget; got {type(state.ema_target).__name__}"
    )

    # Frozen target invariants survive training
    for name, p in state.ema_target.named_parameters():
        assert not p.requires_grad, (
            f"Frozen target parameter {name} became trainable during training."
        )
    assert not state.ema_target.training, (
        "Frozen target should stay in eval() mode after training loop."
    )

    # Online stayed trainable
    assert any(p.requires_grad for p in state.online_encoder.parameters()), (
        "Online encoder has no trainable params; can't fine-tune."
    )

    # Loss produced
    assert "jepa" in state.last_losses, (
        f"Expected 'jepa' loss in last_losses; got {sorted(state.last_losses)}."
    )

    # Metrics contract honoured
    assert metrics.run_id == run_id

    # Story 8.12a: run_jepa now saves a checkpoint so the gate-evaluation
    # orchestrator can reload encoder + predictor + warm-start heads.
    checkpoint_path = project_root() / "results" / run_id / "checkpoints" / "checkpoint.pt"
    assert checkpoint_path.is_file(), (
        f"Expected run_jepa to write checkpoint at {checkpoint_path}; not present."
    )
    assert checkpoint_path.stat().st_size > 0, "Checkpoint file is empty."
