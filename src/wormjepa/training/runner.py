"""JEPA-run orchestration: training + metrics writing (Story 5.11, 8.9).

``build_loader`` + ``ChainedLoader`` moved to
:mod:`wormjepa.data.composition` in Story 8.10 so the baseline runner
can share the same implementation. Re-exported here for backward
compatibility — existing imports still work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn, optim

from wormjepa import DatasetIntegrityError
from wormjepa.data.composition import ChainedLoader, build_loader
from wormjepa.data.loaders.synthetic import SyntheticLoader
from wormjepa.eval.metrics_schema import (
    BootstrapCI,
    MetricEntry,
    MetricsOutput,
)
from wormjepa.models.ema import EMATarget
from wormjepa.models.encoder import WormJEPAEncoder
from wormjepa.models.pose_decoder import PoseDecoderHead
from wormjepa.models.predictor import JEPAPredictor
from wormjepa.models.vjepa_loader import (
    build_frozen_vjepa_target,
    build_trainable_vjepa_encoder,
)
from wormjepa.models.warm_start import (
    BehavioralHead,
    EigenwormHead,
    GraphPriorHead,
    NeuralAuxiliaryHead,
)
from wormjepa.paths import project_root
from wormjepa.training.loop import (
    JEPATrainingConfig,
    JEPATrainingState,
    train_jepa,
)
from wormjepa.training.seeds import set_seeds

if TYPE_CHECKING:
    from wormjepa.configs.jepa_config import JEPARunConfig

__all__ = [
    "ChainedLoader",
    "build_loader",
    "run_jepa",
]


def _select_device() -> torch.device:
    """Auto-pick accelerator: CUDA > MPS > CPU.

    MPS (Apple Silicon Metal backend) is included so the Story 8.9 smoke can
    run on a MacBook Pro and have ``compute.json`` honestly report ``Apple
    GPU (MPS, ...)`` rather than lying about an unused accelerator.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_state(cfg: JEPARunConfig) -> JEPATrainingState:
    device = _select_device()

    # Online + target construction:
    #   - frozen_target=True  → online is a (fine-tuned) V-JEPA 2.1 encoder,
    #     target is a SEPARATE frozen V-JEPA 2.1 encoder. Note (2026-05-20):
    #     this posture has no stable non-collapsed equilibrium — the
    #     predictor can satisfy a fixed foreign target degenerately. See the
    #     R3 collapse diagnosis in CHANGELOG.md.
    #   - frozen_target=False + vjepa_variant set → standard JEPA: online is
    #     a trainable V-JEPA 2.1 encoder, target is its EMA copy. The EMA +
    #     stop-grad is the anti-collapse mechanism. This is the headline
    #     posture as of 2026-05-20.
    #   - frozen_target=False + no vjepa_variant → legacy random-init timm
    #     ViT online + EMA copy. Kept for the non-V-JEPA smoke configs.
    # Predictor + heads size from the encoder's embed_dim (V-JEPA) or
    # cfg.jepa.latent_dim (legacy).
    online: nn.Module
    target: nn.Module
    if cfg.jepa.frozen_target:
        if cfg.jepa.vjepa_variant is None:
            # The schema validator catches this, but keep a defensive guard
            # so a config that bypasses the validator still raises clearly.
            msg = "frozen_target=True requires cfg.jepa.vjepa_variant."
            raise ValueError(msg)
        online_wrap = build_trainable_vjepa_encoder(
            cfg.jepa.vjepa_variant,
            pretrained_checkpoint_sha=cfg.jepa.pretrained_checkpoint_sha,
            random_init=cfg.jepa.online_init == "random",
        )
        target_wrap = build_frozen_vjepa_target(
            cfg.jepa.vjepa_variant,
            pretrained_checkpoint_sha=cfg.jepa.pretrained_checkpoint_sha,
        )
        latent_dim = online_wrap.embed_dim
        online = online_wrap
        target = target_wrap
    elif cfg.jepa.vjepa_variant is not None:
        # Standard EMA-target JEPA on the V-JEPA architecture.
        online_wrap = build_trainable_vjepa_encoder(
            cfg.jepa.vjepa_variant,
            pretrained_checkpoint_sha=cfg.jepa.pretrained_checkpoint_sha,
            random_init=cfg.jepa.online_init == "random",
        )
        latent_dim = online_wrap.embed_dim
        online = online_wrap
        # EMA target deep-copies the online encoder; constructed pre-device-
        # move, moved to device alongside online below.
        target = EMATarget(online_wrap, decay=cfg.jepa.ema_decay)
    else:
        legacy_encoder = WormJEPAEncoder(
            model_name=cfg.jepa.model_name,
            latent_dim=cfg.jepa.latent_dim,
            img_size=cfg.jepa.img_size,
        )
        latent_dim = cfg.jepa.latent_dim
        online = legacy_encoder
        # EMA target deep-copies the random-init online; constructed here
        # pre-device-move so the deepcopy semantics match pre-Story-8.11a
        # behaviour. Moved to device alongside online below.
        target = EMATarget(legacy_encoder, decay=cfg.jepa.ema_decay)

    # Determine head input dimensions from the first pose-providing loader in
    # the dataset chain. The eigenworm head's basis must be fitted to the same
    # n_keypoints the training data produces (otherwise `(flat - pose_mean) @
    # basis` shape-mismatches at the first gradient step); the pose decoder's
    # output must match the training data's keypoint count for the same reason.
    # Real loaders' pose dim is data-dependent and the dry-loop never reaches
    # them in the typical chain (baaiworm leads), so we read from the first
    # synthetic-or-baaiworm spec and fall back to the legacy n_keypoints=4
    # default. Closes the Story 8.11c dry-run discovery of dim 98 vs 8.
    first_spec = cfg.dataset.loaders[0]
    n_keypoints = first_spec.n_keypoints if first_spec.name in ("synthetic", "baaiworm") else 4

    predictor = JEPAPredictor(
        latent_dim=latent_dim,
        n_layers=cfg.jepa.predictor_layers,
        n_heads=cfg.jepa.predictor_heads,
    )
    heads: dict[str, nn.Module] = {
        "eigenworm": EigenwormHead(latent_dim=latent_dim, n_eigen=4),
        "graph_prior": GraphPriorHead(latent_dim=latent_dim, n_neural_aligned_dims=8, n_edges=4),
        "neural": NeuralAuxiliaryHead(latent_dim=latent_dim, n_neurons=8),
        "behavioral": BehavioralHead(latent_dim=latent_dim, n_states=5),
        # Dev-loop visualisation only (not pre-registered); see pose_decoder.py.
        "pose_decoder": PoseDecoderHead(latent_dim=latent_dim, n_keypoints=n_keypoints),
    }
    # Fit eigenworm basis on a sample of synthetic pose data sized to match
    # the training data's n_keypoints. Real-data runs may want to re-fit on a
    # sample of the actual training distribution in a follow-up.
    fit_loader = SyntheticLoader(
        n_worms=2,
        clips_per_worm=1,
        clip_frames=2,
        image_size=(cfg.jepa.img_size, cfg.jepa.img_size),
        n_keypoints=n_keypoints,
        n_neurons=8,
        seed=cfg.jepa.seed,
    )
    flat = []
    for sample in fit_loader:
        if sample.pose is None:
            continue
        t, k, d = sample.pose.shape
        flat.append(sample.pose.reshape(t, k * d))
    eigen = heads["eigenworm"]
    assert isinstance(eigen, EigenwormHead)
    eigen.fit_basis(torch.cat(flat, dim=0))

    # Now move everything to the selected accelerator.
    online = online.to(device)
    target = target.to(device)
    predictor = predictor.to(device)
    for name, head in heads.items():
        heads[name] = head.to(device)

    # Optimiser: two param-groups so the warm-start heads can run at their
    # own learning rate (head_learning_rate) — in the curriculum's phase 2
    # they fit on a frozen encoder and need a higher lr. Each group stores
    # base_lr so the loop's warmup scales each group's own base. The
    # target encoder's params are excluded (requires_grad=False).
    encoder_params = list(online.parameters()) + list(predictor.parameters())
    head_params: list[torch.Tensor] = []
    for head in heads.values():
        head_params += [p for p in head.parameters() if p.requires_grad]
    base_lr = cfg.jepa.learning_rate
    head_lr = cfg.jepa.head_learning_rate
    optimizer: optim.Optimizer = optim.Adam(
        [
            {"params": encoder_params, "lr": base_lr, "base_lr": base_lr},
            {"params": head_params, "lr": head_lr, "base_lr": head_lr},
        ]
    )
    return JEPATrainingState(
        online_encoder=online,
        ema_target=target,
        predictor=predictor,
        optimizer=optimizer,
        warm_start_heads=heads,
    )


def run_jepa(cfg: JEPARunConfig, run_id: str) -> tuple[MetricsOutput, JEPATrainingState]:
    """Train a JEPA model on the configured dataset loaders, return metrics + state.

    Phase 0 v0: produces a single ``jepa_training_loss`` MetricEntry recording
    the final loss values. Future-pose evaluation (Epic 6) consumes the
    trained state.online_encoder.

    Writes per-step JSON-Lines records to
    ``results/<run-id>/log.jsonl`` (Architecture §"results contract").
    """
    set_seeds(cfg.jepa.seed)
    state = _build_state(cfg)

    # Story 8.9: dataset comes from cfg.dataset (was hardcoded SyntheticLoader).
    # The chained loader is consumed lazily by train_jepa, which pulls
    # exactly cfg.jepa.n_steps samples.
    try:
        train_loader = build_loader(cfg.dataset.loaders, seed=cfg.jepa.seed)
    except DatasetIntegrityError:
        # Surface as-is — the CLI prints the message and exits non-zero.
        raise
    training_cfg = JEPATrainingConfig(
        image_size=cfg.jepa.img_size,
        latent_dim=cfg.jepa.latent_dim,
        masking_ratio=cfg.jepa.masking_ratio,
        n_steps=cfg.jepa.n_steps,
        batch_size=cfg.jepa.batch_size,
        learning_rate=cfg.jepa.learning_rate,
        warmup_steps=cfg.jepa.warmup_steps,
        standardize_target=cfg.jepa.standardize_target,
        ema_decay=cfg.jepa.ema_decay,
        warm_start=cfg.jepa.warm_start,
        variance_reg_weight=cfg.jepa.variance_reg_weight,
        covariance_reg_weight=cfg.jepa.covariance_reg_weight,
        warm_start_loss_scale=cfg.jepa.warm_start_loss_scale,
        warm_start_after_step=cfg.jepa.warm_start_after_step,
    )
    log_path = project_root() / "results" / run_id / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    clips_dir = log_path.parent / "clips"
    train_jepa(
        training_cfg,
        train_loader,
        state,
        graph_prior_target_edges=torch.randn(4),
        log_path=log_path,
        clips_dir=clips_dir,
    )

    # Story 8.12a: persist the trained state so the gate-evaluation orchestrator
    # (Story 8.12c) can reload encoder + predictor + warm-start heads. Stored
    # under the results-contract-allowed `checkpoints/` subdir.
    from wormjepa.training.checkpointing import save_checkpoint

    checkpoint_path = log_path.parent / "checkpoints" / "checkpoint.pt"
    save_checkpoint(state, checkpoint_path)

    # Phase 0 v0: report final losses as a single MetricEntry with sub-rows.
    sub_entries = []
    for name, value in sorted(state.last_losses.items()):
        # CI degenerate (point=lower=upper) since we have one final-loss scalar.
        sub_entries.append(
            (
                "loss_" + name,
                BootstrapCI(
                    point=value, lower=value, upper=value, n_samples=1, method="percentile"
                ),
            )
        )
    from wormjepa.eval.metrics_schema import SubEntry

    nan_ci = BootstrapCI(
        point=float("nan"),
        lower=float("nan"),
        upper=float("nan"),
        n_samples=1,
        method="percentile",
    )
    entry = MetricEntry(
        name="jepa_training_loss",
        producer="jepa",
        ci=nan_ci,
        sub_entries=[SubEntry(key=k, ci=v) for k, v in sub_entries],
        notes=f"Final losses after {state.step} steps.",
    )
    return MetricsOutput(run_id=run_id, entries=[entry]), state
