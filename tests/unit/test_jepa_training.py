"""Unit tests for the JEPA training loop, seeds, determinism, checkpoints (Stories 5.8-5.11)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import optim

from wormjepa.data.loaders.synthetic import SyntheticLoader
from wormjepa.models.ema import EMATarget
from wormjepa.models.encoder import WormJEPAEncoder
from wormjepa.models.predictor import JEPAPredictor
from wormjepa.models.warm_start import (
    BehavioralHead,
    EigenwormHead,
    GraphPriorHead,
    NeuralAuxiliaryHead,
)
from wormjepa.training.checkpointing import load_checkpoint, save_checkpoint
from wormjepa.training.determinism import enable_determinism
from wormjepa.training.loop import (
    JEPATrainingConfig,
    JEPATrainingState,
    WarmStartFlags,
    _compute_latent_stats,
    train_jepa,
)
from wormjepa.training.seeds import set_seeds


def _tiny_loader() -> SyntheticLoader:
    return SyntheticLoader(
        n_worms=2,
        clips_per_worm=1,
        clip_frames=2,
        image_size=(64, 64),
        n_keypoints=4,
        n_neurons=8,
        seed=0,
    )


def _build_state(latent_dim: int = 16) -> JEPATrainingState:
    online = WormJEPAEncoder(model_name="vit_tiny_patch16_224", latent_dim=latent_dim, img_size=64)
    ema = EMATarget(online, decay=0.99)
    predictor = JEPAPredictor(latent_dim=latent_dim, n_layers=1, n_heads=2)
    heads = {
        "eigenworm": EigenwormHead(latent_dim=latent_dim, n_eigen=4),
        "graph_prior": GraphPriorHead(latent_dim=latent_dim, n_neural_aligned_dims=8, n_edges=4),
        "neural": NeuralAuxiliaryHead(latent_dim=latent_dim, n_neurons=8),
        "behavioral": BehavioralHead(latent_dim=latent_dim, n_states=5),
    }
    # Fit the eigenworm basis on synthetic data so the head doesn't refuse.
    flat = []
    for sample in _tiny_loader():
        if sample.pose is None:
            continue
        t, k, d = sample.pose.shape
        flat.append(sample.pose.reshape(t, k * d))
    eigenworm_head = heads["eigenworm"]
    assert isinstance(eigenworm_head, EigenwormHead)
    eigenworm_head.fit_basis(torch.cat(flat, dim=0))

    params = list(online.parameters()) + list(predictor.parameters())
    for head in heads.values():
        params += [p for p in head.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1.0e-4)
    return JEPATrainingState(
        online_encoder=online,
        ema_target=ema,
        predictor=predictor,
        optimizer=optimizer,
        warm_start_heads=heads,
    )


def test_set_seeds_returns_record() -> None:
    rec = set_seeds(42)
    assert rec.seed == 42
    assert rec.python_random == 42
    text = rec.to_text()
    assert "seed=42" in text


def test_enable_determinism_does_not_crash() -> None:
    enable_determinism(warn_only=True)


def test_train_jepa_runs_smoke_steps() -> None:
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=2,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
    )
    graph_edges = torch.randn(4)
    train_jepa(
        cfg,
        _tiny_loader(),
        state,
        graph_prior_target_edges=graph_edges,
    )
    assert state.step == 2
    assert "jepa" in state.last_losses


def test_compute_latent_stats_shape_and_keys() -> None:
    """`_compute_latent_stats` returns the documented Phase 2 schema."""
    torch.manual_seed(0)
    latent = torch.randn(2, 4, 8)  # (B, T, D)
    stats = _compute_latent_stats(latent)
    assert set(stats.keys()) == {
        "std_per_dim",
        "std_min",
        "std_mean",
        "cov_offdiag_frobenius",
    }
    std_per_dim = stats["std_per_dim"]
    assert isinstance(std_per_dim, list)
    assert len(std_per_dim) == 8
    assert all(isinstance(v, float) for v in std_per_dim)
    assert isinstance(stats["std_min"], float)
    assert isinstance(stats["std_mean"], float)
    assert isinstance(stats["cov_offdiag_frobenius"], float)
    # std_min ≤ std_mean ≤ max(std_per_dim) — sanity bound.
    assert stats["std_min"] <= stats["std_mean"] + 1e-6
    assert stats["std_mean"] <= max(std_per_dim) + 1e-6


def test_train_jepa_logs_latent_stats(tmp_path: Path) -> None:
    """The Phase 2 training loop writes `extra.latent.*` per step."""
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=2,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
    )
    log_path = tmp_path / "log.jsonl"
    graph_edges = torch.randn(4)
    train_jepa(
        cfg,
        _tiny_loader(),
        state,
        graph_prior_target_edges=graph_edges,
        log_path=log_path,
    )
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[0])
    latent = record["extra"]["latent"]
    assert isinstance(latent["std_per_dim"], list)
    assert len(latent["std_per_dim"]) == cfg.latent_dim
    assert isinstance(latent["std_min"], float)
    assert isinstance(latent["std_mean"], float)
    assert isinstance(latent["cov_offdiag_frobenius"], float)


def test_warm_start_flags_disable_heads() -> None:
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(
            eigenworm=False,
            graph_prior=False,
            neural=False,
            behavioral=False,
        ),
    )
    train_jepa(cfg, _tiny_loader(), state)
    # Only the JEPA loss runs; warm-start heads are disabled.
    assert list(state.last_losses) == ["jepa"]


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
    )
    graph_edges = torch.randn(4)
    train_jepa(cfg, _tiny_loader(), state, graph_prior_target_edges=graph_edges)
    step_before = state.step
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(state, ckpt)

    # Build a fresh state and load.
    fresh = _build_state(latent_dim=16)
    fresh.warm_start_heads["graph_prior"].set_target_edges(graph_edges)
    load_checkpoint(fresh, ckpt)
    assert fresh.step == step_before


def _wide_loader() -> SyntheticLoader:
    """8-sample loader (4 worms x 2 clips) for batched-loop tests."""
    return SyntheticLoader(
        n_worms=4,
        clips_per_worm=2,
        clip_frames=2,
        image_size=(64, 64),
        n_keypoints=4,
        n_neurons=8,
        seed=0,
    )


def test_train_jepa_batched_step_count() -> None:
    """batch_size>1 stacks N clips per gradient step; n_steps is honored."""
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=3,
        batch_size=2,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
    )
    train_jepa(cfg, _wide_loader(), state, graph_prior_target_edges=torch.randn(4))
    # 3 gradient steps of 2 clips each = 6 samples drawn from an 8-sample
    # corpus — no epoching needed, exactly 3 steps.
    assert state.step == 3
    assert "jepa" in state.last_losses


def test_train_jepa_batched_logs_batch_size(tmp_path: Path) -> None:
    """Each log record carries extra.batch_size and cross-sample latent std."""
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=2,
        batch_size=4,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        variance_reg_weight=10.0,
        warm_start=WarmStartFlags(),
    )
    log_path = tmp_path / "log.jsonl"
    train_jepa(
        cfg,
        _wide_loader(),
        state,
        graph_prior_target_edges=torch.randn(4),
        log_path=log_path,
    )
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["extra"]["batch_size"] == 4
    # variance_reg is active and the latent std is now a real cross-sample
    # statistic (B*T = 4*2 = 8 rows), not 2 frames of one clip.
    assert "variance_reg" in record["extra"]["losses"]
    assert len(record["extra"]["latent"]["std_per_dim"]) == cfg.latent_dim


def _video_only_loader(n: int = 12) -> list[object]:
    """n video-only samples (pose=None, neural=None) — mimics WormBehaviorDB /
    OpenWormMovementDB, which carry behavioral video but no pose/neural."""
    from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID

    out: list[object] = []
    for i in range(n):
        out.append(
            DatasetSample(
                video_clip=torch.rand(2, 3, 64, 64),
                pose=None,
                neural=None,
                worm_id=WormID(f"wbdb_w{i:03d}"),
                session_id=SessionID(f"wbdb_w{i:03d}_s00"),
                source_dataset=SourceDataset("wormbehavior_db"),
            )
        )
    return out


def test_train_jepa_trains_on_video_only_samples() -> None:
    """Gate loosening (2026-05-20): JEPA loss runs on pose/neural-free
    samples; warm-start heads skip cleanly when no target is present."""
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=3,
        batch_size=4,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        variance_reg_weight=10.0,
    )
    train_jepa(cfg, _video_only_loader(12), state, graph_prior_target_edges=torch.randn(4))
    # 3 gradient steps ran on video-only data — pre-2026-05-20 this loop
    # skipped every such sample and completed 0 steps.
    assert state.step == 3
    assert "jepa" in state.last_losses
    # variance_reg still runs (needs only video); pose/neural heads do not.
    assert "variance_reg" in state.last_losses
    assert "eigenworm" not in state.last_losses
    assert "neural" not in state.last_losses


def test_train_jepa_epochs_dataset_to_reach_n_steps() -> None:
    """When n_steps exceeds corpus size the loop re-iterates the dataset."""
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=20,  # > 8-sample corpus at batch_size=1
        batch_size=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
    )
    train_jepa(cfg, _wide_loader(), state, graph_prior_target_edges=torch.randn(4))
    # Epoching: the 8-sample loader is re-iterated until 20 steps complete.
    assert state.step == 20


def _build_state_two_groups(
    latent_dim: int = 16,
    *,
    learning_rate: float = 1.0e-4,
    head_learning_rate: float = 1.0e-3,
) -> JEPATrainingState:
    """Build a state matching ``runner._build_state``'s two-param-group optimizer."""
    online = WormJEPAEncoder(model_name="vit_tiny_patch16_224", latent_dim=latent_dim, img_size=64)
    ema = EMATarget(online, decay=0.99)
    predictor = JEPAPredictor(latent_dim=latent_dim, n_layers=1, n_heads=2)
    heads: dict[str, torch.nn.Module] = {
        "eigenworm": EigenwormHead(latent_dim=latent_dim, n_eigen=4),
        "graph_prior": GraphPriorHead(latent_dim=latent_dim, n_neural_aligned_dims=8, n_edges=4),
        "neural": NeuralAuxiliaryHead(latent_dim=latent_dim, n_neurons=8),
        "behavioral": BehavioralHead(latent_dim=latent_dim, n_states=5),
    }
    flat = []
    for sample in _tiny_loader():
        if sample.pose is None:
            continue
        t, k, d = sample.pose.shape
        flat.append(sample.pose.reshape(t, k * d))
    eigenworm_head = heads["eigenworm"]
    assert isinstance(eigenworm_head, EigenwormHead)
    eigenworm_head.fit_basis(torch.cat(flat, dim=0))

    encoder_params = list(online.parameters()) + list(predictor.parameters())
    head_params: list[torch.Tensor] = []
    for head in heads.values():
        head_params += [p for p in head.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        [
            {"params": encoder_params, "lr": learning_rate, "base_lr": learning_rate},
            {"params": head_params, "lr": head_learning_rate, "base_lr": head_learning_rate},
        ]
    )
    return JEPATrainingState(
        online_encoder=online,
        ema_target=ema,
        predictor=predictor,
        optimizer=optimizer,
        warm_start_heads=heads,
    )


def test_warm_start_after_step_gates_heads_until_boundary(tmp_path: Path) -> None:
    """Curriculum boundary: warm-start heads are inactive until step >=
    warm_start_after_step. Pre-boundary log records have only jepa-side
    losses; post-boundary records carry the head keys."""
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=6,
        batch_size=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start_after_step=4,
    )
    log_path = tmp_path / "log.jsonl"
    train_jepa(
        cfg,
        _wide_loader(),
        state,
        graph_prior_target_edges=torch.randn(4),
        log_path=log_path,
    )
    records = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert len(records) == 6
    for rec in records:
        losses = rec["extra"]["losses"]
        if rec["step"] <= 4:
            # Pre-boundary: heads inactive, only the JEPA family runs.
            assert "eigenworm" not in losses
            assert "neural" not in losses
            assert "graph_prior" not in losses
            assert "behavioral" not in losses
        else:
            # Post-boundary: pose/neural heads run on the frozen encoder.
            assert "eigenworm" in losses
            assert "neural" in losses
    # Final losses reflect the post-boundary state.
    assert "eigenworm" in state.last_losses


def test_warm_start_after_step_freezes_online_at_boundary() -> None:
    """At the curriculum boundary the online encoder's params are frozen
    (requires_grad=False) — heads warm-start on a stable representation."""
    state = _build_state(latent_dim=16)
    # Sanity: params start trainable.
    assert all(p.requires_grad for p in state.online_encoder.parameters())
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=5,
        batch_size=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start_after_step=2,
    )
    train_jepa(cfg, _wide_loader(), state, graph_prior_target_edges=torch.randn(4))
    # After training crosses the boundary, every online-encoder param is frozen.
    assert all(not p.requires_grad for p in state.online_encoder.parameters())


def test_head_learning_rate_creates_two_param_groups() -> None:
    """The runner builds Adam with two groups: encoder/predictor at
    learning_rate, heads at head_learning_rate. Each group records base_lr
    so the loop's warmup scales each group's own base."""
    state = _build_state_two_groups(
        latent_dim=16,
        learning_rate=1.0e-4,
        head_learning_rate=5.0e-4,
    )
    groups = state.optimizer.param_groups
    assert len(groups) == 2
    encoder_group, head_group = groups
    assert encoder_group["lr"] == 1.0e-4
    assert encoder_group["base_lr"] == 1.0e-4
    assert head_group["lr"] == 5.0e-4
    assert head_group["base_lr"] == 5.0e-4


def test_warmup_scales_each_group_base_lr() -> None:
    """Linear warmup multiplies each group's current lr by step/warmup_steps,
    each group scaling its own base_lr."""
    state = _build_state_two_groups(
        latent_dim=16,
        learning_rate=1.0e-4,
        head_learning_rate=5.0e-4,
    )
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=1,
        batch_size=1,
        learning_rate=1.0e-4,
        warmup_steps=4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
    )
    train_jepa(cfg, _wide_loader(), state, graph_prior_target_edges=torch.randn(4))
    # After step 1 of warmup=4: factor = 1/4.
    encoder_group, head_group = state.optimizer.param_groups
    assert encoder_group["lr"] == 1.0e-4 * 0.25
    assert head_group["lr"] == 5.0e-4 * 0.25

    # Run to a total of 4 warmup steps (3 more, since `state` is at step 1):
    # `n_steps` is the TOTAL step target, not an increment — `train_jepa` runs
    # `while state.step < n_steps`, so a reused state continues toward the target
    # rather than running n_steps fresh. Both groups reach base_lr at step 4.
    cfg_finish = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=4,
        batch_size=1,
        learning_rate=1.0e-4,
        warmup_steps=4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
    )
    train_jepa(cfg_finish, _wide_loader(), state, graph_prior_target_edges=torch.randn(4))
    encoder_group, head_group = state.optimizer.param_groups
    assert encoder_group["lr"] == encoder_group["base_lr"]
    assert head_group["lr"] == head_group["base_lr"]


def test_resume_continues_to_total_n_steps(tmp_path: Path) -> None:
    """`train_jepa` runs `while step < n_steps`, so resuming from a checkpoint
    continues toward the TOTAL step target instead of running n_steps again —
    the cross-pod resume invariant. A fresh state that loads a step-2 checkpoint
    and runs with n_steps=5 must end at step 5 (3 more), not 7."""
    set_seeds(0)
    state = _build_state(latent_dim=16)
    cfg2 = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=2,
        batch_size=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
    )
    train_jepa(cfg2, _wide_loader(), state, graph_prior_target_edges=torch.randn(4))
    assert state.step == 2
    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(state, ckpt)

    # A freshly-built state stands in for a new pod; load the checkpoint, resume.
    state_resumed = _build_state(latent_dim=16)
    load_checkpoint(state_resumed, ckpt)
    assert state_resumed.step == 2
    cfg5 = cfg2.model_copy(update={"n_steps": 5})
    train_jepa(cfg5, _wide_loader(), state_resumed, graph_prior_target_edges=torch.randn(4))
    assert state_resumed.step == 5


def test_resume_reapplies_curriculum_freeze(tmp_path: Path) -> None:
    """Resuming from a checkpoint at/after `warm_start_after_step` must re-freeze
    the online encoder before any further step: the one-shot `==` freeze event
    already passed on the crashed pod and never re-fires, and load_checkpoint
    restores weights into a freshly-built (trainable) encoder. An unfrozen
    phase-2 encoder collapses the latent (the E5/E6 finding)."""
    set_seeds(0)
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=2,
        batch_size=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
        warm_start_after_step=1,
    )
    train_jepa(cfg, _wide_loader(), state, graph_prior_target_edges=torch.randn(4))
    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(state, ckpt)

    # A fresh encoder is trainable; loading weights does not by itself re-freeze.
    state_resumed = _build_state(latent_dim=16)
    assert all(p.requires_grad for p in state_resumed.online_encoder.parameters())
    load_checkpoint(state_resumed, ckpt)
    cfg_more = cfg.model_copy(update={"n_steps": 4})
    train_jepa(cfg_more, _wide_loader(), state_resumed, graph_prior_target_edges=torch.randn(4))
    assert not any(p.requires_grad for p in state_resumed.online_encoder.parameters())


def test_checkpoint_every_writes_periodically(tmp_path: Path) -> None:
    """`checkpoint_every>0` with a `checkpoint_path` atomic-saves every N steps;
    the file exists, leaves no `.tmp` sibling, and reloads to the latest step."""
    set_seeds(0)
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=4,
        batch_size=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start=WarmStartFlags(),
    )
    ckpt = tmp_path / "checkpoint.pt"
    train_jepa(
        cfg,
        _wide_loader(),
        state,
        graph_prior_target_edges=torch.randn(4),
        checkpoint_path=ckpt,
        checkpoint_every=2,
    )
    assert ckpt.is_file()
    assert not ckpt.with_name(ckpt.name + ".tmp").exists()  # atomic: no leftover tmp
    reloaded = _build_state(latent_dim=16)
    load_checkpoint(reloaded, ckpt)
    assert reloaded.step == 4  # last periodic save lands at step 4 (4 % 2 == 0)


def test_warm_start_loss_scale_only_scales_heads(tmp_path: Path) -> None:
    """``warm_start_loss_scale`` multiplies head-loss weights but not
    jepa / variance_reg / covariance_reg. The heads still appear in
    ``last_losses`` regardless of the scale (they are recorded for
    diagnostics) — only their weighted contribution to ``total`` is
    affected. The step-1 jepa loss is identical between scale=0 and
    scale=1 because at step 1 the encoder has not yet been updated, so
    the forward pass is the same regardless of the head weight."""
    set_seeds(0)
    state_zero = _build_state(latent_dim=16)
    cfg_zero = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=1,
        batch_size=1,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        warm_start_loss_scale=0.0,
    )
    log_zero = tmp_path / "log_zero.jsonl"
    train_jepa(
        cfg_zero,
        _wide_loader(),
        state_zero,
        graph_prior_target_edges=torch.randn(4),
        log_path=log_zero,
    )
    # Heads still recorded (they fire — only the weight is zero).
    assert "eigenworm" in state_zero.last_losses
    assert "jepa" in state_zero.last_losses

    set_seeds(0)
    state_one = _build_state(latent_dim=16)
    cfg_one = cfg_zero.model_copy(update={"warm_start_loss_scale": 1.0})
    log_one = tmp_path / "log_one.jsonl"
    train_jepa(
        cfg_one,
        _wide_loader(),
        state_one,
        graph_prior_target_edges=torch.randn(4),
        log_path=log_one,
    )
    # Step-1 jepa is a pure forward pass on the freshly-seeded encoder:
    # the head weight cannot affect it yet (no backward has run). Identical.
    rec_zero = json.loads(log_zero.read_text().strip().splitlines()[0])
    rec_one = json.loads(log_one.read_text().strip().splitlines()[0])
    assert rec_zero["extra"]["losses"]["jepa"] == rec_one["extra"]["losses"]["jepa"]


def test_covariance_reg_weight_active_in_loss() -> None:
    """``covariance_reg_weight > 0`` adds a ``covariance_reg`` term to the
    per-step loss dict. Inspect ``last_losses`` after the run."""
    state = _build_state(latent_dim=16)
    cfg = JEPATrainingConfig(
        image_size=64,
        latent_dim=16,
        masking_ratio=0.5,
        n_steps=2,
        batch_size=2,
        learning_rate=1.0e-4,
        ema_decay=0.99,
        covariance_reg_weight=1.0,
        warm_start=WarmStartFlags(),
    )
    train_jepa(cfg, _wide_loader(), state, graph_prior_target_edges=torch.randn(4))
    assert "covariance_reg" in state.last_losses
    assert state.last_losses["covariance_reg"] >= 0.0
