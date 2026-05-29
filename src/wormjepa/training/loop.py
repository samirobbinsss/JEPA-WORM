"""JEPA training loop orchestration (Story 5.8).

Composes the online encoder + EMA target + masked-spatiotemporal predictor
+ toggleable warm-start heads (eigenworm regularizer, connectome graph
prior, neural auxiliary head, behavioral classifier). Phase 0 v0 uses a
simple loop on synthetic data; real-data scale comes when Stories 2.4-2.8
populate the loaders.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from wormjepa.data import DatasetSample
from wormjepa.models.ema import EMATarget
from wormjepa.models.masking import random_temporal_mask
from wormjepa.models.pose_decoder import PoseDecoderHead
from wormjepa.models.predictor import JEPAPredictor
from wormjepa.training.clip_writer import RolloutRecorder, write_step_clip

logger = logging.getLogger(__name__)


class WarmStartFlags(BaseModel):
    """Which warm-start heads are enabled for this training run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    eigenworm: bool = True
    graph_prior: bool = True
    neural: bool = True
    behavioral: bool = True


class JEPATrainingConfig(BaseModel):
    """Hyperparameters for a JEPA training run (Phase 0 v0)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_size: int = 224
    latent_dim: int = 64
    masking_ratio: float = 0.5
    n_steps: int = 4
    batch_size: int = Field(default=1, ge=1)
    """Distinct clips stacked per gradient step.

    With ``batch_size=1`` the loop matches the pre-batching v0 behaviour.
    With ``batch_size>1`` the loop accumulates N clips of matching
    pose/neural shape into a ``(B, T, ...)`` batch before the forward
    pass — the VicReg variance regularizer then measures std across
    distinct samples (a real collapse guard) rather than across the 16
    frames of one clip. The dataset is re-iterated (epoched) so the full
    ``n_steps`` gradient steps run regardless of corpus size.
    """
    learning_rate: float = 1.0e-4
    warmup_steps: int = Field(default=0, ge=0)
    """Linear lr warmup over the first ``warmup_steps`` gradient steps."""
    standardize_target: bool = True
    """Standardize target latents to unit variance before the masked MSE."""
    ema_decay: float = 0.996
    warm_start: WarmStartFlags = Field(default_factory=WarmStartFlags)
    variance_reg_weight: float = 0.0
    """VicReg-style variance regularizer weight (Story F3, R3-followup).

    When > 0, adds ``v_loss = mean_d max(0, 1.0 - std(z_d))`` to the total
    JEPA loss, where ``z_d`` is the online encoder's latent dim ``d``
    flattened over batch and time. Penalises dims whose per-dim std drops
    below 1.0 — directly counters the latent-collapse attractor the
    8.11c + R3 sweeps observed. Default 0 preserves pre-R3 loss shape;
    R3-followup headline.yaml sets a positive weight.
    """
    covariance_reg_weight: float = 0.0
    """VICReg-style covariance regularizer weight. When > 0, adds the
    squared off-diagonal of the online-latent covariance (normalised by
    D) to the loss — decorrelates the latent dimensions."""
    warm_start_loss_scale: float = 1.0
    """Global multiplier on every warm-start head's loss weight. Does not
    scale jepa or the variance/covariance regularizers."""
    warm_start_after_step: int = Field(default=0, ge=0)
    """Two-phase curriculum boundary: heads inactive (encoder trains
    pure-JEPA) until this step, then the encoder freezes and heads switch
    on. Default 0 = joint training from step 1."""
    loss_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "jepa": 1.0,
            "eigenworm": 0.1,
            "graph_prior": 0.01,
            "neural": 0.1,
            "behavioral": 0.1,
        }
    )


@dataclass
class JEPATrainingState:
    """Mutable container for a training run's state.

    Used by the loop, the checkpoint module, and the results writer.

    ``online_encoder`` and ``ema_target`` are typed as ``nn.Module`` (not
    the narrower :class:`WormJEPAEncoder` / :class:`EMATarget`) because the
    Story 8.11b ``frozen_target=true`` path substitutes
    :class:`wormjepa.models.vjepa_loader.TrainableVJEPAEncoder` and
    :class:`wormjepa.models.vjepa_loader.FrozenVJEPATarget` respectively;
    both honour the same ``forward(video: (B, T, C, H, W)) -> (B, T, D)``
    contract the training loop consumes.
    """

    online_encoder: nn.Module
    ema_target: nn.Module
    predictor: JEPAPredictor
    optimizer: torch.optim.Optimizer
    warm_start_heads: dict[str, nn.Module] = field(default_factory=dict)
    step: int = 0
    last_losses: dict[str, float] = field(default_factory=dict)


def _compute_latent_stats(online_latent: torch.Tensor) -> dict[str, float | list[float]]:
    """Compute per-step latent-collapse diagnostics from an online-latent tensor.

    Args:
        online_latent: ``(B, T, D)`` latent tensor produced by the online encoder.

    Returns:
        A JSON-serialisable dict with four entries:

        - ``std_per_dim``: list of ``D`` floats — per-dim std of latents flattened
          over batch and time (``(B*T, D) → std(dim=0)``). The minimum of this list
          is the simplest collapse indicator: if a dim's std → 0 the encoder
          stopped using it.
        - ``std_min``: ``min(std_per_dim)`` (scalar).
        - ``std_mean``: ``mean(std_per_dim)`` (scalar).
        - ``cov_offdiag_frobenius``: Frobenius norm of the off-diagonal of the
          centered covariance matrix
          ``Σ = (Z - Z.mean(0))ᵀ (Z - Z.mean(0)) / (N-1)``. High values mean
          dimensions are redundant; combined with a low ``std_min`` this is the
          VICReg-style collapse signal (Growth G1 latent-geometry signal).

    Costs no gradient memory — detaches and moves to CPU under ``no_grad``.
    """
    with torch.no_grad():
        z = online_latent.detach()
        b, t, d = z.shape
        z = z.reshape(b * t, d).to(dtype=torch.float32, device="cpu")
        n = z.shape[0]
        std_per_dim = z.std(dim=0, unbiased=True) if n > 1 else torch.zeros(d)
        std_min = float(std_per_dim.min().item()) if d > 0 else float("nan")
        std_mean = float(std_per_dim.mean().item()) if d > 0 else float("nan")
        if n > 1 and d > 0:
            centered = z - z.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / float(n - 1)
            cov_offdiag = cov - torch.diag(torch.diag(cov))
            cov_offdiag_frobenius = float(cov_offdiag.pow(2).sum().sqrt().item())
        else:
            cov_offdiag_frobenius = 0.0
    return {
        "std_per_dim": [float(x) for x in std_per_dim.tolist()],
        "std_min": std_min,
        "std_mean": std_mean,
        "cov_offdiag_frobenius": cov_offdiag_frobenius,
    }


_TARGET_STD_EPS = 1e-4
"""Per-dim std floor when standardizing frozen-target latents (matches the
1e-4 numerical floor the VicReg variance regularizer uses elsewhere)."""


def _encode_online_tokens(online: nn.Module, video: torch.Tensor) -> torch.Tensor:
    """Encode ``video`` into the online encoder's spatial-token grid.

    Returns ``(B, T, S, D)``. Every encoder the training loop accepts
    (:class:`wormjepa.models.encoder.WormJEPAEncoder`,
    :class:`wormjepa.models.vjepa_loader.TrainableVJEPAEncoder`) exposes
    ``forward_tokens``; the legacy timm path returns a singleton ``S = 1``
    grid. The pooled ``(B, T, D)`` latent the JEPA loss / predictor / non-
    pose warm-start heads consume is ``tokens.mean(dim=2)``.
    """
    forward_tokens = getattr(online, "forward_tokens", None)
    if callable(forward_tokens):
        tokens = forward_tokens(video)
        assert isinstance(tokens, torch.Tensor)
        return tokens
    # Defensive fallback for any encoder predating forward_tokens: treat the
    # pooled (B, T, D) output as a singleton spatial grid.
    return online(video).unsqueeze(2)


def _jepa_step_loss(
    online: nn.Module,
    target: nn.Module,
    predictor: JEPAPredictor,
    video: torch.Tensor,
    masking_ratio: float,
    *,
    standardize_target: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the core JEPA loss for one batch.

    When ``standardize_target`` is True the target latents are standardized
    to zero-mean / unit-variance per dim (statistics over the flattened
    ``B*T`` batch) before the masked MSE. That was added for the
    frozen-foreign-target path, where the raw V-JEPA ViT-L latents are
    unnormalised. Standard I-JEPA / V-JEPA do NOT standardize — the EMA +
    stop-grad is the anti-collapse mechanism, and standardizing a
    collapsing EMA target divides by a vanishing std. The headline-collapse
    research toggles this off via ``config.standardize_target``.

    Returns ``(loss, online_latent, online_tokens, mask)``:

    - ``online_latent`` ``(B, T, D)`` — the spatially-mean-pooled latent the
      JEPA loss, predictor, and non-pose warm-start heads consume.
    - ``online_tokens`` ``(B, T, S, D)`` — the un-pooled spatial-token grid
      the pose-decoder head cross-attends over. ``online_latent`` is exactly
      ``online_tokens.mean(dim=2)``; the online encoder is run once.

    The latent / tokens / mask are returned so warm-start heads can be
    applied to the same forward pass.
    """
    b, t, *_ = video.shape
    online_tokens = _encode_online_tokens(online, video)  # (B, T, S, D)
    online_latent = online_tokens.mean(dim=2)  # (B, T, D)
    with torch.no_grad():
        target_latent = target(video)  # (B, T, D), no grad
        if standardize_target:
            d = target_latent.shape[-1]
            flat_target = target_latent.reshape(b * t, d)
            t_mean = flat_target.mean(dim=0, keepdim=True)
            t_std = flat_target.std(dim=0, unbiased=False, keepdim=True)
            target_used = ((flat_target - t_mean) / (t_std + _TARGET_STD_EPS)).reshape(b, t, d)
        else:
            target_used = target_latent
    mask = random_temporal_mask(n_frames=t, n_batches=b, masking_ratio=masking_ratio).to(
        video.device
    )
    predicted = predictor(online_latent, mask)
    # Loss only at masked positions (predict the target there).
    mask_f = mask.unsqueeze(-1).to(predicted.dtype)
    masked_loss = ((predicted - target_used) ** 2 * mask_f).sum() / mask_f.sum().clamp(min=1.0)
    return masked_loss, online_latent, online_tokens, mask


def train_jepa(
    config: JEPATrainingConfig,
    dataset: Iterable[DatasetSample],
    state: JEPATrainingState,
    *,
    behavioral_labels_per_sample: dict[str, torch.Tensor] | None = None,
    graph_prior_target_edges: torch.Tensor | None = None,
    log_path: Path | None = None,
    clips_dir: Path | None = None,
) -> None:
    """Run the training loop in-place on ``state``.

    Args:
        config: Hyperparameters.
        dataset: Iterable of :class:`DatasetSample`. The loop consumes up to
            ``config.n_steps`` samples (one per gradient step in v0).
        state: Mutable training state.
        behavioral_labels_per_sample: Optional ``{session_id: labels(T,)}`` map
            for the behavioral classifier (only used if the head is enabled).
        graph_prior_target_edges: ``(n_edges,)`` target tensor for the
            connectome head (only used if the head is enabled and the target
            has not already been set).
        log_path: Optional path to a ``log.jsonl`` file. When supplied, one
            JSON-Lines record is appended per gradient step with the schema
            ``{"ts": <UTC ISO 8601>, "step": <int>, "extra": {"loss": <float>,
            "losses": {<name>: <float>, ...}, "latent": {"std_per_dim":
            [<float>, ...], "std_min": <float>, "std_mean": <float>,
            "cov_offdiag_frobenius": <float>}}}``. The ``latent`` block carries
            the per-step collapse diagnostic the GUI's `LatentGeometryPanel`
            (UX spec §"C4 LatentGeometryPanel") consumes; see
            :func:`_compute_latent_stats`. The ``results/<run-id>/`` contract
            names this file (Architecture §"results contract"); this is where
            it lands at training time. Caller is responsible for ensuring the
            parent directory exists.
        clips_dir: Optional path to a ``clips/`` directory. When supplied, one
            ``<step>.png`` (frame strip + pose dots) and one ``<step>.mask.png``
            (magenta-alpha mask overlay) are written per gradient step so the
            GUI's ``ClipViewer`` (UX spec §"C3 ClipViewer") can render against
            synthetic-data runs. Created if missing.
    """
    online = state.online_encoder
    target = state.ema_target
    predictor = state.predictor
    opt = state.optimizer

    # If a graph-prior head needs targets and they were supplied, plug them in.
    if "graph_prior" in state.warm_start_heads and graph_prior_target_edges is not None:
        gp_init = state.warm_start_heads["graph_prior"]
        set_targets = getattr(gp_init, "set_target_edges", None)
        if callable(set_targets):
            set_targets(graph_prior_target_edges)

    online.train()
    predictor.train()
    for head in state.warm_start_heads.values():
        head.train()

    # Models may live on CUDA / MPS / CPU; loader-returned tensors are CPU.
    # Snapshot the model device once and route each sample through it so the
    # forward pass sees co-located tensors.
    model_device = next(online.parameters()).device

    rollout: RolloutRecorder | None = None

    def _all_samples() -> Iterator[DatasetSample]:
        """Yield every sample, re-iterating (epoching) the dataset so the
        loop reaches the full ``config.n_steps`` regardless of corpus size.
        Stops when a fresh epoch yields zero samples (an empty corpus, or a
        single-pass generator already consumed).

        The JEPA loss + variance regularizer need only ``video_clip`` (always
        present), so — unlike the pre-2026-05-20 loop — samples without pose
        or neural are NOT skipped: real video-only loaders (WormBehaviorDB,
        OpenWormMovementDB) now train the encoder. Warm-start heads apply
        per-sub-batch wherever their target tensor exists.
        """
        while True:
            yielded = 0
            for sample in iter(dataset):
                yielded += 1
                yield sample
            if yielded == 0:
                return

    def _video_key(s: DatasetSample) -> tuple[int, ...]:
        """Video-clip shape — samples must match this to stack into a batch."""
        return tuple(s.video_clip.shape)

    def _gather(batch: list[DatasetSample], attr: str) -> tuple[list[int], torch.Tensor | None]:
        """Collect ``(indices, stacked tensor)`` for batch members whose
        ``attr`` is non-None and shares the first such member's shape. A
        warm-start head consuming ``attr`` is applied only to this
        sub-batch; ``([], None)`` means no member carries it."""
        idxs: list[int] = []
        rows: list[torch.Tensor] = []
        ref_shape: torch.Size | None = None
        for i, s in enumerate(batch):
            t = getattr(s, attr)
            if t is None:
                continue
            if ref_shape is None:
                ref_shape = t.shape
            if t.shape != ref_shape:
                continue
            idxs.append(i)
            rows.append(t)
        if not idxs:
            return [], None
        return idxs, torch.stack(rows)

    sample_stream = _all_samples()
    deferred: DatasetSample | None = None  # one-sample lookahead on shape break

    def _next_batch() -> list[DatasetSample]:
        """Collect up to ``config.batch_size`` samples of matching video
        shape. A sample whose video shape differs from the batch head is
        held in ``deferred`` and seeds the next batch — no sample dropped;
        mixed-loader corpora just produce some short batches.
        """
        nonlocal deferred
        batch: list[DatasetSample] = []
        if deferred is not None:
            batch.append(deferred)
            deferred = None
        while len(batch) < config.batch_size:
            try:
                s = next(sample_stream)
            except StopIteration:
                break
            if batch and _video_key(s) != _video_key(batch[0]):
                deferred = s
                break
            batch.append(s)
        return batch

    # variance_reg is weighted by config.variance_reg_weight (its own knob),
    # not by config.loss_weights — the latter is reserved for the per-head
    # composition. Other entries default to loss_weights.get(name, 0.0).
    def _weight(name: str) -> float:
        if name == "variance_reg":
            return config.variance_reg_weight
        if name == "covariance_reg":
            return config.covariance_reg_weight
        if name == "jepa":
            return config.loss_weights.get("jepa", 1.0)
        # Warm-start heads — scaled by warm_start_loss_scale so they
        # nudge rather than drag the encoder (Experiment 5/6).
        return config.loss_weights.get(name, 0.0) * config.warm_start_loss_scale

    for _ in range(config.n_steps):
        batch = _next_batch()
        if not batch:
            break  # dataset fully exhausted (single-pass generator, empty)

        # First pose-bearing sample of the run seeds the rollout reference.
        if clips_dir is not None and rollout is None:
            head = next((s for s in batch if s.pose is not None), None)
            if head is not None and head.pose is not None:
                rollout = RolloutRecorder(
                    video=head.video_clip,
                    pose=head.pose,
                    max_steps=config.n_steps,
                )

        video = torch.stack([s.video_clip for s in batch]).to(model_device)  # (B,T,C,H,W)

        # Linear lr warmup over the first `warmup_steps` gradient steps.
        # Each param-group scales its own base_lr (encoder + heads may
        # differ); fall back to config.learning_rate if base_lr is unset.
        if config.warmup_steps > 0:
            warmup_factor = min(1.0, (state.step + 1) / config.warmup_steps)
            for group in opt.param_groups:
                base = group.get("base_lr", config.learning_rate)
                group["lr"] = base * warmup_factor

        # Two-phase curriculum: at the boundary step freeze the online
        # encoder so the warm-start heads (switched on below) read a stable
        # representation instead of reshaping it toward collapse.
        if config.warm_start_after_step > 0 and state.step == config.warm_start_after_step:
            for p in online.parameters():
                p.requires_grad_(False)
            logger.info("curriculum: froze online encoder at step %d", state.step)
        heads_active = state.step >= config.warm_start_after_step

        opt.zero_grad()
        jepa_loss, online_latent, online_tokens, mask = _jepa_step_loss(
            online,
            target,
            predictor,
            video,
            config.masking_ratio,
            standardize_target=config.standardize_target,
        )

        losses: dict[str, torch.Tensor] = {"jepa": jepa_loss}

        # F3 (R3-followup): VicReg-style variance regularizer. Runs on the
        # full batch — every sample has video. With batch_size>1 the per-dim
        # std is measured across distinct clips (a real collapse guard).
        if config.variance_reg_weight > 0.0:
            bz, tz, dz = online_latent.shape
            flat = online_latent.reshape(bz * tz, dz)
            # Numerical stability: 1e-4 floor under the sqrt matches VicReg.
            std_per_dim = (flat.var(dim=0, unbiased=False) + 1e-4).sqrt()
            v_loss = torch.relu(1.0 - std_per_dim).mean()
            losses["variance_reg"] = v_loss

        # VICReg-style covariance regularizer: squared off-diagonal of the
        # latent covariance, normalised by D. Decorrelates the latent dims.
        # Experiment 2 showed variance_reg alone lets the dims collapse onto
        # a correlated subspace (cov_offdiag 12 -> 1395); this is the fix.
        if config.covariance_reg_weight > 0.0:
            bz, tz, dz = online_latent.shape
            flat = online_latent.reshape(bz * tz, dz)
            n = flat.shape[0]
            centered = flat - flat.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / max(n - 1, 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            losses["covariance_reg"] = off_diag.pow(2).sum() / dz

        # Warm-start heads apply only to the sub-batch carrying their target
        # tensor — video-only samples (WormBehaviorDB, OpenWormMovementDB)
        # still drive the JEPA loss + variance_reg above, just not these
        # heads. This is the 2026-05-20 gate loosening: the encoder now
        # trains on real worm video, not only the baaiworm synthetic clips.
        pose_idxs, pose_t = _gather(batch, "pose")
        neural_idxs, neural_t = _gather(batch, "neural")
        if pose_t is not None:
            pose_t = pose_t.to(model_device)
        if neural_t is not None:
            neural_t = neural_t.to(model_device)

        if (
            heads_active
            and config.warm_start.eigenworm
            and "eigenworm" in state.warm_start_heads
            and pose_t is not None
        ):
            eig = state.warm_start_heads["eigenworm"]
            if hasattr(eig, "fitted") and eig.fitted:
                losses["eigenworm"] = eig(online_latent[pose_idxs], pose_t)

        if (
            heads_active
            and config.warm_start.graph_prior
            and "graph_prior" in state.warm_start_heads
        ):
            gp = state.warm_start_heads["graph_prior"]
            if hasattr(gp, "_target_set") and gp._target_set:
                losses["graph_prior"] = gp.forward(online_latent)

        if (
            heads_active
            and config.warm_start.neural
            and "neural" in state.warm_start_heads
            and neural_t is not None
        ):
            nh = state.warm_start_heads["neural"]
            losses["neural"] = nh(online_latent[neural_idxs], neural_t)

        # Pose-decoder head is dev-loop-only (not pre-registered). Train it
        # jointly when present so the GUI's ClipViewer can show red predicted
        # dots converging onto the green ground-truth dots. It cross-attends
        # over the un-pooled spatial-token grid (online_tokens), not the
        # spatially-mean-pooled latent — keypoint xy-coords are inherently
        # spatial and a pooled vector destroys where things are.
        pose_decoder = state.warm_start_heads.get("pose_decoder")
        if heads_active and isinstance(pose_decoder, PoseDecoderHead) and pose_t is not None:
            losses["pose_decoder"] = pose_decoder(online_tokens[pose_idxs], pose_t)

        # Behavioral head: applied to the sub-batch whose samples carry a
        # label for their session_id. Unlabelled samples are skipped rather
        # than dropping the whole batch.
        if (
            heads_active
            and config.warm_start.behavioral
            and "behavioral" in state.warm_start_heads
            and behavioral_labels_per_sample is not None
        ):
            label_idxs: list[int] = []
            label_rows: list[torch.Tensor] = []
            for i, s in enumerate(batch):
                lbl = behavioral_labels_per_sample.get(str(s.session_id))
                if lbl is not None:
                    label_idxs.append(i)
                    label_rows.append(lbl)
            if label_idxs:
                bh = state.warm_start_heads["behavioral"]
                labels = torch.stack(label_rows).to(model_device)  # (n, T)
                losses["behavioral"] = bh(online_latent[label_idxs], labels)

        total = sum(
            (_weight(name) * loss for name, loss in losses.items()),
            torch.zeros((), device=video.device, dtype=jepa_loss.dtype),
        )
        total.backward()
        opt.step()
        # EMA target tracking is only valid for the EMATarget variant. The
        # Story 8.11a frozen V-JEPA 2.1 target encoder (FrozenVJEPATarget)
        # must not be updated — its weights are the pre-registered transfer
        # source and freezing is the load-bearing pre-reg commitment.
        if isinstance(target, EMATarget):
            target.update(online)  # type: ignore[arg-type]
        state.step += 1
        state.last_losses = {name: float(loss.detach()) for name, loss in losses.items()}

        if log_path is not None:
            latent_stats = _compute_latent_stats(online_latent)
            payload = {
                "ts": datetime.now(tz=UTC).isoformat(),
                "step": state.step,
                "extra": {
                    "loss": state.last_losses.get("jepa", float("nan")),
                    "losses": state.last_losses,
                    "latent": latent_stats,
                    "batch_size": len(batch),
                },
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")

        # Clip writer renders one pose-bearing example (dev-loop visual only).
        if clips_dir is not None and pose_idxs and pose_t is not None:
            j = pose_idxs[0]
            predicted_pose: torch.Tensor | None = None
            if isinstance(pose_decoder, PoseDecoderHead):
                with torch.no_grad():
                    predicted_pose = pose_decoder.predict(online_tokens[j : j + 1].detach())
            write_step_clip(
                clips_dir=clips_dir,
                step=state.step,
                video=video[j : j + 1],
                mask=mask[j : j + 1],
                pose=pose_t[0:1],
                predicted_pose=predicted_pose,
            )
            # Capture the rollout frame on the fixed reference clip.
            if rollout is not None and isinstance(pose_decoder, PoseDecoderHead):
                with torch.no_grad():
                    ref_video_t = rollout.reference_video.unsqueeze(0).to(model_device)
                    ref_tokens = _encode_online_tokens(online, ref_video_t)
                    ref_pred = pose_decoder.predict(ref_tokens)
                    # Recorder serialises to MP4 on CPU; move back before record.
                    rollout.record(state.step, ref_pred.squeeze(0).cpu())

    if state.step == 0:
        logger.warning(
            "train_jepa: completed 0 gradient steps — the dataset yielded "
            "no samples. Check `dataset.loaders` in the run config.",
        )

    # End-of-training: emit the rollup video so it lands as the final
    # write inside `clips/`. Failure here is non-fatal.
    if clips_dir is not None and rollout is not None:
        import contextlib

        with contextlib.suppress(OSError, RuntimeError, ValueError):
            rollout.save(clips_dir / "training_evolution.mp4")
