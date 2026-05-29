"""Gate-evaluation orchestrator (Story 8.12c).

End-to-end glue between a completed training run and the gate-evaluation
verdict. Loads the run's checkpoint, runs the pre-registered probe
suite against the encoder's latents on the eval cohort, aggregates
worm-level bootstrap CIs into a :class:`MetricsOutput`, and invokes
:func:`evaluate_gates` to produce a :class:`GateStatus`.

**Story 8.12c.1+ progressive wiring.** Probes wire in one at a time:

- 8.12c-skeleton: state-load + cfg-load + stub MetricsOutput.
- 8.12c.1: :func:`_build_eval_cache` (one pass over the eval cohort,
  capturing latents + pose + neural + worm_id per clip), plus
  :func:`_run_partial_r2_probe` and :func:`_run_session_id_probe`
  consumed by :func:`evaluate_run`.
- 8.12c.2: :func:`_run_future_pose_probe` — observe-one-
  frame / predict-rest scheme via the trained predictor + the
  pose_decoder head. Wraps into ``future_pose`` MetricEntry with
  per-horizon sub-entries at {0.1s, 0.5s, 1.0s} (assuming the Phase
  0 v0 baaiworm clips render at 10 fps; documented in entry notes).
- 8.12c.2b (this commit): :func:`_run_transformer_eigenworms_future_pose_probe`
  — fits the transformer-eigenworms baseline ON the eval cohort
  (Phase 0 v0 caveat: no separate baseline-eval train cohort wired),
  rolls out per-horizon predictions under the same observe-frame-0 /
  predict-frame-at-offset scheme used by the JEPA-side probe so the
  errors are directly comparable, slices 3D → 2D to match the
  pose_decoder's 2-D outputs, and emits a sibling ``future_pose``
  MetricEntry with ``producer="transformer_eigenworms"``. With both
  entries present, :func:`evaluate_gates._eval_kill_criterion` can
  produce a real verdict (cleared/fired) instead of the previous
  pending.
- 8.12c.3+ (this batch): motif_ari probe + within_state stratified
  partial-R² probe + non-trivial-neuron-subset partial-R² probe, plus
  a Tierpsy-approximating kinematic baseline (centroid + velocity +
  curvature proxy + body-length proxy + raw pose + first-diff) that
  replaces the previous ``pose | diff`` proxy. Synthetic
  behavioral_state from baaiworm's centroid-velocity binning is the
  Phase 0 v0 source for behavioural labels (gamma-deferred Flavell
  cohort is the real source).
- Future: 8.12c.4 ablations, 8.12c.6 Holm correction.

Per-probe wiring uses a held-out baaiworm cohort as a Phase 0 v0
proxy for the pre-registered eval cohort (which depended on
materialised WormID / Flavell data, both gamma-deferred as of
2026-05-18). The proxy is honest about its scope via the MetricEntry
notes field; real eval-cohort plumbing lands once Phase 0 Growth
materialises the deferred data.

Today's deliverables:

- :func:`reconstruct_state_from_run` — load ``cfg.yaml`` from a
  results directory, rebuild :class:`JEPATrainingState` via the same
  runner code path that produced the checkpoint, and call
  :func:`load_checkpoint` to restore weights + RNG state.
- :func:`_build_eval_cache` — iterate the eval loader once, encode
  each clip via the online encoder, snapshot pose + neural + ids.
- :func:`_run_partial_r2_probe` — leave-one-worm-out ridge regression
  on JEPA latent vs a pose-derived kinematic baseline, predicting
  neural activity. Wraps into a ``neural_probe_partial_r2`` MetricEntry.
- :func:`_run_session_id_probe` — logistic-regression probe latent →
  session_id, accuracy reported with chance baseline in notes.
- :func:`evaluate_run` — high-level entry point. Reconstructs state,
  builds the eval cache, runs each probe, assembles MetricsOutput,
  invokes :func:`evaluate_gates`, returns the verdict.
- :func:`evaluate_sweep` — cross-run aggregator: takes ``[run_dir,
  ...]`` (the three pre-committed seeds), evaluates each, and
  combines per-seed verdicts. Stub today.

``wormjepa eval --run <id>`` (cli/eval.py) is the only public surface.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from wormjepa.baselines.base import FuturePoseHorizon
from wormjepa.baselines.transformer_eigenworms import TransformerEigenwormsBaseline
from wormjepa.configs import load_config
from wormjepa.configs.dataset import DatasetLoaderSpec
from wormjepa.configs.jepa_config import JEPARunConfig
from wormjepa.data import WormID
from wormjepa.data.composition import build_loader
from wormjepa.eval.bootstrap import WormGrouping, bootstrap_ci
from wormjepa.eval.future_pose import future_pose_metric
from wormjepa.eval.gates import GateStatus, evaluate_gates
from wormjepa.eval.metrics_schema import BootstrapCI, MetricEntry, MetricsOutput, SubEntry
from wormjepa.eval.motif_ari import motif_ari
from wormjepa.eval.multiple_comparison import apply_correction
from wormjepa.eval.residualization import partial_r2
from wormjepa.models.pose_decoder import PoseDecoderHead

if TYPE_CHECKING:
    from wormjepa.training.loop import JEPATrainingState

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Eval cohort + cache
# ----------------------------------------------------------------------

#: Default eval-cohort sizing knobs. Tuned for ridge regression to have
#: meaningful sample count + leave-one-worm-out to fit ~10 worms.
_EVAL_BAAIWORM_WORMS = 8
_EVAL_BAAIWORM_CLIPS_PER_WORM = 3
_EVAL_BAAIWORM_NEURONS = 8


def _baaiworm_spec_from_train(cfg: JEPARunConfig) -> DatasetLoaderSpec | None:
    """Return the training cfg's baaiworm loader spec if present.

    Used to align the eval cohort's n_keypoints with the training-time
    value so heads sized by the runner (pose_decoder, neural_head) can
    consume eval-cohort samples without dim mismatch.
    """
    for spec in cfg.dataset.loaders:
        if spec.name == "baaiworm":
            return spec
    return None


def _build_eval_loader_spec(cfg: JEPARunConfig) -> tuple[DatasetLoaderSpec, int]:
    """Return the held-out baaiworm eval loader's spec + seed.

    Split from :func:`_build_eval_loader` so the same spec can drive
    both loader construction and the encoder-cache fingerprint. Keeps
    the cache key bytewise-stable across runs that share a (cfg,
    checkpoint) pair.
    """
    train_spec = _baaiworm_spec_from_train(cfg)
    n_keypoints = train_spec.n_keypoints if train_spec is not None else 4
    eval_seed = cfg.jepa.seed + 10000
    spec = DatasetLoaderSpec(
        name="baaiworm",
        clip_frames=16,
        n_worms=_EVAL_BAAIWORM_WORMS,
        clips_per_worm=_EVAL_BAAIWORM_CLIPS_PER_WORM,
        n_keypoints=n_keypoints,
        n_neurons=_EVAL_BAAIWORM_NEURONS,
        image_size=cfg.jepa.img_size,
    )
    return spec, eval_seed


def _build_eval_loader(cfg: JEPARunConfig) -> Any:
    """Build a held-out baaiworm eval loader at a different seed offset.

    The eval cohort's ``n_keypoints`` is inherited from the training
    cfg's baaiworm spec when present — keeps pose_decoder + neural_head
    dims consistent with the trained state. Falls back to 4 (the
    runner's legacy synthetic default) when the training cfg has no
    baaiworm loader (e.g. a synthetic-only smoke run).

    Phase 0 v0 proxy: the pre-registered eval cohort (WormID HL+SF,
    Flavell behavioural cohort) is gamma-deferred at materialization;
    until those are reversed, baaiworm at seed+10000 is the only
    cohort that satisfies the FR8 pose+neural contract for partial-R²
    and session-classifier probes. Documented per-probe in the
    MetricEntry notes.
    """
    spec, eval_seed = _build_eval_loader_spec(cfg)
    return build_loader([spec], seed=eval_seed)


class EvalCache:
    """Per-clip eval cohort cache produced by :func:`_build_eval_cache`.

    All tensors are CPU + numpy after the encoder pass so the downstream
    sklearn / numpy probes don't need to round-trip through torch.
    """

    def __init__(self) -> None:
        self.latents: list[np.ndarray] = []  # each (T, D)
        self.poses: list[np.ndarray] = []  # each (T, K*C)
        self.neural: list[np.ndarray] = []  # each (T, N)
        self.worm_ids: list[str] = []  # per clip
        self.session_ids: list[str] = []  # per clip
        # Story 8.12c.3+: per-clip ``(T,)`` int arrays of behavioral_state
        # class indices, populated when the loader yields a non-None
        # ``DatasetSample.behavioral_state``. baaiworm v1 synthesises these
        # via centroid-velocity binning (see loader docstring); real
        # Flavell labels are gamma-deferred.
        self.behavioral_states: list[np.ndarray] = []  # each (T,) int64


# Legacy alias kept for any in-tree call sites that still reference the
# private name. New code should import :class:`EvalCache` directly.
_EvalCache = EvalCache


def _build_eval_cache(  # pyright: ignore[reportUnusedFunction] - re-exported via wormjepa.eval.encoder_cache
    state: JEPATrainingState,
    cfg: JEPARunConfig,
    max_clips: int = 64,
) -> _EvalCache:
    """Iterate the eval cohort once; encode each clip; capture pose + neural + ids.

    ``max_clips`` caps the cache size to keep ridge-regression compute
    bounded for the Phase 0 v0 proxy cohort. A real-data follow-up
    would remove the cap.
    """
    online = state.online_encoder
    online.eval()
    device = next(online.parameters()).device
    loader = _build_eval_loader(cfg)
    cache = _EvalCache()
    with torch.no_grad():
        for sample in loader:
            if sample.pose is None or sample.neural is None:
                continue
            video = sample.video_clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
            latent = online(video).squeeze(0).cpu().numpy()  # (T, D)
            pose = sample.pose.cpu().numpy()  # (T, K, C)
            t, k, c = pose.shape
            cache.latents.append(latent)
            cache.poses.append(pose.reshape(t, k * c))
            cache.neural.append(sample.neural.cpu().numpy())  # (T, N)
            cache.worm_ids.append(str(sample.worm_id))
            cache.session_ids.append(str(sample.session_id))
            if sample.behavioral_state is not None:
                cache.behavioral_states.append(
                    sample.behavioral_state.cpu().numpy().astype(np.int64)
                )
            if len(cache.latents) >= max_clips:
                break
    logger.info(
        "_build_eval_cache: cached %d clips across %d unique worms",
        len(cache.latents),
        len(set(cache.worm_ids)),
    )
    return cache


def _kinematic_features_from_pose(pose: np.ndarray, pose_dim: int = 2) -> np.ndarray:
    """Phase 0 v0 Tierpsy-approximation kinematic baseline.

    Builds a ``(T, D_kin)`` per-frame feature matrix combining:

    - **centroid**: per-frame ``(x, y)`` mean across keypoints.
    - **centroid velocity**: first-order temporal diff of the centroid.
    - **curvature proxy**: stddev of the flat pose coordinates per frame
      (rough body-bend proxy — high stddev when the worm bends, low
      when it's straight).
    - **body length proxy**: sum of pairwise distances between
      consecutive keypoints (Tierpsy ``length`` analogue).
    - **raw pose**: the original flat pose vector.
    - **pose first-diff**: per-frame first-order temporal diff (kept
      from the previous proxy for backwards-compat).

    Args:
        pose: ``(T, K*C)`` flat pose array (after :func:`_flatten_cache`).
        pose_dim: ``C`` — number of spatial coords per keypoint (2 for
            xy, 3 for xyz). baaiworm v1 pose is 2D; defaulting to 2.

    Real Tierpsy-256 + pose-TCN integration is a Phase 0 Growth
    follow-up. The MetricEntry notes field surfaces this proxy
    explicitly so the headline-claim caveat is visible at audit time.
    """
    t, kc = pose.shape
    if pose_dim <= 0 or kc % pose_dim != 0:
        # Shape doesn't decompose cleanly into keypoints * coords; fall
        # back to the legacy ``pose | diff`` proxy without geometric
        # features (centroid / length / curvature). Should not happen
        # in practice — baaiworm v1 always emits a 2D pose.
        diff = np.diff(pose, axis=0, prepend=pose[:1])
        return np.concatenate([pose, diff], axis=1)
    k = kc // pose_dim
    pose_kc = pose.reshape(t, k, pose_dim)
    # Centroid + centroid velocity.
    centroid = pose_kc.mean(axis=1)  # (T, C)
    centroid_velocity = np.diff(centroid, axis=0, prepend=centroid[:1])
    # Curvature proxy: per-frame stddev of the flat pose coords.
    curvature = pose.std(axis=1, keepdims=True)  # (T, 1)
    # Body length proxy: Σ pairwise distances between consecutive keypoints.
    if k >= 2:
        segs = pose_kc[:, 1:, :] - pose_kc[:, :-1, :]  # (T, K-1, C)
        seg_lens = np.linalg.norm(segs, axis=2)  # (T, K-1)
        body_length = seg_lens.sum(axis=1, keepdims=True)  # (T, 1)
    else:
        body_length = np.zeros((t, 1), dtype=pose.dtype)
    # Pose first-diff (kept for compat).
    pose_diff = np.diff(pose, axis=0, prepend=pose[:1])
    return np.concatenate(
        [centroid, centroid_velocity, curvature, body_length, pose, pose_diff], axis=1
    )


# ----------------------------------------------------------------------
# Probes
# ----------------------------------------------------------------------


def _flatten_cache(
    cache: _EvalCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Concatenate per-clip arrays into per-frame matrices with parallel ids."""
    n_clips = len(cache.latents)
    if n_clips == 0:
        raise ValueError("_flatten_cache: cache is empty")
    latents = np.concatenate(cache.latents, axis=0)  # (Nframes, D_latent)
    poses = np.concatenate(cache.poses, axis=0)  # (Nframes, K*C)
    neural = np.concatenate(cache.neural, axis=0)  # (Nframes, N_neurons)
    worm_per_frame: list[str] = []
    session_per_frame: list[str] = []
    for clip_idx, lat in enumerate(cache.latents):
        worm_per_frame.extend([cache.worm_ids[clip_idx]] * lat.shape[0])
        session_per_frame.extend([cache.session_ids[clip_idx]] * lat.shape[0])
    return latents, poses, neural, worm_per_frame, session_per_frame


def _run_partial_r2_probe(cache: _EvalCache, n_bootstrap: int = 1000) -> MetricEntry:
    """Leave-one-worm-out partial-R² of JEPA latent over a kinematic baseline,
    predicting neural activity. Wraps result into a ``neural_probe_partial_r2``
    MetricEntry consumed by :func:`evaluate_gates`.

    Phase 0 v0 caveats (recorded in entry notes):

    - Eval cohort is held-out baaiworm rather than the pre-registered
      WormID + Flavell cohorts (gamma-deferred at materialization).
    - Kinematic baseline is a hand-engineered Tierpsy-approximation
      (centroid + velocity + curvature proxy + body-length proxy + raw
      pose + first-diff) rather than the pre-registered Tierpsy-256 +
      TCN baseline.
    """
    latents, poses, neural, worm_per_frame, _ = _flatten_cache(cache)
    kin = _kinematic_features_from_pose(poses)
    result = partial_r2(
        jepa_latent=latents,
        kinematic_features=kin,
        neural_target=neural,
        worm_ids=worm_per_frame,
    )
    grouping = WormGrouping(worm_ids=tuple(WormID(w) for w in result.worm_ids))
    per_worm = np.asarray(result.per_worm_partial_r2)
    ci = bootstrap_ci(per_worm, grouping, n_samples=n_bootstrap, method="bca")
    return MetricEntry(
        name="neural_probe_partial_r2",
        producer="jepa",
        ci=ci,
        sub_entries=[
            SubEntry(
                key="r2_jepa",
                ci=BootstrapCI(
                    point=result.r2_jepa,
                    lower=result.r2_jepa,
                    upper=result.r2_jepa,
                    n_samples=1,
                    method="percentile",
                ),
            ),
            SubEntry(
                key="r2_kinematic_baseline",
                ci=BootstrapCI(
                    point=result.r2_kinematic,
                    lower=result.r2_kinematic,
                    upper=result.r2_kinematic,
                    n_samples=1,
                    method="percentile",
                ),
            ),
        ],
        notes=(
            "Phase 0 v0: eval cohort = held-out baaiworm (seed+10000); "
            "kinematic baseline = hand-engineered Tierpsy-approx features "
            "(centroid + velocity + curvature + length + raw pose + diff); "
            "real Tierpsy-256 + pose-TCN integration is Phase 0 Growth. "
            "Worm-level bootstrap over leave-one-worm-out per-worm partial-R²."
        ),
    )


def _run_session_id_probe(cache: _EvalCache, n_bootstrap: int = 1000) -> MetricEntry:
    """Logistic-regression probe ``latent → session_id``; report accuracy with
    chance baseline (1 / n_classes) embedded in the notes per the
    :func:`evaluate_gates._eval_session_id_at_chance` parser contract.

    Leave-one-worm-out CV at the worm grouping (each session is
    per-worm); per-worm accuracy then bootstrapped at the worm level.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    latents, _poses, _neural, worm_per_frame, session_per_frame = _flatten_cache(cache)
    label_enc = LabelEncoder()
    y_all = np.asarray(label_enc.fit_transform(session_per_frame))
    unique_classes = label_enc.classes_
    if unique_classes is None:
        msg = "LabelEncoder produced no classes_; cohort has no sessions."
        raise ValueError(msg)
    chance = 1.0 / max(len(unique_classes), 1)
    worm_array = np.asarray(worm_per_frame)
    unique_worms = list(dict.fromkeys(worm_per_frame))
    per_worm_acc: list[float] = []
    kept_worms: list[str] = []
    for held_out in unique_worms:
        train_mask = worm_array != held_out
        test_mask = worm_array == held_out
        if not train_mask.any() or not test_mask.any():
            continue
        # Need at least 2 classes represented in the train split.
        if len(set(y_all[train_mask].tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=200)
        clf.fit(latents[train_mask], y_all[train_mask])
        preds = clf.predict(latents[test_mask])
        per_worm_acc.append(float(np.mean(preds == y_all[test_mask])))
        kept_worms.append(held_out)
    if not per_worm_acc:
        # Degenerate case: every held-out fold lacked train-side class
        # diversity. Surface a pending-shaped CI; gate evaluator will
        # treat as "could not parse" → pending.
        nan_ci = BootstrapCI(
            point=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n_samples=1,
            method="percentile",
        )
        return MetricEntry(
            name="session_id_classifier",
            producer="jepa",
            ci=nan_ci,
            sub_entries=[],
            notes=(
                f"chance={chance:.4f}; per-worm accuracy bootstrap aborted "
                f"(insufficient class diversity per-fold)."
            ),
        )
    grouping = WormGrouping(worm_ids=tuple(WormID(w) for w in kept_worms))
    ci = bootstrap_ci(np.asarray(per_worm_acc), grouping, n_samples=n_bootstrap, method="bca")
    return MetricEntry(
        name="session_id_classifier",
        producer="jepa",
        ci=ci,
        sub_entries=[],
        notes=(
            f"chance={chance:.4f}; per-worm leave-one-out logistic-regression "
            f"probe accuracy; worm-level bootstrap. Phase 0 v0 eval cohort: "
            f"held-out baaiworm (seed+10000). The session_id_at_chance gate "
            f"clears when the 95% CI contains chance — see _SESSION_ID_AT_CHANCE_PARSE "
            f"contract in eval/gates.py (parses chance=<float> via regex)."
        ),
    )


# Sanity: the notes string above embeds "chance=<float>" matching the
# regex `_eval_session_id_at_chance` parses (`r"chance=(\d*\.?\d+)"`).
_CHANCE_RE = re.compile(r"chance=(\d*\.?\d+)")
assert _CHANCE_RE.search("chance=0.1250") is not None


# ----------------------------------------------------------------------
# Diagnostic probes (Story 8.12c.3+): motif_ari, within_state, neuron_subset
# ----------------------------------------------------------------------


def _flatten_behavioral_states(cache: _EvalCache) -> np.ndarray | None:
    """Per-frame behavioral-state vector aligned with :func:`_flatten_cache`.

    Returns ``None`` when the cache holds no behavioral_state arrays, or
    when the count of behavioral_state arrays does not match the count
    of clips (defensive — the cache only populates this list when the
    loader yielded a non-None ``DatasetSample.behavioral_state``).
    """
    if not cache.behavioral_states:
        return None
    if len(cache.behavioral_states) != len(cache.latents):
        logger.warning(
            "_flatten_behavioral_states: cache has %d behavioral_state arrays vs %d "
            "latent arrays; skipping (loader inconsistency).",
            len(cache.behavioral_states),
            len(cache.latents),
        )
        return None
    return np.concatenate(cache.behavioral_states, axis=0).astype(np.int64)


#: Number of synthetic behavioral-state classes (still / slow / fast) the
#: baaiworm v1 loader emits. Used to size the motif_ari / within_state
#: probes' n_states / n_clusters parameters.
_N_BEHAVIORAL_STATES = 3


def _run_motif_ari_probe(cache: _EvalCache) -> MetricEntry | None:
    """Motif-ARI probe: cluster latents w/ k-means, score Hungarian-matched
    ARI against per-frame behavioral_state labels via leave-one-worm-out.

    Phase 0 v0 caveat (recorded in entry notes): the behavioral_state
    labels are synthetic — derived from baaiworm's pose-centroid
    velocity binning (3 bins: still / slow / fast). Real Flavell
    behavioural labels are gamma-deferred at materialization.

    Returns ``None`` when the cache has no behavioral_state vector
    (e.g. a cohort built from a loader that doesn't synthesise one).
    """
    states = _flatten_behavioral_states(cache)
    if states is None:
        logger.info("_run_motif_ari_probe: no behavioral_state in cache; skipping")
        return None
    latents, _poses, _neural, worm_per_frame, _session = _flatten_cache(cache)
    if len(set(worm_per_frame)) < 2:
        logger.info("_run_motif_ari_probe: <2 distinct worms; skipping")
        return None
    result = motif_ari(
        latents,
        states,
        worm_per_frame,
        n_states=_N_BEHAVIORAL_STATES,
        n_clusters=_N_BEHAVIORAL_STATES,
    )
    # Bootstrap CI from per-worm ARI scores at the worm grouping.
    per_worm = np.asarray(result.per_worm_ari)
    if per_worm.size == 0:
        nan_ci = BootstrapCI(
            point=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n_samples=1,
            method="percentile",
        )
        return MetricEntry(
            name="motif_ari",
            producer="jepa",
            ci=nan_ci,
            sub_entries=[],
            notes=(
                "Phase 0 v0: synthetic behavioral_state from baaiworm velocity "
                "binning; real Flavell behavioural labels are gamma-deferred. "
                "per-worm ARI vector empty — no folds were scoreable."
            ),
        )
    grouping = WormGrouping(worm_ids=tuple(WormID(w) for w in result.worm_ids))
    ci = bootstrap_ci(per_worm, grouping, n_samples=1000, method="bca")
    return MetricEntry(
        name="motif_ari",
        producer="jepa",
        ci=ci,
        sub_entries=[],
        notes=(
            "Phase 0 v0: synthetic behavioral_state from baaiworm velocity "
            "binning (3 classes: still/slow/fast via np.digitize on "
            "centroid-speed quantiles); real Flavell behavioural labels "
            "are gamma-deferred. k-means with n_clusters=n_states=3 + "
            "Hungarian assignment per leave-one-worm-out fold; worm-level "
            "bootstrap over per-worm ARI."
        ),
    )


def _run_within_state_partial_r2_probe(
    cache: _EvalCache, n_bootstrap: int = 1000
) -> MetricEntry | None:
    """Per-state partial-R² stratification probe.

    For each observed behavioral_state class, restricts the cache to
    frames in that state and runs the same leave-one-worm-out
    partial-R² machinery as :func:`_run_partial_r2_probe`. One
    :class:`SubEntry` per state holding its per-state CI; the top-level
    CI is a NaN placeholder.

    Phase 0 v0 caveat (recorded in entry notes): stratification is over
    synthetic behavioral_state from baaiworm velocity-binning; real
    Flavell behavioural labels are gamma-deferred.
    """
    states = _flatten_behavioral_states(cache)
    if states is None:
        logger.info("_run_within_state_partial_r2_probe: no behavioral_state; skipping")
        return None
    latents, poses, neural, worm_per_frame, _session = _flatten_cache(cache)
    if states.shape[0] != latents.shape[0]:
        logger.warning(
            "_run_within_state_partial_r2_probe: states length %d != latents length %d; skipping",
            states.shape[0],
            latents.shape[0],
        )
        return None
    kin = _kinematic_features_from_pose(poses)
    sub_entries: list[SubEntry] = []
    skipped: list[str] = []
    state_names = ("still", "slow", "fast")
    worm_array = np.asarray(worm_per_frame)
    for state_idx, name in enumerate(state_names):
        mask = states == state_idx
        if not mask.any():
            skipped.append(name)
            continue
        sub_worms = worm_array[mask].tolist()
        if len(set(sub_worms)) < 2:
            skipped.append(name)
            continue
        try:
            result = partial_r2(
                latents[mask],
                kin[mask],
                neural[mask],
                sub_worms,
            )
        except ValueError:
            skipped.append(name)
            continue
        per_worm = np.asarray(result.per_worm_partial_r2)
        if per_worm.size == 0:
            skipped.append(name)
            continue
        # Bootstrap CI at the worm grouping for this state's slice.
        try:
            grouping = WormGrouping(worm_ids=tuple(WormID(w) for w in result.worm_ids))
            ci = bootstrap_ci(per_worm, grouping, n_samples=n_bootstrap, method="bca")
        except Exception:
            # Degenerate folds (e.g. only one worm survives) fall back to a
            # min/max-bound placeholder CI from the per-worm values.
            ci = BootstrapCI(
                point=float(per_worm.mean()),
                lower=float(per_worm.min()),
                upper=float(per_worm.max()),
                n_samples=int(per_worm.size),
                method="percentile",
            )
        sub_entries.append(SubEntry(key=name, ci=ci))

    nan_ci = BootstrapCI(
        point=float("nan"),
        lower=float("nan"),
        upper=float("nan"),
        n_samples=1,
        method="percentile",
    )
    skipped_str = ",".join(skipped) if skipped else "none"
    return MetricEntry(
        name="within_state_stratified_r2",
        producer="jepa",
        ci=nan_ci,
        sub_entries=sub_entries,
        notes=(
            f"Phase 0 v0: stratification over synthetic behavioral_state "
            f"from baaiworm velocity binning (3 classes: still/slow/fast); "
            f"real Flavell behavioural labels are gamma-deferred. Per-state "
            f"leave-one-worm-out partial-R² with worm-level bootstrap; states "
            f"skipped: {skipped_str}."
        ),
    )


def _run_neuron_subset_partial_r2_probe(
    cache: _EvalCache, k: int = 4, n_bootstrap: int = 1000
) -> MetricEntry | None:
    """Top-k highest-variance neurons partial-R² diagnostic.

    Picks the ``k`` neurons (columns of the per-frame neural matrix)
    with the highest variance across the cache, restricts the
    partial-R² regression to predict ONLY those neurons, and returns a
    :class:`MetricEntry` named ``non_trivial_neuron_subset_partial_r2``.

    Phase 0 v0 caveat (recorded in entry notes): "top-k variance
    subset" is the Phase 0 v0 stand-in for the pre-registered neuron
    list — that list is gated on Flavell + WormID neuron labels which
    are both gamma-deferred at materialization.
    """
    latents, poses, neural, worm_per_frame, _session = _flatten_cache(cache)
    if neural.shape[1] == 0:
        return None
    if len(set(worm_per_frame)) < 2:
        return None
    kin = _kinematic_features_from_pose(poses)
    # Pick top-k highest-variance columns.
    n_avail = neural.shape[1]
    k_eff = min(int(k), int(n_avail))
    variances = neural.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:k_eff]
    top_idx_sorted = np.sort(top_idx)
    restricted_neural = neural[:, top_idx_sorted]
    try:
        result = partial_r2(
            latents,
            kin,
            restricted_neural,
            worm_per_frame,
        )
    except ValueError:
        logger.exception("_run_neuron_subset_partial_r2_probe: partial_r2 failed")
        return None
    per_worm = np.asarray(result.per_worm_partial_r2)
    if per_worm.size == 0:
        return None
    grouping = WormGrouping(worm_ids=tuple(WormID(w) for w in result.worm_ids))
    try:
        ci = bootstrap_ci(per_worm, grouping, n_samples=n_bootstrap, method="bca")
    except Exception:
        ci = BootstrapCI(
            point=float(per_worm.mean()),
            lower=float(per_worm.min()),
            upper=float(per_worm.max()),
            n_samples=int(per_worm.size),
            method="percentile",
        )
    return MetricEntry(
        name="non_trivial_neuron_subset_partial_r2",
        producer="jepa",
        ci=ci,
        sub_entries=[
            SubEntry(
                key="r2_jepa",
                ci=BootstrapCI(
                    point=result.r2_jepa,
                    lower=result.r2_jepa,
                    upper=result.r2_jepa,
                    n_samples=1,
                    method="percentile",
                ),
            ),
            SubEntry(
                key="r2_kinematic_baseline",
                ci=BootstrapCI(
                    point=result.r2_kinematic,
                    lower=result.r2_kinematic,
                    upper=result.r2_kinematic,
                    n_samples=1,
                    method="percentile",
                ),
            ),
        ],
        notes=(
            f"Phase 0 v0: top-k variance subset (k={k_eff}, neuron_indices="
            f"{top_idx_sorted.tolist()} of {n_avail} available); pre-registered "
            f"neuron list is Phase 0 Growth pending Flavell + WormID labels. "
            f"Same leave-one-worm-out partial-R² machinery as the headline "
            f"probe, restricted to the top-variance neural target columns."
        ),
    )


# ----------------------------------------------------------------------
# Future-pose probe (Story 8.12c.2, kill_criterion gate)
# ----------------------------------------------------------------------

#: Phase 0 v0 baaiworm-clip frame-rate assumption (frames per second).
#: Used to map predicted-frame-offsets to the {0.1s, 0.5s, 1.0s}
#: pre-registered horizons. baaiworm renders without explicit fps;
#: 10 fps is the project's working assumption (documented in the
#: future_pose MetricEntry notes). A real-data eval cohort with
#: ground-truth frame timing would replace this with the loader's
#: actual fps once Phase 0 Growth materialises the deferred data.
_CLIP_ASSUMED_FPS = 10.0

#: Horizons emitted as future_pose sub-entries. Only horizons that fit
#: within the clip length are computed; 5.0s exceeds the 16-frame
#: clip at 10 fps and is therefore omitted from the Phase 0 v0
#: future_pose probe.
_HORIZONS_SECONDS: tuple[float, ...] = (0.1, 0.5, 1.0)


def _run_future_pose_probe(
    state: JEPATrainingState,
    cfg: JEPARunConfig,
    cache: _EvalCache,
    n_bootstrap: int = 1000,
) -> MetricEntry | None:
    """Future-pose error probe via the predictor + pose_decoder head.

    Scheme: for each cached clip we observe the first frame only,
    mask the rest, ask the predictor to forecast the masked latents,
    decode via the pose_decoder, and compare to the recorded
    ground-truth pose at each predicted-frame offset. Per-horizon
    Euclidean keypoint error is then bootstrapped at the worm level
    by :func:`future_pose_metric`.

    Returns ``None`` when the run's training state lacks a
    pose_decoder head (only newer runs have it, and the head is
    "dev-loop only" in its current shipped form). The caller should
    treat ``None`` as "future_pose probe skipped, kill_criterion gate
    stays pending."
    """
    pose_decoder = state.warm_start_heads.get("pose_decoder")
    if not isinstance(pose_decoder, PoseDecoderHead):
        logger.info("_run_future_pose_probe: pose_decoder head absent; skipping")
        return None
    predictor = state.predictor
    online = state.online_encoder
    online.eval()
    pose_decoder.eval()
    predictor.eval()
    device = next(online.parameters()).device

    # Compute predicted-frame offsets that match the requested horizons.
    # offset = round(horizon * fps). Skip horizons whose offset >= T.
    n_clips = len(cache.latents)
    if n_clips == 0:
        return None
    t_total = cache.latents[0].shape[0]
    horizon_offsets: list[tuple[float, int]] = []
    for h in _HORIZONS_SECONDS:
        offset = round(h * _CLIP_ASSUMED_FPS)
        if 1 <= offset < t_total:
            horizon_offsets.append((h, offset))
    if not horizon_offsets:
        logger.info(
            "_run_future_pose_probe: no horizons fit within clip length T=%d "
            "at assumed fps=%.1f; skipping",
            t_total,
            _CLIP_ASSUMED_FPS,
        )
        return None

    # Reload sample clips so we can re-encode under the masked-predict
    # scheme. We cannot re-use cache.latents directly because the cache
    # was built without a mask; the predictor needs us to substitute
    # mask tokens at the future positions before encoding.
    loader = _build_eval_loader(cfg)
    predictions: list[FuturePoseHorizon] = []
    with torch.no_grad():
        for sample in loader:
            if sample.pose is None or sample.neural is None:
                continue
            video = sample.video_clip.unsqueeze(0).to(device)  # (1, T, C, H, W)
            online_latent = online(video)  # (1, T, D)
            t = online_latent.shape[1]
            # Mask everything past frame 0: observe one, predict the rest.
            mask = torch.zeros((1, t), dtype=torch.bool, device=device)
            mask[:, 1:] = True
            predicted_latent = predictor(online_latent, mask)  # (1, T, D)
            # The predictor forecasts pooled latents; the spatial-token-aware
            # PoseDecoderHead expects (B, T, S, D). The forecast has no
            # spatial grid, so present it as a singleton-S grid — the
            # cross-attention degenerates to attending over one token.
            predicted_pose = pose_decoder.predict(predicted_latent.unsqueeze(2))  # (1, T, K, 2)
            gt_pose = sample.pose.unsqueeze(0).to(device)  # (1, T, K, C)
            # pose_decoder is 2-D; if ground-truth is 3-D, slice the first
            # two coordinates. Phase 0 v0 baaiworm is 3-D (x, y, z); the
            # decoder ignores z. Document via the entry notes.
            if gt_pose.shape[-1] != predicted_pose.shape[-1]:
                gt_pose = gt_pose[..., : predicted_pose.shape[-1]]
            worm_id = WormID(str(sample.worm_id))
            session_id = str(sample.session_id)
            for h_seconds, offset in horizon_offsets:
                predictions.append(
                    FuturePoseHorizon(
                        worm_id=worm_id,
                        session_id=session_id,
                        horizon_seconds=h_seconds,
                        predicted=predicted_pose[0, offset].cpu(),
                        ground_truth=gt_pose[0, offset].cpu(),
                    )
                )
            if len(predictions) >= n_clips * len(horizon_offsets):
                break

    if not predictions:
        return None

    entry = future_pose_metric(predictions, producer="jepa", n_bootstrap=n_bootstrap)
    notes_suffix = (
        f"  Phase 0 v0 scheme: observe-frame-0 / predict-the-rest via the "
        f"trained JEPAPredictor + PoseDecoderHead. Horizon mapping assumes "
        f"baaiworm clips render at {_CLIP_ASSUMED_FPS:.1f} fps (no native "
        f"fps in the synthetic generator); a real-data eval cohort would "
        f"use the loader's recorded frame timing. The kill_criterion gate "
        f"also requires a sibling transformer_eigenworms future_pose entry "
        f"to fire a verdict; without that the gate stays 'pending'. The "
        f"pose_decoder head is dev-loop-only in its current shipped form "
        f"so this entry is honestly framed as the Phase 0 v0 form of "
        f"future-pose probing; a Phase 0 Growth follow-up may train a "
        f"pre-registered pose-decoder head per Story 5.x."
    )
    return MetricEntry(
        name=entry.name,
        producer=entry.producer,
        ci=entry.ci,
        sub_entries=entry.sub_entries,
        notes=(entry.notes or "") + notes_suffix,
    )


def _run_transformer_eigenworms_future_pose_probe(
    cfg: JEPARunConfig,
    cache: _EvalCache,
    n_bootstrap: int = 1000,
) -> MetricEntry | None:
    """Future-pose error probe for the transformer-eigenworms baseline.

    Fits :class:`TransformerEigenwormsBaseline` on the eval cohort's pose
    sequences (Phase 0 v0 train-test caveat — see notes below), then
    rolls out per-horizon predictions under the SAME observe-frame-0 /
    predict-frame-at-offset scheme as :func:`_run_future_pose_probe`, so
    the per-clip errors are directly comparable for the kill_criterion
    gate. The baseline's default :meth:`predict` uses a different scheme
    (last-frame ground truth, lookback-from-end rollout) which would
    produce error magnitudes that aren't apples-to-apples with the JEPA
    side, so we bypass it here in favour of a tailored rollout loop
    that calls the baseline's projection / model machinery directly.

    Returns ``None`` if the cache is empty or no horizon fits within
    the clip length (mirrors the JEPA-side probe's contract).

    Phase 0 v0 train-test contamination caveat (recorded in entry notes):
    the baseline is fit on the same cohort it's evaluated on because a
    separate baseline-eval train cohort isn't wired yet. A Phase 0
    Growth follow-up will materialise a held-out train split for the
    baseline.
    """
    # Reuse the JEPA-side horizon mapping so both producers report the
    # same sub-entry keys ("0.1s", "0.5s", "1s") at matching offsets.
    n_clips = len(cache.latents)
    if n_clips == 0:
        logger.info("_run_transformer_eigenworms_future_pose_probe: cache empty; skipping")
        return None
    t_total = cache.latents[0].shape[0]
    horizon_offsets: list[tuple[float, int]] = []
    for h in _HORIZONS_SECONDS:
        offset = round(h * _CLIP_ASSUMED_FPS)
        if 1 <= offset < t_total:
            horizon_offsets.append((h, offset))
    if not horizon_offsets:
        logger.info(
            "_run_transformer_eigenworms_future_pose_probe: no horizons fit within "
            "clip length T=%d at assumed fps=%.1f; skipping",
            t_total,
            _CLIP_ASSUMED_FPS,
        )
        return None

    # Collect eval-cohort samples once. The baseline fits on this same
    # set (train-test caveat documented in entry notes) and then we roll
    # out the JEPA-comparable scheme on it.
    loader = _build_eval_loader(cfg)
    samples = [s for s in loader if s.pose is not None]
    if not samples:
        logger.info(
            "_run_transformer_eigenworms_future_pose_probe: no pose-bearing samples; skipping"
        )
        return None

    baseline = TransformerEigenwormsBaseline(
        horizons_seconds=tuple(h for h, _ in horizon_offsets),
        frame_dt_seconds=1.0 / _CLIP_ASSUMED_FPS,
    )
    baseline.fit(samples)

    # Slice predicted + ground-truth to 2D to align with the JEPA-side
    # pose_decoder (which is 2D-only). baaiworm pose is (T, K, 3); the
    # JEPA-side probe drops the z coordinate. We mirror that slice here
    # so kill_criterion's "lower error is better" comparison is on the
    # same dimensionality.
    pose_dims = 2
    predictions: list[FuturePoseHorizon] = []
    with torch.no_grad():
        for sample in samples:
            if sample.pose is None:
                continue
            t, k, d = sample.pose.shape
            flat = sample.pose.reshape(t, k * d)
            eigen_seq = baseline._project(flat)  # pyright: ignore[reportPrivateUsage]
            # Observe frame 0 only; roll forward `offset` steps; compare
            # against ground-truth frame at the same offset (mirrors the
            # JEPA-side scheme).
            history0 = eigen_seq[:1]
            for h_seconds, offset in horizon_offsets:
                eigen_pred = baseline._rollout(history0, offset)  # pyright: ignore[reportPrivateUsage]
                flat_pred = baseline._unproject(eigen_pred.unsqueeze(0))[0]  # pyright: ignore[reportPrivateUsage]
                predicted = flat_pred.reshape(k, d)[:, :pose_dims].contiguous()
                ground_truth = sample.pose[offset, :, :pose_dims].clone().contiguous()
                predictions.append(
                    FuturePoseHorizon(
                        worm_id=sample.worm_id,
                        session_id=str(sample.session_id),
                        horizon_seconds=h_seconds,
                        predicted=predicted,
                        ground_truth=ground_truth,
                    )
                )

    if not predictions:
        return None

    entry = future_pose_metric(
        predictions, producer="transformer_eigenworms", n_bootstrap=n_bootstrap
    )
    notes_suffix = (
        f"  Phase 0 v0 scheme: observe-frame-0 / predict-frame-at-offset "
        f"via TransformerEigenwormsBaseline (autoregressive transformer over "
        f"eigenworm coefficients, fit on the eval cohort's pose). Horizon "
        f"mapping assumes baaiworm clips render at {_CLIP_ASSUMED_FPS:.1f} fps. "
        f"Predicted + ground-truth pose sliced to first 2 coords (x, y) to "
        f"align with the JEPA-side PoseDecoderHead's 2D output so the "
        f"kill_criterion gate compares apples-to-apples. "
        f"Train-test caveat: baseline fit on eval cohort because Phase 0 v0 "
        f"has no separate baseline-eval train cohort wired; a Phase 0 Growth "
        f"follow-up will materialise a held-out train split for the baseline."
    )
    return MetricEntry(
        name=entry.name,
        producer=entry.producer,
        ci=entry.ci,
        sub_entries=entry.sub_entries,
        notes=(entry.notes or "") + notes_suffix,
    )


def reconstruct_state_from_run(run_dir: Path) -> tuple[JEPARunConfig, JEPATrainingState]:
    """Load ``cfg.yaml`` + ``checkpoints/checkpoint.pt`` from a results directory.

    The state is rebuilt via the same :func:`_build_state` runner code
    that produced the checkpoint at training time, so the load is exact
    (architecture-, dim-, head-shape-identical).

    Args:
        run_dir: Path to a ``results/<run-id>/`` directory containing
            both ``config.yaml`` (the run's frozen config) and
            ``checkpoints/checkpoint.pt`` (Story 8.12a-saved weights).

    Returns:
        ``(cfg, state)``: the loaded :class:`JEPARunConfig` and a
        :class:`JEPATrainingState` whose encoder + predictor + warm-
        start heads + optimizer + RNG are all restored to checkpoint
        snapshot.

    Raises:
        FileNotFoundError: ``config.yaml`` or ``checkpoint.pt`` absent.
        ConfigSchemaError: the run's config fails schema validation
            (e.g. a future-incompatible field).
    """
    # Lazy imports: orchestrator runs at eval time, not at every import
    # of wormjepa.eval. Avoids heavy torch / runner deps in the import
    # path of the metric modules.
    from wormjepa.training.checkpointing import load_checkpoint
    from wormjepa.training.runner import _build_state  # pyright: ignore[reportPrivateUsage]

    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "checkpoint.pt"
    if not cfg_path.is_file():
        msg = f"reconstruct_state_from_run: {cfg_path} not found."
        raise FileNotFoundError(msg)
    if not ckpt_path.is_file():
        msg = (
            f"reconstruct_state_from_run: {ckpt_path} not found. "
            f"Was Story 8.12a's save_checkpoint wired when this run ran? "
            f"(Pre-8.12a runs produced no checkpoint.pt.)"
        )
        raise FileNotFoundError(msg)

    # CLI may have prepended a "--seed override" provenance comment line;
    # yaml.safe_load handles that transparently.
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{cfg_path}: top-level YAML must be a mapping, got {type(raw).__name__}."
        raise TypeError(msg)
    cfg = load_config(raw, JEPARunConfig)

    state = _build_state(cfg)
    load_checkpoint(state, ckpt_path)
    logger.info(
        "reconstruct_state_from_run: state restored from %s at step %d",
        ckpt_path,
        state.step,
    )
    return cfg, state


def _build_eval_cache_and_run_partial_r2(
    run_dir: Path,
) -> tuple[MetricEntry | None, _EvalCache | None]:
    """Helper: reconstruct + cache + partial_R² for ablation runs.

    Used by :func:`evaluate_run_with_ablation` to compute partial_R² for
    a *control* run without running the full probe suite. Returns
    (entry, cache) so the caller can reuse the cache for downstream
    diagnostics if needed. Returns (None, None) on any failure.
    """
    from wormjepa.eval.encoder_cache import load_or_build_cache

    try:
        cfg, state = reconstruct_state_from_run(run_dir)
        cache = load_or_build_cache(state, cfg, run_dir=run_dir)
        if len(cache.latents) < 2:
            return None, None
        entry = _run_partial_r2_probe(cache)
        return entry, cache
    except Exception:
        logger.exception("_build_eval_cache_and_run_partial_r2 failed for %s", run_dir)
        return None, None


def evaluate_run_with_ablation(
    primary_run_dir: Path, control_run_dir: Path
) -> tuple[MetricsOutput, GateStatus]:
    """Evaluate ``primary_run_dir`` and append a neural_prior_ablation
    entry computed against ``control_run_dir``.

    Phase 0 v0 ablation: delta_R² = primary.partial_R² - control.partial_R²
    where the control run was trained without the warm_start.neural
    contribution. The orchestrator does NOT verify that the control
    run actually had warm_start.neural=False — that's a caller
    contract (see ``wormjepa eval --primary X --control Y`` docs in
    cli/eval.py).

    The neural_prior_ablation gate's threshold is 0.02 (FR34); the
    delta_R² entry's CI is computed as primary.r2 - control.r2 with
    a Wald-approximated CI by summing variances (independence
    assumption between the two runs' bootstrap distributions —
    appropriate when seed and data shards differ).
    """
    metrics, gate_status = evaluate_run(primary_run_dir)
    control_entry, _ = _build_eval_cache_and_run_partial_r2(control_run_dir)
    if control_entry is None:
        logger.warning(
            "evaluate_run_with_ablation: control run %s produced no partial_R² "
            "entry; neural_prior_ablation gate stays pending.",
            control_run_dir,
        )
        return metrics, gate_status

    primary_entry = next((e for e in metrics.entries if e.name == "neural_probe_partial_r2"), None)
    if primary_entry is None:
        return metrics, gate_status

    # delta_R² = primary - control. CI via Wald-summed variance (rough).
    p_point = float(primary_entry.ci.point)
    c_point = float(control_entry.ci.point)
    p_se = max(
        (float(primary_entry.ci.upper) - float(primary_entry.ci.lower)) / (2 * 1.96),
        1e-12,
    )
    c_se = max(
        (float(control_entry.ci.upper) - float(control_entry.ci.lower)) / (2 * 1.96),
        1e-12,
    )
    delta = p_point - c_point
    delta_se = float(np.sqrt(p_se**2 + c_se**2))
    delta_ci = BootstrapCI(
        point=delta,
        lower=delta - 1.96 * delta_se,
        upper=delta + 1.96 * delta_se,
        n_samples=int(primary_entry.ci.n_samples),
        method="percentile",
    )
    ablation_entry = MetricEntry(
        name="neural_prior_ablation_delta_r2",
        producer="jepa",
        ci=delta_ci,
        sub_entries=[],
        notes=(
            f"Phase 0 v0 ablation: delta_R² = primary.partial_R² ({p_point:.4f}) "
            f"- control.partial_R² ({c_point:.4f}) = {delta:.4f}. "
            f"control_run_dir={control_run_dir.name}. CI via Wald-summed-variance "
            f"normal approximation (independence assumed between the two runs' "
            f"bootstrap distributions; the assumption holds when the runs differ "
            f"by seed + shard). Caller contract: control run must have been "
            f"trained with `warm_start.neural=False`."
        ),
    )
    new_entries = [*metrics.entries, ablation_entry]
    new_metrics = MetricsOutput(run_id=metrics.run_id, entries=new_entries)
    new_gate_status = evaluate_gates(new_metrics)
    new_gate_status = _apply_holm_correction(new_metrics, new_gate_status)
    return new_metrics, new_gate_status


def evaluate_run(run_dir: Path) -> tuple[MetricsOutput, GateStatus]:
    """Run the pre-registered probe suite + gate evaluation on a single run.

    **Skeleton today (Story 8.12c first commit).** Returns a stub
    :class:`MetricsOutput` (empty entries list) so the gate evaluator
    surfaces every primary gate as ``pending`` with informative notes.
    Real probes wire in at 8.12c.1+ commits (one per probe / cohort
    pair). The state-load + cfg-load path is exercised end-to-end now so
    the integration smoke catches breakage early.

    Args:
        run_dir: Path to a ``results/<run-id>/`` directory.

    Returns:
        ``(metrics, gate_status)``: the aggregated metrics output and
        the gate-evaluator verdict.
    """
    from wormjepa.eval.encoder_cache import load_or_build_cache

    cfg, state = reconstruct_state_from_run(run_dir)
    logger.info(
        "evaluate_run: cfg.jepa.vjepa_variant=%s online_init=%s n_steps=%d",
        cfg.jepa.vjepa_variant,
        cfg.jepa.online_init,
        cfg.jepa.n_steps,
    )

    entries: list[MetricEntry] = []

    # 8.12c.1: build (or load cached) eval cohort cache once, dispatch each probe.
    try:
        cache = load_or_build_cache(state, cfg, run_dir=run_dir)
    except Exception:
        logger.exception("evaluate_run: eval cache build failed; gates stay pending")
        cache = None

    if cache is not None and len(cache.latents) >= 2:
        try:
            entries.append(_run_partial_r2_probe(cache))
        except Exception:
            logger.exception("evaluate_run: partial_r2 probe failed")
        try:
            entries.append(_run_session_id_probe(cache))
        except Exception:
            logger.exception("evaluate_run: session_id probe failed")
        try:
            fp_entry = _run_future_pose_probe(state, cfg, cache)
            if fp_entry is not None:
                entries.append(fp_entry)
        except Exception:
            logger.exception("evaluate_run: future_pose probe failed")
        try:
            fp_baseline_entry = _run_transformer_eigenworms_future_pose_probe(cfg, cache)
            if fp_baseline_entry is not None:
                entries.append(fp_baseline_entry)
        except Exception:
            logger.exception("evaluate_run: transformer_eigenworms future_pose probe failed")
        # 8.12c.3+: motif_ari + within_state + neuron_subset diagnostics.
        # Each is wrapped in try/except so a single failure doesn't nuke the
        # whole eval; the probes don't fire any primary gate, only diagnostics.
        try:
            motif_entry = _run_motif_ari_probe(cache)
            if motif_entry is not None:
                entries.append(motif_entry)
        except Exception:
            logger.exception("evaluate_run: motif_ari probe failed")
        try:
            within_entry = _run_within_state_partial_r2_probe(cache)
            if within_entry is not None:
                entries.append(within_entry)
        except Exception:
            logger.exception("evaluate_run: within_state_stratified_r2 probe failed")
        try:
            subset_entry = _run_neuron_subset_partial_r2_probe(cache)
            if subset_entry is not None:
                entries.append(subset_entry)
        except Exception:
            logger.exception("evaluate_run: neuron_subset_partial_r2 probe failed")
    else:
        logger.warning(
            "evaluate_run: eval cache empty / too small (%d clips); "
            "skipping probes — gates surface as pending.",
            0 if cache is None else len(cache.latents),
        )

    # TODO (Story 8.12c.2+): future_pose (needs predictor + pose decoder
    # + horizon scaffolding), motif_ari (needs Flavell behavioural labels,
    # currently deferred), neural_prior_ablation (needs ablation runner
    # against a no-warm-start sibling run), within_state stratified R²,
    # non-trivial-neuron subset R², BAAIWorm-augmentation ablation, plus
    # Holm correction at alpha=0.05 across the full primary + diagnostic
    # gate set.

    metrics = MetricsOutput(run_id=run_dir.name, entries=entries)
    gate_status = evaluate_gates(metrics)
    gate_status = _apply_holm_correction(metrics, gate_status)
    return metrics, gate_status


# ----------------------------------------------------------------------
# Holm correction (Story 8.12c.6 — pre-registered FR34 / NFR17)
# ----------------------------------------------------------------------

#: Pre-registered thresholds + tail-direction per gate, used to derive
#: one-sided p-values from each gate's MetricEntry CI for the Holm pass.
#: Only gates where the verdict logic reduces to "effect exceeds a
#: positive threshold" are in the Holm family today; session_id_at_chance
#: uses a containment test that doesn't map cleanly to a one-sided p,
#: and kill_criterion compares two producers' point estimates without an
#: absolute threshold. Those gates are excluded from the family + flagged
#: in the Holm note.
_HOLM_GATES: dict[str, tuple[float, str, str]] = {
    # gate name → (threshold, direction "above"/"below", entry name)
    "neural_probe_partial_r2": (0.05, "above", "neural_probe_partial_r2"),
    "neural_prior_ablation": (0.02, "above", "neural_prior_ablation_delta_r2"),
}


def _approx_p_value_from_ci(
    point: float, lower: float, upper: float, threshold: float, direction: str
) -> float:
    """Approximate one-sided p-value from a bootstrap CI via normal approx.

    Standard error ~ (upper - lower) / (2 * 1.96) — Wald-CI inversion.
    Direction "above" tests H0: effect ≤ threshold (small p → effect IS
    above threshold). Direction "below" tests H0: effect ≥ threshold.

    Phase 0 v0 caveat: exact bootstrap p-values would re-resample the
    underlying per-worm vector and count threshold violations directly.
    The normal approximation here is appropriate when the bootstrap
    distribution is roughly symmetric; for skewed distributions a real
    bootstrap p ships at Phase 0 Growth.
    """
    from scipy.stats import norm

    se = max((upper - lower) / (2 * 1.96), 1e-12)
    z = (point - threshold) / se
    if direction == "above":
        return float(norm.cdf(-z))  # P(effect ≤ threshold | normal model)
    return float(norm.cdf(z))


def _apply_holm_correction(
    metrics: MetricsOutput, gate_status: GateStatus, alpha: float = 0.05
) -> GateStatus:
    """Apply Holm correction to gates with directional-threshold tests.

    Appends a multi-line note to ``gate_status.notes`` listing the
    family, raw + Holm-adjusted p-values per gate, and the
    Holm-reject-null decisions. Does NOT modify gate verdicts —
    that's a methodology decision for Phase 0 Growth (the gate
    evaluator's CI-lower-bound logic and Holm's reject decision are
    different inferential machinery; reconciling them needs explicit
    pre-reg alignment).
    """
    from dataclasses import replace

    p_values: list[float] = []
    gate_names: list[str] = []
    for gate, (threshold, direction, entry_name) in _HOLM_GATES.items():
        entry = next(
            (e for e in metrics.entries if e.name == entry_name and e.producer == "jepa"),
            None,
        )
        if entry is None:
            continue
        point = float(entry.ci.point)
        lower = float(entry.ci.lower)
        upper = float(entry.ci.upper)
        # Skip NaN-bearing entries (degenerate cohorts).
        if not (point == point and lower == lower and upper == upper):
            continue
        p = _approx_p_value_from_ci(point, lower, upper, threshold, direction)
        p_values.append(p)
        gate_names.append(gate)

    if not p_values:
        return gate_status

    result = apply_correction(p_values, method="holm", alpha=alpha)
    new_notes = list(gate_status.notes)
    new_notes.append(
        f"Holm correction at alpha={alpha} over directional-threshold gates "
        f"(family: {gate_names}; session_id_at_chance + kill_criterion excluded "
        f"— see _HOLM_GATES rationale in eval/orchestrator.py):"
    )
    for gate, p, p_adj, rej in zip(
        gate_names, p_values, result.corrected_pvalues, result.reject, strict=True
    ):
        new_notes.append(f"  {gate}: p={p:.4f}, holm_adj_p={p_adj:.4f}, reject_null={rej}")
    new_notes.append(
        "(Phase 0 v0: p-values approximated from CI via normal approx. Exact "
        "bootstrap p-values that re-resample per-worm vectors land at Phase 0 "
        "Growth alongside the pose-TCN kinematic baseline.)"
    )
    return replace(gate_status, notes=new_notes)


def evaluate_sweep(
    run_dirs: list[Path],
) -> tuple[list[MetricsOutput], list[GateStatus], SweepSummary]:
    """Cross-run (cross-seed) aggregator. Returns per-run metrics + verdicts
    plus a :class:`SweepSummary` aggregate.

    The summary captures the NFR9 cross-seed-spread quantities for each
    primary gate: point estimate per seed and (mean, min, max) across
    seeds. The summary does NOT attempt a CI over CIs — combining
    per-seed bootstrap distributions into a single seed-spread-aware
    CI is its own methodology question (Phase 0 Growth follow-up).

    A gate's "consensus_verdict" is:
      - "cleared" if every seed returned "cleared"
      - "fired" if any seed returned "fired"
      - "pending" if every seed was pending
      - "split" otherwise (mixed cleared/pending without firing)
    """
    metrics: list[MetricsOutput] = []
    gates: list[GateStatus] = []
    for run_dir in run_dirs:
        m, g = evaluate_run(run_dir)
        metrics.append(m)
        gates.append(g)
    summary = _summarise_sweep(run_dirs, metrics, gates)
    return metrics, gates, summary


@dataclass(frozen=True, slots=True)
class _GateSeedSpread:
    """Per-gate cross-seed summary."""

    gate: str
    per_seed_verdict: list[str]
    consensus_verdict: str
    per_seed_point: list[float]  # NaN where the gate's metric is missing
    point_mean: float
    point_min: float
    point_max: float


@dataclass(frozen=True, slots=True)
class SweepSummary:
    """Aggregate of :func:`evaluate_sweep`.

    Attributes:
        run_ids: parallel to ``per_seed_verdict[*]`` lists.
        per_gate: one :class:`_GateSeedSpread` entry per gate name the
            evaluator emitted across the sweep.
    """

    run_ids: list[str]
    per_gate: list[_GateSeedSpread]


# Map gate-evaluator gate name → MetricsOutput entry name to find the
# matching point estimate. Only the gates whose verdicts come from a
# single MetricEntry's top-level CI are listed; future_pose's verdict
# (kill_criterion) depends on a sub-entry + a sibling baseline entry
# and is handled with a per-sub-entry path below.
_GATE_TO_ENTRY: dict[str, str] = {
    "neural_probe_partial_r2": "neural_probe_partial_r2",
    "neural_prior_ablation": "neural_prior_ablation_delta_r2",
    "session_id_at_chance": "session_id_classifier",
}


def _point_for_gate(metrics: MetricsOutput, gate: str) -> float:
    """Return the per-seed point estimate for the given gate, or NaN."""
    entry_name = _GATE_TO_ENTRY.get(gate)
    if entry_name is None:
        return float("nan")
    for e in metrics.entries:
        if e.name == entry_name:
            return float(e.ci.point)
    if gate == "kill_criterion":
        # kill_criterion reads the "1s" sub-entry of future_pose (jepa).
        for e in metrics.entries:
            if e.name == "future_pose" and e.producer == "jepa":
                for sub in e.sub_entries:
                    if sub.key == "1s":
                        return float(sub.ci.point)
    return float("nan")


def _consensus(per_seed: list[str]) -> str:
    s = set(per_seed)
    if s == {"cleared"}:
        return "cleared"
    if "fired" in s:
        return "fired"
    if s == {"pending"}:
        return "pending"
    return "split"


def _summarise_sweep(
    run_dirs: list[Path],
    metrics: list[MetricsOutput],
    gates: list[GateStatus],
) -> SweepSummary:
    if not gates:
        return SweepSummary(run_ids=[d.name for d in run_dirs], per_gate=[])
    gate_names = sorted(gates[0].gates)
    spreads: list[_GateSeedSpread] = []
    for gate in gate_names:
        per_seed_verdict = [g.gates.get(gate, "pending") for g in gates]
        per_seed_point = [_point_for_gate(m, gate) for m in metrics]
        finite = [p for p in per_seed_point if p == p]  # NaN != NaN; this filters NaN
        if finite:
            point_mean = float(np.mean(finite))
            point_min = float(np.min(finite))
            point_max = float(np.max(finite))
        else:
            point_mean = point_min = point_max = float("nan")
        spreads.append(
            _GateSeedSpread(
                gate=gate,
                per_seed_verdict=per_seed_verdict,
                consensus_verdict=_consensus(per_seed_verdict),
                per_seed_point=per_seed_point,
                point_mean=point_mean,
                point_min=point_min,
                point_max=point_max,
            )
        )
    return SweepSummary(run_ids=[d.name for d in run_dirs], per_gate=spreads)
