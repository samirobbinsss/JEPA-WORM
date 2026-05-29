"""BAAIWorm synthetic loader — Story 8.7 v1 implementation.

BAAIWorm (Zhao et al. 2024, *Nature Computational Science*) is a code-only
release: a Metaworm/MuJoCo physics simulator + neural-controller model
that emits synthetic worm video + pose + neural activity. The canonical
"source" is a GitHub commit + a generator config file, pinned by SHA in
``pre-registration/MANIFEST.lock`` under the ``github_commit_pin``
canonicalization.

**What v1 does**

- **Local deterministic synthesis.** A self-contained generator written
  against numpy + torch produces BAAIWorm-shaped output (worm body video,
  ``(T, K, 2)`` skeleton, simulated neural traces) from a fixed seed.
  Repeated iteration under the same seed produces bit-identical record
  sequences — the AC's determinism clause.
- **MANIFEST.lock provenance verification.** On first iteration the
  loader reads ``pre-registration/MANIFEST.lock`` and verifies the
  ``baaiworm`` artifact's ``commit_sha`` + ``config_sha256`` match the
  SPEC at :mod:`wormjepa.data.sources.baaiworm`. Drift between the SPEC
  and the lock fails loudly (FR7); the v1 stand-in is methodologically
  honest about which commitment it represents.

**What v1 does NOT do**

- It does NOT invoke the upstream Metaworm/MuJoCo simulator. Adding
  MuJoCo to the project's dependency surface is a multi-day integration
  and is deliberately deferred. The stand-in's output dimensions and
  general shape mirror what Metaworm emits so a future drop-in of the
  real generator (Phase-0-Growth follow-up story) does not require
  changes to downstream training code.
- It does NOT redistribute BAAIWorm bytes. Reproducers who want the real
  upstream output can ``git clone`` the pinned commit; the loader will
  switch to reading their on-disk export when that follow-up lands.

The output's ``source_dataset`` is ``"baaiworm"``. Worm ids are
``f"baaiworm_w{idx:03d}"`` where ``idx`` is the iteration index — stable
under repeated iteration with the same seed.

**Synthetic behavioral_state (Story 8.12c.3+).** Each yielded
:class:`DatasetSample` carries a ``behavioral_state`` ``(T,)`` long
tensor derived from the pose centroid's per-frame velocity. Computation
is purely synthetic — no real behavioural label is available from
BAAIWorm v1's local stand-in. The per-frame centroid (mean over
keypoints) is differenced in time to get a scalar speed; that speed is
binned with :func:`numpy.digitize` against two thresholds (33rd / 66th
percentiles of the clip's speed distribution) into three classes:
``0`` = still, ``1`` = slow, ``2`` = fast. This is consumed by the
eval orchestrator's ``motif_ari`` + ``within_state`` probes as a Phase
0 v0 proxy for the pre-registered Flavell behavioural cohort, which is
gamma-deferred at materialization. Real Flavell behavioural labels
land at Phase 0 Growth.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import numpy as np
import torch

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID
from wormjepa.data.sources.baaiworm import SPEC
from wormjepa.paths import project_root

logger = logging.getLogger(__name__)


class BAAIWormLoader:
    """Iterate over BAAIWorm-shaped synthetic samples (Story 8.7 v1).

    Args:
        n_worms: How many distinct synthetic worms to emit. Each gets a
            unique :class:`WormID` and a unique RNG sub-stream.
        clips_per_worm: Clips emitted per worm.
        clip_frames: Frames per clip (T dimension).
        image_size: ``(H, W)`` of the rendered worm-body video.
        n_keypoints: Number of skeleton keypoints (``K``). Default 49
            matches the upstream BAAIWorm body discretisation.
        n_neurons: Width of the simulated neural-activity vector.
        seed: Master RNG seed. Two loaders constructed with the same seed
            yield bit-identical record sequences.
        verify_provenance: Read ``pre-registration/MANIFEST.lock`` on
            first iteration and fail loudly if the locked ``commit_sha``
            or ``config_sha256`` does not match the SPEC. Default True.
    """

    def __init__(
        self,
        n_worms: int = 4,
        clips_per_worm: int = 3,
        clip_frames: int = 8,
        image_size: tuple[int, int] = (64, 64),
        n_keypoints: int = 49,
        n_neurons: int = 8,
        seed: int = 0,
        verify_provenance: bool = True,
    ) -> None:
        self.n_worms = n_worms
        self.clips_per_worm = clips_per_worm
        self.clip_frames = clip_frames
        self.image_size = image_size
        self.n_keypoints = n_keypoints
        self.n_neurons = n_neurons
        self.seed = seed
        self.verify_provenance = verify_provenance

    def __iter__(self) -> Iterator[DatasetSample]:
        if self.verify_provenance:
            _verify_baaiworm_pin()
        h, w = self.image_size
        gen = torch.Generator().manual_seed(self.seed)
        # Per-worm phase + travel direction so each worm has visible motion;
        # neural projection is a fixed linear map from flattened pose, mirroring
        # SyntheticLoader's pattern. The whole pipeline is deterministic in `gen`.
        proj = torch.randn((self.n_keypoints * 2, self.n_neurons), generator=gen)
        for worm_idx in range(self.n_worms):
            phase = float(torch.rand((), generator=gen).item()) * 6.283
            centre = torch.tensor(
                [
                    float(w) * 0.25 + float(torch.rand((), generator=gen).item()) * (w * 0.5),
                    float(h) * 0.5 + float(torch.randn((), generator=gen).item()) * (h * 0.1),
                ]
            )
            # Per-worm morphology + posture so the corpus has real visual
            # diversity (a single fixed S-curve gave the encoder nothing to
            # spread latents across — see the R3 collapse diagnosis). Each
            # worm gets its own body length, body-wave amplitude, number of
            # waves, and a free orientation (not all horizontal).
            body_length_frac = 0.45 + float(torch.rand((), generator=gen).item()) * 0.35
            lateral_amp = 0.05 + float(torch.rand((), generator=gen).item()) * 0.22
            n_waves = 1.0 + float(torch.rand((), generator=gen).item()) * 1.5
            orientation = float(torch.rand((), generator=gen).item()) * 2.0 * math.pi
            worm_id = WormID(f"baaiworm_w{worm_idx:03d}")
            for clip_idx in range(self.clips_per_worm):
                pose = _build_worm_pose(
                    n_frames=self.clip_frames,
                    n_keypoints=self.n_keypoints,
                    image_size=(h, w),
                    phase=phase + float(clip_idx),
                    centre=centre,
                    gen=gen,
                    body_length_frac=body_length_frac,
                    lateral_amp=lateral_amp,
                    n_waves=n_waves,
                    orientation=orientation,
                )
                video = _rasterise_worm(pose, image_size=(h, w), gen=gen)
                pose_flat = pose.reshape(self.clip_frames, -1)
                neural = (
                    pose_flat @ proj
                    + torch.randn((self.clip_frames, self.n_neurons), generator=gen) * 0.05
                )
                behavioral_state = _velocity_binned_states(pose)
                yield DatasetSample(
                    video_clip=video,
                    pose=pose,
                    neural=neural,
                    worm_id=worm_id,
                    session_id=SessionID(f"{worm_id}_s{clip_idx:02d}"),
                    source_dataset=SourceDataset("baaiworm"),
                    behavioral_state=behavioral_state,
                )


# ---------------------------------------------------------------------------
# Provenance verification
# ---------------------------------------------------------------------------


def _verify_baaiworm_pin() -> None:
    """Confirm the BAAIWorm pin in MANIFEST.lock matches the SPEC.

    Raises :class:`DatasetIntegrityError` on any mismatch. Silently
    succeeds (logs INFO) when the lockfile is absent — this lets the
    loader be exercised in dev workflows that have not yet committed a
    lock.
    """
    lock_path = project_root() / "pre-registration" / "MANIFEST.lock"
    if not lock_path.is_file():
        logger.info(
            "BAAIWormLoader: no MANIFEST.lock at %s; skipping provenance check.",
            lock_path,
        )
        return
    # Local import keeps the loader's import-time cost low for non-verifying paths.
    from wormjepa.manifest.lock import read_manifest

    manifest = read_manifest(lock_path)
    entry = next((a for a in manifest.artifacts if a.dataset == "baaiworm"), None)
    if entry is None:
        msg = (
            "BAAIWormLoader: pre-registration/MANIFEST.lock has no entry for "
            "dataset='baaiworm'. The Story 8.7 AC requires the lock to pin "
            "the generator commit_sha + config_sha256."
        )
        raise DatasetIntegrityError(msg)
    if entry.commit_sha != SPEC.commit_sha:
        msg = (
            f"BAAIWormLoader: MANIFEST.lock commit_sha {entry.commit_sha!r} does "
            f"not match SPEC commit_sha {SPEC.commit_sha!r}. Bumping the pin "
            f"requires a CHANGELOG entry under 'Frozen-artifact changes' "
            f"(FR3/FR48)."
        )
        raise DatasetIntegrityError(msg)
    if entry.config_sha256 != SPEC.config_sha256:
        msg = (
            f"BAAIWormLoader: MANIFEST.lock config_sha256 {entry.config_sha256!r} "
            f"does not match SPEC config_sha256 {SPEC.config_sha256!r}. "
            f"Generator config drift requires a frozen-artifact CHANGELOG entry."
        )
        raise DatasetIntegrityError(msg)


# ---------------------------------------------------------------------------
# Local synthesis primitives (mirrored from SyntheticLoader; kept here so
# BAAIWormLoader is self-contained and a future Metaworm integration can
# replace them without touching SyntheticLoader's mature path)
# ---------------------------------------------------------------------------


def _build_worm_pose(
    n_frames: int,
    n_keypoints: int,
    image_size: tuple[int, int],
    phase: float,
    centre: torch.Tensor,
    gen: torch.Generator,
    *,
    body_length_frac: float,
    lateral_amp: float,
    n_waves: float,
    orientation: float,
) -> torch.Tensor:
    """Build a (T, K, 2) pose tensor with a worm-shaped sinusoid body.

    The worm is parameterised by arclength s ∈ [0, 1]; lateral offset is a
    sine wave with phase advancing in time (swimming locomotion); centre
    drifts in a slow random walk so the worm visibly translates across the
    frame. The per-worm morphology knobs give the corpus visual diversity:

    - ``body_length_frac``: body length as a fraction of ``min(h, w)``.
    - ``lateral_amp``: body-wave amplitude as a fraction of body length.
    - ``n_waves``: number of sinusoid waves along the body.
    - ``orientation``: rotation of the whole worm (radians); without it
      every worm lies horizontally.
    """
    h, w = image_size
    body_length = float(min(h, w)) * body_length_frac
    cos_o, sin_o = math.cos(orientation), math.sin(orientation)
    pose = torch.empty((n_frames, n_keypoints, 2))
    drift = torch.zeros(2)
    for t in range(n_frames):
        drift = drift + torch.randn(2, generator=gen) * 0.5
        s = torch.linspace(0.0, 1.0, n_keypoints)
        lateral = (
            lateral_amp
            * body_length
            * torch.sin(n_waves * 2.0 * torch.pi * s + torch.tensor(phase + t * 0.4))
        )
        along = (s - 0.5) * body_length
        # Rotate the (along, lateral) body frame by `orientation`, translate.
        x = along * cos_o - lateral * sin_o
        y = along * sin_o + lateral * cos_o
        pose[t, :, 0] = centre[0] + drift[0] + x
        pose[t, :, 1] = centre[1] + drift[1] + y
    return pose


def _rasterise_worm(
    pose: torch.Tensor,
    image_size: tuple[int, int],
    gen: torch.Generator,
) -> torch.Tensor:
    """Rasterise a (T, K, 2) pose into a (T, 3, H, W) float video in [0, 1].

    Renders a continuous tapered worm body: each spine keypoint is splatted
    as a soft Gaussian disk whose radius tapers head -> tail (fattest at
    mid-body, like *C. elegans*), and the dense overlapping disks fuse into
    a smooth curved tube. The worm is dark on a light, gently vignetted,
    mildly textured background — a brightfield-microscopy look. Channels
    are identical so the output reads as grayscale-RGB.

    This replaces the Phase 0 v0 placeholder (49 hard white disks on a
    pure-black field). That placeholder gave a V-JEPA encoder near-zero
    visual diversity — near-identical sparse-dot frames map to
    near-identical latents — and was the root cause of the latent collapse
    the 2026-05-19/20 headline runs exhibited (see the R3 diagnosis in
    CHANGELOG.md). A structured worm body gives the encoder real spatial
    variation to spread its latents across.
    """
    n_frames, n_kp, _ = pose.shape
    h, w = image_size
    r_max = max(2.0, min(h, w) * 0.030)
    r_min = max(1.0, min(h, w) * 0.006)

    # Background coordinate grids + a gentle radial vignette.
    yy_full, xx_full = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    vignette = 1.0 - 0.20 * (((xx_full - w / 2.0) / w) ** 2 + ((yy_full - h / 2.0) / h) ** 2)

    video = torch.empty((n_frames, h, w), dtype=torch.float32)
    margin = r_max * 3.0 + 2.0
    for t in range(n_frames):
        spine = pose[t]  # (K, 2) — (x, y) in pixels
        sx = spine[:, 0]
        sy = spine[:, 1]

        # Per-keypoint normalised arclength -> tapered radius. The
        # 4*s*(1-s) parabola peaks at mid-body; the **0.4 keeps the worm
        # fat over most of its length and thins only near head + tail.
        seg = (spine[1:] - spine[:-1]).pow(2).sum(-1).sqrt() if n_kp > 1 else torch.zeros(1)
        cum = torch.cat([torch.zeros(1), seg.cumsum(0)])
        s_kp = cum / cum[-1].clamp(min=1e-6)
        taper = (4.0 * s_kp * (1.0 - s_kp)).clamp(min=0.0) ** 0.4
        r_kp = (r_min + (r_max - r_min) * taper).clamp(min=r_min)  # (K,)

        # Bounding box around the worm (+margin) — rasterise only there so
        # a full-frame 49 x H x W tensor is never materialised.
        x0 = int((sx.min() - margin).clamp(0, w - 1).item())
        x1 = int((sx.max() + margin).clamp(0, w - 1).item()) + 1
        y0 = int((sy.min() - margin).clamp(0, h - 1).item())
        y1 = int((sy.max() + margin).clamp(0, h - 1).item()) + 1

        body = torch.zeros((h, w), dtype=torch.float32)
        if x1 > x0 and y1 > y0:
            bx = xx_full[y0:y1, x0:x1]  # (bh, bw)
            by = yy_full[y0:y1, x0:x1]
            d2 = (bx[None] - sx[:, None, None]) ** 2 + (by[None] - sy[:, None, None]) ** 2
            sigma2 = (r_kp[:, None, None] ** 2).clamp(min=1.0)
            disks = torch.exp(-d2 / (2.0 * sigma2))  # (K, bh, bw)
            body[y0:y1, x0:x1] = disks.max(dim=0).values

        bg = 0.72 * vignette + torch.randn((h, w), generator=gen) * 0.015
        worm_val = 0.20 + torch.randn((h, w), generator=gen) * 0.015
        video[t] = (bg * (1.0 - body) + worm_val * body).clamp(0.0, 1.0)

    return video.unsqueeze(1).expand(n_frames, 3, h, w).contiguous()


def _velocity_binned_states(pose: torch.Tensor) -> torch.Tensor:
    """Per-frame synthetic behavioral state via centroid-velocity binning.

    Returns a ``(T,)`` long tensor with class indices in ``{0, 1, 2}``
    (``still`` / ``slow`` / ``fast``). Derived from the pose centroid's
    per-frame speed magnitude, binned with :func:`numpy.digitize` against
    the 33rd / 66th percentiles of the clip's speed distribution. The
    very-first frame inherits the second frame's bin (no preceding
    frame to difference against) so the output length matches ``T``.

    This is a Phase 0 v0 stand-in for the pre-registered Flavell
    behavioural labels (gamma-deferred at materialization); the
    derivation is documented in the loader's module docstring.
    """
    t = pose.shape[0]
    if t < 2:
        return torch.zeros(t, dtype=torch.long)
    centroid = pose.mean(dim=1).numpy()  # (T, C)
    diff = np.diff(centroid, axis=0)  # (T-1, C)
    speed = np.linalg.norm(diff, axis=1)  # (T-1,)
    # Pad with the first speed so the (T,) output length matches T.
    speed_full = np.concatenate([speed[:1], speed], axis=0)  # (T,)
    # Thresholds: 33rd + 66th percentiles of the in-clip speed distribution.
    q33, q66 = np.quantile(speed_full, [1.0 / 3.0, 2.0 / 3.0])
    # np.digitize with bins = [q33, q66] returns 0 (≤q33), 1 (q33<.≤q66), 2 (>q66).
    bins = np.digitize(speed_full, bins=np.asarray([q33, q66]))
    return torch.from_numpy(np.asarray(bins, dtype=np.int64))
