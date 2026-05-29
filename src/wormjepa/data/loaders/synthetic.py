"""Synthetic in-memory dataset loader.

Useful for smoke tests, baseline development, and end-to-end pipeline
validation before real data loaders (Stories 2.4-2.8) come online. Produces
deterministic :class:`DatasetSample` instances given a seed.

This is **not** a real dataset — it does not pretend to model worm behavior
accurately. It exists so that downstream code paths (training loop, baseline
fit/predict, evaluation, reporting) can be exercised end-to-end on something
that obeys the :class:`DatasetSample` contract.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID


class SyntheticLoader:
    """Yields deterministic synthetic ``DatasetSample`` instances.

    Each worm gets a randomly-initialized "pose state" that evolves over
    frames via Brownian motion. Video clips are gradient-coded views of the
    pose. Neural activity is a low-dimensional linear projection of pose
    (so neural-decoding probes have non-zero signal in tests).

    Args:
        n_worms: How many distinct worms to emit.
        clips_per_worm: How many clips per worm.
        clip_frames: Frames per clip (T dimension).
        image_size: ``(H, W)`` of each video frame.
        n_keypoints: Number of pose keypoints (K dimension).
        n_neurons: Number of neural-activity dimensions.
        seed: Master RNG seed for full reproducibility.
    """

    def __init__(
        self,
        n_worms: int = 4,
        clips_per_worm: int = 3,
        clip_frames: int = 8,
        image_size: tuple[int, int] = (32, 32),
        n_keypoints: int = 10,
        n_neurons: int = 16,
        *,
        seed: int = 0,
    ) -> None:
        self.n_worms = n_worms
        self.clips_per_worm = clips_per_worm
        self.clip_frames = clip_frames
        self.image_size = image_size
        self.n_keypoints = n_keypoints
        self.n_neurons = n_neurons
        self.seed = seed

    def __iter__(self) -> Iterator[DatasetSample]:
        gen = torch.Generator().manual_seed(self.seed)
        h, w = self.image_size
        # Linear projection from pose -> neural, fixed across the loader.
        proj = torch.randn((self.n_keypoints * 2, self.n_neurons), generator=gen)
        for worm_idx in range(self.n_worms):
            # Per-worm phase + travel direction so the worm-shaped video has
            # visible motion. Phase is a random offset on the sinusoid that
            # parameterises the worm body; centre is the head pixel position
            # at t=0.
            phase = float(torch.rand((), generator=gen).item()) * 6.283
            centre = torch.tensor(
                [
                    float(w) * 0.25 + float(torch.rand((), generator=gen).item()) * (w * 0.5),
                    float(h) * 0.5 + float(torch.randn((), generator=gen).item()) * (h * 0.1),
                ]
            )
            for clip_idx in range(self.clips_per_worm):
                pose = _build_worm_pose(
                    n_frames=self.clip_frames,
                    n_keypoints=self.n_keypoints,
                    image_size=(h, w),
                    phase=phase + float(clip_idx),
                    centre=centre,
                    gen=gen,
                )
                video = _rasterise_worm(pose, image_size=(h, w), gen=gen)
                # Neural: linear projection of flattened pose, plus noise.
                pose_flat = pose.reshape(self.clip_frames, -1)
                neural = (
                    pose_flat @ proj
                    + torch.randn((self.clip_frames, self.n_neurons), generator=gen) * 0.05
                )
                yield DatasetSample(
                    video_clip=video,
                    pose=pose,
                    neural=neural,
                    worm_id=WormID(f"synth_w{worm_idx:03d}"),
                    session_id=SessionID(f"synth_w{worm_idx:03d}_s{clip_idx:02d}"),
                    source_dataset=SourceDataset("synthetic"),
                )


def _build_worm_pose(
    n_frames: int,
    n_keypoints: int,
    image_size: tuple[int, int],
    phase: float,
    centre: torch.Tensor,
    gen: torch.Generator,
) -> torch.Tensor:
    """Build a (T, K, 2) pose tensor whose keypoints trace a worm body in pixel space.

    The worm is parameterised by an arclength s in [0, 1] along its body;
    the lateral offset is a sine wave with phase advancing in time
    (locomotion), and the centre drifts in a slow random walk so the worm
    visibly swims across the frame.
    """
    h, w = image_size
    body_length = float(min(h, w)) * 0.7
    pose = torch.empty((n_frames, n_keypoints, 2))
    drift = torch.zeros(2)
    for t in range(n_frames):
        # Small per-frame drift so the worm visibly moves.
        drift = drift + torch.randn(2, generator=gen) * 0.5
        s = torch.linspace(0.0, 1.0, n_keypoints)
        # Lateral sinusoid creates the worm's S-shape; phase advances with time
        # to give the worm a swimming gait.
        lateral = torch.sin(2.0 * torch.pi * s * 2.0 + phase + t * 0.6) * (body_length * 0.18)
        # Body axis: x runs along the body; small tilt per frame.
        tilt = 0.15 * torch.sin(torch.tensor(phase + t * 0.3))
        x = s * body_length - body_length * 0.5
        y = lateral + tilt * x
        # Rotate (x, y) so the body axis is mostly horizontal; centre + drift.
        cx = centre[0] + drift[0]
        cy = centre[1] + drift[1]
        pose[t, :, 0] = x + cx
        pose[t, :, 1] = y + cy
    return pose


def _rasterise_worm(
    pose: torch.Tensor, image_size: tuple[int, int], gen: torch.Generator
) -> torch.Tensor:
    """Rasterise a (T, K, 2) pose into a (T, 3, H, W) float32 video in [0, 1].

    Rendering is upsampled internally (oversampling factor 4) so the body
    has sub-pixel-smooth edges; the worm body is drawn as a sequence of
    tapered, intensity-graded discs from head (thick + bright) to tail
    (thin + dim), with a brighter head spot. Result is downsampled with
    box averaging to give visible antialiasing.
    """
    from PIL import Image, ImageDraw

    t_n, k_n, _ = pose.shape
    h, w = image_size
    upsample = 4
    uh, uw = h * upsample, w * upsample

    out = torch.empty((t_n, 3, h, w), dtype=torch.float32)
    for t in range(t_n):
        # Background: faint vertical gradient + per-frame noise. Renders on
        # the downsampled canvas so the noise pattern is at the final
        # resolution (not blurred by the box-average).
        bg = torch.linspace(0.05, 0.20, h, dtype=torch.float32).unsqueeze(1).expand(h, w).clone()
        bg = bg.unsqueeze(0).expand(3, h, w).clone()
        bg += torch.randn(bg.shape, generator=gen) * 0.015

        # Body rasterisation on the upsampled canvas.
        canvas = Image.new("L", (uw, uh), color=0)
        draw = ImageDraw.Draw(canvas)
        for k in range(k_n - 1):
            x0 = float(pose[t, k, 0].item()) * upsample
            y0 = float(pose[t, k, 1].item()) * upsample
            x1 = float(pose[t, k + 1, 0].item()) * upsample
            y1 = float(pose[t, k + 1, 1].item()) * upsample
            # Taper head→tail. Head radius (k=0) thickest; tail (k=K-1) thinnest.
            r_start = _body_radius(k, k_n, upsample)
            r_end = _body_radius(k + 1, k_n, upsample)
            i_start = _body_intensity(k, k_n)
            i_end = _body_intensity(k + 1, k_n)
            _draw_tapered_segment(draw, x0, y0, x1, y1, r_start, r_end, i_start, i_end)
        # Head spotlight: brighter disc at keypoint 0.
        hx = float(pose[t, 0, 0].item()) * upsample
        hy = float(pose[t, 0, 1].item()) * upsample
        head_r = _body_radius(0, k_n, upsample) * 1.1
        draw.ellipse(
            (hx - head_r, hy - head_r, hx + head_r, hy + head_r),
            fill=255,
        )

        # Downsample by box averaging (Image.BOX gives clean antialiasing).
        small = canvas.resize((w, h), resample=Image.Resampling.BOX)
        body = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0)
        # Add the body onto the background; clamp.
        composite = (bg + body.unsqueeze(0).expand(3, h, w) * 0.9).clamp(0.0, 1.0)
        out[t] = composite
    return out


def _body_radius(k: int, n_keypoints: int, upsample: int) -> float:
    """Per-keypoint body radius in upsampled pixels: head thickest, tail thinnest."""
    s = k / max(n_keypoints - 1, 1)
    # Quadratic taper so the head reads as a head, not just slightly thicker.
    radius = (1.0 - 0.8 * s) * 2.2 * upsample
    return max(radius, 0.5 * upsample)


def _body_intensity(k: int, n_keypoints: int) -> float:
    """Per-keypoint body brightness, head brightest."""
    s = k / max(n_keypoints - 1, 1)
    return 1.0 - 0.35 * s


def _draw_tapered_segment(
    draw: object,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    r0: float,
    r1: float,
    i0: float,
    i1: float,
) -> None:
    """Approximate a tapered, intensity-graded line by stamping interpolated discs."""
    # Number of stamps proportional to segment length.
    dx, dy = x1 - x0, y1 - y0
    seg_len = (dx * dx + dy * dy) ** 0.5
    steps = max(int(seg_len), 2)
    for i in range(steps + 1):
        a = i / steps
        cx = x0 + a * dx
        cy = y0 + a * dy
        r = r0 + a * (r1 - r0)
        intensity = i0 + a * (i1 - i0)
        fill = max(0, min(255, int(intensity * 255.0)))
        draw.ellipse(  # type: ignore[attr-defined]
            (cx - r, cy - r, cx + r, cy + r),
            fill=fill,
        )
