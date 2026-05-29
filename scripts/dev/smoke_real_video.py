"""Dev-loop smoke: does the JEPA model train on a real C. elegans video?

Bypasses ``wormjepa run`` and its frozen-artifact / MANIFEST.lock machinery.
Runs the core JEPA loss (encoder + EMA target + predictor) for N steps on
clips decoded from a single local video, prints per-step loss, and reports
whether loss decreased first → last.

**Dev-only. Not a reportable run. Not part of pytest.** Pure smoke for
"architecture compiles, forward pass works, gradients flow, loss decreases."

Warm-start heads are skipped entirely because arbitrary internet video has
no pose or neural-activity annotations.

Usage:
    uv run python scripts/dev/smoke_real_video.py PATH_TO_VIDEO.mp4

Example:
    uv run python scripts/dev/smoke_real_video.py tests/fixtures/dev_local/sample.mp4
"""
# T201 (no print) is the project's *production* anti-pattern; print is the
# correct tool for a dev smoke script.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import optim

from wormjepa.data import DatasetSample
from wormjepa.data.loaders.dev_local import DevLocalLoader
from wormjepa.models.ema import EMATarget
from wormjepa.models.encoder import WormJEPAEncoder
from wormjepa.models.masking import random_temporal_mask
from wormjepa.models.predictor import JEPAPredictor


def _resolve_device() -> torch.device:
    """Pick CUDA > MPS (Apple Silicon) > CPU. Smoke runs anywhere."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _jepa_loss(
    online: WormJEPAEncoder,
    target: EMATarget,
    predictor: JEPAPredictor,
    video: torch.Tensor,
    masking_ratio: float,
) -> torch.Tensor:
    """Core JEPA loss: predict EMA-target embeddings at masked time steps."""
    b, t, *_ = video.shape
    online_latent = online(video)
    with torch.no_grad():
        target_latent = target(video)
    mask = random_temporal_mask(n_frames=t, n_batches=b, masking_ratio=masking_ratio).to(
        video.device
    )
    predicted = predictor(online_latent, mask)
    mask_f = mask.unsqueeze(-1).to(predicted.dtype)
    return ((predicted - target_latent) ** 2 * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Path to a PyAV-decodable video.")
    parser.add_argument("--n-steps", type=int, default=8, help="Gradient steps to run.")
    parser.add_argument("--clip-frames", type=int, default=8, help="Frames per clip.")
    parser.add_argument("--img-size", type=int, default=64, help="Square resize H=W=img_size.")
    parser.add_argument("--latent-dim", type=int, default=32, help="Encoder output dim.")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Adam learning rate.")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA target decay.")
    parser.add_argument("--masking-ratio", type=float, default=0.5, help="Frame mask fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Torch seed.")
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    device = _resolve_device()
    print(f"device: {device}")

    online = WormJEPAEncoder(
        model_name="vit_tiny_patch16_224",
        latent_dim=args.latent_dim,
        img_size=args.img_size,
    ).to(device)
    ema = EMATarget(online, decay=args.ema_decay)
    predictor = JEPAPredictor(latent_dim=args.latent_dim, n_layers=1, n_heads=2).to(device)
    opt = optim.Adam(list(online.parameters()) + list(predictor.parameters()), lr=args.lr)

    # No max_clips cap — we cycle the loader when exhausted so we can run
    # more steps than the video has clips. Cycling reuses the same clips with
    # fresh random masks, which is itself an informative noise floor: same
    # content, different mask, observed-variance attributable to the mask.
    def _cycle() -> Iterator[DatasetSample]:
        while True:
            yield from DevLocalLoader(
                args.video,
                clip_frames=args.clip_frames,
                image_size=(args.img_size, args.img_size),
            )

    online.train()
    predictor.train()
    losses: list[float] = []
    samples_iter = _cycle()
    for step in range(args.n_steps):
        try:
            sample = next(samples_iter)
        except StopIteration:
            print(f"warning: cycle iterator unexpectedly exhausted at step {step}")
            break
        video = sample.video_clip.unsqueeze(0).to(device)  # (1, T, C, H, W)

        opt.zero_grad()
        loss = _jepa_loss(online, ema, predictor, video, args.masking_ratio)
        loss.backward()
        opt.step()
        ema.update(online)

        value = float(loss.detach().cpu())
        losses.append(value)
        print(f"step {step:3d}  loss={value:.6f}")

    if len(losses) < 2:
        print("error: need at least 2 steps to assess trend", file=sys.stderr)
        return 3

    delta = losses[-1] - losses[0]
    direction = "decreased" if delta < 0 else ("increased" if delta > 0 else "flat")
    print(f"\nfirst={losses[0]:.6f}  last={losses[-1]:.6f}  delta={delta:+.6f}  ({direction})")
    print("note: a few steps on one short clip is not convergent training.")
    print("'decreased' = expected sanity signal; 'increased' on this scale also possible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
