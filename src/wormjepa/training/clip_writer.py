"""Per-step training-clip writer for the dev-loop GUI's `ClipViewer`.

Writes one ``<step>.mp4`` per logged training step into
``results/<run-id>/clips/`` so the GUI's :mod:`scripts.dev.gui.components.clip_viewer`
can play it via ``st.video``.

Each frame in the MP4 carries:

- The grayscale source video (the encoder's perceptual view).
- Green dots at the ground-truth pose keypoints.
- Red dots at the pose-decoder head's predicted keypoints (when present).
- A semi-transparent magenta tint on frames the predictor was asked to
  reconstruct (the masked frames).

PyAV is used for h264/yuv420p encoding (already a project dependency). The
output is small (a few KB at 64x64 resolution).

Phase 0 v0 dumps one MP4 per gradient step. Long runs that don't need this
verbosity can pass ``clips_dir=None`` to :func:`wormjepa.training.loop.train_jepa`
to disable; future stories may add a stride.
"""

from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Magenta tint applied to masked frames in the per-step MP4. RGB; alpha
# is baked in via alpha_composite when the frame is built. Hex 0xD81B60
# matches the GUI's documented overlay colour.
_MASK_RGBA: tuple[int, int, int, int] = (216, 27, 96, int(0.35 * 255))

# Default playback rate. Synthetic clips run at clip_frames=16 by default;
# 8 fps gives a 2-second loop that reads as motion rather than a slideshow.
_DEFAULT_FPS: int = 8

# Resolution-multiplier so even tiny source frames (32x32, 64x64) render at
# a comfortable size in `st.video`. h264 also requires even-sized dims.
_OUTPUT_UPSCALE: int = 6


def write_step_clip(
    clips_dir: Path,
    step: int,
    video: torch.Tensor,
    mask: torch.Tensor,
    pose: torch.Tensor | None = None,
    predicted_pose: torch.Tensor | None = None,
    fps: int = _DEFAULT_FPS,
) -> None:
    """Write ``<step>.mp4`` under ``clips_dir``.

    Args:
        clips_dir: Output directory. Created if missing.
        step: Training step (used as the filename stem).
        video: ``(1, T, C, H, W)`` or ``(T, C, H, W)`` float tensor in [0, 1].
        mask: ``(1, T)`` or ``(T,)`` 0/1 tensor — 1 = masked (renders with a
            magenta tint baked into the frame).
        pose: Optional ``(1, T, K, 2)`` or ``(T, K, 2)`` ground-truth keypoint
            coordinates. Drawn as **green** dots over each frame.
        predicted_pose: Optional ``(1, T, K, 2)`` or ``(T, K, 2)`` *predicted*
            keypoint coordinates. Drawn as **red** dots so the user can
            watch them converge onto the green dots across training steps.
        fps: Playback rate. Default 2 fps to keep short synthetic clips
            readable.

    Failures are logged at WARNING and swallowed — the training loop must
    not crash because we couldn't write a debug video.
    """
    try:
        clips_dir.mkdir(parents=True, exist_ok=True)
        video_arr = _to_numpy(video)
        mask_arr = _to_numpy(mask)
        pose_arr = _to_numpy(pose) if pose is not None else None
        pred_arr = _to_numpy(predicted_pose) if predicted_pose is not None else None
        frames = _build_video_frames(video_arr, mask_arr, pose_arr, pred_arr)
        _encode_mp4(clips_dir / f"{step}.mp4", frames, fps=fps)
    except (OSError, ValueError, RuntimeError) as exc:
        # av.error.FFmpegError exists at runtime but is not in pyright's stubs;
        # the broader OSError/RuntimeError catches encoding failures too.
        logger.warning(
            "clip_writer: failed to write step %d clip to %s (%s: %s)",
            step,
            clips_dir,
            type(exc).__name__,
            exc,
        )


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(dtype=torch.float32, device="cpu").numpy()


def _video_to_frames(video: np.ndarray) -> np.ndarray:
    """Coerce ``(1, T, C, H, W)`` or ``(T, C, H, W)`` → ``(T, H, W)`` uint8 grayscale."""
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        msg = f"clip_writer: unexpected video ndim={video.ndim}, shape={video.shape}"
        raise ValueError(msg)
    frames = video.mean(axis=1)
    frames = np.clip(frames, 0.0, 1.0)
    return (frames * 255.0).astype(np.uint8)


def _mask_to_flags(mask: np.ndarray, n_frames: int) -> np.ndarray:
    arr = mask.squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.shape[0] != n_frames:
        return np.zeros(n_frames, dtype=bool)
    return arr.astype(bool)


def _pose_to_pixels(pose: np.ndarray | None, n_frames: int, h: int, w: int) -> np.ndarray | None:
    """Map ``(T, K, 2)`` pose to per-frame pixel coordinates, or return None."""
    if pose is None:
        return None
    arr: np.ndarray = pose[0] if pose.ndim == 4 else pose
    if arr.ndim != 3 or arr.shape[-1] != 2:
        return None
    if arr.shape[0] != n_frames:
        return None
    out = np.empty_like(arr)
    finite: np.ndarray = np.asarray(np.isfinite(arr).all(axis=(1, 2)))
    for t in range(n_frames):
        if not bool(finite[t]):
            out[t] = np.nan
            continue
        xy = arr[t]
        xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
        ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
        if (xmax - xmin) > 2.0 and (ymax - ymin) > 2.0:
            out[t, :, 0] = np.clip(xy[:, 0], 0, w - 1)
            out[t, :, 1] = np.clip(xy[:, 1], 0, h - 1)
        elif xmax > xmin and ymax > ymin:
            out[t, :, 0] = (xy[:, 0] - xmin) / (xmax - xmin) * (w - 4) + 2
            out[t, :, 1] = (xy[:, 1] - ymin) / (ymax - ymin) * (h - 4) + 2
        else:
            out[t] = np.array([w / 2.0, h / 2.0])
    return out


def _build_video_frames(
    video: np.ndarray,
    mask: np.ndarray,
    pose: np.ndarray | None,
    predicted_pose: np.ndarray | None,
) -> list[np.ndarray]:
    """Return a list of ``(H, W, 3)`` uint8 RGB frames ready for h264 encoding."""
    raw = _video_to_frames(video)
    t_n, h, w = raw.shape
    mask_flags = _mask_to_flags(mask, t_n)
    pose_px = _pose_to_pixels(pose, t_n, h, w)
    pred_px = _pose_to_pixels(predicted_pose, t_n, h, w)

    out_h = h * _OUTPUT_UPSCALE
    out_w = w * _OUTPUT_UPSCALE
    # h264 needs even dims; the upscale guarantees that for any source >=1.
    if out_h % 2:
        out_h += 1
    if out_w % 2:
        out_w += 1

    frames: list[np.ndarray] = []
    for t in range(t_n):
        rgb = (
            Image.fromarray(raw[t], mode="L")
            .convert("RGBA")
            .resize((out_w, out_h), resample=Image.Resampling.NEAREST)
        )
        if mask_flags[t]:
            overlay = Image.new("RGBA", (out_w, out_h), color=_MASK_RGBA)
            rgb = Image.alpha_composite(rgb, overlay)
        draw = ImageDraw.Draw(rgb)
        if pose_px is not None and np.isfinite(pose_px[t]).all():
            _draw_dots(draw, pose_px[t], scale=_OUTPUT_UPSCALE, fill=(57, 211, 83), radius=3)
        if pred_px is not None and np.isfinite(pred_px[t]).all():
            _draw_dots(draw, pred_px[t], scale=_OUTPUT_UPSCALE, fill=(220, 50, 47), radius=3)
        frames.append(np.asarray(rgb.convert("RGB"), dtype=np.uint8))
    return frames


def _draw_dots(
    draw: ImageDraw.ImageDraw,
    pose_xy: np.ndarray,
    scale: int,
    fill: tuple[int, int, int],
    radius: int,
) -> None:
    """Draw scaled keypoint dots into the upscaled frame canvas."""
    for kx, ky in pose_xy:
        cx, cy = int(kx * scale), int(ky * scale)
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=fill,
        )


def _encode_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Encode a list of RGB frames to ``path`` as h264 yuv420p MP4."""
    if not frames:
        return
    h, w, _ = frames[0].shape
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("h264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Training-evolution rollup
# ---------------------------------------------------------------------------


class RolloutRecorder:
    """Capture predicted-pose frames on a fixed reference clip across training.

    One instance is created at training-loop start (when ``clips_dir`` is
    set). It holds a single reference clip (the first valid sample from the
    training iterator) and, at every step, the predictor's current
    keypoint estimates on that clip. At end-of-training :meth:`save` emits
    ``training_evolution.mp4`` — one frame per training step, all rendered
    on the same reference clip's first frame, so the user can watch the
    red predicted dots converge onto the green ground-truth dots in a
    single short video.

    A per-frame step label is burned into the bottom of each frame using
    PIL's default bitmap font (no font file shipped with the project).
    """

    def __init__(
        self,
        video: torch.Tensor,
        pose: torch.Tensor,
        max_steps: int = 200,
    ) -> None:
        """Snapshot the reference clip.

        Args:
            video: ``(T, C, H, W)`` source clip. The recorder uses frame 0.
            pose: ``(T, K, 2)`` ground-truth pose for the same clip.
            max_steps: Hard cap on how many predictions to record. The
                training loop's typical default is small (a few dozen),
                but pin it so a runaway loop can't exhaust memory.
        """
        self._video_arr = _to_numpy(video)
        self._pose_arr = _to_numpy(pose)
        self._predictions: list[tuple[int, np.ndarray]] = []
        self._max_steps = max_steps

    @property
    def reference_video(self) -> torch.Tensor:
        """``(T, C, H, W)`` snapshot of the reference clip's video, as a tensor."""
        return torch.from_numpy(self._video_arr.copy())

    def record(self, step: int, predicted_pose: torch.Tensor) -> None:
        """Stash ``predicted_pose`` for the current training step."""
        if len(self._predictions) >= self._max_steps:
            return
        self._predictions.append((step, _to_numpy(predicted_pose)))

    def save(self, path: Path, fps: int = _DEFAULT_FPS) -> None:
        """Encode the rollup MP4 to ``path``.

        One video frame per recorded step. The reference clip's frame 0 is
        the canvas; ground-truth dots (green) and that step's predicted
        dots (red) sit on top. A small step label burns into the bottom-left
        corner so you can tell where you are in training.
        """
        if not self._predictions:
            return
        # video_arr is (T, C, H, W) or (1, T, C, H, W); _video_to_frames handles both.
        all_frames = _video_to_frames(self._video_arr)
        t_n, h, w = all_frames.shape
        if t_n == 0:
            return
        ref_frame = all_frames[0]
        # Ground-truth pose for frame 0 (broadcast across all rollup frames).
        gt_px = _pose_to_pixels(self._pose_arr, t_n, h, w)
        gt0 = gt_px[0] if gt_px is not None and np.isfinite(gt_px[0]).all() else None

        out_h = h * _OUTPUT_UPSCALE
        out_w = w * _OUTPUT_UPSCALE
        if out_h % 2:
            out_h += 1
        if out_w % 2:
            out_w += 1

        rendered: list[np.ndarray] = []
        for step, pred_arr in self._predictions:
            pred_px = _pose_to_pixels(pred_arr, t_n, h, w)
            pred0 = pred_px[0] if pred_px is not None and np.isfinite(pred_px[0]).all() else None
            rgb = (
                Image.fromarray(ref_frame, mode="L")
                .convert("RGB")
                .resize((out_w, out_h), resample=Image.Resampling.NEAREST)
            )
            draw = ImageDraw.Draw(rgb)
            if pred0 is not None:
                _draw_dots(draw, pred0, scale=_OUTPUT_UPSCALE, fill=(220, 50, 47), radius=4)
            if gt0 is not None:
                _draw_dots(draw, gt0, scale=_OUTPUT_UPSCALE, fill=(57, 211, 83), radius=3)
            # Step label in the bottom-left corner.
            label = f"step {step}"
            text_y = out_h - 18
            # Outline + fill for readability against any background.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    draw.text((6 + dx, text_y + dy), label, fill=(0, 0, 0))
            draw.text((6, text_y), label, fill=(255, 255, 255))
            rendered.append(np.asarray(rgb, dtype=np.uint8))

        path.parent.mkdir(parents=True, exist_ok=True)
        _encode_mp4(path, rendered, fps=fps)
