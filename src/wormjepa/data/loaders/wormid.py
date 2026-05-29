"""WormID corpus loader — reads local NWB files from a DANDI federation root.

The WormID corpus per Sprague et al. (2025, *Cell Reports Methods*) is a
federation of seven DANDI dandisets harmonised into a unified NWB-format
corpus of 118 worms across five labs. The pre-registered SPEC lives at
:mod:`wormjepa.data.sources.wormid`; the per-lab train/eval cohort policy
lives at ``pre-registration/splits/wormid_train_eval.yaml``.

This loader assumes ``.nwb`` files have already been fetched to disk by
``download.py`` (downstream work) and are organised as
``<local_dandi_root>/<dandiset_id>/<filename>.nwb``. Given that layout it
iterates every file in the active cohort, decodes one ``ImageSeries`` (the
video / image stack), optionally aligns one ``TimeSeries`` of neural traces
to the video frame rate (simple nearest-neighbor resample for v1), and
yields non-overlapping ``DatasetSample`` clips of length ``clip_frames``.

**Cohort filter.** ``cohort="train" | "eval" | "all"`` selects which
dandisets are iterated, using the disjoint partition declared in
``wormid_train_eval.yaml``. The cohort mapping is duplicated here as a
module-level constant rather than parsed from the YAML at every
construction — that keeps the loader free of a YAML dep at import time and
the constant is short enough that a mismatch is trivially auditable.

**What v1 does not do.**

- No DANDI fetch. ``download.py`` (Story 8.x not yet wired) is responsible
  for placing files under ``local_dandi_root``. The loader fails loudly if
  the root is empty or absent.
- No pose extraction. WormID NWB pose conventions vary per lab; pose
  surfaces as ``None`` in every emitted sample until per-lab readers land.
- No per-neuron NeuroPAL label propagation. Neural traces are emitted as
  ``(T, N)`` but the column-to-neuron mapping is not surfaced. The
  downstream warm-start head (FR16) consumes the raw matrix; per-neuron
  identity work belongs to a follow-up story.
- No multi-channel video discovery. The first ``ImageSeries`` found in
  ``acquisition`` is used. Files with multiple imaging channels merge to
  the channel-1 stream for v1.

If an NWB file lacks a usable ``ImageSeries`` or otherwise fails to parse,
the loader logs a warning and skips it — one malformed file in 118 worms
must not crash the whole iterator.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from pynwb import NWBHDF5IO
from pynwb.base import TimeSeries
from pynwb.image import ImageSeries

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID
from wormjepa.data.sources.wormid import SPEC

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# pynwb does not ship type stubs; pyright-strict cannot reason about NWBFile,
# ImageSeries, or TimeSeries dynamic attributes (acquisition, processing, rate,
# data, subject, ...). Treat the parsed file and its time-series children as
# ``Any`` at the boundary — the runtime isinstance checks and the deliberate
# small surface area we touch are the real safety net here.
NWBFileLike = Any
TimeSeriesLike = Any


# Per-lab dandiset partition. Mirrors pre-registration/splits/wormid_train_eval.yaml.
# Duplicated here intentionally: the YAML is the canonical methodological commitment;
# this constant is the implementation detail that *consumes* it. Mismatch would be
# caught by tests + pre-commit lock-check; both authorities sit in the same repo.
_TRAIN_DANDISETS: frozenset[str] = frozenset({"000472", "000541", "000565", "000692", "000715"})
_EVAL_DANDISETS: frozenset[str] = frozenset({"000714", "000776"})

Cohort = Literal["train", "eval", "all"]


class WormIDLoader:
    """Iterates over WormID NWB files under a local DANDI federation root.

    Args:
        local_dandi_root: Directory containing downloaded ``.nwb`` files
            organised as ``<dandiset_id>/<filename>.nwb``. The loader looks
            for the dandiset subdirectories listed in
            :data:`wormjepa.data.sources.wormid.SPEC.dandisets`.
        cohort: Which lab cohort to iterate. ``"train"`` restricts to the
            train dandisets per the split YAML; ``"eval"`` to the eval
            dandisets; ``"all"`` to the union.
        clip_frames: Frames per emitted clip (T dimension). Default 8.
        image_size: ``(H, W)`` to bilinearly resize each frame to. ``None``
            keeps the source resolution. Defaults to ``(64, 64)`` so the
            shape matches the smoke-config encoder by default.

    Raises:
        DatasetIntegrityError: ``local_dandi_root`` does not exist, is not
            a directory, or contains no ``.nwb`` file for any dandiset in
            the active cohort.
    """

    def __init__(
        self,
        local_dandi_root: Path | str,
        cohort: Cohort = "all",
        clip_frames: int = 8,
        image_size: tuple[int, int] | None = (64, 64),
    ) -> None:
        self.local_dandi_root = Path(local_dandi_root)
        self.cohort: Cohort = cohort
        self.clip_frames = clip_frames
        self.image_size = image_size
        self._active_dandisets = _select_dandisets(cohort)

    def __iter__(self) -> Iterator[DatasetSample]:
        if not self.local_dandi_root.exists() or not self.local_dandi_root.is_dir():
            msg = (
                f"WormIDLoader: local_dandi_root not found or not a directory: "
                f"{self.local_dandi_root}"
            )
            raise DatasetIntegrityError(msg)

        files = list(self._collect_nwb_files())
        if not files:
            msg = (
                f"WormIDLoader: no .nwb files found under {self.local_dandi_root} for "
                f"cohort={self.cohort!r} (active dandisets: {sorted(self._active_dandisets)}). "
                f"Expected layout: <local_dandi_root>/<dandiset_id>/*.nwb"
            )
            raise DatasetIntegrityError(msg)

        for dandiset_id, nwb_path in files:
            try:
                yield from self._iter_file(dandiset_id, nwb_path)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                # Don't crash the whole iterator on one malformed file.
                logger.warning(
                    "WormIDLoader: skipping unreadable NWB file %s (%s: %s)",
                    nwb_path,
                    type(exc).__name__,
                    exc,
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_nwb_files(self) -> Iterable[tuple[str, Path]]:
        """Yield ``(dandiset_id, nwb_path)`` pairs in deterministic order.

        Order: dandisets sorted ascending by id (matches SPEC canonicalization),
        then files sorted ascending by name within each dandiset. This is the
        single source of truth for iteration order — every test that asserts
        determinism relies on it.
        """
        for dandiset_id in sorted(self._active_dandisets):
            subdir = self.local_dandi_root / dandiset_id
            if not subdir.is_dir():
                continue
            # DANDI's canonical layout is `<dandiset>/sub-<X>/sub-<X>_ophys.nwb`
            # (per-subject subdirectory). Story 8.3 originally globbed flat
            # `*.nwb` against fixture data; that misses real DANDI downloads.
            # Story 8.12c-prep (2026-05-18) switched to recursive rglob to
            # catch both layouts deterministically (rglob is sorted explicitly).
            for nwb_path in sorted(subdir.rglob("*.nwb")):
                yield dandiset_id, nwb_path

    def _iter_file(self, dandiset_id: str, nwb_path: Path) -> Iterator[DatasetSample]:
        """Decode one NWB file into one or more clips.

        Streams via ``pynwb.NWBHDF5IO``; the file handle is kept open for the
        duration of iteration so HDF5-backed lazy reads stay valid.
        """
        # pynwb's @docval-decorated __init__ is invisible to pyright; the runtime
        # signature is (path, mode, load_namespaces=False, ...).
        with NWBHDF5IO(str(nwb_path), "r") as io:  # pyright: ignore[reportCallIssue]
            nwbfile: NWBFileLike = io.read()
            video_array, video_rate = _find_first_image_series(nwbfile, nwb_path)
            if video_array is None:
                logger.warning(
                    "WormIDLoader: no ImageSeries found in %s; skipping",
                    nwb_path,
                )
                return

            neural_array, neural_rate = _find_first_neural_time_series(nwbfile)

            worm_subject = (
                nwbfile.subject.subject_id
                if nwbfile.subject is not None and nwbfile.subject.subject_id is not None
                else nwb_path.stem
            )
            worm_id = WormID(f"wormid_{dandiset_id}_{worm_subject}")
            session_id = SessionID(f"{worm_id}_{nwb_path.stem}")

            total_frames = int(video_array.shape[0])
            n_clips = total_frames // self.clip_frames
            if n_clips == 0:
                logger.warning(
                    "WormIDLoader: %s has only %d frames < clip_frames=%d; skipping",
                    nwb_path,
                    total_frames,
                    self.clip_frames,
                )
                return

            for clip_idx in range(n_clips):
                start = clip_idx * self.clip_frames
                stop = start + self.clip_frames
                # HDF5-backed lazy read returns a numpy array on slicing.
                clip_raw = np.asarray(video_array[start:stop])
                video_clip = _normalize_video_clip(clip_raw, self.image_size)

                neural_clip: torch.Tensor | None = None
                if neural_array is not None:
                    neural_clip = _resample_neural_to_video(
                        neural_array=neural_array,
                        neural_rate=neural_rate,
                        video_rate=video_rate,
                        video_start_frame=start,
                        clip_frames=self.clip_frames,
                    )

                yield DatasetSample(
                    video_clip=video_clip,
                    pose=None,  # WormID pose extraction is per-lab; deferred.
                    neural=neural_clip,
                    worm_id=worm_id,
                    session_id=session_id,
                    source_dataset=SourceDataset("wormid"),
                )


def _select_dandisets(cohort: Cohort) -> frozenset[str]:
    """Resolve cohort label to the set of dandiset ids it covers."""
    spec_ids = frozenset(p.dandiset_id for p in SPEC.dandisets)
    if cohort == "train":
        return _TRAIN_DANDISETS & spec_ids
    if cohort == "eval":
        return _EVAL_DANDISETS & spec_ids
    return spec_ids


def _find_first_image_series(
    nwbfile: NWBFileLike, nwb_path: Path
) -> tuple[Any, float] | tuple[None, float]:
    """Locate the first ``ImageSeries`` in ``acquisition`` or processing modules.

    Returns a tuple ``(data_handle, frame_rate)`` where ``data_handle`` is the
    HDF5-backed dataset (lazy) or ``None`` if no ``ImageSeries`` is present.
    Frame rate falls back to 1.0 Hz if neither ``rate`` nor ``timestamps`` is
    populated — sampling-rate-aware downstream code (neural resampling) must
    treat 1.0 as a sentinel.
    """
    candidates: list[TimeSeriesLike] = []
    for ts in nwbfile.acquisition.values():
        if isinstance(ts, ImageSeries):
            candidates.append(ts)
    for module in nwbfile.processing.values():
        for ts in module.data_interfaces.values():
            if isinstance(ts, ImageSeries):
                candidates.append(ts)
    if not candidates:
        return None, 1.0

    # Deterministic pick: first by name to make iteration order reproducible
    # when more than one ImageSeries is present.
    candidates.sort(key=lambda x: cast(str, x.name))
    chosen = candidates[0]
    rate_raw = cast(float | None, chosen.rate)
    rate = float(rate_raw) if rate_raw is not None else 1.0
    if len(candidates) > 1:
        logger.info(
            "WormIDLoader: %s has %d ImageSeries; using %r",
            nwb_path,
            len(candidates),
            chosen.name,
        )
    # chosen.data is an h5py.Dataset (lazy) — np.asarray() materialises on slice.
    return chosen.data, rate


def _find_first_neural_time_series(
    nwbfile: NWBFileLike,
) -> tuple[Any, float] | tuple[None, float]:
    """Locate a 2-D ``TimeSeries`` likely to carry neural activity traces.

    Heuristic for v1: pick the first ``TimeSeries`` (not an ``ImageSeries``)
    with 2-D data in ``acquisition`` or under any processing module. Real
    WormID NWB files use ``RoiResponseSeries`` (a TimeSeries subclass) in an
    ``ophys`` processing module — the isinstance check catches both.
    """
    candidates: list[TimeSeriesLike] = []
    for ts in nwbfile.acquisition.values():
        if (
            isinstance(ts, TimeSeries)
            and not isinstance(ts, ImageSeries)
            and len(ts.data.shape) == 2
        ):
            candidates.append(ts)
    for module in nwbfile.processing.values():
        for ts in module.data_interfaces.values():
            if (
                isinstance(ts, TimeSeries)
                and not isinstance(ts, ImageSeries)
                and len(ts.data.shape) == 2
            ):
                candidates.append(ts)
    if not candidates:
        return None, 1.0

    candidates.sort(key=lambda x: cast(str, x.name))
    chosen = candidates[0]
    rate_raw = cast(float | None, chosen.rate)
    rate = float(rate_raw) if rate_raw is not None else 1.0
    return chosen.data, rate


def _normalize_video_clip(
    clip_raw: np.ndarray[Any, Any], image_size: tuple[int, int] | None
) -> torch.Tensor:
    """Convert a raw NWB image clip to ``(T, C, H, W)`` float32 in [0, 1].

    Accepts input shapes:
      - ``(T, H, W)`` — grayscale; channel axis is inserted and replicated 3x.
      - ``(T, H, W, C)`` — channel-last RGB; permuted to channel-first.
      - ``(T, C, H, W)`` — already correct.
    """
    arr = clip_raw
    if arr.ndim == 3:
        # Grayscale (T, H, W) -> (T, 1, H, W) -> replicate to 3 channels.
        arr = arr[:, np.newaxis, :, :]
        arr = np.broadcast_to(arr, (arr.shape[0], 3, arr.shape[2], arr.shape[3]))
    elif arr.ndim == 4:
        # Detect channel-last (T, H, W, C) by looking at the smallest trailing dim.
        # A real (T, C, H, W) tensor has C ∈ {1, 3, 4}; a real (T, H, W, C) has
        # C ∈ {1, 3, 4} as the last axis. Disambiguate by checking which axis is small.
        if arr.shape[-1] in (1, 3, 4) and arr.shape[1] not in (1, 3, 4):
            # Channel-last -> permute to channel-first.
            arr = np.transpose(arr, (0, 3, 1, 2))
        # If neither axis is obviously small, assume channel-first as-is.
    else:
        msg = f"WormIDLoader: unexpected video clip ndim={arr.ndim}, shape={arr.shape}"
        raise ValueError(msg)

    # If single-channel, replicate to 3 (encoder expects RGB).
    if arr.shape[1] == 1:
        arr = np.broadcast_to(arr, (arr.shape[0], 3, arr.shape[2], arr.shape[3]))

    arr = np.ascontiguousarray(arr)
    tensor = torch.from_numpy(arr).to(torch.float32)

    # Normalise to [0, 1]. uint8 is the common case; for other dtypes fall back
    # to per-clip min-max to keep the range bounded.
    if clip_raw.dtype == np.uint8:
        tensor = tensor / 255.0
    else:
        tmin = float(tensor.min())
        tmax = float(tensor.max())
        tensor = (tensor - tmin) / (tmax - tmin) if tmax > tmin else torch.zeros_like(tensor)

    if image_size is not None:
        h, w = image_size
        tensor = torch.nn.functional.interpolate(
            tensor, size=(h, w), mode="bilinear", align_corners=False
        )
    return tensor


def _resample_neural_to_video(
    neural_array: Any,
    neural_rate: float,
    video_rate: float,
    video_start_frame: int,
    clip_frames: int,
) -> torch.Tensor:
    """Nearest-neighbor align a neural ``(T_n, N)`` trace to the video clip.

    Returns ``(clip_frames, N)`` float32. The neural timestamp of video frame
    ``k`` is ``(video_start_frame + k) / video_rate`` seconds; the nearest
    neural sample is ``round(t * neural_rate)``.
    """
    neural_t_n = int(neural_array.shape[0])
    if neural_t_n == 0:
        return torch.zeros((clip_frames, int(neural_array.shape[1])), dtype=torch.float32)

    out_rows: list[np.ndarray] = []
    for k in range(clip_frames):
        t_sec = (video_start_frame + k) / max(video_rate, 1e-9)
        idx = round(t_sec * neural_rate)
        idx = max(0, min(idx, neural_t_n - 1))
        out_rows.append(np.asarray(neural_array[idx]))
    stacked = np.stack(out_rows, axis=0)
    return torch.from_numpy(np.ascontiguousarray(stacked)).to(torch.float32)
