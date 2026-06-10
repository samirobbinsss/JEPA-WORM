"""Shared Schafer-lab HDF5 reader for WormBehaviorDB and OpenWormMovement.

Both datasets are per-experiment Zenodo archives derived from the Schafer
worm-tracker (Yemini 2013 / Javer 2018). Their HDF5 files share a common
shape: a per-frame mask/video dataset, a 49-point skeleton coordinate
dataset, an ``fps`` attribute, and various engineered-feature timeseries
we ignore for v1.

This module factors out:

- :class:`SchaferFile` — a lightweight handle around an open HDF5 file
  exposing the picked video dataset, the picked skeleton dataset, and the
  per-file video min/max for non-uint8 normalisation. The handle stays
  open for the duration of one file's iteration so per-clip ``[start:stop]``
  slicing reads from disk lazily — never the whole video stack at once.
- :func:`normalize_video_clip` — ``(T, H, W)`` or ``(T, H, W, C)`` raw
  bytes to ``(T, 3, H, W)`` float32 in [0, 1], optional bilinear resize.
- :class:`SchaferZenodoSubsetLoader` — base loader class parameterised
  on (spec, source_dataset, worm_id_prefix); WormBehaviorDB and
  OpenWormMovement are thin subclasses that pass their SPEC + naming.

Records that have a skeleton but no real video stream are **skipped** —
emitting a rasterised stick-figure as ``video_clip`` would silently
substitute synthetic input for a downstream encoder that the FR17
contract says operates on real video. Skipped count is logged at INFO at
end-of-iteration.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import torch

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID
from wormjepa.data.sources.base import ZenodoSubsetSource

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

_VIDEO_CANDIDATES: tuple[str, ...] = (
    "mask",
    "full_data",
    "video",
    "frames",
)
_SKELETON_CANDIDATES: tuple[str, ...] = (
    "skeleton",
    "skeletons",
)
_FRAME_RATE_ATTRS: tuple[str, ...] = ("fps", "frame_rate", "rate")

_DEFAULT_FRAME_RATE: float = 30.0


@dataclass(frozen=True, slots=True)
class _OpenedSchaferFile:
    """In-memory handle to an open HDF5 file with picked datasets + cached stats.

    Held inside :func:`opened_schafer_file` for the duration of one file's
    iteration. Kept private — callers go through the context manager.
    """

    file: h5py.File
    video: h5py.Dataset
    skeleton: h5py.Dataset | None
    frame_rate: float
    video_min: float
    video_max: float


@contextmanager
def opened_schafer_file(path: Path) -> Generator[_OpenedSchaferFile | None, None, None]:
    """Open one Schafer HDF5 file and yield a handle, or ``None`` to skip.

    Yields ``None`` when the file has no usable video dataset (skeleton-only
    records are skipped — see module docstring for rationale). The h5py file
    handle is closed on context exit, including when the consumer raises.
    """
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        logger.warning("Schafer reader: cannot open %s (%s)", path, exc)
        yield None
        return
    try:
        video_ds = _pick_first_dataset(f, _VIDEO_CANDIDATES)
        skeleton_ds = _pick_first_dataset(f, _SKELETON_CANDIDATES)
        if video_ds is None:
            if skeleton_ds is not None:
                logger.info(
                    "Schafer reader: %s has skeleton but no video dataset; "
                    "skipping (rasterised-skeleton fallback removed — encoder "
                    "operates on real video per FR17).",
                    path,
                )
            else:
                logger.warning(
                    "Schafer reader: %s has neither video nor skeleton; cannot decode.",
                    path,
                )
            yield None
            return

        if video_ds.ndim not in (3, 4):
            logger.warning(
                "Schafer reader: %s video has ndim=%d (expected 3 or 4); skipping",
                path,
                video_ds.ndim,
            )
            yield None
            return

        frame_rate = _read_frame_rate(video_ds, _DEFAULT_FRAME_RATE)
        t_video = int(video_ds.shape[0])
        video_min, video_max = _compute_video_minmax(video_ds, t_video)

        # Per-file skeleton sanity check: shape must be (T, K, 2) or we ignore it.
        valid_skeleton: h5py.Dataset | None = skeleton_ds
        if skeleton_ds is not None:
            sk_shape = skeleton_ds.shape
            if not (len(sk_shape) == 3 and sk_shape[-1] == 2):
                logger.warning(
                    "Schafer reader: %s skeleton shape %s is not (T, K, 2); ignoring pose",
                    path,
                    sk_shape,
                )
                valid_skeleton = None

        yield _OpenedSchaferFile(
            file=f,
            video=video_ds,
            skeleton=valid_skeleton,
            frame_rate=frame_rate,
            video_min=video_min,
            video_max=video_max,
        )
    finally:
        f.close()


def _pick_first_dataset(f: h5py.File, candidate_names: tuple[str, ...]) -> h5py.Dataset | None:
    """Return the dataset matching the most-preferred candidate.

    Top-level lookup honours candidate-preference order. The recursive
    fallback also honours preference order *and* sorts by full HDF5 path so
    the result is deterministic across h5py builds (``visititems`` order is
    not guaranteed stable).
    """
    for name in candidate_names:
        if name in f and isinstance(f[name], h5py.Dataset):
            return cast(h5py.Dataset, f[name])
    found: list[tuple[int, str, h5py.Dataset]] = []
    candidate_index = {name: i for i, name in enumerate(candidate_names)}

    def _visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            base = name.rsplit("/", 1)[-1]
            if base in candidate_index:
                found.append((candidate_index[base], name, obj))

    f.visititems(_visit)
    if not found:
        return None
    found.sort(key=lambda t: (t[0], t[1]))
    return found[0][2]


def verify_file_sha(path: Path, expected: str) -> None:
    """Stream-hash ``path`` and compare to ``expected`` (FR7).

    No-op when ``expected`` is empty. On mismatch raises
    :class:`DatasetIntegrityError` naming the file, the expected digest, and
    the actual digest. Streamed in 1 MiB blocks so very large HDF5 files do
    not balloon RAM during verification.
    """
    if not expected:
        return
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    actual = h.hexdigest()
    if actual != expected.lower():
        msg = (
            f"DatasetIntegrityError: SHA-256 mismatch for {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"FR7: refusing to iterate a file whose bytes do not match the "
            f"pre-registered SPEC."
        )
        raise DatasetIntegrityError(msg)


def _read_frame_rate(dataset: h5py.Dataset, default: float) -> float:
    for attr in _FRAME_RATE_ATTRS:
        if attr in dataset.attrs:
            try:
                return float(cast(Any, dataset.attrs[attr]))
            except (TypeError, ValueError):
                continue
    return default


def _compute_video_minmax(
    video_ds: h5py.Dataset,
    t_total: int,
) -> tuple[float, float]:
    """Per-file video min/max for non-uint8 normalisation.

    Computed in 64-frame chunks so the whole video stack is never resident
    in memory. Returns ``(0.0, 255.0)`` for uint8 (the normaliser
    short-circuits and divides by 255 directly — these values are unused).
    """
    if video_ds.dtype == np.uint8:
        return 0.0, 255.0
    if t_total == 0:
        return 0.0, 1.0
    chunk = 64
    vmin = float("inf")
    vmax = float("-inf")
    for start in range(0, t_total, chunk):
        block = np.asarray(video_ds[start : min(start + chunk, t_total)])
        if block.size == 0:
            continue
        bmin = float(block.min())
        bmax = float(block.max())
        if bmin < vmin:
            vmin = bmin
        if bmax > vmax:
            vmax = bmax
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    return vmin, vmax


def normalize_video_clip(
    clip_raw: np.ndarray,
    image_size: tuple[int, int] | None,
    file_min: float = 0.0,
    file_max: float = 1.0,
) -> torch.Tensor:
    """Convert a raw image clip to ``(T, 3, H, W)`` float32 in [0, 1].

    Accepts ``(T, H, W)`` (grayscale → 3-channel replicate) and
    ``(T, H, W, C)`` (channel-last permuted to channel-first). For non-uint8
    inputs the caller passes the **per-file** min/max so neighbouring clips
    share the same intensity scale.
    """
    arr = clip_raw
    if arr.ndim == 3:
        arr = arr[:, np.newaxis, :, :]
        arr = np.broadcast_to(arr, (arr.shape[0], 3, arr.shape[2], arr.shape[3]))
    elif arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4) and arr.shape[1] not in (1, 3, 4):
            arr = np.transpose(arr, (0, 3, 1, 2))
    else:
        msg = f"Schafer reader: unexpected clip ndim={arr.ndim}, shape={arr.shape}"
        raise ValueError(msg)

    if arr.shape[1] == 1:
        arr = np.broadcast_to(arr, (arr.shape[0], 3, arr.shape[2], arr.shape[3]))

    # .copy() so the resulting tensor owns its memory after the source HDF5
    # file handle closes.
    arr = np.ascontiguousarray(arr).copy()
    tensor = torch.from_numpy(arr).to(torch.float32)
    if clip_raw.dtype == np.uint8:
        tensor = tensor / 255.0
    elif file_max > file_min:
        tensor = (tensor - file_min) / (file_max - file_min)
    else:
        tensor = torch.zeros_like(tensor)
    if image_size is not None:
        h, w = image_size
        tensor = torch.nn.functional.interpolate(
            tensor, size=(h, w), mode="bilinear", align_corners=False
        )
    return tensor


class SchaferZenodoSubsetLoader:
    """Base loader for per-experiment Zenodo subsets in Schafer HDF5 format.

    Subclassed by :class:`wormjepa.data.loaders.wormbehavior_db.WormBehaviorDBLoader`
    and :class:`wormjepa.data.loaders.openworm_movement.OpenWormMovementLoader`.

    Args:
        local_root: Directory layout:
            ``<local_root>/<zenodo_record_id>/*.hdf5`` (or ``*.h5``).
        spec: The :class:`ZenodoSubsetSource` SPEC for this dataset; record
            iteration order matches ``spec.records``.
        source_dataset: Canonical ``SourceDataset`` value (e.g. ``"wormbehavior_db"``).
        worm_id_prefix: Prefix used to construct ``WormID`` strings; ensures
            no overlap with other loaders (e.g. ``"wormbehavior_"``).
        clip_frames: Frames per emitted clip.
        image_size: ``(H, W)`` to bilinearly resize each frame to.

    Raises:
        DatasetIntegrityError: ``local_root`` does not exist, contains no
            readable HDF5 file in any active record subdirectory, or every
            discovered file failed to yield any clips.
    """

    def __init__(
        self,
        local_root: Path | str,
        spec: ZenodoSubsetSource,
        source_dataset: SourceDataset,
        worm_id_prefix: str,
        clip_frames: int = 8,
        image_size: tuple[int, int] | None = (64, 64),
    ) -> None:
        self.local_root = Path(local_root)
        self.spec = spec
        self.source_dataset = source_dataset
        self.worm_id_prefix = worm_id_prefix
        self.clip_frames = clip_frames
        self.image_size = image_size

    def __iter__(self) -> Iterator[DatasetSample]:
        if not self.local_root.exists() or not self.local_root.is_dir():
            msg = (
                f"{type(self).__name__}: local_root not found or not a directory: {self.local_root}"
            )
            raise DatasetIntegrityError(msg)

        files = list(self._collect_files())
        if not files:
            record_ids = [p.zenodo_record_id for p in self.spec.records]
            msg = (
                f"{type(self).__name__}: no .h5/.hdf5 files found under "
                f"{self.local_root} for any active record (records: {record_ids}). "
                f"Expected layout: <local_root>/<zenodo_record_id>/*.hdf5"
            )
            raise DatasetIntegrityError(msg)

        # Per-record SHA lookup for FR7 verification. Records without a
        # populated sha256 in the SPEC are still allowed (backward compat),
        # but each is logged INFO once per loader instance so the FR7
        # enforcement gap is visible.
        record_sha = {r.zenodo_record_id: r.sha256 for r in self.spec.records}
        warned_unverified: set[str] = set()

        emitted = 0
        skipped_skeleton_only = 0
        for record_id, path in files:
            expected_sha = record_sha.get(record_id, "")
            if expected_sha:
                # Raises DatasetIntegrityError on mismatch — fail loudly.
                verify_file_sha(path, expected_sha)
            elif record_id not in warned_unverified:
                logger.info(
                    "%s: record %s has no sha256 in SPEC; FR7 verification "
                    "skipped for files under that record. Populate "
                    "ZenodoRecordPin.sha256 to enable.",
                    type(self).__name__,
                    record_id,
                )
                warned_unverified.add(record_id)
            try:
                pre = emitted
                for sample in self._iter_file(record_id, path):
                    emitted += 1
                    yield sample
                if emitted == pre:
                    # File opened cleanly but yielded nothing — most often a
                    # skeleton-only Schafer record (logged at INFO inside
                    # opened_schafer_file).
                    skipped_skeleton_only += 1
            except OSError as exc:
                # h5py read failure (truncated archive, permission, etc.) —
                # skip but keep iterating others. Bug-class exceptions
                # (ValueError from a normaliser, KeyError from our own dict
                # access) are NOT swallowed.
                logger.warning(
                    "%s: skipping unreadable file %s (%s: %s)",
                    type(self).__name__,
                    path,
                    type(exc).__name__,
                    exc,
                )

        if skipped_skeleton_only > 0:
            logger.info(
                "%s: skipped %d skeleton-only file(s) under %s (no real video stream).",
                type(self).__name__,
                skipped_skeleton_only,
                self.local_root,
            )

        if emitted == 0:
            msg = (
                f"{type(self).__name__}: every discovered file under "
                f"{self.local_root} failed to decode or yielded zero clips. "
                f"Refusing to return an empty iterator (FR7: fail loudly on "
                f"integrity issues)."
            )
            raise DatasetIntegrityError(msg)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_files(self) -> Iterable[tuple[str, Path]]:
        """Yield ``(record_id, path)`` pairs in deterministic order.

        Order: ``spec.records`` order (preserves pre-registered ordering), then
        files sorted by name within each record subdir.
        """
        for record in self.spec.records:
            subdir = self.local_root / record.zenodo_record_id
            if not subdir.is_dir():
                continue
            paths: list[Path] = []
            for pat in ("*.h5", "*.hdf5"):
                paths.extend(subdir.glob(pat))
            for path in sorted(set(paths), key=lambda p: p.name):
                yield record.zenodo_record_id, path

    def _iter_file(self, record_id: str, path: Path) -> Iterator[DatasetSample]:
        with opened_schafer_file(path) as opened:
            if opened is None:
                return
            video_ds = opened.video
            skeleton_ds = opened.skeleton

            t_video = int(video_ds.shape[0])
            total_frames = t_video
            if skeleton_ds is not None:
                t_pose = int(skeleton_ds.shape[0])
                if t_pose != t_video:
                    logger.info(
                        "%s: %s frame-count mismatch video=%d pose=%d → iterating common prefix=%d",
                        type(self).__name__,
                        path,
                        t_video,
                        t_pose,
                        min(t_video, t_pose),
                    )
                    total_frames = min(t_video, t_pose)

            n_clips = total_frames // self.clip_frames
            if n_clips == 0:
                logger.warning(
                    "%s: %s has %d frames < clip_frames=%d; skipping",
                    type(self).__name__,
                    path,
                    total_frames,
                    self.clip_frames,
                )
                return

            worm_id = WormID(f"{self.worm_id_prefix}{record_id}_{path.stem}")
            session_id = SessionID(f"{worm_id}_{path.stem}")
            logger.debug(
                "%s: %s record=%s clips=%d frame_rate=%.3fHz",
                type(self).__name__,
                path,
                record_id,
                n_clips,
                opened.frame_rate,
            )

            for clip_idx in range(n_clips):
                start = clip_idx * self.clip_frames
                stop = start + self.clip_frames
                # Lazy h5py slicing — no whole-video materialisation.
                video_raw = np.asarray(video_ds[start:stop])
                video_clip = normalize_video_clip(
                    video_raw,
                    self.image_size,
                    file_min=opened.video_min,
                    file_max=opened.video_max,
                )
                pose_tensor: torch.Tensor | None = None
                if skeleton_ds is not None:
                    pose_slice = np.asarray(skeleton_ds[start:stop]).astype(np.float32)
                    if np.isnan(pose_slice).any():
                        # Schafer tracking output marks occluded frames with NaN.
                        # Propagating NaN to downstream loss yields NaN gradients;
                        # zero-fill with a per-clip warning so the user knows.
                        nan_count = int(np.isnan(pose_slice).any(axis=(1, 2)).sum())
                        logger.warning(
                            "%s: %s clip %d has %d NaN-pose frames; zero-filling",
                            type(self).__name__,
                            path,
                            clip_idx,
                            nan_count,
                        )
                        pose_slice = np.nan_to_num(pose_slice, nan=0.0)
                    pose_tensor = torch.from_numpy(np.ascontiguousarray(pose_slice).copy())
                yield DatasetSample(
                    video_clip=video_clip,
                    pose=pose_tensor,
                    neural=None,
                    worm_id=worm_id,
                    session_id=session_id,
                    source_dataset=self.source_dataset,
                    frame_rate=opened.frame_rate,
                )
