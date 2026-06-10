"""Atanas/Flavell-2023 loader — Story 8.4 real implementation.

The brain-wide-representations corpus (Atanas et al. 2023, *Cell*) ships
on Zenodo at DOI ``10.5281/zenodo.8150515`` as an archive of per-worm HDF5
files. The substantive payload is the *behavioral-state label series*
paired with neural traces and a video stack — Story 6.3's motif-ARI
metric is the downstream consumer.

This loader follows the same model as :class:`wormjepa.data.loaders.wormid.WormIDLoader`:
files are assumed to be already extracted to a local root by an upstream
fetcher (deliberately out of scope here — a follow-up story owns the
Zenodo unzip). The loader iterates ``.h5`` / ``.hdf5`` files under a
local root, decodes neural / video / behavioral-state datasets via
best-effort name heuristics, and yields non-overlapping clips of
``clip_frames`` length.

Behavioral-state labels surface as :attr:`DatasetSample.behavioral_state`
(a ``(T,)`` long tensor). A file with no behavioral-state dataset is a
hard error — Story 8.4's AC requires labels on every emitted sample.

**What v1 does not do.**

- No Zenodo fetch. The archive must be on disk already.
- No pose extraction (Flavell-2023 doesn't ship pose; ``pose=None``).
- No multi-channel video discovery. The first matching video dataset is
  used; multi-channel files merge to channel-1 for v1.
- No NeuroPAL label propagation (Flavell-2023 is calcium-only at the
  ROI level; per-neuron identity is not surfaced).

**Layout heuristics.** Accepts either flat
(``<local_root>/*.h5``) or per-worm-subdir (``<local_root>/<worm>/*.h5``)
layouts. ``worm_id`` is derived from the file's path *relative to the
local root* so files with the same stem in different subdirs do not
collide (LOWO-CV correctness).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import torch

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SessionID, SourceDataset, WormID
from wormjepa.data.loaders._schafer_hdf5 import verify_file_sha

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Candidate dataset basenames inside an HDF5 file. Ordered by preference; the
# first match wins. The names below cover the conventions seen in
# Atanas-2023 derivatives and the wormwideweb.org export format.
_NEURAL_CANDIDATES: tuple[str, ...] = (
    "neural_activity",
    "traces",
    "neuron_traces",
    "gcamp_traces",
    "F",
    "activity",
)
_BEHAVIOR_CANDIDATES: tuple[str, ...] = (
    "behavioral_state",
    "state_labels",
    "behavior",
    "labels",
    "behavior_state",
)
_VIDEO_CANDIDATES: tuple[str, ...] = (
    "video",
    "frames",
    "image_stack",
    "gfp_video",
    "tiff_stack",
)
_FRAME_RATE_ATTRS: tuple[str, ...] = ("frame_rate", "rate", "fs", "fps")

# Atanas-2023 publishes ~3 Hz volumetric imaging. Used as the default
# frame rate when no attribute is present on the video dataset.
_DEFAULT_FRAME_RATE: float = 3.0


class Flavell2023Loader:
    """Iterates over Atanas/Flavell-2023 HDF5 files under a local root.

    Args:
        local_root: Directory containing extracted ``.h5`` / ``.hdf5`` files.
            Either flat (``<local_root>/*.h5``) or per-worm-subdir
            (``<local_root>/<worm>/*.h5``); both are scanned.
        clip_frames: Frames per emitted clip (T dimension). Default 8.
        image_size: ``(H, W)`` to bilinearly resize each frame to. ``None``
            keeps the source resolution. Defaults to ``(64, 64)`` so the
            shape matches the smoke-config encoder by default.

    Raises:
        DatasetIntegrityError: ``local_root`` does not exist, is not a
            directory, contains no readable HDF5 files, or every discovered
            file failed to decode.
    """

    def __init__(
        self,
        local_root: Path | str,
        clip_frames: int = 8,
        image_size: tuple[int, int] | None = (64, 64),
        expected_file_shas: dict[str, str] | None = None,
    ) -> None:
        """Args (continued):

        expected_file_shas: Optional ``{relative_path: hex_sha256}`` mapping
            verified per file before iteration (FR7). Keys are paths
            *relative to local_root* in posix form (e.g.,
            ``"w01/data.h5"``). When the map is ``None`` or a file's
            relative path is missing from the map, verification is skipped
            and an INFO log records the gap.
        """
        self.local_root = Path(local_root)
        self.clip_frames = clip_frames
        self.image_size = image_size
        self.expected_file_shas = expected_file_shas or {}

    def __iter__(self) -> Iterator[DatasetSample]:
        if not self.local_root.exists() or not self.local_root.is_dir():
            msg = f"Flavell2023Loader: local_root not found or not a directory: {self.local_root}"
            raise DatasetIntegrityError(msg)

        files = list(self._collect_hdf5_files())
        if not files:
            msg = (
                f"Flavell2023Loader: no .h5/.hdf5 files found under {self.local_root}. "
                f"Expected flat (<local_root>/*.h5) or per-worm "
                f"(<local_root>/<worm>/*.h5) layout."
            )
            raise DatasetIntegrityError(msg)

        warned_unverified = False
        emitted = 0
        for path in files:
            rel = path.relative_to(self.local_root).as_posix()
            expected_sha = self.expected_file_shas.get(rel, "")
            if expected_sha:
                # Raises DatasetIntegrityError on mismatch — fail loudly.
                verify_file_sha(path, expected_sha)
            elif self.expected_file_shas and not warned_unverified:
                # Caller provided a partial map; flag the gap once.
                logger.info(
                    "Flavell2023Loader: %s not in expected_file_shas; FR7 verification skipped.",
                    rel,
                )
                warned_unverified = True
            elif not self.expected_file_shas and not warned_unverified:
                logger.info(
                    "Flavell2023Loader: no expected_file_shas provided; "
                    "FR7 per-file verification disabled. Pass "
                    "expected_file_shas={rel: sha256, ...} to enable.",
                )
                warned_unverified = True
            try:
                for sample in self._iter_file(path):
                    emitted += 1
                    yield sample
            except OSError as exc:
                # h5py read failure (truncated archive, permission, etc.) — skip
                # the file but keep iterating others. Bug-class exceptions
                # (ValueError from a normaliser, KeyError from our own dict access)
                # are NOT swallowed — those should surface to the test suite.
                logger.warning(
                    "Flavell2023Loader: skipping unreadable file %s (%s: %s)",
                    path,
                    type(exc).__name__,
                    exc,
                )

        if emitted == 0:
            msg = (
                f"Flavell2023Loader: every discovered file under {self.local_root} "
                f"failed to decode or yielded zero clips. Refusing to return an "
                f"empty iterator (FR7: fail loudly on integrity issues)."
            )
            raise DatasetIntegrityError(msg)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_hdf5_files(self) -> Iterable[Path]:
        """Yield ``.h5`` / ``.hdf5`` files under the local root in deterministic order.

        Order: sorted by relative path. Single source of truth for iteration
        order — tests that assert determinism rely on it.
        """
        candidates: list[Path] = []
        for pat in ("*.h5", "*.hdf5"):
            candidates.extend(self.local_root.glob(pat))
            candidates.extend(self.local_root.glob(f"*/{pat}"))
        return sorted(set(candidates), key=lambda p: p.relative_to(self.local_root).as_posix())

    def _worm_id_for(self, path: Path) -> WormID:
        """Build a worm_id from the file's path relative to the local root.

        Uses the relative path (with separators replaced by ``_`` and the
        ``.h5`` suffix stripped) so files with the same stem in different
        subdirs (e.g., ``w01/data.h5`` vs ``w02/data.h5``) get distinct IDs —
        LOWO-CV would otherwise collapse them onto the same worm.
        """
        rel = path.relative_to(self.local_root).with_suffix("").as_posix()
        return WormID(f"flavell_{rel.replace('/', '_')}")

    def _iter_file(self, path: Path) -> Iterator[DatasetSample]:
        """Decode one HDF5 file into clips. Raises on missing required fields."""
        with h5py.File(path, "r") as f:
            video_ds = _pick_first_dataset(f, _VIDEO_CANDIDATES)
            if video_ds is None:
                msg = (
                    f"Flavell2023Loader: no video dataset in {path} "
                    f"(looked for {_VIDEO_CANDIDATES})."
                )
                raise DatasetIntegrityError(msg)
            neural_ds = _pick_first_dataset(f, _NEURAL_CANDIDATES)
            if neural_ds is None:
                msg = (
                    f"Flavell2023Loader: no neural dataset in {path} "
                    f"(looked for {_NEURAL_CANDIDATES})."
                )
                raise DatasetIntegrityError(msg)
            behavior_ds = _pick_first_dataset(f, _BEHAVIOR_CANDIDATES)
            if behavior_ds is None:
                # Story 8.4 AC requires behavioral-state labels on every sample.
                msg = (
                    f"Flavell2023Loader: no behavioral-state dataset in {path} "
                    f"(looked for {_BEHAVIOR_CANDIDATES}). Story 8.4 requires "
                    f"labels on every emitted sample."
                )
                raise DatasetIntegrityError(msg)

            if neural_ds.ndim != 2:
                msg = (
                    f"Flavell2023Loader: neural dataset in {path} has "
                    f"ndim={neural_ds.ndim} (expected 2)."
                )
                raise DatasetIntegrityError(msg)
            if behavior_ds.ndim != 1:
                msg = (
                    f"Flavell2023Loader: behavior dataset in {path} has "
                    f"ndim={behavior_ds.ndim} (expected 1)."
                )
                raise DatasetIntegrityError(msg)

            frame_rate = _read_frame_rate(video_ds, _DEFAULT_FRAME_RATE)
            t_video = int(video_ds.shape[0])
            t_neural = int(neural_ds.shape[0])
            t_behavior = int(behavior_ds.shape[0])
            # The three series share the same frame rate; if their lengths
            # differ we conservatively iterate the common prefix and surface
            # the mismatch so a real misalignment doesn't go unnoticed.
            t_total = min(t_video, t_neural, t_behavior)
            if not (t_video == t_neural == t_behavior):
                logger.info(
                    "Flavell2023Loader: %s frame-count mismatch "
                    "video=%d neural=%d behavior=%d → iterating common prefix=%d",
                    path,
                    t_video,
                    t_neural,
                    t_behavior,
                    t_total,
                )
            n_clips = t_total // self.clip_frames
            if n_clips == 0:
                logger.warning(
                    "Flavell2023Loader: %s has %d aligned frames < clip_frames=%d; skipping",
                    path,
                    t_total,
                    self.clip_frames,
                )
                return

            worm_id = self._worm_id_for(path)
            session_id = SessionID(f"{worm_id}_{path.stem}")

            # Per-file video min/max for non-uint8 dtypes — computed once at
            # file open and reused across all clips so neighbouring clips share
            # the same intensity scale (per-clip min/max would inject a
            # synthetic clip-boundary artifact into pretraining).
            video_dtype = video_ds.dtype
            video_min, video_max = _compute_video_minmax(video_ds, t_total)

            logger.debug(
                "Flavell2023Loader: %s clips=%d frame_rate=%.3fHz dtype=%s",
                path,
                n_clips,
                frame_rate,
                video_dtype,
            )

            for clip_idx in range(n_clips):
                start = clip_idx * self.clip_frames
                stop = start + self.clip_frames
                # h5py-backed lazy reads — slice per clip rather than loading
                # the whole video / neural / behavior arrays into RAM. Real
                # Atanas-2023 worms are small, but the same pattern keeps the
                # Schafer loaders memory-safe.
                clip_raw = np.asarray(video_ds[start:stop])
                video_clip = _normalize_video_clip(
                    clip_raw,
                    self.image_size,
                    video_min,
                    video_max,
                )
                neural_slice = np.ascontiguousarray(
                    np.asarray(neural_ds[start:stop]).astype(np.float32)
                )
                neural_tensor = torch.from_numpy(neural_slice.copy())

                behavior_slice = np.ascontiguousarray(
                    np.asarray(behavior_ds[start:stop]).astype(np.int64)
                )
                behavioral_state = torch.from_numpy(behavior_slice.copy())

                yield DatasetSample(
                    video_clip=video_clip,
                    pose=None,
                    neural=neural_tensor,
                    worm_id=worm_id,
                    session_id=session_id,
                    source_dataset=SourceDataset("flavell_2023"),
                    behavioral_state=behavioral_state,
                    frame_rate=frame_rate,
                )


def _pick_first_dataset(f: h5py.File, candidate_names: tuple[str, ...]) -> h5py.Dataset | None:
    """Walk the HDF5 file; return the dataset matching the most-preferred candidate.

    Looks at top-level keys first (one direct lookup per candidate, in
    preference order), then falls back to a recursive visit. The recursive
    fallback honours candidate-preference order *and* sorts by full HDF5 path
    so the result is deterministic across h5py builds (``visititems`` order
    is not guaranteed stable).
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


def _read_frame_rate(dataset: h5py.Dataset, default: float) -> float:
    """Read frame rate from a dataset's attrs; fall back to ``default``."""
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
    """Per-file min/max for non-uint8 video, computed in chunks to bound memory.

    Returns ``(0.0, 255.0)`` for uint8 (the normaliser short-circuits and
    divides by 255 directly — these values are unused). For other dtypes the
    range is scanned in chunks of 64 frames to avoid materialising the whole
    stack.
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


def _normalize_video_clip(
    clip_raw: np.ndarray[Any, Any],
    image_size: tuple[int, int] | None,
    file_min: float,
    file_max: float,
) -> torch.Tensor:
    """Convert a raw HDF5 image clip to ``(T, C, H, W)`` float32 in [0, 1].

    Accepts ``(T, H, W)`` (grayscale, replicated to 3 channels) and
    ``(T, H, W, C)`` (channel-last RGB, permuted to channel-first).
    Non-uint8 dtypes are normalised with the **per-file** min/max passed in
    by the caller — neighbouring clips therefore share the same intensity
    scale.
    """
    arr = clip_raw
    if arr.ndim == 3:
        arr = arr[:, np.newaxis, :, :]
        arr = np.broadcast_to(arr, (arr.shape[0], 3, arr.shape[2], arr.shape[3]))
    elif arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4) and arr.shape[1] not in (1, 3, 4):
            arr = np.transpose(arr, (0, 3, 1, 2))
    else:
        msg = f"Flavell2023Loader: unexpected video clip ndim={arr.ndim}, shape={arr.shape}"
        raise ValueError(msg)

    if arr.shape[1] == 1:
        arr = np.broadcast_to(arr, (arr.shape[0], 3, arr.shape[2], arr.shape[3]))

    # .copy() so the resulting tensor owns its memory — the source h5py
    # file handle goes out of scope when the with-block exits, and a
    # zero-stride view from broadcast_to would otherwise alias freed memory.
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
