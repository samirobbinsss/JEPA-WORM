"""Unit tests for WormBehaviorDBLoader (Story 8.5)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SourceDataset
from wormjepa.data.loaders.wormbehavior_db import WormBehaviorDBLoader

# Anchor record ids from src/wormjepa/data/sources/wormbehavior_db.py.
ANCHOR_RECORD_IDS = ("1031550", "1029149")


def _stable_seed(path: Path) -> int:
    return int.from_bytes(hashlib.md5(str(path).encode()).digest()[:4], "big")


def _write_minimal_schafer_h5(
    path: Path,
    *,
    n_frames: int = 16,
    h: int = 16,
    w: int = 16,
    with_pose: bool = True,
    fps: float = 30.0,
) -> None:
    """Write a minimal Schafer-format HDF5 fixture: ``mask`` + optional skeleton."""
    rng = np.random.default_rng(seed=_stable_seed(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        mask = f.create_dataset("mask", data=rng.integers(0, 255, (n_frames, h, w), dtype=np.uint8))
        mask.attrs["fps"] = fps
        if with_pose:
            f.create_dataset(
                "skeleton",
                data=rng.standard_normal((n_frames, 49, 2)).astype(np.float32),
            )


def _materialise_root(tmp_path: Path, record_ids: tuple[str, ...]) -> Path:
    root = tmp_path / "wormbehavior_local"
    root.mkdir()
    for rid in record_ids:
        _write_minimal_schafer_h5(root / rid / f"experiment_{rid}.hdf5", n_frames=16)
    return root


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_loader_raises_when_root_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    loader = WormBehaviorDBLoader(missing, clip_frames=4, image_size=(8, 8))
    with pytest.raises(DatasetIntegrityError, match="local_root not found"):
        list(loader)


def test_loader_raises_when_no_record_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(8, 8))
    with pytest.raises(DatasetIntegrityError, match=r"no \.h5/\.hdf5 files"):
        list(loader)


def test_loader_raises_when_all_files_yield_nothing(tmp_path: Path) -> None:
    """Skeleton-only records skip but should produce a final raise (FR7)."""
    root = tmp_path / "skeleton_only_root"
    root.mkdir()
    rid = ANCHOR_RECORD_IDS[0]
    sub = root / rid
    sub.mkdir()
    # Skeleton but no video → loader skips this record entirely.
    with h5py.File(sub / "skel.hdf5", "w") as f:
        f.create_dataset(
            "skeleton",
            data=np.zeros((8, 49, 2), dtype=np.float32),
        )
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(4, 4))
    with pytest.raises(DatasetIntegrityError):
        list(loader)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_loader_yields_correct_shape_and_pose(tmp_path: Path) -> None:
    root = _materialise_root(tmp_path, ANCHOR_RECORD_IDS)
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    # 2 records * 1 file * 16 frames / 4 = 8 samples.
    assert len(samples) == 8
    s = samples[0]
    assert isinstance(s, DatasetSample)
    assert s.video_clip.shape == (4, 3, 8, 8)
    assert s.video_clip.dtype is torch.float32
    assert s.neural is None
    assert s.behavioral_state is None
    assert s.pose is not None
    assert s.pose.shape == (4, 49, 2)
    assert s.source_dataset == SourceDataset("wormbehavior_db")
    assert s.worm_id.startswith("wormbehavior_")


def test_pose_none_when_skeleton_absent(tmp_path: Path) -> None:
    root = tmp_path / "no_pose"
    root.mkdir()
    rid = ANCHOR_RECORD_IDS[0]
    _write_minimal_schafer_h5(root / rid / "no_pose.hdf5", n_frames=8, with_pose=False)
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(4, 4))
    samples = list(loader)
    assert samples
    assert all(s.pose is None for s in samples)


def test_skeleton_only_record_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A file with skeleton but no video is skipped with INFO; other files still emit."""
    root = tmp_path
    rid_skel = ANCHOR_RECORD_IDS[0]
    rid_good = ANCHOR_RECORD_IDS[1]
    sub_skel = root / rid_skel
    sub_skel.mkdir()
    with h5py.File(sub_skel / "skel_only.hdf5", "w") as f:
        f.create_dataset("skeleton", data=np.zeros((8, 49, 2), dtype=np.float32))
    _write_minimal_schafer_h5(root / rid_good / "good.hdf5", n_frames=8)
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(4, 4))
    with caplog.at_level("INFO", logger="wormjepa.data.loaders._schafer_hdf5"):
        samples = list(loader)
    # Only the real record contributes (skeleton-only record is silently skipped).
    assert len(samples) == 2
    assert any("skeleton but no video" in r.message for r in caplog.records)


def test_pose_nan_is_zero_filled_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """NaN pose frames are zero-filled (rather than propagating NaN gradients downstream)."""
    root = tmp_path
    rid = ANCHOR_RECORD_IDS[0]
    sub = root / rid
    sub.mkdir()
    pose = np.zeros((8, 49, 2), dtype=np.float32)
    pose[3, :, :] = np.nan
    with h5py.File(sub / "nan_pose.hdf5", "w") as f:
        mask = f.create_dataset("mask", data=np.zeros((8, 8, 8), dtype=np.uint8))
        mask.attrs["fps"] = 30.0
        f.create_dataset("skeleton", data=pose)
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(4, 4))
    with caplog.at_level("WARNING", logger="wormjepa.data.loaders._schafer_hdf5"):
        samples = list(loader)
    assert len(samples) == 2
    # No NaN should propagate to the emitted tensor.
    for s in samples:
        assert s.pose is not None
        assert not torch.isnan(s.pose).any()
    assert any("NaN-pose" in r.message for r in caplog.records)


def test_unknown_record_subdirs_are_ignored(tmp_path: Path) -> None:
    """Subdirs whose name is not in the SPEC record set are silently ignored."""
    root = tmp_path / "mixed"
    root.mkdir()
    _write_minimal_schafer_h5(root / "999999" / "stray.hdf5", n_frames=8)
    _write_minimal_schafer_h5(root / ANCHOR_RECORD_IDS[0] / "real.hdf5", n_frames=8)
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(4, 4))
    samples = list(loader)
    assert len(samples) == 2  # only the SPEC-listed record contributes
    assert all(ANCHOR_RECORD_IDS[0] in s.worm_id for s in samples)
    # And no sample is sourced from the unknown record id.
    assert not any("999999" in s.worm_id for s in samples)


def test_iteration_is_deterministic(tmp_path: Path) -> None:
    root = _materialise_root(tmp_path, ANCHOR_RECORD_IDS)
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(4, 4))
    first = [s.worm_id for s in loader]
    second = [s.worm_id for s in loader]
    assert first == second


def test_video_range_normalised(tmp_path: Path) -> None:
    root = _materialise_root(tmp_path, (ANCHOR_RECORD_IDS[0],))
    loader = WormBehaviorDBLoader(root, clip_frames=4, image_size=(4, 4))
    samples = list(loader)
    s = samples[0]
    assert float(s.video_clip.min()) >= 0.0
    assert float(s.video_clip.max()) <= 1.0
