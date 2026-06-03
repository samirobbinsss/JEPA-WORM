"""Unit tests for OpenWormMovementLoader (Story 8.6)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SourceDataset
from wormjepa.data.loaders.openworm_movement import OpenWormMovementLoader

# Record ids pinned in src/wormjepa/data/sources/openworm_movement.py: the
# N2 anchor 1031550 plus a stratified N2 record. The loader filters local
# record dirs against SPEC membership, so both ids must be present in SPEC.
# (The former mutant anchor 1033265 was dropped in Story 9.6 — video-less,
# silently loader-skipped — so it can no longer stand in as a second record.)
PINNED_RECORD_IDS = ("1031550", "1033159")


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
    root = tmp_path / "openworm_local"
    root.mkdir()
    for rid in record_ids:
        _write_minimal_schafer_h5(root / rid / f"experiment_{rid}.hdf5", n_frames=16)
    return root


def test_loader_raises_when_root_missing(tmp_path: Path) -> None:
    loader = OpenWormMovementLoader(tmp_path / "nope", clip_frames=4, image_size=(8, 8))
    with pytest.raises(DatasetIntegrityError, match="local_root not found"):
        list(loader)


def test_loader_raises_when_no_record_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    loader = OpenWormMovementLoader(root, clip_frames=4, image_size=(8, 8))
    with pytest.raises(DatasetIntegrityError, match=r"no \.h5/\.hdf5 files"):
        list(loader)


def test_loader_yields_correct_shape_and_pose(tmp_path: Path) -> None:
    root = _materialise_root(tmp_path, PINNED_RECORD_IDS)
    loader = OpenWormMovementLoader(root, clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    assert len(samples) == 8  # 2 records * 1 file * 4 clips
    s = samples[0]
    assert isinstance(s, DatasetSample)
    assert s.video_clip.shape == (4, 3, 8, 8)
    assert s.video_clip.dtype is torch.float32
    assert s.neural is None
    assert s.behavioral_state is None
    assert s.pose is not None
    assert s.pose.shape == (4, 49, 2)
    assert s.source_dataset == SourceDataset("openworm_movement")
    assert s.worm_id.startswith("openworm_")
    # Distinct namespace from WormBehaviorDBLoader (avoids cross-dataset collisions).
    assert not s.worm_id.startswith("wormbehavior_")


def test_pose_none_when_skeleton_absent(tmp_path: Path) -> None:
    root = tmp_path / "no_pose"
    root.mkdir()
    rid = PINNED_RECORD_IDS[0]
    _write_minimal_schafer_h5(root / rid / "no_pose.hdf5", n_frames=8, with_pose=False)
    loader = OpenWormMovementLoader(root, clip_frames=4, image_size=(4, 4))
    samples = list(loader)
    assert samples
    assert all(s.pose is None for s in samples)


def test_iteration_is_deterministic(tmp_path: Path) -> None:
    root = _materialise_root(tmp_path, PINNED_RECORD_IDS)
    loader = OpenWormMovementLoader(root, clip_frames=4, image_size=(4, 4))
    first = [s.worm_id for s in loader]
    second = [s.worm_id for s in loader]
    assert first == second
