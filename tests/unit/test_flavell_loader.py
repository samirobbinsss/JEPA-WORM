"""Unit tests for :class:`wormjepa.data.loaders.flavell_2023.Flavell2023Loader`.

Fixtures are minimal in-process HDF5 files written via ``h5py`` so the
loader's full path runs end-to-end without Zenodo access. Real-data smoke
lives at ``tests/smoke/test_flavell_loader.py`` (auto-skipped when no
fixture directory is provided).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SourceDataset
from wormjepa.data.loaders.flavell_2023 import Flavell2023Loader


def _stable_seed(path: Path) -> int:
    """Deterministic per-path seed independent of PYTHONHASHSEED."""
    return int.from_bytes(hashlib.md5(str(path).encode()).digest()[:4], "big")


def _write_minimal_flavell_h5(
    path: Path,
    *,
    n_frames: int = 16,
    n_neurons: int = 5,
    h: int = 16,
    w: int = 16,
    include_neural: bool = True,
    include_behavior: bool = True,
    include_video: bool = True,
    frame_rate: float = 3.0,
) -> None:
    """Write a minimal Flavell-format HDF5 fixture with grayscale (T, H, W) video."""
    rng = np.random.default_rng(seed=_stable_seed(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if include_neural:
            f.create_dataset(
                "neural_activity",
                data=rng.standard_normal((n_frames, n_neurons)).astype(np.float32),
            )
        if include_behavior:
            f.create_dataset(
                "behavioral_state",
                data=rng.integers(0, 4, size=n_frames, dtype=np.int32),
            )
        if include_video:
            video = f.create_dataset(
                "video",
                data=rng.integers(0, 255, size=(n_frames, h, w), dtype=np.uint8),
            )
            video.attrs["frame_rate"] = frame_rate


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_loader_raises_when_root_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    loader = Flavell2023Loader(missing, clip_frames=4, image_size=(8, 8))
    with pytest.raises(DatasetIntegrityError, match="local_root not found"):
        list(loader)


def test_loader_raises_when_root_empty(tmp_path: Path) -> None:
    root = tmp_path / "empty_root"
    root.mkdir()
    loader = Flavell2023Loader(root, clip_frames=4, image_size=(8, 8))
    with pytest.raises(DatasetIntegrityError, match=r"no \.h5/\.hdf5 files"):
        list(loader)


def test_loader_raises_when_all_files_fail(tmp_path: Path) -> None:
    """All files missing required dataset → raise rather than yield empty (FR7)."""
    root = tmp_path
    for name in ("a.h5", "b.h5"):
        with h5py.File(root / name, "w") as f:
            f.create_dataset("foo", data=np.zeros(4))
    loader = Flavell2023Loader(root, clip_frames=4, image_size=(4, 4))
    with pytest.raises(DatasetIntegrityError):
        list(loader)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_loader_yields_correct_shape(tmp_path: Path) -> None:
    """One file, 16 frames, clip_frames=4 → 4 samples with documented shapes."""
    root = tmp_path / "flat"
    root.mkdir()
    _write_minimal_flavell_h5(root / "worm0.h5", n_frames=16, n_neurons=5)
    loader = Flavell2023Loader(root, clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    assert len(samples) == 4
    s = samples[0]
    assert isinstance(s, DatasetSample)
    assert s.video_clip.shape == (4, 3, 8, 8)
    assert s.video_clip.dtype is torch.float32
    assert float(s.video_clip.min()) >= 0.0
    assert float(s.video_clip.max()) <= 1.0
    # Neural is now plain (T, N) — labels surface separately on behavioral_state.
    assert s.neural is not None
    assert s.neural.shape == (4, 5)
    assert s.behavioral_state is not None
    assert s.behavioral_state.shape == (4,)
    assert s.behavioral_state.dtype is torch.int64
    assert s.pose is None
    assert s.source_dataset == SourceDataset("flavell_2023")
    assert s.worm_id.startswith("flavell_")
    assert not s.worm_id.startswith("wormid_")
    # frame_rate propagates from the video dataset's frame_rate attr.
    assert s.frame_rate == 3.0


def test_frame_rate_falls_back_to_default_when_attr_absent(tmp_path: Path) -> None:
    """No frame_rate attr → the loader's nominal ~3 Hz default is surfaced."""
    root = tmp_path / "no_rate"
    root.mkdir()
    rng = np.random.default_rng(seed=7)
    with h5py.File(root / "w.h5", "w") as f:
        f.create_dataset("neural_activity", data=rng.standard_normal((8, 3)).astype(np.float32))
        f.create_dataset("behavioral_state", data=rng.integers(0, 4, size=8, dtype=np.int32))
        # No frame_rate attr on the video dataset.
        f.create_dataset("video", data=np.zeros((8, 8, 8), dtype=np.uint8))
    loader = Flavell2023Loader(root, clip_frames=4, image_size=(4, 4))
    samples = list(loader)
    assert samples
    assert samples[0].frame_rate == 3.0


def test_behavioral_state_surfaces_as_dedicated_field(tmp_path: Path) -> None:
    """Labels are exposed as DatasetSample.behavioral_state (long tensor, shape (T,))."""
    root = tmp_path
    rng = np.random.default_rng(seed=42)
    n_frames = 8
    expected_labels = rng.integers(0, 4, size=n_frames, dtype=np.int32)
    with h5py.File(root / "w.h5", "w") as f:
        f.create_dataset(
            "neural_activity", data=rng.standard_normal((n_frames, 3)).astype(np.float32)
        )
        f.create_dataset("behavioral_state", data=expected_labels)
        vid = f.create_dataset("video", data=np.zeros((n_frames, 8, 8), dtype=np.uint8))
        vid.attrs["frame_rate"] = 3.0
    loader = Flavell2023Loader(root, clip_frames=4, image_size=(4, 4))
    samples = list(loader)
    assert len(samples) == 2
    s0, s1 = samples
    assert s0.behavioral_state is not None
    assert s1.behavioral_state is not None
    np.testing.assert_array_equal(s0.behavioral_state.numpy().astype(np.int32), expected_labels[:4])
    np.testing.assert_array_equal(
        s1.behavioral_state.numpy().astype(np.int32), expected_labels[4:8]
    )


def test_worm_id_namespace_does_not_collide_with_wormid(tmp_path: Path) -> None:
    _write_minimal_flavell_h5(tmp_path / "subject-0001.h5", n_frames=8)
    loader = Flavell2023Loader(tmp_path, clip_frames=4, image_size=(4, 4))
    ids = {s.worm_id for s in loader}
    assert all(rid.startswith("flavell_") for rid in ids)
    # The wormid loader uses prefix "wormid_"; no overlap is possible.
    assert not any(rid.startswith("wormid_") for rid in ids)


def test_worm_id_disjoint_from_wormid_loader_empirically(tmp_path: Path) -> None:
    """Construct the WormID prefix scheme and confirm no Flavell ID can match it.

    Story 8.4 AC says worm IDs must not overlap with the WormID corpus naming.
    Beyond the prefix convention check, instantiate both loaders' ID schemes
    and verify the produced sets are disjoint.
    """
    # Match wormid.py's WormID(f"wormid_{dandiset_id}_{worm_subject}") scheme.
    wormid_corpus_ids = {
        f"wormid_{dandiset}_{subj}"
        for dandiset in ("000472", "000776", "000714")
        for subj in ("worm-001", "worm-002")
    }
    _write_minimal_flavell_h5(tmp_path / "worm-001.h5", n_frames=8)
    _write_minimal_flavell_h5(tmp_path / "wormid_000472_worm-001.h5", n_frames=8)
    loader = Flavell2023Loader(tmp_path, clip_frames=4, image_size=(4, 4))
    flavell_ids = {s.worm_id for s in loader}
    # Even the deliberately-evil filename gets the "flavell_" prefix, so
    # collision with wormid scheme is structurally impossible.
    assert flavell_ids.isdisjoint(wormid_corpus_ids)


def test_per_worm_subdir_layout_disambiguates_collisions(tmp_path: Path) -> None:
    """``<root>/<worm>/*.h5`` layout: same-stem files in different subdirs get distinct IDs."""
    root = tmp_path
    for worm in ("w01", "w02"):
        _write_minimal_flavell_h5(root / worm / "data.h5", n_frames=8)
    loader = Flavell2023Loader(root, clip_frames=4, image_size=(4, 4))
    samples = list(loader)
    assert len(samples) == 4  # 2 files * 2 clips
    ids = sorted({s.worm_id for s in samples})
    # Each subdir produces a distinct worm_id — LOWO-CV correctness depends on it.
    assert len(ids) == 2
    assert ids == ["flavell_w01_data", "flavell_w02_data"]


def test_malformed_file_raises(tmp_path: Path) -> None:
    """A file missing the video dataset raises (bug-class exceptions are not swallowed)."""
    root = tmp_path
    # malformed: no video dataset
    with h5py.File(root / "bad.h5", "w") as f:
        f.create_dataset("neural_activity", data=np.zeros((8, 3), dtype=np.float32))
        f.create_dataset("behavioral_state", data=np.zeros(8, dtype=np.int32))
    loader = Flavell2023Loader(root, clip_frames=4, image_size=(4, 4))
    with pytest.raises(DatasetIntegrityError, match="no video dataset"):
        list(loader)


def test_unreadable_file_logged_others_emit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An OSError on one file is logged and skipped; siblings still emit."""
    root = tmp_path
    # Write a non-HDF5 file with .h5 extension — h5py will raise OSError on open.
    (root / "junk.h5").write_bytes(b"not an hdf5 file")
    _write_minimal_flavell_h5(root / "good.h5", n_frames=8)
    loader = Flavell2023Loader(root, clip_frames=4, image_size=(4, 4))
    with caplog.at_level("WARNING", logger="wormjepa.data.loaders.flavell_2023"):
        samples = list(loader)
    assert len(samples) == 2  # only the good file emits
    assert any("unreadable" in r.message for r in caplog.records)


def test_iteration_is_deterministic(tmp_path: Path) -> None:
    _write_minimal_flavell_h5(tmp_path / "a.h5", n_frames=8)
    _write_minimal_flavell_h5(tmp_path / "b.h5", n_frames=8)
    loader = Flavell2023Loader(tmp_path, clip_frames=4, image_size=(4, 4))
    first = [s.worm_id for s in loader]
    second = [s.worm_id for s in loader]
    assert first == second


def test_clip_frames_larger_than_total_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_minimal_flavell_h5(tmp_path / "tiny.h5", n_frames=3)
    loader = Flavell2023Loader(tmp_path, clip_frames=8, image_size=(4, 4))
    # With one file too short and no other files, the empty-iterator guard
    # surfaces a DatasetIntegrityError.
    with (
        caplog.at_level("WARNING", logger="wormjepa.data.loaders.flavell_2023"),
        pytest.raises(DatasetIntegrityError),
    ):
        list(loader)
    assert any("< clip_frames" in r.message for r in caplog.records)


def test_sha_verification_raises_on_mismatch(tmp_path: Path) -> None:
    """If expected_file_shas is provided and bytes mismatch, raise (FR7)."""
    _write_minimal_flavell_h5(tmp_path / "w.h5", n_frames=8)
    bogus = "0" * 64
    loader = Flavell2023Loader(
        tmp_path,
        clip_frames=4,
        image_size=(4, 4),
        expected_file_shas={"w.h5": bogus},
    )
    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        list(loader)


def test_sha_verification_passes_on_match(tmp_path: Path) -> None:
    """If expected_file_shas is provided and matches, iteration proceeds normally."""
    p = tmp_path / "w.h5"
    _write_minimal_flavell_h5(p, n_frames=8)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    loader = Flavell2023Loader(
        tmp_path,
        clip_frames=4,
        image_size=(4, 4),
        expected_file_shas={"w.h5": sha},
    )
    samples = list(loader)
    assert len(samples) == 2
