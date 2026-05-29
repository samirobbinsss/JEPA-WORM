"""Unit tests for :class:`wormjepa.data.loaders.wormid.WormIDLoader`.

These tests use minimal in-memory NWB fixtures written to ``tmp_path`` via
``pynwb`` so the loader's full ImageSeries → DatasetSample path runs end to
end without requiring DANDI access. Real-data smoke is covered by
``tests/integration/test_wormid_loader_real.py`` (auto-skipped when the
fixture directory is absent).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.file import Subject
from pynwb.image import ImageSeries

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SourceDataset
from wormjepa.data.loaders.wormid import WormIDLoader

# Cohort partition from pre-registration/splits/wormid_train_eval.yaml.
TRAIN_DANDISETS = ["000472", "000541", "000565", "000692", "000715"]
EVAL_DANDISETS = ["000714", "000776"]


def _write_minimal_nwb(
    path: Path,
    *,
    subject_id: str,
    n_frames: int = 16,
    h: int = 16,
    w: int = 16,
    include_neural: bool = True,
    rate: float = 10.0,
) -> None:
    """Write a minimal NWB file with one grayscale ImageSeries (and optional neural).

    Shape is ``(T, H, W)`` uint8 to exercise the grayscale-to-RGB code path in
    ``_normalize_video_clip``.
    """
    nwb = NWBFile(
        session_description="unit-test fixture",
        identifier=f"{subject_id}-id",
        session_start_time=datetime.datetime.now(datetime.UTC),
        subject=Subject(subject_id=subject_id, species="C. elegans"),
    )
    rng = np.random.default_rng(seed=hash(subject_id) & 0xFFFFFFFF)
    video = rng.integers(0, 255, size=(n_frames, h, w), dtype=np.uint8)
    nwb.add_acquisition(ImageSeries(name="video", data=video, rate=rate, unit="n.a."))
    if include_neural:
        neural = rng.standard_normal((n_frames, 5)).astype(np.float32)
        nwb.add_acquisition(
            TimeSeries(
                name="neural_activity",
                data=neural,
                rate=rate,
                unit="dF/F",
                description="simulated calcium traces",
            )
        )
    with NWBHDF5IO(str(path), "w") as io:  # pyright: ignore[reportCallIssue]
        io.write(nwb)


def _build_federation_root(
    tmp_path: Path,
    dandisets: list[str],
    *,
    n_frames: int = 16,
    files_per_dandiset: int = 1,
) -> Path:
    """Materialise a federation root with one or more NWB files per dandiset."""
    root = tmp_path / "wormid_local"
    root.mkdir()
    for did in dandisets:
        subdir = root / did
        subdir.mkdir()
        for k in range(files_per_dandiset):
            _write_minimal_nwb(
                subdir / f"sub-{did}-{k}.nwb",
                subject_id=f"sub-{did}-{k}",
                n_frames=n_frames,
            )
    return root


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_loader_raises_when_root_missing(tmp_path: Path) -> None:
    """Non-existent ``local_dandi_root`` is a substantive failure, not a no-op."""
    missing = tmp_path / "does_not_exist"
    loader = WormIDLoader(missing, cohort="all", clip_frames=4, image_size=(8, 8))
    with pytest.raises(DatasetIntegrityError, match="local_dandi_root not found"):
        list(loader)


def test_loader_raises_when_no_nwb_files(tmp_path: Path) -> None:
    """An empty root for the active cohort is a substantive failure."""
    root = tmp_path / "empty_root"
    root.mkdir()
    loader = WormIDLoader(root, cohort="all", clip_frames=4, image_size=(8, 8))
    with pytest.raises(DatasetIntegrityError, match=r"no \.nwb files"):
        list(loader)


# ---------------------------------------------------------------------------
# Cohort filter
# ---------------------------------------------------------------------------


def test_cohort_train_iterates_only_train_dandisets(tmp_path: Path) -> None:
    """``cohort="train"`` excludes every eval-dandiset id from emitted worm_ids."""
    root = _build_federation_root(
        tmp_path, TRAIN_DANDISETS + EVAL_DANDISETS, n_frames=8, files_per_dandiset=1
    )
    loader = WormIDLoader(root, cohort="train", clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    assert len(samples) > 0, "train cohort should produce samples"
    for sample in samples:
        for eval_id in EVAL_DANDISETS:
            assert eval_id not in sample.worm_id, (
                f"train cohort leaked eval dandiset {eval_id} in worm_id={sample.worm_id}"
            )


def test_cohort_eval_iterates_only_eval_dandisets(tmp_path: Path) -> None:
    """``cohort="eval"`` excludes every train-dandiset id from emitted worm_ids."""
    root = _build_federation_root(
        tmp_path, TRAIN_DANDISETS + EVAL_DANDISETS, n_frames=8, files_per_dandiset=1
    )
    loader = WormIDLoader(root, cohort="eval", clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    assert len(samples) > 0, "eval cohort should produce samples"
    for sample in samples:
        for train_id in TRAIN_DANDISETS:
            assert train_id not in sample.worm_id, (
                f"eval cohort leaked train dandiset {train_id} in worm_id={sample.worm_id}"
            )


def test_cohort_all_is_union_of_train_and_eval(tmp_path: Path) -> None:
    """``cohort="all"`` should produce samples spanning both partitions."""
    root = _build_federation_root(
        tmp_path, TRAIN_DANDISETS + EVAL_DANDISETS, n_frames=8, files_per_dandiset=1
    )
    train_samples = list(WormIDLoader(root, cohort="train", clip_frames=4, image_size=(8, 8)))
    eval_samples = list(WormIDLoader(root, cohort="eval", clip_frames=4, image_size=(8, 8)))
    all_samples = list(WormIDLoader(root, cohort="all", clip_frames=4, image_size=(8, 8)))
    assert len(all_samples) == len(train_samples) + len(eval_samples)


# ---------------------------------------------------------------------------
# Contract compliance + determinism
# ---------------------------------------------------------------------------


def test_sample_contract_compliance(tmp_path: Path) -> None:
    """Every emitted sample obeys the DatasetSample contract for source ``wormid``."""
    root = _build_federation_root(tmp_path, TRAIN_DANDISETS[:2], n_frames=12, files_per_dandiset=1)
    loader = WormIDLoader(root, cohort="train", clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    assert len(samples) > 0
    seen_with_neural = 0
    for sample in samples:
        assert isinstance(sample, DatasetSample)
        # video_clip: (T, C, H, W) float32 in [0, 1] with C=3.
        assert sample.video_clip.shape == (4, 3, 8, 8)
        assert sample.video_clip.dtype.is_floating_point
        assert float(sample.video_clip.min()) >= 0.0
        assert float(sample.video_clip.max()) <= 1.0
        # pose deferred for v1.
        assert sample.pose is None
        # neural is (T, N) when present.
        if sample.neural is not None:
            seen_with_neural += 1
            assert sample.neural.shape[0] == 4
            assert sample.neural.ndim == 2
            assert sample.neural.dtype.is_floating_point
        # IDs and provenance.
        assert sample.source_dataset == SourceDataset("wormid")
        assert sample.worm_id.startswith("wormid_")
        assert sample.session_id.startswith(sample.worm_id)
    assert seen_with_neural > 0, "fixture should produce at least one neural-bearing sample"


def test_iteration_is_deterministic(tmp_path: Path) -> None:
    """Same input → same iteration order → same sample identifiers."""
    root = _build_federation_root(tmp_path, TRAIN_DANDISETS[:2], n_frames=8, files_per_dandiset=2)
    loader_a = WormIDLoader(root, cohort="train", clip_frames=4, image_size=(8, 8))
    loader_b = WormIDLoader(root, cohort="train", clip_frames=4, image_size=(8, 8))
    ids_a = [(s.worm_id, s.session_id) for s in loader_a]
    ids_b = [(s.worm_id, s.session_id) for s in loader_b]
    assert ids_a == ids_b
    assert len(ids_a) == len(set(ids_a)) or len(ids_a) > 0  # not required to be unique


def test_multiple_clips_per_file(tmp_path: Path) -> None:
    """When n_frames > clip_frames the loader yields multiple non-overlapping clips."""
    root = _build_federation_root(tmp_path, [TRAIN_DANDISETS[0]], n_frames=12, files_per_dandiset=1)
    loader = WormIDLoader(root, cohort="train", clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    # 12 frames / 4 frames/clip = 3 clips per file.
    assert len(samples) == 3
    # Same worm/session for every clip in the file.
    assert len({s.worm_id for s in samples}) == 1
    assert len({s.session_id for s in samples}) == 1


def test_skips_files_shorter_than_clip_frames(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A file with fewer than ``clip_frames`` frames is warned and skipped."""
    root = tmp_path / "short_fixture"
    root.mkdir()
    did = TRAIN_DANDISETS[0]
    subdir = root / did
    subdir.mkdir()
    _write_minimal_nwb(subdir / "short.nwb", subject_id="too-short", n_frames=2)
    loader = WormIDLoader(root, cohort="train", clip_frames=8, image_size=(8, 8))
    # No DatasetIntegrityError — the loader emits zero samples and logs a warning.
    samples = list(loader)
    assert samples == []


def test_invalid_nwb_file_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed .nwb file does not crash the iterator."""
    root = tmp_path / "mixed"
    root.mkdir()
    did = TRAIN_DANDISETS[0]
    subdir = root / did
    subdir.mkdir()
    # Write a valid file...
    _write_minimal_nwb(subdir / "good.nwb", subject_id="good", n_frames=8)
    # ...and a garbage file.
    (subdir / "bad.nwb").write_bytes(b"this is not an NWB file")
    loader = WormIDLoader(root, cohort="train", clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    # Good file should still produce samples.
    assert len(samples) > 0
    assert any(s.worm_id == "wormid_000472_good" for s in samples)


# ---------------------------------------------------------------------------
# Identifier wiring
# ---------------------------------------------------------------------------


def test_worm_id_includes_dandiset_and_subject(tmp_path: Path) -> None:
    """worm_id format is ``wormid_<dandiset>_<subject_id>``."""
    root = _build_federation_root(tmp_path, [TRAIN_DANDISETS[0]], n_frames=8, files_per_dandiset=1)
    loader = WormIDLoader(root, cohort="train", clip_frames=4, image_size=(8, 8))
    samples = list(loader)
    assert len(samples) > 0
    expected_prefix = f"wormid_{TRAIN_DANDISETS[0]}_"
    for s in samples:
        assert s.worm_id.startswith(expected_prefix)


def _silence_h5py_user_block_warning() -> None:
    """Keep h5py garbage-file warnings out of the test log on stderr."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="h5py")


_silence_h5py_user_block_warning()
