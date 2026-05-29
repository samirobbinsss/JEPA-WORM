"""Unit tests for ``wormjepa.data.loaders.baaiworm.BAAIWormLoader`` (Story 8.7)."""

from __future__ import annotations

import pytest
import torch

from wormjepa import DatasetIntegrityError
from wormjepa.data import DatasetSample, SourceDataset
from wormjepa.data.loaders.baaiworm import BAAIWormLoader


def test_baaiworm_yields_contract_compliant_samples() -> None:
    loader = BAAIWormLoader(
        n_worms=2,
        clips_per_worm=2,
        clip_frames=4,
        image_size=(8, 8),
        n_keypoints=8,
        n_neurons=4,
        seed=0,
        verify_provenance=False,
    )
    samples = list(loader)
    assert len(samples) == 4  # 2 worms * 2 clips
    s = samples[0]
    assert isinstance(s, DatasetSample)
    assert s.video_clip.shape == (4, 3, 8, 8)
    assert s.video_clip.dtype is torch.float32
    assert float(s.video_clip.min()) >= 0.0
    assert float(s.video_clip.max()) <= 1.0
    assert s.pose is not None
    assert s.pose.shape == (4, 8, 2)
    assert s.neural is not None
    assert s.neural.shape == (4, 4)
    assert s.source_dataset == SourceDataset("baaiworm")
    assert s.worm_id.startswith("baaiworm_w")


def test_baaiworm_iteration_is_bit_identical_under_same_seed() -> None:
    """Story 8.7 AC: repeated iteration under the same seed → bit-identical sequences."""
    cfg = dict(
        n_worms=2,
        clips_per_worm=2,
        clip_frames=4,
        image_size=(8, 8),
        n_keypoints=8,
        n_neurons=4,
        seed=42,
        verify_provenance=False,
    )
    a = list(BAAIWormLoader(**cfg))
    b = list(BAAIWormLoader(**cfg))
    assert len(a) == len(b)
    for sa, sb in zip(a, b, strict=True):
        assert sa.worm_id == sb.worm_id
        assert sa.session_id == sb.session_id
        assert torch.equal(sa.video_clip, sb.video_clip)
        assert sa.pose is not None and sb.pose is not None
        assert torch.equal(sa.pose, sb.pose)
        assert sa.neural is not None and sb.neural is not None
        assert torch.equal(sa.neural, sb.neural)


def test_baaiworm_different_seeds_produce_different_output() -> None:
    cfg = dict(
        n_worms=1,
        clips_per_worm=1,
        clip_frames=4,
        image_size=(8, 8),
        n_keypoints=8,
        n_neurons=4,
        verify_provenance=False,
    )
    a = next(iter(BAAIWormLoader(**cfg, seed=0)))
    b = next(iter(BAAIWormLoader(**cfg, seed=1)))
    assert a.pose is not None and b.pose is not None
    assert not torch.equal(a.pose, b.pose)


def test_baaiworm_provenance_check_passes_with_real_lock() -> None:
    """When MANIFEST.lock matches the SPEC, the loader iterates cleanly."""
    loader = BAAIWormLoader(
        n_worms=1,
        clips_per_worm=1,
        clip_frames=4,
        image_size=(8, 8),
        n_keypoints=8,
        n_neurons=4,
        seed=0,
        verify_provenance=True,
    )
    samples = list(loader)
    assert len(samples) == 1


def test_baaiworm_provenance_fails_loudly_on_commit_sha_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPEC commit_sha differing from MANIFEST.lock is a frozen-artifact violation."""
    import wormjepa.data.loaders.baaiworm as baaiworm_mod
    from wormjepa.data.sources.base import GithubGeneratorSource

    drifted_spec = GithubGeneratorSource(
        name="baaiworm",
        repo=baaiworm_mod.SPEC.repo,
        commit_sha="0" * 40,  # not what MANIFEST.lock pins
        config_path=baaiworm_mod.SPEC.config_path,
        config_sha256=baaiworm_mod.SPEC.config_sha256,
        license=baaiworm_mod.SPEC.license,
        citation=baaiworm_mod.SPEC.citation,
    )
    monkeypatch.setattr(baaiworm_mod, "SPEC", drifted_spec)
    loader = BAAIWormLoader(
        n_worms=1,
        clips_per_worm=1,
        clip_frames=4,
        image_size=(8, 8),
        n_keypoints=8,
        n_neurons=4,
        seed=0,
        verify_provenance=True,
    )
    with pytest.raises(DatasetIntegrityError, match="commit_sha"):
        list(loader)


def test_baaiworm_worm_id_namespace() -> None:
    """worm_ids must be unique across worms and not collide with other loaders' prefixes."""
    loader = BAAIWormLoader(
        n_worms=3,
        clips_per_worm=1,
        clip_frames=4,
        image_size=(8, 8),
        n_keypoints=8,
        n_neurons=4,
        seed=0,
        verify_provenance=False,
    )
    ids = sorted({s.worm_id for s in loader})
    assert len(ids) == 3
    assert all(wid.startswith("baaiworm_") for wid in ids)
    # Distinct from other loaders' prefixes.
    for wid in ids:
        assert not wid.startswith(("flavell_", "wormbehavior_", "openworm_", "wormid_"))
