"""Per-loader sanity test — instantiate every shipped loader via
:func:`wormjepa.data.composition.build_loader` and assert the
:class:`DatasetSample` contract holds.

Catches loader regressions (signature drift, default-arg breakage,
contract violations) at CI time without needing real-data fixtures.
File-based loaders (``flavell_2023``, ``wormbehavior_db``,
``openworm_movement``, ``wormid``) are skipped when their on-disk root
is absent — that path is covered by their dedicated unit tests with
fixture HDF5/NWB files.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wormjepa.configs.dataset import DatasetLoaderSpec
from wormjepa.data import DatasetSample
from wormjepa.data.composition import build_loader

# Loaders that synthesise data in-process — always exercisable in CI.
_SYNTH_LOADERS = ("synthetic", "baaiworm", "two_camera_mock")

# Loaders that require a populated on-disk root — env-var override lets a
# local dev box opt in. Default behaviour in CI is to skip.
_FILE_LOADERS = {
    "flavell_2023": "WORMJEPA_TEST_FLAVELL_ROOT",
    "wormbehavior_db": "WORMJEPA_TEST_WORMBEHAVIOR_ROOT",
    "openworm_movement": "WORMJEPA_TEST_OPENWORM_ROOT",
    "wormid": "WORMJEPA_TEST_WORMID_ROOT",
}


def _assert_sample_contract(sample: object) -> None:
    """Every loader must yield :class:`DatasetSample` instances satisfying
    the documented contract: 4-D video_clip, non-empty worm_id, set
    source_dataset.
    """
    assert isinstance(sample, DatasetSample)
    assert sample.video_clip.ndim == 4, (
        f"expected (T, C, H, W) video_clip, got shape {tuple(sample.video_clip.shape)}"
    )
    assert sample.worm_id, "worm_id must be non-empty"
    assert sample.source_dataset, "source_dataset must be set"


@pytest.mark.parametrize("loader_name", _SYNTH_LOADERS)
def test_synthesising_loader_yields_at_least_four_samples(loader_name: str) -> None:
    """n_worms=2 * clips_per_worm=2 → ≥ 4 samples for every in-process loader."""
    spec = DatasetLoaderSpec(
        name=loader_name,  # type: ignore[arg-type]
        n_worms=2,
        clips_per_worm=2,
        clip_frames=4,
        image_size=8,
    )
    loader = build_loader([spec], seed=0)
    samples = list(loader)
    assert len(samples) >= 4, (
        f"{loader_name}: expected ≥ 4 samples from n_worms=2, clips_per_worm=2; got {len(samples)}"
    )
    for s in samples:
        _assert_sample_contract(s)


@pytest.mark.parametrize(("loader_name", "env_var"), list(_FILE_LOADERS.items()))
def test_file_backed_loader_yields_at_least_one_sample(loader_name: str, env_var: str) -> None:
    """File-based loaders need a real-data root. Skip when absent in CI."""
    root = os.environ.get(env_var)
    if not root:
        pytest.skip(
            f"{loader_name}: set {env_var} to an on-disk root to exercise this loader; "
            f"file-based loaders need real-data fixtures (NWB/HDF5)."
        )
    root_path = Path(root)
    if not root_path.exists():
        pytest.skip(f"{loader_name}: {env_var}={root!r} does not exist on disk.")
    spec = DatasetLoaderSpec(
        name=loader_name,  # type: ignore[arg-type]
        local_root=str(root_path),
        clip_frames=4,
        image_size=8,
    )
    loader = build_loader([spec], seed=0)
    samples = list(loader)
    assert len(samples) >= 1, f"{loader_name}: expected ≥ 1 sample from {root_path}"
    for s in samples:
        _assert_sample_contract(s)
