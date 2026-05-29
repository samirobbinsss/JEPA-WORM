"""Real-format smoke test for the Flavell-2023 loader.

Default fixture: ``tests/fixtures/flavell_2023/`` (committed minimal HDF5,
~6 KB, real Flavell schema). The committed fixture is what Story 8.4's
"per-loader smoke test on a fixture subset passes" AC enforces in CI.

Override with the real Atanas/Flavell-2023 Zenodo extract by setting
``JEPA_WORM_FLAVELL_FIXTURE_DIR=/path/to/extracted/zenodo/archive`` —
the env var, when present and a directory, replaces the default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wormjepa.data import DatasetSample, SourceDataset
from wormjepa.data.loaders.flavell_2023 import Flavell2023Loader

_ENV_VAR = "JEPA_WORM_FLAVELL_FIXTURE_DIR"
_COMMITTED_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "flavell_2023"
_MIN_SAMPLES = 2


def _resolve_fixture_dir() -> Path:
    override = os.environ.get(_ENV_VAR)
    if override:
        p = Path(override).expanduser()
        if not p.is_dir():
            pytest.fail(f"{_ENV_VAR}={p} is set but not a directory")
        return p
    if not _COMMITTED_FIXTURE.is_dir():
        pytest.fail(
            f"committed fixture missing at {_COMMITTED_FIXTURE} "
            f"(regenerate via the fixture generator)."
        )
    return _COMMITTED_FIXTURE


def test_flavell_loader_on_real_subset() -> None:
    fixture = _resolve_fixture_dir()
    loader = Flavell2023Loader(fixture, clip_frames=8, image_size=(64, 64))
    samples: list[DatasetSample] = []
    for sample in loader:
        samples.append(sample)
        if len(samples) >= 4:
            break
    # The generator goes out of scope when this function returns; Python GC
    # then drives the with-block exit and the h5py file handle closes.
    assert len(samples) >= _MIN_SAMPLES, (
        f"Flavell2023Loader yielded only {len(samples)} samples (expected >= {_MIN_SAMPLES})"
    )
    for s in samples:
        assert s.source_dataset == SourceDataset("flavell_2023")
        assert s.video_clip.shape[0] == 8
        assert s.video_clip.shape[1] == 3
        assert s.neural is not None
        assert s.behavioral_state is not None
        assert s.behavioral_state.shape[0] == 8
