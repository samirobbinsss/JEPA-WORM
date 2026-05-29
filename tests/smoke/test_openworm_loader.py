"""Real-format smoke test for OpenWormMovementLoader.

Default fixture: ``tests/fixtures/openworm_movement/`` (committed minimal
HDF5 per anchor record). Override with real Zenodo extract by setting
``JEPA_WORM_OPENWORM_FIXTURE_DIR`` to a directory laid out as
``<root>/<record_id>/*.hdf5``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wormjepa.data import DatasetSample, SourceDataset
from wormjepa.data.loaders.openworm_movement import OpenWormMovementLoader

_ENV_VAR = "JEPA_WORM_OPENWORM_FIXTURE_DIR"
_COMMITTED_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "openworm_movement"
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


def test_openworm_loader_on_real_subset() -> None:
    fixture = _resolve_fixture_dir()
    loader = OpenWormMovementLoader(fixture, clip_frames=8, image_size=(64, 64))
    samples: list[DatasetSample] = []
    for sample in loader:
        samples.append(sample)
        if len(samples) >= 4:
            break
    assert len(samples) >= _MIN_SAMPLES
    for s in samples:
        assert s.source_dataset == SourceDataset("openworm_movement")
        assert s.video_clip.shape[0] == 8
        assert s.video_clip.shape[1] == 3
        assert s.neural is None
