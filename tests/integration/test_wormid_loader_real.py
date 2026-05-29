"""Real-data integration test for :class:`WormIDLoader`.

Auto-skipped when no fixture is present at
``tests/fixtures/wormid_local/<dandiset_id>/*.nwb``. To enable, drop one or
more downloaded WormID NWB files under that path (gitignored) and re-run.

This mirrors the auto-skip pattern in ``tests/smoke/test_dev_local_loader.py``
so CI stays green on a clean clone while still giving the project lead a
fast end-to-end sanity check on real bytes when the fixture is available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wormjepa.data import SourceDataset
from wormjepa.data.loaders.wormid import WormIDLoader

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "wormid_local"


def _fixture_has_any_nwb() -> bool:
    if not FIXTURE_ROOT.is_dir():
        return False
    return any(FIXTURE_ROOT.rglob("*.nwb"))


@pytest.mark.skipif(
    not _fixture_has_any_nwb(),
    reason=(
        "No WormID NWB fixture present. Drop one or more .nwb files at "
        "tests/fixtures/wormid_local/<dandiset_id>/*.nwb to enable this test."
    ),
)
def test_wormid_loader_yields_contract_compliant_samples_on_real_nwb() -> None:
    """Real WormID NWB files iterate cleanly and obey the DatasetSample contract."""
    loader = WormIDLoader(
        FIXTURE_ROOT,
        cohort="all",
        clip_frames=4,
        image_size=(32, 32),
    )
    # Pull only the first few samples — real WormID files are long; we only need
    # to validate the contract, not consume the corpus.
    samples = []
    for sample in loader:
        samples.append(sample)
        if len(samples) >= 4:
            break
    assert samples, "loader produced no samples on real fixture"
    for sample in samples:
        assert sample.video_clip.shape == (4, 3, 32, 32)
        assert sample.video_clip.dtype.is_floating_point
        assert float(sample.video_clip.min()) >= 0.0
        assert float(sample.video_clip.max()) <= 1.0
        assert sample.source_dataset == SourceDataset("wormid")
        assert sample.worm_id.startswith("wormid_")
        assert sample.session_id.startswith(sample.worm_id)
        if sample.neural is not None:
            assert sample.neural.shape[0] == 4
            assert sample.neural.ndim == 2
