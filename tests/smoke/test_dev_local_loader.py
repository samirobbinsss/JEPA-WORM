"""Smoke test for the dev-local single-MP4 loader.

The test is auto-skipped if no real video fixture is present, so CI stays
green on a clean clone. To enable, drop an MP4 at
``tests/fixtures/dev_local/sample.mp4`` (gitignored) and re-run the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wormjepa.data import SourceDataset
from wormjepa.data.loaders.dev_local import DevLocalLoader

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "dev_local" / "sample.mp4"


def test_dev_local_loader_raises_when_file_missing(tmp_path: Path) -> None:
    """The loader fails loudly (FileNotFoundError) on a missing path."""
    missing = tmp_path / "does_not_exist.mp4"
    loader = DevLocalLoader(missing, clip_frames=2, image_size=(16, 16), max_clips=1)
    with pytest.raises(FileNotFoundError, match="video file not found"):
        next(iter(loader))


@pytest.mark.skipif(
    not FIXTURE_PATH.is_file(),
    reason=(
        "No dev_local sample present. Drop an MP4 at "
        "tests/fixtures/dev_local/sample.mp4 to enable this smoke test."
    ),
)
def test_dev_local_loader_yields_contract_compliant_samples() -> None:
    """Decoded samples match the DatasetSample contract on real video."""
    loader = DevLocalLoader(
        FIXTURE_PATH,
        clip_frames=4,
        image_size=(32, 32),
        max_clips=2,
    )
    samples = list(loader)
    assert len(samples) >= 1, "loader produced no clips — video shorter than clip_frames?"
    for sample in samples:
        assert sample.video_clip.shape == (4, 3, 32, 32)
        assert sample.video_clip.dtype.is_floating_point
        assert float(sample.video_clip.min()) >= 0.0
        assert float(sample.video_clip.max()) <= 1.0
        assert sample.pose is None
        assert sample.neural is None
        assert sample.source_dataset == SourceDataset("dev_local")
