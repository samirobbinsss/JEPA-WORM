"""Unit tests for ``configs.jepa_config.DatasetSection`` + ``build_loader``."""

from __future__ import annotations

from pathlib import Path

import pytest

from wormjepa import WormJEPAError
from wormjepa.configs.jepa_config import (
    DatasetLoaderSpec,
    DatasetSection,
    JEPARunConfig,
    JEPASection,
)
from wormjepa.data.loaders.flavell_2023 import Flavell2023Loader
from wormjepa.data.loaders.openworm_movement import OpenWormMovementLoader
from wormjepa.data.loaders.synthetic import SyntheticLoader
from wormjepa.data.loaders.wormbehavior_db import WormBehaviorDBLoader
from wormjepa.training.runner import ChainedLoader, build_loader

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def test_dataset_section_defaults_to_single_synthetic_loader() -> None:
    section = DatasetSection()
    assert len(section.loaders) == 1
    assert section.loaders[0].name == "synthetic"


def test_jepa_run_config_validates_with_no_dataset_block() -> None:
    """Pre-Story-8.9 configs (no ``dataset:``) still validate via the default."""
    cfg = JEPARunConfig(schema_version=1, jepa=JEPASection())
    assert cfg.dataset.loaders[0].name == "synthetic"


def test_dataset_section_rejects_unknown_loader_name() -> None:
    with pytest.raises(Exception):  # pydantic ValidationError  # noqa: B017
        DatasetLoaderSpec.model_validate({"name": "not_a_real_loader"})


def test_build_loader_synthetic_returns_synthetic_loader() -> None:
    spec = DatasetLoaderSpec(name="synthetic", clip_frames=4, n_worms=2)
    loader = build_loader([spec], seed=42)
    assert isinstance(loader, SyntheticLoader)


def test_build_loader_real_loader_requires_local_root() -> None:
    spec = DatasetLoaderSpec(name="flavell_2023")
    with pytest.raises(WormJEPAError, match="local_root"):
        build_loader([spec], seed=42)


def test_build_loader_flavell_returns_flavell_loader() -> None:
    spec = DatasetLoaderSpec(
        name="flavell_2023",
        local_root=str(_FIXTURE_ROOT / "flavell_2023"),
        clip_frames=4,
        image_size=8,
    )
    loader = build_loader([spec], seed=42)
    assert isinstance(loader, Flavell2023Loader)


def test_build_loader_wormbehavior_returns_wormbehavior_loader() -> None:
    spec = DatasetLoaderSpec(
        name="wormbehavior_db",
        local_root=str(_FIXTURE_ROOT / "wormbehavior_db"),
        clip_frames=4,
        image_size=8,
    )
    loader = build_loader([spec], seed=42)
    assert isinstance(loader, WormBehaviorDBLoader)


def test_build_loader_openworm_returns_openworm_loader() -> None:
    spec = DatasetLoaderSpec(
        name="openworm_movement",
        local_root=str(_FIXTURE_ROOT / "openworm_movement"),
        clip_frames=4,
        image_size=8,
    )
    loader = build_loader([spec], seed=42)
    assert isinstance(loader, OpenWormMovementLoader)


def test_build_loader_chains_multiple_loaders() -> None:
    specs = [
        DatasetLoaderSpec(name="synthetic", clip_frames=4, n_worms=1, clips_per_worm=1),
        DatasetLoaderSpec(
            name="flavell_2023",
            local_root=str(_FIXTURE_ROOT / "flavell_2023"),
            clip_frames=4,
            image_size=8,
        ),
    ]
    loader = build_loader(specs, seed=42)
    assert isinstance(loader, ChainedLoader)
    samples = list(loader)
    # Synthetic (1 worm * 1 clip = 1 sample) + Flavell (16 frames / 4 = 4 samples).
    assert len(samples) >= 2
    sources = {s.source_dataset for s in samples}
    assert "synthetic" in sources
    assert "flavell_2023" in sources


def test_build_loader_rejects_empty_specs() -> None:
    with pytest.raises(WormJEPAError, match="at least one"):
        build_loader([], seed=42)
