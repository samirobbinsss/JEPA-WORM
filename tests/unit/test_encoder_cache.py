"""Unit tests for the eval-time encoder forward-pass disk cache.

Story: encoder-cache. Validates :func:`load_or_build_cache`'s
read-through contract — first call builds, second call loads, key
varies with cfg, schema-bump invalidates, env var bypasses.

The encoder is never actually invoked: every test monkeypatches the
underlying ``_build_eval_cache`` builder with a counted fake so we can
both (a) assert call counts and (b) avoid setting up a real encoder
just to test cache plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from wormjepa.configs.dataset import DatasetLoaderSpec, DatasetSection
from wormjepa.configs.jepa_config import JEPARunConfig
from wormjepa.eval import encoder_cache as encoder_cache_mod
from wormjepa.eval import orchestrator as orchestrator_mod
from wormjepa.eval.encoder_cache import (
    DISABLE_ENV_VAR,
    SCHEMA_VERSION,
    EvalCache,
    load_or_build_cache,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_cfg(*, seed: int = 42, image_size: int = 64, n_worms: int = 4) -> JEPARunConfig:
    """Build a minimal but valid :class:`JEPARunConfig` for the cache tests.

    The cfg only needs to be valid enough for
    :func:`_build_eval_loader_spec` to derive a stable fingerprint;
    nothing in this test ever calls the real encoder.
    """
    dataset = DatasetSection(
        loaders=[
            DatasetLoaderSpec(
                name="baaiworm",
                clip_frames=16,
                n_worms=n_worms,
                clips_per_worm=3,
                n_keypoints=4,
                n_neurons=8,
                image_size=image_size,
            )
        ]
    )
    cfg = JEPARunConfig.model_validate(
        {
            "schema_version": 1,
            "jepa": {
                "model_name": "vit_tiny_patch16_224",
                "img_size": image_size,
                "latent_dim": 32,
                "masking_ratio": 0.5,
                "n_steps": 2,
                "learning_rate": 1e-4,
                "ema_decay": 0.99,
                "seed": seed,
            },
            "dataset": dataset.model_dump(),
        }
    )
    return cfg


def _make_cache() -> EvalCache:
    """Construct a small but non-empty :class:`EvalCache` for save/load tests."""
    cache = EvalCache()
    rng = np.random.default_rng(0)
    for clip_idx in range(3):
        cache.latents.append(rng.normal(size=(8, 32)).astype(np.float32))
        cache.poses.append(rng.normal(size=(8, 8)).astype(np.float32))
        cache.neural.append(rng.normal(size=(8, 8)).astype(np.float32))
        cache.worm_ids.append(f"worm_{clip_idx % 2}")
        cache.session_ids.append(f"sess_{clip_idx}")
        cache.behavioral_states.append(rng.integers(0, 3, size=(8,), dtype=np.int64))
    return cache


def _write_fake_checkpoint(run_dir: Path, contents: bytes = b"fake-checkpoint") -> Path:
    """Write a placeholder checkpoint file under ``run_dir/checkpoints/``.

    The SHA-256 of this byte string is what the encoder cache will key
    on. Tests that need a stable SHA can pass the same bytes; tests that
    want a different SHA pass different bytes.
    """
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"
    ckpt_path.write_bytes(contents)
    return ckpt_path


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run-fake-001"
    d.mkdir()
    _write_fake_checkpoint(d)
    return d


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "encoder_cache"


@pytest.fixture
def cfg() -> JEPARunConfig:
    return _make_cfg()


@pytest.fixture
def fake_state() -> Any:
    """Stand-in for :class:`JEPATrainingState`.

    The cache wrapper passes ``state`` straight to the underlying
    builder, which we always monkeypatch — so the only requirement is
    that the value be addressable; its contents are never inspected.
    """
    return object()


@pytest.fixture
def counted_builder(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch :func:`_build_eval_cache` with a counted fake.

    Returns a dict whose ``"calls"`` key records how many times the
    builder was invoked. Tests assert against this counter to detect
    cache hits (no call) vs cache misses (one call).
    """
    counter: dict[str, int] = {"calls": 0}

    def _fake_build(state: Any, cfg: Any, max_clips: int = 64) -> EvalCache:
        counter["calls"] += 1
        return _make_cache()

    monkeypatch.setattr(encoder_cache_mod, "_build_eval_cache", _fake_build)
    # Also patch the orchestrator's reference so any helper-side calls
    # route through the counted fake too (belt-and-braces).
    monkeypatch.setattr(orchestrator_mod, "_build_eval_cache", _fake_build)
    return counter


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_cache_miss_builds_and_saves(
    fake_state: Any,
    cfg: JEPARunConfig,
    run_dir: Path,
    cache_dir: Path,
    counted_builder: dict[str, int],
) -> None:
    """First call: builds via the underlying builder, writes a .pt file."""
    cache = load_or_build_cache(fake_state, cfg, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    assert isinstance(cache, EvalCache)
    assert len(cache.latents) == 3
    assert counted_builder["calls"] == 1, "first call should invoke the builder exactly once"
    # Exactly one .pt file should exist in the cache dir.
    pt_files = list(cache_dir.glob("*.pt"))
    assert len(pt_files) == 1, f"expected one cache file, got {pt_files}"
    # The .pt round-trips and bears the documented schema_version.
    blob = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    assert isinstance(blob, dict)
    assert blob.get("schema_version") == SCHEMA_VERSION
    for key in ("latents", "poses", "neural", "worm_ids", "session_ids", "behavioral_states"):
        assert key in blob, f"missing field {key!r} in cache blob"


def test_cache_hit_skips_encoder(
    fake_state: Any,
    cfg: JEPARunConfig,
    run_dir: Path,
    cache_dir: Path,
    counted_builder: dict[str, int],
) -> None:
    """Second call with identical key: loads from disk, does NOT call builder."""
    first = load_or_build_cache(fake_state, cfg, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    assert counted_builder["calls"] == 1
    second = load_or_build_cache(fake_state, cfg, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    # Builder must not have been called a second time.
    assert counted_builder["calls"] == 1, "second call should be a cache hit"
    # Contents are identical (per-clip arrays element-wise equal).
    assert len(second.latents) == len(first.latents)
    for a, b in zip(first.latents, second.latents, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(first.poses, second.poses, strict=True):
        np.testing.assert_array_equal(a, b)
    assert second.worm_ids == first.worm_ids
    assert second.session_ids == first.session_ids


def test_cache_key_changes_with_cfg(
    fake_state: Any,
    run_dir: Path,
    cache_dir: Path,
    counted_builder: dict[str, int],
) -> None:
    """Different loader spec → different key → builder runs twice.

    Varies ``img_size`` because that's a cfg field that flows into the
    eval-loader spec (see :func:`_build_eval_loader_spec`), so a change
    here is guaranteed to change the fingerprint.
    """
    cfg_a = _make_cfg(image_size=64)
    cfg_b = _make_cfg(image_size=96)
    load_or_build_cache(fake_state, cfg_a, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    load_or_build_cache(fake_state, cfg_b, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    assert counted_builder["calls"] == 2, (
        "differing loader spec should produce different cache keys → two builder invocations"
    )
    # Two distinct .pt files exist.
    pt_files = list(cache_dir.glob("*.pt"))
    assert len(pt_files) == 2


def test_schema_version_invalidates(
    fake_state: Any,
    cfg: JEPARunConfig,
    run_dir: Path,
    cache_dir: Path,
    counted_builder: dict[str, int],
) -> None:
    """A v0 file at the cache path is overwritten + a v1 blob is returned."""
    # First, populate the cache so we know the on-disk filename.
    load_or_build_cache(fake_state, cfg, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    pt_files = list(cache_dir.glob("*.pt"))
    assert len(pt_files) == 1
    cache_path = pt_files[0]
    # Overwrite with a v0-shaped blob.
    torch.save(
        {
            "schema_version": 0,  # intentionally stale
            "latents": [],
            "poses": [],
            "neural": [],
            "worm_ids": [],
            "session_ids": [],
            "behavioral_states": [],
        },
        cache_path,
    )
    # Reset the call counter so we can detect the rebuild.
    counted_builder["calls"] = 0
    cache = load_or_build_cache(fake_state, cfg, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    assert counted_builder["calls"] == 1, "stale schema should force a rebuild"
    # Re-read from disk — should now be schema_version=SCHEMA_VERSION.
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert isinstance(blob, dict)
    assert blob.get("schema_version") == SCHEMA_VERSION
    assert len(cache.latents) > 0


def test_disable_env_var_bypasses(
    fake_state: Any,
    cfg: JEPARunConfig,
    run_dir: Path,
    cache_dir: Path,
    counted_builder: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``WORMJEPA_DISABLE_ENCODER_CACHE=1`` forces a rebuild every call.

    Asserts both halves of the contract:
      - the builder is invoked on every call (no read short-circuit), and
      - no .pt is written (no spurious cache pollution on disabled mode).
    """
    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    load_or_build_cache(fake_state, cfg, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    load_or_build_cache(fake_state, cfg, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    load_or_build_cache(fake_state, cfg, run_dir=run_dir, cache_dir=cache_dir, max_clips=4)
    assert counted_builder["calls"] == 3, "disabled cache should rebuild on every call"
    # No .pt files should be written when the cache is disabled.
    pt_files = list(cache_dir.glob("*.pt")) if cache_dir.exists() else []
    assert pt_files == [], f"disabled cache should not write files, found {pt_files}"
