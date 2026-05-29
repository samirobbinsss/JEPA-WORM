"""Disk-side cache of the eval-time encoder forward pass.

The eval orchestrator's :func:`wormjepa.eval.orchestrator._build_eval_cache`
encodes every eval-cohort clip through the trained encoder. With V-JEPA 2.1
ViT-L this dominates eval wall-time and is paid afresh on every invocation
of ``wormjepa eval`` — even when nothing about the encoder or eval cohort
has changed (e.g. re-running after a probe-suite-only change).

This module wraps that hot path with a read-through disk cache keyed on
``(checkpoint_sha256, eval_loader_fingerprint, max_clips, latent_dim)``:

- **checkpoint_sha256** — SHA-256 of ``<run_dir>/checkpoints/checkpoint.pt``.
  Captures any change to the trained weights (a re-train invalidates the
  cache automatically). Computed via chunked read to avoid loading the
  full checkpoint into memory just for the digest.
- **eval_loader_fingerprint** — SHA-256 of the eval :class:`DatasetLoaderSpec`'s
  pydantic JSON serialisation. Captures any change to the eval cohort
  shape (n_worms, clip_frames, image_size, n_keypoints, …) — including
  changes the user can't see directly because the spec is derived from
  cfg inside :func:`_build_eval_loader_spec`.
- **max_clips** — the cap that :func:`_build_eval_cache` enforces. Cached
  caches are sized to this value, so a different max_clips needs a
  different key.
- **latent_dim** — captures encoder-architecture changes that produce a
  different latent shape even when the checkpoint SHA happens to match
  (paranoid belt-and-braces alongside the SHA).

Default ``cache_dir`` is ``<project_root>/.cache/encoder_cache/`` (gitignored).

Bump :data:`SCHEMA_VERSION` to invalidate every existing on-disk entry
(stale files are silently overwritten on the next miss).

Opt-out: set ``WORMJEPA_DISABLE_ENCODER_CACHE=1`` to force a rebuild on
every call (useful for debugging cache-correctness issues).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from wormjepa.eval.orchestrator import (
    EvalCache,
    _build_eval_cache,  # pyright: ignore[reportPrivateUsage]
    _build_eval_loader_spec,  # pyright: ignore[reportPrivateUsage]
)
from wormjepa.paths import project_root

if TYPE_CHECKING:
    from wormjepa.configs.jepa_config import JEPARunConfig
    from wormjepa.training.loop import JEPATrainingState

logger = logging.getLogger(__name__)


#: Disk-side serialisation schema version. Bump to invalidate every cached
#: entry on disk; the next call will rebuild + overwrite.
SCHEMA_VERSION = 1

#: Env var name that disables the cache entirely (forces rebuild every
#: call, mirroring pre-cache behaviour). Useful for debugging.
DISABLE_ENV_VAR = "WORMJEPA_DISABLE_ENCODER_CACHE"


def _checkpoint_sha256(checkpoint_path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of ``checkpoint_path`` via chunked read.

    Chunked rather than ``Path.read_bytes()`` because the V-JEPA 2.1
    ViT-L checkpoint runs into the GB range; loading the whole file into
    memory just to hash it is wasteful.
    """
    h = hashlib.sha256()
    with checkpoint_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _eval_loader_fingerprint(cfg: JEPARunConfig) -> str:
    """SHA-256 of the eval loader spec's canonical JSON form.

    Uses pydantic's ``model_dump_json`` (with field defaults populated)
    and wraps it in :func:`json.dumps` with ``sort_keys=True`` so the
    fingerprint is bytewise-stable across pydantic minor versions that
    may permute key ordering.
    """
    spec, _eval_seed = _build_eval_loader_spec(cfg)
    raw = spec.model_dump_json()
    # Re-serialise via json with sort_keys to make the fingerprint
    # bytewise-stable regardless of pydantic's key ordering. Per the
    # contract spec, the hash input is `json.dumps(spec.model_dump_json(),
    # sort_keys=True)` — the inner model_dump_json is a string so this
    # wraps it in JSON-encoded quotes (still deterministic).
    canonical = json.dumps(raw, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compose_cache_key(
    checkpoint_sha: str,
    loader_fingerprint: str,
    max_clips: int,
    latent_dim: int,
) -> str:
    """Compose the file-name-friendly cache key string.

    The components are joined into a single SHA-256-derived hex string
    so the on-disk file name is always a fixed length (good for both
    OS limits and quick visual inspection).
    """
    payload = json.dumps(
        {
            "checkpoint_sha": checkpoint_sha,
            "loader_fingerprint": loader_fingerprint,
            "max_clips": int(max_clips),
            "latent_dim": int(latent_dim),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialise_cache(cache: EvalCache) -> dict[str, object]:
    """Convert an :class:`EvalCache` into a torch-saveable dict.

    Stored as plain lists of numpy arrays + lists of strings. The
    ``schema_version`` field is used by the loader to detect stale
    on-disk format (any value other than :data:`SCHEMA_VERSION`
    triggers a cache miss).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "latents": cache.latents,
        "poses": cache.poses,
        "neural": cache.neural,
        "worm_ids": cache.worm_ids,
        "session_ids": cache.session_ids,
        "behavioral_states": cache.behavioral_states,
    }


def _deserialise_cache(blob: dict[str, object]) -> EvalCache:
    """Inverse of :func:`_serialise_cache`. Caller must verify schema."""
    cache = EvalCache()
    cache.latents = list(blob["latents"])  # type: ignore[arg-type]
    cache.poses = list(blob["poses"])  # type: ignore[arg-type]
    cache.neural = list(blob["neural"])  # type: ignore[arg-type]
    cache.worm_ids = list(blob["worm_ids"])  # type: ignore[arg-type]
    cache.session_ids = list(blob["session_ids"])  # type: ignore[arg-type]
    cache.behavioral_states = list(blob["behavioral_states"])  # type: ignore[arg-type]
    return cache


def _default_cache_dir() -> Path:
    """Project-root-relative default cache directory.

    ``<project_root>/.cache/encoder_cache/``. Created on first write.
    ``.cache/`` is gitignored at the repo root.
    """
    return project_root() / ".cache" / "encoder_cache"


def load_or_build_cache(
    state: JEPATrainingState,
    cfg: JEPARunConfig,
    *,
    run_dir: Path,
    cache_dir: Path | None = None,
    max_clips: int = 64,
) -> EvalCache:
    """Read-through disk cache for :func:`_build_eval_cache`.

    On a cache hit, loads the previously-saved :class:`EvalCache` from
    disk without touching the encoder. On a miss, calls the underlying
    :func:`_build_eval_cache`, writes the result to disk, and returns
    it.

    Args:
        state: trained :class:`JEPATrainingState` (encoder + heads +
            predictor restored from the run's checkpoint).
        cfg: the run's :class:`JEPARunConfig` (used to derive the eval
            loader spec for fingerprinting).
        run_dir: path to the run's results directory. Used to locate
            the checkpoint file for SHA-256 derivation.
        cache_dir: optional override for the cache root. Defaults to
            :func:`_default_cache_dir`.
        max_clips: cap on clips per cache build (forwarded to
            :func:`_build_eval_cache`). Cache key includes this value.

    Returns:
        :class:`EvalCache` — either loaded from disk (cache hit) or
        freshly built and written (cache miss).

    Environment:
        ``WORMJEPA_DISABLE_ENCODER_CACHE=1`` skips both the read and
        the write paths, forcing a fresh build every time.
    """
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        logger.info(
            "encoder cache disabled via %s=1; rebuilding from encoder",
            DISABLE_ENV_VAR,
        )
        return _build_eval_cache(state, cfg, max_clips=max_clips)

    # Compute the cache key. Any failure here (missing checkpoint,
    # missing config field) falls through to a fresh build — the cache
    # is a perf optimisation, not a correctness gate.
    try:
        checkpoint_path = run_dir / "checkpoints" / "checkpoint.pt"
        if not checkpoint_path.is_file():
            logger.warning(
                "encoder cache miss — checkpoint missing at %s; building without cache",
                checkpoint_path,
            )
            return _build_eval_cache(state, cfg, max_clips=max_clips)
        checkpoint_sha = _checkpoint_sha256(checkpoint_path)
        loader_fingerprint = _eval_loader_fingerprint(cfg)
        latent_dim = int(cfg.jepa.latent_dim)
        key = _compose_cache_key(checkpoint_sha, loader_fingerprint, max_clips, latent_dim)
    except Exception:
        logger.exception("encoder cache key derivation failed; building without cache")
        return _build_eval_cache(state, cfg, max_clips=max_clips)

    cache_root = cache_dir if cache_dir is not None else _default_cache_dir()
    cache_path = cache_root / f"{key}.pt"

    # Read path.
    if cache_path.is_file():
        try:
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            logger.exception("encoder cache hit at %s but load failed; rebuilding", cache_path)
        else:
            if isinstance(blob, dict) and blob.get("schema_version") == SCHEMA_VERSION:
                logger.info(
                    "encoder cache hit",
                    extra={"key": key, "path": str(cache_path)},
                )
                return _deserialise_cache(blob)
            logger.info(
                "encoder cache schema mismatch at %s (have %r, want %d); rebuilding",
                cache_path,
                blob.get("schema_version") if isinstance(blob, dict) else None,
                SCHEMA_VERSION,
            )

    # Write path (cache miss).
    logger.info(
        "encoder cache miss — rebuilding",
        extra={"key": key, "path": str(cache_path)},
    )
    cache = _build_eval_cache(state, cfg, max_clips=max_clips)
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        torch.save(_serialise_cache(cache), cache_path)
    except Exception:
        # Cache write failures must NEVER break eval — log + move on.
        logger.exception("encoder cache write failed for %s", cache_path)
    return cache


__all__ = [
    "DISABLE_ENV_VAR",
    "SCHEMA_VERSION",
    "EvalCache",
    "load_or_build_cache",
]
