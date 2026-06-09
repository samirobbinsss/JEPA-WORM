"""Unit tests for ``wormjepa.data.prefetch.PrefetchLoader`` (FIX 1).

The prefetch loader is a *scheduling-only* change: a single ordered producer
thread runs one step ahead of the consumer over a bounded FIFO queue. These
tests pin the two integrity properties that make the change safe:

1. STREAM byte-equality — the emitted sample sequence (order + every
   ``video_clip`` / ``pose`` / ``neural`` tensor + ``worm_id``) is identical to
   the synchronous iterator, holds under re-iteration, and an early consumer
   break leaves no thread blocked.
2. DETERMINISM — the global ``torch``/``numpy``/``random`` RNG state is
   untouched by prefetch iteration (the property that guarantees identical
   downstream model RNG), and a tiny end-to-end CPU training loop produces a
   byte-identical per-step loss trajectory with prefetch on vs. off.
"""

from __future__ import annotations

import json
import os
import random
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from wormjepa.data.loaders.baaiworm import BAAIWormLoader
from wormjepa.data.loaders.synthetic import SyntheticLoader
from wormjepa.data.prefetch import PrefetchLoader

if TYPE_CHECKING:
    from collections.abc import Iterator

    from wormjepa.data import DatasetSample


# A tiny, fully-deterministic loader: fixed seed + tiny dims so the whole test
# is fast. SyntheticLoader seeds its own local torch.Generator on every
# __iter__, so it is re-iterable and bit-identical across passes.
def _tiny_loader() -> SyntheticLoader:
    return SyntheticLoader(
        n_worms=4,
        clips_per_worm=3,
        clip_frames=4,
        image_size=(8, 8),
        n_keypoints=4,
        n_neurons=6,
        seed=1234,
    )


def _assert_sample_equal(a: DatasetSample, b: DatasetSample) -> None:
    """Assert two DatasetSamples are byte-identical across every field."""
    assert torch.equal(a.video_clip, b.video_clip)
    if a.pose is None or b.pose is None:
        assert a.pose is None and b.pose is None
    else:
        assert torch.equal(a.pose, b.pose)
    if a.neural is None or b.neural is None:
        assert a.neural is None and b.neural is None
    else:
        assert torch.equal(a.neural, b.neural)
    if a.behavioral_state is None or b.behavioral_state is None:
        assert a.behavioral_state is None and b.behavioral_state is None
    else:
        assert torch.equal(a.behavioral_state, b.behavioral_state)
    assert a.worm_id == b.worm_id
    assert a.session_id == b.session_id
    assert a.source_dataset == b.source_dataset


# ---------------------------------------------------------------------------
# (1) STREAM byte-equality
# ---------------------------------------------------------------------------


def test_prefetch_stream_is_byte_identical_to_raw() -> None:
    """The prefetched stream equals the synchronous stream sample-for-sample."""
    raw = list(_tiny_loader())
    assert len(raw) == 12  # 4 worms * 3 clips — sanity on the fixture

    prefetched = list(PrefetchLoader(_tiny_loader(), depth=8))

    assert len(prefetched) == len(raw)
    # ORDER + every tensor identical.
    for expect, got in zip(raw, prefetched, strict=True):
        _assert_sample_equal(expect, got)


def test_prefetch_stream_byte_identical_with_baaiworm() -> None:
    """Same byte-equality guarantee for the BAAIWorm loader (the real FIX-1
    target: pure-Python clip synthesis on a fixed seed)."""
    common = dict(
        n_worms=3,
        clips_per_worm=2,
        clip_frames=4,
        image_size=(8, 8),
        n_keypoints=4,
        n_neurons=4,
        seed=7,
        verify_provenance=False,
    )
    raw = list(BAAIWormLoader(**common))  # type: ignore[arg-type]
    prefetched = list(PrefetchLoader(BAAIWormLoader(**common), depth=4))  # type: ignore[arg-type]

    assert len(prefetched) == len(raw) == 6
    for expect, got in zip(raw, prefetched, strict=True):
        _assert_sample_equal(expect, got)


@pytest.mark.parametrize("depth", [1, 2, 8, 64])
def test_prefetch_byte_identical_across_depths(depth: int) -> None:
    """Queue depth is a buffering knob only — never changes the stream."""
    raw = list(_tiny_loader())
    prefetched = list(PrefetchLoader(_tiny_loader(), depth=depth))
    assert len(prefetched) == len(raw)
    for expect, got in zip(raw, prefetched, strict=True):
        _assert_sample_equal(expect, got)


def test_prefetch_is_reiterable_two_passes_identical() -> None:
    """Each __iter__ spawns a fresh producer over a fresh iter(inner); two
    passes over the SAME PrefetchLoader give identical streams (matches the
    training loop's per-epoch iter(dataset))."""
    pf = PrefetchLoader(_tiny_loader(), depth=8)
    first = list(pf)
    second = list(pf)

    assert len(first) == len(second) == 12
    for a, b in zip(first, second, strict=True):
        _assert_sample_equal(a, b)


def test_prefetch_early_break_cleans_up_producer() -> None:
    """Consumer breaks after k samples — must not hang and must leave no live
    prefetch producer thread blocked on a full queue."""
    before = {t.name for t in threading.enumerate()}

    pf = PrefetchLoader(_tiny_loader(), depth=2)
    collected: list[DatasetSample] = []
    for i, sample in enumerate(pf):
        collected.append(sample)
        if i == 2:  # break well before exhaustion (12 samples available)
            break

    assert len(collected) == 3

    # The producer is a daemon named "wormjepa-prefetch"; the generator's
    # finally-block signals stop + drains, so it should die promptly. Poll
    # briefly to avoid a flaky race on slow CI.
    deadline = threading.Event()
    timer = threading.Timer(2.0, deadline.set)
    timer.start()
    try:
        while not deadline.is_set():
            live = {t.name for t in threading.enumerate() if t.name == "wormjepa-prefetch"}
            if not live:
                break
        else:  # pragma: no cover - only hit if cleanup regressed
            pytest.fail("prefetch producer thread still alive after early break")
    finally:
        timer.cancel()

    after = {t.name for t in threading.enumerate()}
    # No net new non-daemon threads leaked.
    assert "wormjepa-prefetch" not in after - before


def test_prefetch_propagates_producer_exception() -> None:
    """A failure raised while building samples surfaces on the consumer thread,
    preserving the synchronous path's exception semantics."""

    class _BoomError(RuntimeError):
        pass

    class _ExplodingLoader:
        def __iter__(self) -> Iterator[DatasetSample]:
            # Yield one good sample, then explode mid-stream.
            yield from list(_tiny_loader())[:1]
            raise _BoomError("synthesis failed")

    pf = PrefetchLoader(_ExplodingLoader(), depth=4)
    seen = 0
    with pytest.raises(_BoomError, match="synthesis failed"):
        for _ in pf:
            seen += 1
    assert seen == 1  # the good sample came through before the error propagated


def test_prefetch_rejects_zero_depth() -> None:
    with pytest.raises(ValueError, match="depth must be >= 1"):
        PrefetchLoader(_tiny_loader(), depth=0)


# ---------------------------------------------------------------------------
# (2) DETERMINISM
# ---------------------------------------------------------------------------


def test_prefetch_does_not_touch_global_rng_state() -> None:
    """Draining the prefetch queue must not perturb the global torch / numpy /
    random RNG. This is the invariant that guarantees identical downstream
    model RNG: train_jepa consumes the global RNG on the consumer thread, while
    sample building uses the loader's OWN local generators on the producer
    thread.
    """
    # Seed all three global RNGs to a known point.
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)

    torch_before = torch.random.get_rng_state().clone()
    np_before = np.random.get_state()
    py_before = random.getstate()

    # Fully drain a prefetch pass (forces the producer to iterate end-to-end).
    drained = list(PrefetchLoader(_tiny_loader(), depth=8))
    assert len(drained) == 12

    torch_after = torch.random.get_rng_state()
    np_after = np.random.get_state()
    py_after = random.getstate()

    assert torch.equal(torch_before, torch_after), "global torch RNG state changed"
    # numpy legacy state is a tuple; element [1] is the uint32 key array.
    assert np.array_equal(np_before[1], np_after[1]), "global numpy RNG state changed"  # type: ignore[index]
    assert py_before == py_after, "global python random state changed"


def _run_tiny_jepa(tmp_path: Path, *, prefetch: bool, seed: int = 42) -> list[float]:
    """Run a few real JEPA training steps on CPU and return the per-step loss
    trajectory read back from log.jsonl. Imports are local so test collection
    stays cheap and any heavy import failure skips only this test."""
    from wormjepa.configs.dataset import DatasetLoaderSpec, DatasetSection
    from wormjepa.configs.jepa_config import JEPARunConfig, JEPASection
    from wormjepa.training import runner as runner_mod

    # Force CPU so the trajectory is reproducible regardless of host accel.
    monkey_device = torch.device("cpu")

    cfg = JEPARunConfig(
        schema_version=1,
        jepa=JEPASection(
            model_name="vit_tiny_patch16_224",
            img_size=16,
            latent_dim=16,
            masking_ratio=0.5,
            n_steps=6,
            learning_rate=1e-4,
            ema_decay=0.99,
            seed=seed,
        ),
        dataset=DatasetSection(
            loaders=[
                DatasetLoaderSpec(
                    name="synthetic",
                    clip_frames=4,
                    image_size=16,
                    n_worms=4,
                    clips_per_worm=4,
                    n_keypoints=4,
                    # Runner builds the neural warm-start head with a fixed
                    # n_neurons=8 (runner._build_state); match it so the head
                    # accepts the synthetic neural target.
                    n_neurons=8,
                )
            ]
        ),
    )

    run_id = f"prefetch-det-{'on' if prefetch else 'off'}"
    results_dir = tmp_path / ("on" if prefetch else "off")

    # Pin device to CPU and redirect results into tmp_path via project_root.
    import wormjepa.training.runner as rmod

    orig_device = rmod._select_device  # type: ignore[attr-defined]
    orig_root = rmod.project_root

    def _cpu_device() -> torch.device:
        return monkey_device

    def _tmp_root() -> Path:
        return results_dir

    if prefetch:
        os.environ["WORMJEPA_PREFETCH"] = "1"
    else:
        os.environ.pop("WORMJEPA_PREFETCH", None)

    rmod._select_device = _cpu_device  # type: ignore[attr-defined]
    rmod.project_root = _tmp_root  # type: ignore[assignment]
    try:
        runner_mod.run_jepa(cfg, run_id=run_id)
    finally:
        rmod._select_device = orig_device  # type: ignore[attr-defined]
        rmod.project_root = orig_root  # type: ignore[assignment]
        os.environ.pop("WORMJEPA_PREFETCH", None)

    log_path = results_dir / "results" / run_id / "log.jsonl"
    losses: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        losses.append(float(rec["extra"]["loss"]))
    return losses


def test_prefetch_end_to_end_loss_trajectory_identical(tmp_path: Path) -> None:
    """End-to-end determinism: the per-step loss sequence with prefetch ON is
    byte-identical (atol=0) to prefetch OFF under the same seed."""
    # Confirm the runner exposes the symbols we monkeypatch; if its internals
    # have moved, skip rather than silently passing on a no-op patch.
    import wormjepa.training.runner as rmod

    if not hasattr(rmod, "_select_device") or not hasattr(rmod, "project_root"):
        pytest.skip("runner internals (_select_device/project_root) not patchable")

    losses_off = _run_tiny_jepa(tmp_path, prefetch=False, seed=42)
    losses_on = _run_tiny_jepa(tmp_path, prefetch=True, seed=42)

    assert len(losses_off) == len(losses_on)
    assert len(losses_off) >= 1
    for step, (off, on) in enumerate(zip(losses_off, losses_on, strict=True)):
        assert off == on, f"step {step}: prefetch-off loss {off!r} != prefetch-on {on!r}"
