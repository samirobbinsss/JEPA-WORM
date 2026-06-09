"""Single-producer ordered background prefetch for dataset loaders (FIX 1).

The headline training loop iterates a single synchronous ``DatasetSample``
stream on the calling thread; for the BAAIWorm corpus each clip is synthesised
in pure Python (rasterisation + projection), so the GPU sits idle while the
next sample is built. :class:`PrefetchLoader` interposes a bounded queue and
*exactly one* producer thread that runs one (or ``depth``) steps ahead, giving
producer/consumer overlap without changing the emitted stream.

Integrity contract (this is a scheduling change only):

* The producer iterates a single ``iter(inner)`` in declared order and pushes
  each sample onto a FIFO :class:`queue.Queue`; the consumer yields strictly in
  that order. The emitted sample sequence is therefore byte-identical to
  ``iter(inner)``.
* All RNG consumed while building samples lives inside the wrapped loader
  (each loader seeds its own local ``torch.Generator``). The producer thread
  touches no global ``torch``/``numpy``/``random`` state, so the downstream
  model RNG — consumed on the consumer thread by ``train_jepa`` — is identical
  to the synchronous path under the same seed.

Re-iterability matches the training loop's per-epoch ``iter(dataset)``
semantics: every ``__iter__`` call spawns a *fresh* producer over a *fresh*
``iter(inner)``. Producer exceptions are propagated to the consumer; on early
consumer exit (``break`` or exception) the producer is signalled to stop and
drained so no daemon thread is left blocked on a full queue.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from wormjepa.data import DatasetSample


class _Sentinel:
    """Marker types placed on the queue to signal end-of-stream / error.

    Distinct singleton instances let the consumer disambiguate a normal end
    (``_END``) from a propagated producer exception (``_Error``) without any
    chance of colliding with a real :class:`DatasetSample`.
    """


_END = _Sentinel()
"""Singleton placed on the queue once the producer exhausts ``iter(inner)``."""


class _Error:
    """Wraps a producer-side exception so it can travel through the queue."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


class PrefetchLoader:
    """Ordered single-thread prefetch wrapper around an iterable of samples.

    Args:
        inner: The wrapped iterable (e.g. a ``ChainedLoader`` /
            ``BAAIWormLoader``). Must be re-iterable: ``iter(inner)`` is called
            afresh on every :meth:`__iter__` so per-epoch iteration matches the
            synchronous loop.
        depth: Bounded queue capacity, i.e. how many samples the producer may
            run ahead of the consumer. ``depth=8`` overlaps ~8 clips of
            synthesis with GPU compute; the producer blocks once the queue is
            full, so memory is bounded.
    """

    def __init__(self, inner: Iterable[DatasetSample], depth: int = 8) -> None:
        if depth < 1:
            msg = f"PrefetchLoader depth must be >= 1, got {depth}"
            raise ValueError(msg)
        self._inner = inner
        self._depth = depth

    def __iter__(self) -> Iterator[DatasetSample]:
        # Bounded FIFO: the producer blocks once `depth` samples are queued, so
        # it runs at most `depth` steps ahead and memory stays bounded.
        q: queue.Queue[DatasetSample | _Sentinel | _Error] = queue.Queue(maxsize=self._depth)
        # Set by the consumer on early exit (break / exception) to tell the
        # producer to stop pushing and unwind.
        stop = threading.Event()

        def _produce() -> None:
            try:
                for sample in iter(self._inner):
                    # Re-check stop before every (potentially blocking) put so
                    # an early consumer exit cannot leave us parked forever on a
                    # full queue. Use a timeout-poll loop rather than an
                    # unbounded blocking put for the same reason.
                    while not stop.is_set():
                        try:
                            q.put(sample, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    else:
                        return  # stop requested mid-stream
            except BaseException as exc:
                # Surface ANY producer-side failure on the consumer thread,
                # preserving the synchronous path's exception semantics.
                self._safe_put(q, _Error(exc), stop)
                return
            # Normal exhaustion: signal end-of-stream (best-effort if stopping).
            self._safe_put(q, _END, stop)

        producer = threading.Thread(target=_produce, name="wormjepa-prefetch", daemon=True)
        producer.start()

        try:
            while True:
                item = q.get()
                if item is _END:
                    return
                if isinstance(item, _Error):
                    raise item.exc
                # `item` is a DatasetSample here.
                yield item  # type: ignore[misc]
        finally:
            # Covers normal completion, an exception raised here, and — most
            # importantly — early consumer exit (the caller `break`s out of the
            # loop, e.g. once n_steps samples are pulled). Signal the producer
            # to stop, then drain the queue so a producer parked on a full
            # queue can make progress and exit. Daemon status is a backstop;
            # this is the deterministic cleanup.
            stop.set()
            while producer.is_alive():
                try:
                    q.get_nowait()
                except queue.Empty:
                    # Producer may be between `stop` checks; give it a moment
                    # to observe `stop` and unwind, then re-drain.
                    producer.join(timeout=0.1)

    @staticmethod
    def _safe_put(
        q: queue.Queue[DatasetSample | _Sentinel | _Error],
        item: _Sentinel | _Error,
        stop: threading.Event,
    ) -> None:
        """Put a terminal marker without deadlocking if the consumer has left.

        Once the consumer exits it stops draining, so a plain blocking ``put``
        on a full queue would hang the producer. Poll with a short timeout and
        bail the moment ``stop`` is set.
        """
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue
