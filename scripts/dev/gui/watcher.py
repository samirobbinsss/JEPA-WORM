"""File-watcher stub for live-tail mode.

Phase 1 deliberately ships a **polling stub** rather than the full
`watchdog`-based debounced rerun pipeline. The full design (300 ms debounce
+ `st.rerun()` on `log.jsonl` / `metrics.json` / checkpoints/ events) is
specified in §"Live-tail mode" of the UX spec and lands in Phase 2.

The polling stub is sufficient for the first-open journey (J1) because:

- The Phase 1 trajectory chart only needs the per-rerender `mtime` to
  invalidate the cache; the `cache.cached_*` layer already keys on mtime.
- Driving updates from the user toggling live-tail (which forces a rerun)
  + Streamlit's natural reruns on widget interaction covers the J1 case
  without a background thread.
- A background `watchdog` Observer + thread-safe `st.rerun()` requires
  careful Streamlit thread coordination that's out of scope for the
  Phase 1 demo.

The stub still exposes the watch / unwatch API the Phase 2 implementation
will replace, so callers in `main.py` are forward-compatible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class WatcherStatus:
    """Last-observed state for a watched directory.

    `last_observed_ts` is `None` until the first poll.
    `latest_mtime` is the max(mtime) of `log.jsonl` and `metrics.json`,
    or 0.0 if neither file exists.
    """

    run_dir: Path
    last_observed_ts: float | None
    latest_mtime: float


def poll_run_dir(run_dir: Path) -> WatcherStatus:
    """One-shot poll: return the most recent mtime among the watched files.

    Phase 2 will replace the body with a `watchdog` observer; the
    return-shape is stable so the call sites do not change.
    """
    candidates = (run_dir / "log.jsonl", run_dir / "metrics.json")
    mtimes: list[float] = []
    for path in candidates:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    latest = max(mtimes) if mtimes else 0.0
    return WatcherStatus(
        run_dir=run_dir,
        last_observed_ts=time.time(),
        latest_mtime=latest,
    )


def seconds_since(ts: float | None) -> float | None:
    """How many wall-clock seconds since `ts`? `None` if `ts` is `None`."""
    if ts is None:
        return None
    return max(0.0, time.time() - ts)
