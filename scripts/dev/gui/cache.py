"""Streamlit cache wrappers for the dev-loop GUI.

`data.load_*` is the source of truth for shape; this module wraps it with
`@st.cache_data` keyed by (path, mtime) so repeated Streamlit re-renders
don't re-parse the same file.

The on-disk cache directory lives under `scripts/dev/.gui_cache/` (gitignored)
for future Phase-2 use cases like decoded-clip MP4s. Phase 1 only needs
the in-memory `@st.cache_data` layer; the directory is initialised eagerly
so downstream phases don't need to re-add the setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from scripts.dev.gui import data
from scripts.dev.gui.data import LogEntry, RunSummary

# Cache directory under the dev tree; never inside `results/`.
GUI_CACHE_DIR: Path = Path(__file__).resolve().parents[1] / ".gui_cache"


def ensure_cache_dir() -> Path:
    """Create the on-disk cache dir if it doesn't exist; return its path."""
    GUI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return GUI_CACHE_DIR


def _mtime_or_zero(path: Path) -> float:
    """Stat helper that tolerates missing files (returns 0.0)."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def cached_list_runs(results_root_str: str, _mtime_key: float) -> list[str]:
    """Return run-id names under `results_root_str`.

    `_mtime_key` is the root's mtime; passing it through the cache key means
    new run directories invalidate cache without manual refresh.
    """
    return [p.name for p in data.list_run_directories(Path(results_root_str))]


@st.cache_data(show_spinner=False)
def cached_run_summary(run_dir_str: str, _mtime_key: float) -> RunSummary:
    """Cached `RunSummary` keyed by (run_dir, run_dir mtime)."""
    return data.load_run_summary(Path(run_dir_str))


@st.cache_data(show_spinner=False)
def cached_log_entries(
    log_path_str: str, _mtime_key: float, max_lines: int | None = None
) -> list[LogEntry]:
    """Cached `log.jsonl` parse keyed by (path, file mtime)."""
    return data.load_log_entries(Path(log_path_str).parent, max_lines=max_lines)


@st.cache_data(show_spinner=False)
def cached_metrics(metrics_path_str: str, _mtime_key: float) -> Any:
    """Cached `metrics.json` parse keyed by (path, mtime).

    The return value type is `MetricsOutput | None`; declared as `Any` to
    avoid forcing every caller through a pydantic import. The shape is
    stable and documented in `data.load_metrics`.
    """
    return data.load_metrics(Path(metrics_path_str).parent)


def invalidate_all() -> None:
    """Drop every entry from all `@st.cache_data` caches in this module.

    Wired to the "Refresh cache" button (Phase 2+); kept here so that
    component code has a single function to call.
    """
    cached_list_runs.clear()
    cached_run_summary.clear()
    cached_log_entries.clear()
    cached_metrics.clear()


def list_runs(results_root: Path) -> list[str]:
    """Convenience: cached list of run-ids under `results_root`."""
    return cached_list_runs(str(results_root), _mtime_or_zero(results_root))


def run_summary(run_dir: Path) -> RunSummary:
    """Convenience: cached `RunSummary` for `run_dir`."""
    return cached_run_summary(str(run_dir), _mtime_or_zero(run_dir))


def log_entries(run_dir: Path, max_lines: int | None = None) -> list[LogEntry]:
    """Convenience: cached log entries for `run_dir`."""
    log_path = run_dir / data.FILE_LOG
    return cached_log_entries(str(log_path), _mtime_or_zero(log_path), max_lines=max_lines)


def metrics(run_dir: Path) -> Any:
    """Convenience: cached `MetricsOutput | None` for `run_dir`."""
    metrics_path = run_dir / data.FILE_METRICS
    return cached_metrics(str(metrics_path), _mtime_or_zero(metrics_path))
