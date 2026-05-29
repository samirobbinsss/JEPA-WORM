"""Session-state schema and helpers for the dev-loop inspection GUI.

All cross-component coupling (selected run, pinned step, live-tail toggle,
run-id-to-color mapping) routes through `st.session_state` so individual
components stay pure functions of state. The keys are typed via constants
to keep agents from inventing parallel keys.

Phase 1 scope: pinned step + multi-select cross-run are declared but not
fully wired (those land in Phase 2 / Phase 3 per the implementation roadmap
in the UX spec).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import streamlit as st

from scripts.dev.gui import theme

# ---------------------------------------------------------------------------
# Session-state keys (string constants → typo-resistant cross-module access)
# ---------------------------------------------------------------------------

KEY_RESULTS_ROOT: Final[str] = "gui.results_root"
KEY_SELECTED_RUN: Final[str] = "gui.selected_run"
KEY_LIVE_TAIL: Final[str] = "gui.live_tail"
KEY_PINNED_STEP: Final[str] = "gui.pinned_step"
KEY_LAST_UPDATE_TS: Final[str] = "gui.last_update_ts"
KEY_RUN_COLOR_MAP: Final[str] = "gui.run_color_map"
KEY_SMOOTHING: Final[str] = "gui.smoothing"

# Phase 3 — multi-run + filter-bar state.
KEY_SELECTED_RUNS: Final[str] = "gui.selected_runs"
KEY_FILTER_TEXT: Final[str] = "gui.filter.text"
KEY_FILTER_GATE_STATUSES: Final[str] = "gui.filter.gate_statuses"
KEY_FILTER_MTIME_AFTER: Final[str] = "gui.filter.mtime_after"

# Phase 2 default for the EMA smoothing slider (`C10 SmoothingSlider`). 0.0
# means raw trajectory; matches TensorBoard's default-off behaviour.
SMOOTHING_DEFAULT: Final[float] = 0.0
SMOOTHING_MIN: Final[float] = 0.0
SMOOTHING_MAX: Final[float] = 0.99


@dataclass(slots=True)
class GuiSessionState:
    """Typed view over the relevant `st.session_state` slice.

    Intentionally not a pydantic model — Streamlit's session_state is a
    dict-like, mutable proxy, and we want a lightweight read/write facade
    without runtime validation overhead per re-render.
    """

    results_root: str | None = None
    selected_run: str | None = None
    live_tail: bool = False
    pinned_step: int | None = None
    last_update_ts: float | None = None
    run_color_map: dict[str, str] = field(default_factory=dict)
    smoothing: float = SMOOTHING_DEFAULT
    selected_runs: list[str] = field(default_factory=list)
    filter_text: str = ""
    filter_gate_statuses: list[str] = field(default_factory=list)
    filter_mtime_after: float | None = None


def init_session_state(results_root: str) -> None:
    """Seed Streamlit session state on first run.

    Idempotent — re-invoking on a hot reload leaves user-set values in place.
    The results root is always overwritten from the CLI argument because
    it is the source of truth for the current invocation.
    """
    st.session_state[KEY_RESULTS_ROOT] = results_root
    st.session_state.setdefault(KEY_SELECTED_RUN, None)
    st.session_state.setdefault(KEY_LIVE_TAIL, False)
    st.session_state.setdefault(KEY_PINNED_STEP, None)
    st.session_state.setdefault(KEY_LAST_UPDATE_TS, None)
    st.session_state.setdefault(KEY_RUN_COLOR_MAP, {})
    st.session_state.setdefault(KEY_SMOOTHING, SMOOTHING_DEFAULT)
    st.session_state.setdefault(KEY_SELECTED_RUNS, [])
    st.session_state.setdefault(KEY_FILTER_TEXT, "")
    st.session_state.setdefault(KEY_FILTER_GATE_STATUSES, [])
    st.session_state.setdefault(KEY_FILTER_MTIME_AFTER, None)


def get_state() -> GuiSessionState:
    """Return a typed snapshot of the GUI's session state."""
    return GuiSessionState(
        results_root=st.session_state.get(KEY_RESULTS_ROOT),
        selected_run=st.session_state.get(KEY_SELECTED_RUN),
        live_tail=bool(st.session_state.get(KEY_LIVE_TAIL, False)),
        pinned_step=st.session_state.get(KEY_PINNED_STEP),
        last_update_ts=st.session_state.get(KEY_LAST_UPDATE_TS),
        run_color_map=dict(st.session_state.get(KEY_RUN_COLOR_MAP, {})),
        smoothing=float(st.session_state.get(KEY_SMOOTHING, SMOOTHING_DEFAULT)),
        selected_runs=list(st.session_state.get(KEY_SELECTED_RUNS, [])),
        filter_text=str(st.session_state.get(KEY_FILTER_TEXT, "")),
        filter_gate_statuses=list(st.session_state.get(KEY_FILTER_GATE_STATUSES, [])),
        filter_mtime_after=st.session_state.get(KEY_FILTER_MTIME_AFTER),
    )


def assign_run_color(run_id: str) -> str:
    """Get-or-assign a tableau-colorblind10 color for `run_id`.

    Assignment order is the order in which runs are first observed; the
    mapping persists in session state so the same run keeps the same color
    across navigation (§"Cross-run color stability").
    """
    color_map: dict[str, str] = st.session_state.setdefault(KEY_RUN_COLOR_MAP, {})
    if run_id in color_map:
        return color_map[run_id]
    color = theme.run_color(len(color_map))
    color_map[run_id] = color
    return color


def set_selected_run(run_id: str | None) -> None:
    """Update the actively selected run."""
    st.session_state[KEY_SELECTED_RUN] = run_id


def set_live_tail(enabled: bool) -> None:
    """Toggle live-tail mode."""
    st.session_state[KEY_LIVE_TAIL] = enabled


def set_pinned_step(step: int | None) -> None:
    """Update the pinned-step (the focus point of the right-column panels)."""
    st.session_state[KEY_PINNED_STEP] = step


def get_pinned_step() -> int | None:
    """Read the pinned step. Returns `None` if no step is pinned."""
    value = st.session_state.get(KEY_PINNED_STEP)
    if isinstance(value, int):
        return value
    return None


def set_smoothing(alpha: float) -> None:
    """Update the EMA smoothing factor; clamped to ``[SMOOTHING_MIN, SMOOTHING_MAX]``."""
    st.session_state[KEY_SMOOTHING] = max(SMOOTHING_MIN, min(SMOOTHING_MAX, float(alpha)))


def get_smoothing() -> float:
    """Read the EMA smoothing factor (default `SMOOTHING_DEFAULT`)."""
    return float(st.session_state.get(KEY_SMOOTHING, SMOOTHING_DEFAULT))


# ---------------------------------------------------------------------------
# Phase 3 — multi-run selection + filter-bar helpers
# ---------------------------------------------------------------------------


def get_selected_runs() -> list[str]:
    """Return a copy of the multi-select run-id list (order-preserving)."""
    return list(st.session_state.get(KEY_SELECTED_RUNS, []))


def set_selected_runs(run_ids: list[str]) -> None:
    """Replace the multi-select run-id list (deduped, order-preserving)."""
    seen: set[str] = set()
    deduped: list[str] = []
    for rid in run_ids:
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(rid)
    st.session_state[KEY_SELECTED_RUNS] = deduped


def add_selected_run(run_id: str) -> None:
    """Add ``run_id`` to the multi-select list if not already present."""
    current = list(st.session_state.get(KEY_SELECTED_RUNS, []))
    if run_id not in current:
        current.append(run_id)
        st.session_state[KEY_SELECTED_RUNS] = current


def remove_selected_run(run_id: str) -> None:
    """Drop ``run_id`` from the multi-select list (no-op if absent)."""
    current = [r for r in st.session_state.get(KEY_SELECTED_RUNS, []) if r != run_id]
    st.session_state[KEY_SELECTED_RUNS] = current


def clear_selected_runs() -> None:
    """Empty the multi-select list (disengage cross-run mode)."""
    st.session_state[KEY_SELECTED_RUNS] = []


def get_filter_text() -> str:
    """Read the current substring filter."""
    return str(st.session_state.get(KEY_FILTER_TEXT, ""))


def set_filter_text(text: str) -> None:
    """Write the substring filter."""
    st.session_state[KEY_FILTER_TEXT] = text


def get_filter_gate_statuses() -> list[str]:
    """Read the active gate-status filter set."""
    return list(st.session_state.get(KEY_FILTER_GATE_STATUSES, []))


def set_filter_gate_statuses(statuses: list[str]) -> None:
    """Write the gate-status multiselect filter."""
    st.session_state[KEY_FILTER_GATE_STATUSES] = list(statuses)


def get_filter_mtime_after() -> float | None:
    """Read the lower-bound mtime filter (epoch seconds; ``None`` for no filter)."""
    value = st.session_state.get(KEY_FILTER_MTIME_AFTER)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def set_filter_mtime_after(mtime: float | None) -> None:
    """Write the lower-bound mtime filter."""
    st.session_state[KEY_FILTER_MTIME_AFTER] = mtime


def clear_filters() -> None:
    """Reset all RunFilterBar inputs to defaults."""
    st.session_state[KEY_FILTER_TEXT] = ""
    st.session_state[KEY_FILTER_GATE_STATUSES] = []
    st.session_state[KEY_FILTER_MTIME_AFTER] = None


def active_filter_count() -> int:
    """Return how many RunFilterBar filters are currently active."""
    count = 0
    if get_filter_text().strip():
        count += 1
    if get_filter_gate_statuses():
        count += 1
    if get_filter_mtime_after() is not None:
        count += 1
    return count
