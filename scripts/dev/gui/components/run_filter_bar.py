"""C11 — RunFilterBar component (Phase 3).

Filter the run-table by substring (run_id / config_slug / git_sha),
gate-status outcome, and a lower-bound mtime. All state lives in
``st.session_state`` via :mod:`scripts.dev.gui.state` so component code
stays free of widget-key bookkeeping.

The gate-status filter is the only one that needs a per-run side load
(reading ``metrics.json``). Cached behind ``cache.metrics``; the cache
key is the file mtime so freshly-written runs invalidate naturally.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time

import streamlit as st

from scripts.dev.gui import cache, state
from scripts.dev.gui.data import RunSummary
from wormjepa.eval.gates import evaluate_gates

logger = logging.getLogger(__name__)

_GATE_STATUS_OPTIONS: tuple[str, ...] = (
    "cleared",
    "kill_criterion_fired",
    "reframed",
    "pending",
)


def _outcome_for_summary(summary: RunSummary) -> str:
    """Resolve a run's gate-outcome category for filtering purposes.

    Runs without ``metrics.json`` (or unparseable ones) collapse to
    ``"pending"`` — which is the same bucket the verdict table uses.
    """
    if not summary.has_metrics:
        return "pending"
    metrics = cache.metrics(summary.path)
    if metrics is None:
        return "pending"
    try:
        return evaluate_gates(metrics).outcome
    except (ValueError, KeyError, AttributeError) as exc:
        logger.warning("evaluate_gates failed on %s: %s", summary.path, exc)
        return "pending"


def _substring_matches(summary: RunSummary, needle: str) -> bool:
    haystack = " ".join(
        s for s in (summary.run_id, summary.config_slug, summary.git_sha) if s is not None
    ).lower()
    return needle.lower() in haystack


def _apply_filters(summaries: list[RunSummary]) -> list[RunSummary]:
    text = state.get_filter_text().strip()
    gate_filter = set(state.get_filter_gate_statuses())
    mtime_after = state.get_filter_mtime_after()
    out: list[RunSummary] = []
    for s in summaries:
        if text and not _substring_matches(s, text):
            continue
        if gate_filter and _outcome_for_summary(s) not in gate_filter:
            continue
        if mtime_after is not None and s.mtime < mtime_after:
            continue
        out.append(s)
    return out


def render_run_filter_bar(summaries: list[RunSummary]) -> list[RunSummary]:
    """Render the filter bar above the run-table; return the filtered list.

    The original ``summaries`` order (mtime-descending) is preserved through
    the filter.
    """
    text_col, gate_col, date_col, count_col = st.columns([3, 3, 2, 1])
    with text_col:
        new_text = st.text_input(
            "Filter (run_id / config / sha)",
            value=state.get_filter_text(),
            placeholder="e.g. seed-42",
            key="gui.filter.text.widget",
            label_visibility="collapsed",
        )
        state.set_filter_text(new_text)
    with gate_col:
        new_statuses = st.multiselect(
            "Gate status",
            options=list(_GATE_STATUS_OPTIONS),
            default=state.get_filter_gate_statuses(),
            key="gui.filter.gate.widget",
            label_visibility="collapsed",
            placeholder="Gate status…",
        )
        state.set_filter_gate_statuses(new_statuses)
    with date_col:
        current_after = state.get_filter_mtime_after()
        default_date = (
            datetime.fromtimestamp(current_after, tz=UTC).date()
            if current_after is not None
            else None
        )
        chosen = st.date_input(
            "Modified on / after",
            value=default_date,
            key="gui.filter.date.widget",
            label_visibility="collapsed",
        )
        if isinstance(chosen, date):
            dt = datetime.combine(chosen, time.min, tzinfo=UTC)
            state.set_filter_mtime_after(dt.timestamp())
        else:
            state.set_filter_mtime_after(None)
    with count_col:
        active = state.active_filter_count()
        if active:
            st.caption(f"{active} filter{'s' if active != 1 else ''}")
            if st.button("Clear", key="gui.filter.clear", help="Clear all filters"):
                state.clear_filters()
                st.rerun()

    filtered = _apply_filters(summaries)
    if not filtered and summaries:
        st.info("No runs match these filters. Use the Clear button to reset.")
    return filtered
