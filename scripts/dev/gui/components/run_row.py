"""C1 — RunRow (basic variant) component.

Phase 1 ships the run-table as a Streamlit `st.dataframe` populated by
mapping each `RunSummary` to a row dict (the spec's C1.basic variant). The
hover/selection states from the spec live on `st.dataframe`'s native row
selection; the detailed-variant (config-diff expansion) is a Phase 4 item.

`render_run_table` returns the user's currently-selected `run_id` (or
`None`), which the caller stores in session state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import streamlit as st

from scripts.dev.gui import state
from scripts.dev.gui.data import RunSummary


def render_run_table(summaries: list[RunSummary], selected: str | None) -> str | None:
    """Render the left-column run-table.

    Args:
        summaries: Run summaries already loaded by the caller (cached via
            `cache.run_summary`).
        selected: Currently-selected run-id; pre-selected in the dataframe.

    Returns:
        The newly-selected run-id (or `None` if no row is selected).

    Notes:
        - We use `st.dataframe` rather than `st.data_editor` because the GUI
          is read-only (§"Read-only against `results/<run-id>/`").
        - Run-id coloring is assigned via `state.assign_run_color` so colors
          stay sticky across navigation; a `Color` column displays the
          assigned hex as a swatch through `column_config.TextColumn`'s
          `help` field.
    """
    if not summaries:
        st.info("No runs to display.")
        return selected

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        color = state.assign_run_color(summary.run_id)
        rows.append(
            {
                "run_id": summary.run_id,
                "color": color,
                "seed": summary.seed or "—",
                "config": summary.config_slug or "—",
                "git_sha": _short_sha(summary.git_sha),
                "gpu_hours": (f"{summary.gpu_hours:.2f}" if summary.gpu_hours is not None else "—"),
                "metrics": _flag(summary.has_metrics),
                "log": _flag(summary.has_log),
                "mtime": _fmt_mtime(summary.mtime),
            }
        )

    # Determine the default selection index so the dataframe pre-highlights it.
    default_index: list[int] = []
    if selected is not None:
        for idx, summary in enumerate(summaries):
            if summary.run_id == selected:
                default_index = [idx]
                break

    event = st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "run_id": st.column_config.TextColumn("run id", help="Run identifier"),
            "color": st.column_config.TextColumn(
                "color", help="Sticky run-id color (tableau-colorblind10)"
            ),
            "seed": st.column_config.TextColumn("seed"),
            "config": st.column_config.TextColumn("config"),
            "git_sha": st.column_config.TextColumn("git sha"),
            "gpu_hours": st.column_config.TextColumn("gpu·h"),
            "metrics": st.column_config.TextColumn("metrics"),
            "log": st.column_config.TextColumn("log"),
            "mtime": st.column_config.TextColumn("modified"),
        },
        on_select="rerun",
        selection_mode="single-row",
        key="gui.run_table",
    )

    rows_selected = _extract_selected_rows(event)
    if rows_selected:
        return summaries[rows_selected[0]].run_id
    if default_index:
        return summaries[default_index[0]].run_id
    return selected


def _extract_selected_rows(event: Any) -> list[int]:
    """Pull selected row indices out of an `st.dataframe` selection event.

    Streamlit's selection-event shape is mildly inconsistent across versions
    (sometimes a dict-like, sometimes attribute-accessible). This helper
    isolates the version-sniffing.
    """
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return []
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows", [])
    if rows is None:
        return []
    return list(rows)


def _short_sha(sha: str | None) -> str:
    if not sha:
        return "—"
    return sha[:8]


def _flag(present: bool) -> str:
    """Visible icon for presence/absence; pairs ✓/— with text per a11y rules."""
    return "✓" if present else "—"


def _fmt_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
