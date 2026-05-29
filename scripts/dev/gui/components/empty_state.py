"""C9 — EmptyState component.

Renders a workable, non-alarming empty state. Two variants per the UX spec:

- `waiting`: results dir exists but has no runs yet. Pair with
  `LiveTailIndicator` to communicate "I'm watching for new runs."
- `permanent`: results dir does not exist or path is broken. Caller passes
  `hint` to suggest a fix (e.g., "Check that the path exists").

Forbidden by §"Empty and Loading States": indeterminate spinners,
"Loading..." without ETA. This component uses static text + an emoji icon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import streamlit as st

EmptyStateVariant = Literal["waiting", "permanent"]


def render_empty_state(
    title: str,
    message: str,
    variant: EmptyStateVariant = "waiting",
    hint: str | None = None,
    icon: str = "📂",
) -> None:
    """Render the empty state.

    The body region remains visible behind this; the empty state is content,
    not an overlay (§"Modal / Overlay Patterns — Forbidden").
    """
    with st.container(border=True):
        st.markdown(f"### {icon} {title}")
        st.write(message)
        if hint:
            st.caption(hint)
        if variant == "waiting":
            st.caption("Auto-refresh is on; new runs will appear here without reloading.")
        else:
            st.caption("Pass a different path on the CLI to inspect another results root.")


def render_results_root_empty(results_root: Path | None) -> None:
    """Specialised empty-state for the GUI's top-level 'no runs found' case."""
    if results_root is None or not results_root.exists():
        render_empty_state(
            title="No results path found",
            message=(
                f"The path `{results_root}` does not exist."
                if results_root is not None
                else "No results path was provided on the CLI."
            ),
            variant="permanent",
            hint=(
                "Pass an existing directory, e.g. "
                "uv run streamlit run scripts/dev/gui/main.py -- /path/to/results/"
            ),
        )
        return
    render_empty_state(
        title="No runs in results/ yet",
        message=f"Watching `{results_root}` for new run directories.",
        variant="waiting",
    )
