"""C7 — ProvenanceFooter component.

Always-visible footer that cites the file + step + timestamp the
currently-selected panel reads from (§"Flow Optimization Principles #4 —
Provenance is always visible").

Phase 1 covers the selected-run + step + file path triple. Cross-run
provenance (Phase 3) extends `render_provenance_footer` to accept a list
of source-paths; the signature is forward-compatible.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from scripts.dev.gui import theme


def render_provenance_footer(
    run_id: str | None,
    run_dir: Path | None,
    step: int | None,
    log_line: int | None,
    timestamp: datetime | None = None,
) -> None:
    """Render the provenance footer.

    `run_id=None` renders the "no run selected" variant so the footer is
    never absent from the viewport.
    """
    st.divider()
    if run_id is None or run_dir is None:
        st.markdown(
            f'<div role="status" aria-live="polite" '
            f'style="color:{theme.COLOR_NEUTRAL}; font-family:monospace; font-size:12px;">'
            "no run selected · provenance footer waiting"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    rel_log = f"{run_dir}/log.jsonl"
    step_token = f"step {step}" if step is not None else "step —"
    line_token = f"L{log_line}" if log_line is not None else "L—"
    ts = (timestamp or datetime.now(tz=UTC)).strftime("%Y-%m-%d %H:%M:%S UTC")

    st.markdown(
        f'<div role="status" aria-live="polite" '
        f'style="color:{theme.COLOR_NEUTRAL}; font-family:monospace; font-size:12px;">'
        f"{run_id} · {step_token} · {rel_log} {line_token} · {ts}"
        "</div>",
        unsafe_allow_html=True,
    )
