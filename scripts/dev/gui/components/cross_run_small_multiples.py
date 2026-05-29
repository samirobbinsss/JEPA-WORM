"""C6 — CrossRunSmallMultiples component (Phase 3).

When ≥ 2 runs are multi-selected the right column's single-run panels are
replaced with N side-by-side panels, each rendering ClipViewer +
LatentGeometryPanel + GateVerdictTable for one run.

State machine per the UX spec §"C6 — `CrossRunSmallMultiples`":

- 1 run → no-op (caller falls back to the single-run layout).
- 2 to 4 runs → `st.columns(N)` side-by-side.
- ≥ 5 runs → Streamlit's column layout overflows badly; we fall back to
  one `st.expander` per run (open by default) so the page stays readable.
  Documented in the docstring; the UX spec calls for horizontal scroll
  which Streamlit doesn't support, so the expander cascade is the closest
  workable equivalent.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from scripts.dev.gui import cache
from scripts.dev.gui.components import (
    clip_viewer,
    gate_verdict_table,
    latent_geometry,
)

_SIDE_BY_SIDE_MAX: int = 4


def _run_header(run_id: str, run_dir: Path) -> None:
    """Render the per-column header carrying run-id + seed + git-sha."""
    summary = cache.run_summary(run_dir)
    seed = summary.seed or "—"
    sha = summary.git_sha[:8] if summary.git_sha else "—"
    st.markdown(f"**{run_id}**")
    st.caption(f"seed `{seed}` · sha `{sha}`")


def _render_per_run_panels(run_dir: Path, run_id: str) -> None:
    """Render the per-run panel stack used inside each small multiple."""
    _run_header(run_id, run_dir)
    clip_viewer.render_clip_viewer(run_dir, run_id)
    latent_geometry.render_latent_geometry_panel(run_dir, run_id)
    gate_verdict_table.render_gate_verdict_table(run_dir, run_id)


def render_cross_run_small_multiples(results_root: Path, run_ids: list[str]) -> None:
    """Render N side-by-side panels for ``run_ids``.

    Args:
        results_root: Root of the ``results/`` tree.
        run_ids: Multi-selected run ids (deduped; caller's responsibility).
    """
    if len(run_ids) < 2:
        return

    st.markdown("##### Cross-run small multiples")

    if len(run_ids) <= _SIDE_BY_SIDE_MAX:
        cols = st.columns(len(run_ids))
        for col, run_id in zip(cols, run_ids, strict=True):
            with col:
                _render_per_run_panels(results_root / run_id, run_id)
        return

    st.caption(
        f"{len(run_ids)} runs selected — side-by-side columns would overflow; "
        "showing as expander cascade (open by default)."
    )
    for run_id in run_ids:
        with st.expander(run_id, expanded=True):
            _render_per_run_panels(results_root / run_id, run_id)
