"""Entry point for the dev-loop inspection GUI (Phase 1).

Invocation:

    uv run streamlit run scripts/dev/gui/main.py -- /path/to/results/

The trailing `--` is Streamlit's separator between its own flags and the
script's argparse args. The argparse layer reads positional args from
`sys.argv` after Streamlit has stripped its own flags.

Phase 1 + Phase 2 scope (per UX spec §"Implementation Roadmap"):

Phase 1 — J1 first-open works end-to-end:
- C7 ProvenanceFooter
- C8 LiveTailIndicator
- C9 EmptyState
- C1 RunRow (basic)
- C2 TrajectoryChart (single-run)

Phase 2 — J3 collapse detection:
- C3 ClipViewer (single-run, no-pose variant)
- C4 LatentGeometryPanel
- C10 SmoothingSlider

Cross-run, gate verdicts, pose-overlay variants of ClipViewer land in
Phases 3/4.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

# Streamlit launches `main.py` as a top-level script; the project root is
# not on sys.path by default. We push it in front so `from scripts.dev.gui
# import ...` resolves both here and inside the package modules.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402  — must follow sys.path setup

from scripts.dev.gui import cache, data, state, theme, watcher  # noqa: E402
from scripts.dev.gui.components import (  # noqa: E402
    clip_viewer,
    cross_run_small_multiples,
    empty_state,
    gate_verdict_table,
    latent_geometry,
    live_tail_indicator,
    provenance_footer,
    run_filter_bar,
    run_row,
    smoothing_slider,
    trajectory_chart,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse the script's argparse args from the post-`--` tail of `sys.argv`."""
    parser = argparse.ArgumentParser(
        prog="jepa-worm-gui",
        description="Dev-loop inspection GUI for the JEPA-WORM model (Phase 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path("results"),
        help="Path to the results/ directory containing one or more <run-id>/ subdirs.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Pre-select this run-id at startup (optional).",
    )
    return parser.parse_args(argv)


def _render_top_bar(results_root: Path, run_ids: list[str]) -> None:
    """Render the top-bar: title + live-tail toggle + run selector."""
    title_col, toggle_col, selector_col = st.columns([4, 2, 4])
    with title_col:
        st.markdown("### JEPA-WORM dev-loop GUI")
        st.caption(f"watching `{results_root}` · phase 1 (run-table + trajectory)")
    with toggle_col:
        live = st.toggle(
            "Live tail",
            value=st.session_state.get(state.KEY_LIVE_TAIL, False),
            key="gui.live_tail.widget",
            help="When on, re-reads files each rerun; off for static post-run review.",
        )
        state.set_live_tail(live)
        # LiveTailIndicator sits directly under the toggle.
        s = state.get_state()
        selected = s.selected_run
        last_seconds: float | None = None
        if live and selected is not None:
            status = watcher.poll_run_dir(results_root / selected)
            last_seconds = (
                watcher.seconds_since(status.last_observed_ts) if status.latest_mtime > 0 else None
            )
        live_tail_indicator.render_live_tail_indicator(
            live=live,
            seconds_since_update=last_seconds,
            loaded_at=datetime.now(tz=UTC) if not live else None,
        )
    with selector_col:
        if run_ids:
            current = st.session_state.get(state.KEY_SELECTED_RUN)
            default_index = run_ids.index(current) if current in run_ids else 0
            choice = st.selectbox(
                "Select run",
                options=run_ids,
                index=default_index,
                key="gui.run_selector.widget",
            )
            state.set_selected_run(choice)
        else:
            st.selectbox("Select run", options=["(no runs)"], disabled=True, index=0)


def _render_left_column(summaries: list[data.RunSummary], run_ids: list[str]) -> None:
    """Left column: run-table + cross-run multi-select.

    Phase 3 adds a ``st.multiselect`` widget under the run-table so the user
    can engage cross-run mode without inventing a per-row checkbox in the
    dataframe (Streamlit's row-selection events are awkward to track across
    reruns). Selecting ≥ 2 runs swaps the middle/right columns to the
    small-multiples layout.
    """
    st.markdown("##### Runs")
    selected_now = run_row.render_run_table(
        summaries, selected=st.session_state.get(state.KEY_SELECTED_RUN)
    )
    if selected_now is not None:
        state.set_selected_run(selected_now)

    current_multi = state.get_selected_runs()
    chosen_multi = st.multiselect(
        "Cross-run mode (select 2+ runs)",
        options=run_ids,
        default=[r for r in current_multi if r in run_ids],
        key="gui.multiselect.runs.widget",
        help="Select ≥ 2 runs to engage the small-multiples cross-run layout.",
    )
    if chosen_multi != current_multi:
        state.set_selected_runs(chosen_multi)
    if len(chosen_multi) >= 2 and st.button(
        "Disengage cross-run mode",
        type="primary",
        key="gui.disengage_cross_run",
    ):
        state.clear_selected_runs()
        st.rerun()


def _render_middle_column(run_dir: Path | None, run_id: str | None) -> None:
    """Middle column: smoothing slider + stacked trajectory charts.

    The smoothing slider sits *above* the charts so its value applies to
    every chart underneath in the same Streamlit render pass.
    """
    smoothing_slider.render_smoothing_slider()
    st.markdown("##### Trajectories")
    if run_dir is None or run_id is None:
        st.info("Select a run from the left to populate the trajectories.")
        return

    entries = cache.log_entries(run_dir)
    if not entries:
        st.info(f"`{run_dir}/log.jsonl` has no entries yet. Waiting for the first step.")
        return

    color = state.assign_run_color(run_id)
    steps, losses = data.trajectory_from_log(entries)
    trajectory_chart.render_trajectory_chart(
        run_id=run_id,
        label="Loss",
        steps=steps,
        values=losses,
        color=color,
        y_axis_title="loss",
        key="gui.trajectory.loss",
    )

    latent = data.latent_norm_trajectory_from_log(entries)
    if latent is not None:
        latent_steps, latent_values = latent
        trajectory_chart.render_trajectory_chart(
            run_id=run_id,
            label="Latent norm",
            steps=latent_steps,
            values=latent_values,
            color=theme.COLOR_WARNING,
            y_axis_title="latent norm",
            key="gui.trajectory.latent",
        )
    else:
        st.caption("No latent-norm field found in `log.jsonl`; chart hidden.")


def _render_cross_run_middle(results_root: Path, run_ids: list[str]) -> None:
    """Middle column when cross-run mode is engaged: overlay trajectories.

    Smoothing slider sits above the overlay charts (same Streamlit render
    pass, value applies to all traces).
    """
    smoothing_slider.render_smoothing_slider()
    st.markdown("##### Trajectories (overlay)")

    loss_series: list[tuple[str, list[int], list[float], str]] = []
    latent_series: list[tuple[str, list[int], list[float], str]] = []
    for run_id in run_ids:
        run_dir = results_root / run_id
        entries = cache.log_entries(run_dir)
        color = state.assign_run_color(run_id)
        steps, losses = data.trajectory_from_log(entries)
        if steps and losses:
            loss_series.append((run_id, steps, losses, color))
        latent = data.latent_norm_trajectory_from_log(entries)
        if latent is not None:
            lsteps, lvalues = latent
            if lsteps and lvalues:
                latent_series.append((run_id, lsteps, lvalues, color))

    trajectory_chart.render_trajectory_chart_overlay(
        series=loss_series,
        label="Loss",
        y_axis_title="loss",
        key="gui.trajectory.loss.overlay",
    )
    if latent_series:
        trajectory_chart.render_trajectory_chart_overlay(
            series=latent_series,
            label="Latent norm",
            y_axis_title="latent norm",
            key="gui.trajectory.latent.overlay",
        )
    else:
        st.caption("No latent-norm series across the selected runs.")


def _render_right_column(run_dir: Path | None, run_id: str | None) -> None:
    """Right column: ClipViewer (top) + LatentGeometryPanel (bottom) + GateVerdictTable.

    All three components handle the "no data / no run selected" cases
    internally so the column always renders something.
    """
    clip_viewer.render_clip_viewer(run_dir, run_id)
    latent_geometry.render_latent_geometry_panel(run_dir, run_id)
    if run_dir is not None and run_id is not None:
        gate_verdict_table.render_gate_verdict_table(run_dir, run_id)


def _pinned_log_line(run_dir: Path | None) -> int | None:
    """Phase 1 has no pin-on-hover — surface the last log line as a stand-in."""
    if run_dir is None:
        return None
    entries = cache.log_entries(run_dir)
    if not entries:
        return None
    return entries[-1].line_number


def _pinned_step(run_dir: Path | None) -> int | None:
    """Phase 1 stand-in for a pinned step: the most recent step in the log."""
    if run_dir is None:
        return None
    entries = cache.log_entries(run_dir)
    steps, _ = data.trajectory_from_log(entries)
    return steps[-1] if steps else None


def _streamlit_argv_tail() -> list[str]:
    """Return the script-arg tail from `sys.argv`.

    Streamlit forwards everything after `--` to the script as `sys.argv[1:]`,
    so this is effectively `sys.argv[1:]`. Wrapping it in a function makes
    unit-testing argument parsing trivial.
    """
    return sys.argv[1:]


def run() -> None:
    """Main entry point — composes the three-column cockpit."""
    args = _parse_args(_streamlit_argv_tail())
    results_root: Path = args.results_root.expanduser().resolve()

    # Lay out the page first so the top-bar always renders before any
    # potentially-slow file IO. Streamlit's render order matches code order.
    st.set_page_config(
        page_title="JEPA-WORM dev GUI",
        page_icon="🪱",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    state.init_session_state(str(results_root))
    cache.ensure_cache_dir()

    if args.run is not None:
        state.set_selected_run(args.run)

    run_ids = cache.list_runs(results_root)
    if not run_ids:
        _render_top_bar(results_root, run_ids)
        empty_state.render_results_root_empty(results_root)
        provenance_footer.render_provenance_footer(
            run_id=None, run_dir=None, step=None, log_line=None
        )
        return

    # Default selection: first run by mtime if nothing chosen yet.
    if st.session_state.get(state.KEY_SELECTED_RUN) not in run_ids:
        state.set_selected_run(run_ids[0])

    _render_top_bar(results_root, run_ids)

    all_summaries = [cache.run_summary(results_root / rid) for rid in run_ids]
    filtered_summaries = run_filter_bar.render_run_filter_bar(all_summaries)
    filtered_run_ids = [s.run_id for s in filtered_summaries]

    selected_runs = state.get_selected_runs()
    cross_run_engaged = len(selected_runs) >= 2

    if cross_run_engaged:
        # Cross-run mode: left column for navigation, full-width small-multiples
        # below for the per-run panels. Trajectory overlay sits above.
        left, content = st.columns([2, 10])
        with left:
            _render_left_column(filtered_summaries, filtered_run_ids)
        with content:
            _render_cross_run_middle(results_root, selected_runs)
            cross_run_small_multiples.render_cross_run_small_multiples(results_root, selected_runs)
            gate_verdict_table.render_gate_verdict_diff(
                [(rid, results_root / rid) for rid in selected_runs]
            )
        provenance_footer.render_provenance_footer(
            run_id=", ".join(selected_runs),
            run_dir=results_root,
            step=None,
            log_line=None,
        )
        return

    left, middle, right = st.columns(theme.COCKPIT_COLUMN_RATIOS)
    with left:
        _render_left_column(filtered_summaries, filtered_run_ids)

    selected = st.session_state.get(state.KEY_SELECTED_RUN)
    run_dir = results_root / selected if isinstance(selected, str) else None

    with middle:
        _render_middle_column(run_dir, selected if isinstance(selected, str) else None)
    with right:
        _render_right_column(run_dir, selected if isinstance(selected, str) else None)

    provenance_footer.render_provenance_footer(
        run_id=selected if isinstance(selected, str) else None,
        run_dir=run_dir,
        step=_pinned_step(run_dir),
        log_line=_pinned_log_line(run_dir),
    )


# Streamlit imports this module; running `run()` at import time is the
# canonical pattern (Streamlit does not call a `main()` for you).
run()
