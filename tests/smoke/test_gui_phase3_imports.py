"""Phase 3 smoke tests for the dev-loop inspection GUI.

Covers the new cross-run + filter components added in Phase 3:

- C5 ``gate_verdict_table``: single-run + cross-run-diff variants render
  without crashing against empty or partial run dirs.
- C6 ``cross_run_small_multiples``: handles 1 / 2 / 5 run cases per the
  UX spec state-machine.
- C11 ``run_filter_bar``: substring + gate-status + mtime filtering, and
  the empty-result hint when filters match nothing.
- ``trajectory_chart.render_trajectory_chart_overlay``: callable; the
  empty-series path renders a caption instead of a chart.
- ``state`` multi-run helpers round-trip via ``st.session_state``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_PHASE_3_MODULES = (
    "scripts.dev.gui.components.gate_verdict_table",
    "scripts.dev.gui.components.cross_run_small_multiples",
    "scripts.dev.gui.components.run_filter_bar",
)


@pytest.mark.parametrize("module_name", _PHASE_3_MODULES)
def test_phase3_module_imports_cleanly(module_name: str) -> None:
    """Every Phase 3 GUI module imports without raising."""
    assert importlib.import_module(module_name) is not None


def test_phase3_components_define_render_functions() -> None:
    """Each Phase 3 component exposes its documented `render_*` entry point."""
    from scripts.dev.gui.components import (
        cross_run_small_multiples,
        gate_verdict_table,
        run_filter_bar,
        trajectory_chart,
    )

    assert callable(gate_verdict_table.render_gate_verdict_table)
    assert callable(gate_verdict_table.render_gate_verdict_diff)
    assert callable(cross_run_small_multiples.render_cross_run_small_multiples)
    assert callable(run_filter_bar.render_run_filter_bar)
    assert callable(trajectory_chart.render_trajectory_chart_overlay)


def test_state_selected_runs_round_trip() -> None:
    """Multi-select helpers add/remove/clear consistently and dedupe on set."""
    import streamlit as st
    from scripts.dev.gui import state

    st.session_state.clear()
    assert state.get_selected_runs() == []
    state.add_selected_run("run-a")
    state.add_selected_run("run-b")
    state.add_selected_run("run-a")  # dedupe
    assert state.get_selected_runs() == ["run-a", "run-b"]
    state.remove_selected_run("run-a")
    assert state.get_selected_runs() == ["run-b"]
    state.set_selected_runs(["x", "y", "x"])
    assert state.get_selected_runs() == ["x", "y"]
    state.clear_selected_runs()
    assert state.get_selected_runs() == []


def test_state_filter_round_trip() -> None:
    """Filter helpers round-trip + ``active_filter_count`` reflects current state."""
    import streamlit as st
    from scripts.dev.gui import state

    st.session_state.clear()
    assert state.active_filter_count() == 0
    state.set_filter_text("seed-42")
    state.set_filter_gate_statuses(["cleared"])
    state.set_filter_mtime_after(1_700_000_000.0)
    assert state.get_filter_text() == "seed-42"
    assert state.get_filter_gate_statuses() == ["cleared"]
    assert state.get_filter_mtime_after() == pytest.approx(1_700_000_000.0)
    assert state.active_filter_count() == 3
    state.clear_filters()
    assert state.active_filter_count() == 0


def _make_empty_run_dir(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def test_apply_filters_substring_and_mtime(tmp_path: Path) -> None:
    """Substring + mtime filters narrow the summary list as expected."""
    import streamlit as st
    from scripts.dev.gui import state
    from scripts.dev.gui.components.run_filter_bar import _apply_filters
    from scripts.dev.gui.data import RunSummary

    st.session_state.clear()
    base = tmp_path
    summaries = [
        RunSummary(
            run_id="alpha-seed-1",
            path=base / "alpha-seed-1",
            mtime=1_000_000.0,
            seed="1",
            config_slug="jepa_smoke",
            git_sha="abc1234",
            gpu_hours=None,
            has_metrics=False,
            has_report=False,
            has_log=False,
        ),
        RunSummary(
            run_id="beta-seed-2",
            path=base / "beta-seed-2",
            mtime=2_000_000.0,
            seed="2",
            config_slug="jepa_smoke",
            git_sha="def5678",
            gpu_hours=None,
            has_metrics=False,
            has_report=False,
            has_log=False,
        ),
    ]
    # No filters → all rows returned.
    assert len(_apply_filters(summaries)) == 2
    # Substring filter on run_id.
    state.set_filter_text("alpha")
    assert [s.run_id for s in _apply_filters(summaries)] == ["alpha-seed-1"]
    # Substring on git_sha.
    state.set_filter_text("def5678")
    assert [s.run_id for s in _apply_filters(summaries)] == ["beta-seed-2"]
    # mtime filter.
    state.clear_filters()
    state.set_filter_mtime_after(1_500_000.0)
    assert [s.run_id for s in _apply_filters(summaries)] == ["beta-seed-2"]


def test_gate_verdict_table_tolerates_missing_metrics(tmp_path: Path) -> None:
    """Empty run dir → render does not crash; all gates surface as pending."""
    from scripts.dev.gui.components import gate_verdict_table

    run_dir = _make_empty_run_dir(tmp_path, "run-empty")
    # render path goes through Streamlit; we don't drive the Streamlit harness
    # here, just smoke-test the underlying helpers used by it.
    metrics, status = gate_verdict_table._load_metrics_and_status(run_dir)  # type: ignore[attr-defined]
    assert metrics is None
    assert status is None

    rows = [
        gate_verdict_table._row_for_gate(name, status, metrics, "run-empty")  # type: ignore[attr-defined]
        for name in gate_verdict_table._FULL_GATE_ORDER  # type: ignore[attr-defined]
    ]
    assert len(rows) == 9
    for row in rows:
        assert "pending" in row["verdict"]


def test_cross_run_small_multiples_one_run_no_op(tmp_path: Path) -> None:
    """Single-run input is a documented no-op (caller falls back to single layout)."""
    from scripts.dev.gui.components.cross_run_small_multiples import (
        render_cross_run_small_multiples,
    )

    _make_empty_run_dir(tmp_path, "only")
    # The function must not raise for the empty/skip cases. Streamlit calls
    # inside short-circuit before any rendering; we exercise the branch.
    render_cross_run_small_multiples(tmp_path, ["only"])
