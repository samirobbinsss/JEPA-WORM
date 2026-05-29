"""Phase 1 smoke tests for the dev-loop inspection GUI.

Goals:
- Verify every GUI module imports without error (catches syntax + missing-dep
  regressions).
- Verify `theme.py` exports the documented constants.
- Verify `data.py` parses a synthetic minimal `metrics.json` written to a
  tmp `results/<run-id>/` directory.

Out of scope:
- Actually running a Streamlit server (Streamlit is awkward to drive
  headlessly in pytest; we stop at the import + pure-Python boundary).
- Asserting on `RunRow` / `TrajectoryChart` output shape — those touch the
  Streamlit runtime, which requires `streamlit.testing.v1`, which is a
  Phase 2 expansion target.

Follows the `tests/smoke/test_dev_local_loader.py` skip-on-missing-input
pattern in spirit: smoke-tests must keep CI green on a clean clone.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `import scripts.dev.gui.*` resolves
# without depending on conftest wiring. Mirrors `main.py`'s prelude.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Every Phase 1 module — these names drive a single parametrised
# "imports cleanly" test so adding modules later is one-line.
_PHASE_1_MODULES = (
    "scripts.dev.gui",
    "scripts.dev.gui.cache",
    "scripts.dev.gui.data",
    "scripts.dev.gui.state",
    "scripts.dev.gui.theme",
    "scripts.dev.gui.watcher",
    "scripts.dev.gui.components",
    "scripts.dev.gui.components.empty_state",
    "scripts.dev.gui.components.live_tail_indicator",
    "scripts.dev.gui.components.provenance_footer",
    "scripts.dev.gui.components.run_row",
    "scripts.dev.gui.components.trajectory_chart",
)


@pytest.mark.parametrize("module_name", _PHASE_1_MODULES)
def test_gui_module_imports_cleanly(module_name: str) -> None:
    """Every Phase 1 GUI module imports without raising."""
    assert importlib.import_module(module_name) is not None


def test_theme_exports_documented_constants() -> None:
    """`theme.py` exports the §"Visual Design Foundation" hex tokens.

    Spec values for the semantic palette (cross-reference with
    `.streamlit/config.toml`):
        primary / accent  = #1565C0
        background        = #FAFAFA
        surface           = #FFFFFF
        text / neutral    = #424242
    """
    from scripts.dev.gui import theme

    assert theme.COLOR_ACCENT == "#1565C0"
    assert theme.COLOR_BACKGROUND == "#FAFAFA"
    assert theme.COLOR_SURFACE == "#FFFFFF"
    assert theme.COLOR_NEUTRAL == "#424242"
    assert theme.COLOR_HEALTHY == "#2E7D32"
    assert theme.COLOR_WARNING == "#E68A00"
    assert theme.COLOR_CRITICAL == "#C62828"
    assert theme.COCKPIT_COLUMN_RATIOS == (2, 5, 5)

    assert len(theme.TABLEAU_COLORBLIND_10) == 10
    # Colors cycle modulo 10
    assert theme.run_color(0) == theme.TABLEAU_COLORBLIND_10[0]
    assert theme.run_color(11) == theme.TABLEAU_COLORBLIND_10[1]
    with pytest.raises(ValueError):
        theme.run_color(-1)


def test_streamlit_theme_config_matches_python_tokens() -> None:
    """`.streamlit/config.toml` agrees with `theme.py` on the load-bearing colors."""
    from scripts.dev.gui import theme

    config_path = _PROJECT_ROOT / "scripts" / "dev" / "gui" / ".streamlit" / "config.toml"
    assert config_path.is_file(), f"missing {config_path}"

    text = config_path.read_text(encoding="utf-8")
    assert f'primaryColor = "{theme.COLOR_ACCENT}"' in text
    assert f'backgroundColor = "{theme.COLOR_BACKGROUND}"' in text
    assert f'secondaryBackgroundColor = "{theme.COLOR_SURFACE}"' in text
    assert f'textColor = "{theme.COLOR_NEUTRAL}"' in text
    assert 'base = "light"' in text


def test_data_loads_minimal_synthetic_metrics(tmp_path: Path) -> None:
    """`data.load_metrics` parses a synthetic minimal `metrics.json`."""
    from scripts.dev.gui import data

    run_dir = tmp_path / "results" / "synthetic-run-0"
    run_dir.mkdir(parents=True)

    payload = {
        "run_id": "synthetic-run-0",
        "entries": [
            {
                "name": "future_pose",
                "producer": "jepa",
                "ci": {
                    "point": 0.42,
                    "lower": 0.35,
                    "upper": 0.49,
                    "level": 0.95,
                    "method": "bca",
                    "n_samples": 1000,
                    "grouping": "worm",
                },
                "sub_entries": [],
                "notes": "",
            }
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")

    parsed = data.load_metrics(run_dir)
    assert parsed is not None
    assert parsed.run_id == "synthetic-run-0"
    assert len(parsed.entries) == 1
    assert parsed.entries[0].name == "future_pose"
    assert parsed.entries[0].ci.point == pytest.approx(0.42)


def test_data_load_metrics_returns_none_when_missing(tmp_path: Path) -> None:
    """Missing `metrics.json` returns `None` (live-tail mid-run case)."""
    from scripts.dev.gui import data

    run_dir = tmp_path / "results" / "empty-run"
    run_dir.mkdir(parents=True)
    assert data.load_metrics(run_dir) is None


def test_data_load_log_entries_parses_synthetic_jsonl(tmp_path: Path) -> None:
    """`data.load_log_entries` reads (step, loss) out of a fake `log.jsonl`."""
    from scripts.dev.gui import data

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "log.jsonl"

    def _entry(step: int, loss: float) -> str:
        return json.dumps(
            {"ts": "2026-05-12T10:00:00", "level": "INFO", "extra": {"step": step, "loss": loss}}
        )

    lines = [
        _entry(0, 1.5),
        _entry(1, 1.2),
        "garbage line that should be skipped",
        _entry(2, 1.0),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    entries = data.load_log_entries(run_dir)
    assert len(entries) == 3
    steps, losses = data.trajectory_from_log(entries)
    assert steps == [0, 1, 2]
    assert losses == [1.5, 1.2, 1.0]


def test_data_list_run_directories_orders_by_mtime(tmp_path: Path) -> None:
    """Most-recently-modified run appears first (newest-first sort)."""
    import os
    import time

    from scripts.dev.gui import data

    results_root = tmp_path / "results"
    results_root.mkdir()
    older = results_root / "run-older"
    newer = results_root / "run-newer"
    older.mkdir()
    newer.mkdir()

    # Force a deterministic mtime ordering even on filesystems with
    # coarse-grained mtime resolution.
    os.utime(older, (time.time() - 1000, time.time() - 1000))
    os.utime(newer, (time.time(), time.time()))

    dirs = data.list_run_directories(results_root)
    assert [p.name for p in dirs] == ["run-newer", "run-older"]


def test_data_handles_nonexistent_results_root(tmp_path: Path) -> None:
    """Missing results root returns empty list (empty-state handling)."""
    from scripts.dev.gui import data

    assert data.list_run_directories(tmp_path / "nope") == []


def test_load_run_summary_with_minimal_files(tmp_path: Path) -> None:
    """`load_run_summary` populates fields from whatever files exist."""
    from scripts.dev.gui import data

    run_dir = tmp_path / "abc123"
    run_dir.mkdir()
    (run_dir / "seed.txt").write_text("42\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text("name: headline\nschema_version: 1\n", encoding="utf-8")
    (run_dir / "compute.json").write_text(
        json.dumps({"gpu_hours": 1.25, "wall_clock_seconds": 4500.0, "python_version": "3.11.0"}),
        encoding="utf-8",
    )
    (run_dir / "manifest_at_run.lock").write_text(
        "schema_version: 1\ngit_sha_at_lock: deadbeefcafef00d\n", encoding="utf-8"
    )

    summary = data.load_run_summary(run_dir)
    assert summary.run_id == "abc123"
    assert summary.seed == "42"
    assert summary.config_slug == "headline"
    assert summary.git_sha == "deadbeefcafef00d"
    assert summary.gpu_hours == pytest.approx(1.25)
    assert summary.has_metrics is False
    assert summary.has_log is False


def test_watcher_poll_returns_zero_for_missing_files(tmp_path: Path) -> None:
    """`watcher.poll_run_dir` tolerates absent log/metrics files."""
    from scripts.dev.gui import watcher

    status = watcher.poll_run_dir(tmp_path)
    assert status.latest_mtime == 0.0
    assert status.last_observed_ts is not None


def test_components_define_render_functions() -> None:
    """Each Phase 1 component exposes its documented `render_*` entry point."""
    from scripts.dev.gui.components import (
        empty_state,
        live_tail_indicator,
        provenance_footer,
        run_row,
        trajectory_chart,
    )

    assert callable(empty_state.render_empty_state)
    assert callable(empty_state.render_results_root_empty)
    assert callable(live_tail_indicator.render_live_tail_indicator)
    assert callable(provenance_footer.render_provenance_footer)
    assert callable(run_row.render_run_table)
    assert callable(trajectory_chart.render_trajectory_chart)
