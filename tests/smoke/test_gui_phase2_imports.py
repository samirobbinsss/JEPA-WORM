"""Phase 2 smoke tests for the dev-loop inspection GUI.

Goals (mirrors `test_gui_imports.py` for Phase 1):
- Every Phase 2 module imports without error.
- `data.latent_stats_from_log` parses the new `extra.latent.*` fields out
  of a synthetic JSONL fixture.
- `data.apply_ema_smoothing` matches the documented TensorBoard semantics.
- Each Phase 2 component exposes its documented `render_*` entry point.

Out of scope (same as Phase 1):
- Streamlit headless rendering — `streamlit.testing.v1` would let us drive
  pages programmatically but Phase 2 stays at the import + pure-Python
  boundary like Phase 1 did.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Phase 2 modules — three new components + the extended data/state layer.
_PHASE_2_MODULES = (
    "scripts.dev.gui.components.clip_viewer",
    "scripts.dev.gui.components.latent_geometry",
    "scripts.dev.gui.components.smoothing_slider",
)


@pytest.mark.parametrize("module_name", _PHASE_2_MODULES)
def test_phase2_module_imports_cleanly(module_name: str) -> None:
    """Every Phase 2 GUI module imports without raising."""
    assert importlib.import_module(module_name) is not None


def test_phase2_components_define_render_functions() -> None:
    """Each Phase 2 component exposes its documented `render_*` entry point."""
    from scripts.dev.gui.components import (
        clip_viewer,
        latent_geometry,
        smoothing_slider,
    )

    assert callable(clip_viewer.render_clip_viewer)
    assert callable(latent_geometry.render_latent_geometry_panel)
    assert callable(smoothing_slider.render_smoothing_slider)


def test_apply_ema_smoothing_zero_returns_input() -> None:
    """`alpha = 0` is the documented pass-through identity."""
    from scripts.dev.gui import data

    raw = [1.0, 2.0, 3.0, 4.0]
    assert data.apply_ema_smoothing(raw, 0.0) == raw


def test_apply_ema_smoothing_matches_tensorboard_recurrence() -> None:
    """`s[i] = alpha * s[i-1] + (1 - alpha) * x[i]` with `s[0] = x[0]`."""
    from scripts.dev.gui import data

    raw = [1.0, 2.0, 3.0]
    alpha = 0.5
    smoothed = data.apply_ema_smoothing(raw, alpha)
    expected = [1.0, 0.5 * 1.0 + 0.5 * 2.0, 0.5 * (0.5 * 1.0 + 0.5 * 2.0) + 0.5 * 3.0]
    for a, b in zip(smoothed, expected, strict=True):
        assert a == pytest.approx(b)


def test_apply_ema_smoothing_clamps_alpha() -> None:
    """`alpha` outside [0, 0.99] is silently clamped."""
    from scripts.dev.gui import data

    # alpha=2.0 → clamped to 0.99 → very smooth, still finite.
    smoothed = data.apply_ema_smoothing([1.0, 100.0], 2.0)
    assert smoothed[0] == 1.0
    assert smoothed[1] == pytest.approx(0.99 * 1.0 + 0.01 * 100.0)


def test_apply_ema_smoothing_empty_returns_empty() -> None:
    """Empty input must round-trip to empty output (no crashes)."""
    from scripts.dev.gui import data

    assert data.apply_ema_smoothing([], 0.5) == []


def test_latent_stats_from_log_parses_synthetic_jsonl(tmp_path: Path) -> None:
    """`data.latent_stats_from_log` reads the Phase 2 latent block correctly."""
    from scripts.dev.gui import data

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / "log.jsonl"

    def _entry(step: int, std_per_dim: list[float], cov_off: float) -> str:
        return json.dumps(
            {
                "ts": "2026-05-15T10:00:00",
                "step": step,
                "extra": {
                    "loss": 1.0,
                    "losses": {"jepa": 1.0},
                    "latent": {
                        "std_per_dim": std_per_dim,
                        "std_min": min(std_per_dim),
                        "std_mean": sum(std_per_dim) / len(std_per_dim),
                        "cov_offdiag_frobenius": cov_off,
                    },
                },
            }
        )

    lines = [
        _entry(1, [0.5, 0.6, 0.7], 0.1),
        _entry(2, [0.4, 0.6, 0.6], 0.2),
        _entry(3, [0.3, 0.5, 0.5], 0.3),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    entries = data.load_log_entries(run_dir)
    series = data.latent_stats_from_log(entries)
    assert series.steps == [1, 2, 3]
    assert series.n_dims == 3
    assert series.std_min == pytest.approx([0.5, 0.4, 0.3])
    assert series.std_mean[0] == pytest.approx((0.5 + 0.6 + 0.7) / 3.0)
    assert series.cov_offdiag_frobenius == pytest.approx([0.1, 0.2, 0.3])
    assert series.is_empty is False


def test_latent_stats_from_log_skips_missing_block(tmp_path: Path) -> None:
    """Older log entries without `extra.latent` are skipped."""
    from scripts.dev.gui import data

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / "log.jsonl"

    # Mix: one row with the block, one without.
    lines = [
        json.dumps(
            {
                "ts": "2026-05-15T10:00:00",
                "step": 1,
                "extra": {"loss": 1.0, "losses": {"jepa": 1.0}},
            }
        ),
        json.dumps(
            {
                "ts": "2026-05-15T10:00:01",
                "step": 2,
                "extra": {
                    "loss": 0.5,
                    "losses": {"jepa": 0.5},
                    "latent": {
                        "std_per_dim": [0.1, 0.2],
                        "std_min": 0.1,
                        "std_mean": 0.15,
                        "cov_offdiag_frobenius": 0.01,
                    },
                },
            }
        ),
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    entries = data.load_log_entries(run_dir)
    series = data.latent_stats_from_log(entries)
    assert series.steps == [2]
    assert series.n_dims == 2


def test_latent_stats_from_log_empty_when_no_block(tmp_path: Path) -> None:
    """Entirely-missing latent block yields an empty series flagged via `is_empty`."""
    from scripts.dev.gui import data

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / "log.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "ts": "2026-05-15T10:00:00",
                "step": 1,
                "extra": {"loss": 1.0, "losses": {"jepa": 1.0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    entries = data.load_log_entries(run_dir)
    series = data.latent_stats_from_log(entries)
    assert series.is_empty is True
    assert series.steps == []
    assert series.n_dims == 0


def test_latent_stats_from_log_drops_ragged_rows(tmp_path: Path) -> None:
    """Rows whose `std_per_dim` width disagrees with the first row are dropped."""
    from scripts.dev.gui import data

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / "log.jsonl"

    def _entry(step: int, std_per_dim: list[float]) -> str:
        return json.dumps(
            {
                "ts": "2026-05-15T10:00:00",
                "step": step,
                "extra": {
                    "loss": 1.0,
                    "losses": {"jepa": 1.0},
                    "latent": {
                        "std_per_dim": std_per_dim,
                        "std_min": min(std_per_dim) if std_per_dim else 0.0,
                        "std_mean": (sum(std_per_dim) / len(std_per_dim)) if std_per_dim else 0.0,
                        "cov_offdiag_frobenius": 0.0,
                    },
                },
            }
        )

    log_path.write_text(
        "\n".join([_entry(1, [0.1, 0.2]), _entry(2, [0.1, 0.2, 0.3])]) + "\n",
        encoding="utf-8",
    )
    entries = data.load_log_entries(run_dir)
    series = data.latent_stats_from_log(entries)
    # Only the first row is accepted; the second row is ragged.
    assert series.steps == [1]
    assert series.n_dims == 2


def test_state_smoothing_round_trip() -> None:
    """`state.set_smoothing` / `state.get_smoothing` clamp to `[0, 0.99]`."""
    import streamlit as st
    from scripts.dev.gui import state

    # Use the real session_state — Streamlit's testing harness is heavy and
    # we only need a dict-like behaviour here.
    st.session_state.clear()
    state.set_smoothing(0.5)
    assert state.get_smoothing() == pytest.approx(0.5)
    state.set_smoothing(-1.0)
    assert state.get_smoothing() == pytest.approx(0.0)
    state.set_smoothing(2.0)
    assert state.get_smoothing() == pytest.approx(0.99)


def test_state_pinned_step_round_trip() -> None:
    """`set_pinned_step` accepts int or None; `get_pinned_step` mirrors it."""
    import streamlit as st
    from scripts.dev.gui import state

    st.session_state.clear()
    assert state.get_pinned_step() is None
    state.set_pinned_step(42)
    assert state.get_pinned_step() == 42
    state.set_pinned_step(None)
    assert state.get_pinned_step() is None
