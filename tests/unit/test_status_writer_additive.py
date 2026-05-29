"""Tests for the additive STATUS.md merger (Story 8.12c.d)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from wormjepa.eval.gates import GateStatus
from wormjepa.manifest.status_writer import update_status_additive


def _gs(outcome: str = "cleared") -> GateStatus:
    return GateStatus(
        gates={
            "kill_criterion": "cleared",
            "neural_probe_partial_r2": "cleared",
            "neural_prior_ablation": "cleared",
            "session_id_at_chance": "cleared",
        },
        outcome=outcome,  # type: ignore[arg-type]
        notes=["all good"],
    )


def test_update_status_additive_writes_fresh_when_file_absent(tmp_path: Path):
    status = tmp_path / "STATUS.md"
    update_status_additive(
        "20260519T000000Z__abc__test",
        _gs("cleared"),
        status_path=status,
        now=datetime(2026, 5, 19, tzinfo=UTC),
    )
    text = status.read_text()
    assert "phase: 0" in text
    assert "gate_status: cleared" in text
    assert "## Gate verdicts" in text


def test_update_status_additive_preserves_caller_owned_sections(tmp_path: Path):
    existing = """# JEPA-WORM Status

phase: 0
gate_status: pending
last_updated: 2026-05-18
last_run_id: old_run

## Gate verdicts

old verdicts table

## Phase 0 milestone progress

milestone table CALLER-OWNED preserve this verbatim

## Custom narrative

Sami's hand-written notes about story 8.11b carryforward.
"""
    status = tmp_path / "STATUS.md"
    status.write_text(existing)
    update_status_additive(
        "new_run",
        _gs("cleared"),
        status_path=status,
        now=datetime(2026, 5, 19, tzinfo=UTC),
    )
    merged = status.read_text()
    # Writer-owned: replaced
    assert "last_run_id: new_run" in merged
    assert "gate_status: cleared" in merged
    assert "old verdicts table" not in merged  # ## Gate verdicts replaced
    # Caller-owned: preserved
    assert "milestone table CALLER-OWNED preserve this verbatim" in merged
    assert "Sami's hand-written notes about story 8.11b carryforward." in merged


def test_update_status_additive_handles_missing_owned_sections(tmp_path: Path):
    existing = """# JEPA-WORM Status

phase: 0
gate_status: pending
last_updated: 2026-05-18
last_run_id: old

## Phase 0 milestone progress

milestone caller-owned
"""
    status = tmp_path / "STATUS.md"
    status.write_text(existing)
    update_status_additive(
        "new",
        _gs("kill_criterion_fired"),
        status_path=status,
        now=datetime(2026, 5, 19, tzinfo=UTC),
    )
    merged = status.read_text()
    # New owned section gets appended
    assert "## Gate verdicts" in merged
    # Caller-owned preserved
    assert "milestone caller-owned" in merged


def test_update_status_additive_idempotent_for_writer_owned(tmp_path: Path):
    status = tmp_path / "STATUS.md"
    fixed = datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC)
    update_status_additive("run1", _gs("cleared"), status_path=status, now=fixed)
    first = status.read_text()
    update_status_additive("run1", _gs("cleared"), status_path=status, now=fixed)
    second = status.read_text()
    assert first == second
