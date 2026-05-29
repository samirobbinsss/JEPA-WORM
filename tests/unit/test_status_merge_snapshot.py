"""Snapshot tests for `_merge_status` additive-merge semantics.

The writer owns three section headings (`## Gate verdicts`, `## Notes`,
`## Cross-seed sweep`); every other top-level section is caller-owned
and must survive verbatim through additive updates. These tests pin the
contract via exact-output snapshots so a regression in the merge logic
trips immediately.
"""

from __future__ import annotations

from wormjepa.manifest.status_writer import _merge_status


def test_merge_preserves_caller_owned_section() -> None:
    """Caller-owned `## Phase 0 milestone progress` section survives merge."""
    existing = (
        "# STATUS\n\n"
        "last_run_id: prior-run\n\n"
        "## Phase 0 milestone progress\n\n"
        "- Milestone 0.1 cleared 2026-05-12\n"
        "- Milestone 0.2 in flight\n\n"
        "## Gate verdicts\n\n"
        "| gate | verdict |\n"
        "|---|---|\n"
        "| kill_criterion | pending |\n"
    )
    rendered = (
        "# STATUS\n\n"
        "last_run_id: new-run\n\n"
        "## Gate verdicts\n\n"
        "| gate | verdict |\n"
        "|---|---|\n"
        "| kill_criterion | cleared |\n"
    )
    merged = _merge_status(existing, rendered)

    assert "## Phase 0 milestone progress" in merged
    assert "Milestone 0.1 cleared 2026-05-12" in merged
    assert "Milestone 0.2 in flight" in merged
    assert "last_run_id: new-run" in merged
    assert "last_run_id: prior-run" not in merged
    assert "| kill_criterion | cleared |" in merged
    assert "| kill_criterion | pending |" not in merged


def test_merge_overwrites_writer_owned_section() -> None:
    """Writer-owned `## Gate verdicts` table is replaced wholesale, not appended."""
    existing = (
        "# STATUS\n\n"
        "## Gate verdicts\n\n"
        "| gate | verdict |\n"
        "|---|---|\n"
        "| neural_probe_partial_r2 | pending |\n"
        "| kill_criterion | pending |\n"
    )
    rendered = (
        "# STATUS\n\n"
        "## Gate verdicts\n\n"
        "| gate | verdict |\n"
        "|---|---|\n"
        "| neural_probe_partial_r2 | cleared |\n"
        "| kill_criterion | cleared |\n"
    )
    merged = _merge_status(existing, rendered)

    assert merged.count("## Gate verdicts") == 1
    assert "| neural_probe_partial_r2 | cleared |" in merged
    assert "| neural_probe_partial_r2 | pending |" not in merged


def test_merge_appends_new_writer_section_to_existing() -> None:
    """Writer-owned section absent from existing file is appended on merge."""
    existing = "# STATUS\n\n## Phase 0 milestone progress\n\n- Milestone narrative\n"
    rendered = (
        "# STATUS\n\n"
        "## Gate verdicts\n\n"
        "| gate | verdict |\n"
        "|---|---|\n"
        "| kill_criterion | cleared |\n"
    )
    merged = _merge_status(existing, rendered)

    assert "## Phase 0 milestone progress" in merged
    assert "Milestone narrative" in merged
    assert "## Gate verdicts" in merged
    assert "kill_criterion | cleared" in merged


def test_merge_handles_empty_existing() -> None:
    """Empty existing file: merge returns the rendered output verbatim header."""
    existing = ""
    rendered = (
        "# STATUS\n\n"
        "last_run_id: only-run\n\n"
        "## Gate verdicts\n\n"
        "| gate | verdict |\n"
        "|---|---|\n"
        "| kill_criterion | cleared |\n"
    )
    merged = _merge_status(existing, rendered)
    assert "last_run_id: only-run" in merged
    assert "kill_criterion | cleared" in merged


def test_merge_preserves_multiple_caller_sections_in_order() -> None:
    """Two caller-owned sections survive in their original order."""
    existing = (
        "# STATUS\n\n"
        "## Phase 0 milestone progress\n\n- Milestone\n\n"
        "## Open questions\n\n- Question 1\n\n"
        "## Gate verdicts\n\n| gate | verdict |\n|---|---|\n| x | pending |\n"
    )
    rendered = "# STATUS\n\n## Gate verdicts\n\n| gate | verdict |\n|---|---|\n| x | cleared |\n"
    merged = _merge_status(existing, rendered)

    milestone_idx = merged.index("## Phase 0 milestone progress")
    open_q_idx = merged.index("## Open questions")
    gates_idx = merged.index("## Gate verdicts")
    assert milestone_idx < open_q_idx < gates_idx
    assert "| x | cleared |" in merged
    assert "| x | pending |" not in merged
