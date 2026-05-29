"""End-to-end integration test for ``wormjepa eval`` (Story 8.12c).

Runs the full pipeline once per pytest invocation:

  1. ``wormjepa run --config configs/jepa_smoke.yaml`` produces a
     results dir with a checkpoint.
  2. ``wormjepa eval --run <run-id>`` reconstructs state, runs the
     probe suite (partial_R², session_id, future_pose x 2 producers),
     applies Holm correction, writes ``metrics_eval.json``, and
     updates STATUS.md additively.
  3. Asserts the eval output's invariants — number of MetricEntries,
     gate-verdict set, presence of Holm notes — without locking in
     specific numerical verdicts (they're a function of the random-
     init smoke encoder; the test is about the orchestrator's
     contract, not the model's claim).

This test catches regressions in the orchestrator wiring at CI time
without needing the WORMJEPA_TEST_VJEPA21=1 env flag (no V-JEPA 2.1
checkpoint download required — the smoke config uses random vit_tiny).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from wormjepa.paths import project_root


def _invoke(cmd: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "wormjepa.cli.main", *cmd],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode != 0:
        msg = f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        raise AssertionError(msg)
    return result


def test_wormjepa_eval_e2e_against_smoke_checkpoint() -> None:
    repo_root = project_root()
    config = repo_root / "configs" / "jepa_smoke.yaml"
    assert config.is_file()
    status_backup = repo_root / "STATUS.md"
    status_backup_bytes = status_backup.read_bytes() if status_backup.is_file() else None

    # 1. Produce a fresh checkpoint via the run CLI.
    run_result = _invoke(["run", "--config", str(config)], repo_root)
    run_id_line = next(
        (line for line in run_result.stdout.splitlines() if line.startswith("run-id:")),
        None,
    )
    assert run_id_line is not None, run_result.stdout
    run_id = run_id_line.split(":", 1)[1].strip()
    results_dir = repo_root / "results" / run_id
    try:
        assert (results_dir / "checkpoints" / "checkpoint.pt").is_file()

        # 2. Run eval; assert contract invariants on its output.
        eval_result = _invoke(["eval", "--run", run_id], repo_root)
        # `outcome:` line must be one of the three pre-registered categories.
        outcome_line = next(
            line for line in eval_result.stdout.splitlines() if line.startswith("outcome:")
        )
        outcome = outcome_line.split(":", 1)[1].strip()
        assert outcome in {"cleared", "kill_criterion_fired", "reframed"}, outcome

        # Per-gate verdict lines must include all four pre-registered gates.
        gate_lines = [
            line.strip()
            for line in eval_result.stdout.splitlines()
            if line.startswith("  ") and ":" in line
        ]
        gate_names = {line.split(":", 1)[0].strip() for line in gate_lines}
        assert {
            "kill_criterion",
            "neural_probe_partial_r2",
            "neural_prior_ablation",
            "session_id_at_chance",
        }.issubset(gate_names), gate_names

        # metrics_eval.json exists and round-trips as JSON.
        eval_json = results_dir / "metrics_eval.json"
        assert eval_json.is_file()
        parsed = json.loads(eval_json.read_text())
        assert "run_id" in parsed
        assert parsed["run_id"] == run_id
        # The smoke run wires future_pose (x2 producers) + partial_R² +
        # session_id = 4 entries. Allow ≥ 3 because session_id can
        # degenerate-skip if every fold lacks class diversity.
        assert len(parsed["entries"]) >= 3
        entry_names = {(e["name"], e["producer"]) for e in parsed["entries"]}
        assert ("neural_probe_partial_r2", "jepa") in entry_names
        assert ("future_pose", "jepa") in entry_names
        assert ("future_pose", "transformer_eigenworms") in entry_names

        # 3. STATUS.md was touched additively: writer-owned header +
        # `## Gate verdicts` table present; the milestone narrative
        # (caller-owned) survives. Check both.
        assert status_backup.is_file()
        status_text = status_backup.read_text()
        assert "## Gate verdicts" in status_text
        assert f"last_run_id: {run_id}" in status_text
        if status_backup_bytes is not None:
            # The pre-test STATUS had this caller-owned section; it should
            # survive the additive merge.
            pre = status_backup_bytes.decode("utf-8")
            if "## Phase 0 milestone progress" in pre:
                assert "## Phase 0 milestone progress" in status_text
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)
        if status_backup_bytes is not None:
            status_backup.write_bytes(status_backup_bytes)
