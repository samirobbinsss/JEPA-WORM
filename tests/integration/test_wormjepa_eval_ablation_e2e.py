"""End-to-end integration test for ``wormjepa eval --control`` (ablation mode).

Mirrors :mod:`tests.integration.test_wormjepa_eval_e2e` but exercises the
neural-prior ablation path: a primary run (with ``warm_start.neural=true``)
is paired with a control run (``warm_start.neural=false``); the eval
orchestrator emits a ``neural_prior_ablation_delta_r2`` MetricEntry
whose CI feeds the ``neural_prior_ablation`` gate.

The test asserts the orchestrator's contract, not the model's claim:
- 5 MetricEntries (the 4 from single-mode + neural_prior_ablation_delta_r2).
- outcome ∈ {cleared, kill_criterion_fired, reframed}.
- gate set matches the pre-registered four primary gates.
- metrics_eval.json round-trips and contains the ablation entry.
- STATUS.md restored byte-stably in ``finally`` regardless of the test
  outcome.

Wall-clock budget: ~20s on a typical dev laptop (two smoke runs at
2 train steps each + eval over both).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

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


def _run_id_from_stdout(stdout: str) -> str:
    line = next(
        (line for line in stdout.splitlines() if line.startswith("run-id:")),
        None,
    )
    assert line is not None, stdout
    return line.split(":", 1)[1].strip()


def _write_control_config(smoke_cfg_path: Path, dest: Path) -> None:
    """Write a sibling smoke config with ``warm_start.neural=false``."""
    raw = yaml.safe_load(smoke_cfg_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw.setdefault("jepa", {}).setdefault("warm_start", {})
    raw["jepa"]["warm_start"]["neural"] = False
    # Bump the seed so the run-id hash differs from the primary's; otherwise
    # the ResultsWriter would clobber the primary's results directory.
    raw["jepa"]["seed"] = int(raw["jepa"].get("seed", 42)) + 1
    dest.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def test_wormjepa_eval_ablation_e2e_against_smoke_checkpoints() -> None:
    repo_root = project_root()
    primary_cfg = repo_root / "configs" / "jepa_smoke.yaml"
    assert primary_cfg.is_file()
    status_backup = repo_root / "STATUS.md"
    status_backup_bytes = status_backup.read_bytes() if status_backup.is_file() else None

    primary_run_id: str | None = None
    control_run_id: str | None = None
    tmp_dir = Path(tempfile.mkdtemp(prefix="wormjepa-ablation-e2e-"))
    try:
        control_cfg = tmp_dir / "jepa_smoke_control.yaml"
        _write_control_config(primary_cfg, control_cfg)

        # 1. Produce primary + control checkpoints.
        primary_result = _invoke(["run", "--config", str(primary_cfg)], repo_root)
        primary_run_id = _run_id_from_stdout(primary_result.stdout)
        control_result = _invoke(["run", "--config", str(control_cfg)], repo_root)
        control_run_id = _run_id_from_stdout(control_result.stdout)
        assert primary_run_id != control_run_id, (primary_run_id, control_run_id)

        primary_dir = repo_root / "results" / primary_run_id
        control_dir = repo_root / "results" / control_run_id
        assert (primary_dir / "checkpoints" / "checkpoint.pt").is_file()
        assert (control_dir / "checkpoints" / "checkpoint.pt").is_file()

        # 2. Run eval in ablation mode.
        eval_result = _invoke(
            ["eval", "--run", primary_run_id, "--control", control_run_id],
            repo_root,
        )

        # outcome is one of the pre-registered categories.
        outcome_line = next(
            line for line in eval_result.stdout.splitlines() if line.startswith("outcome:")
        )
        outcome = outcome_line.split(":", 1)[1].strip()
        assert outcome in {"cleared", "kill_criterion_fired", "reframed"}, outcome

        # Gate verdict lines: all four pre-registered primary gates present.
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

        # metrics_eval.json contract: round-trips, 5 entries (4 + ablation),
        # contains neural_prior_ablation_delta_r2 / jepa.
        eval_json = primary_dir / "metrics_eval.json"
        assert eval_json.is_file()
        parsed = json.loads(eval_json.read_text())
        assert parsed["run_id"] == primary_run_id
        # The smoke run produces partial_R2 + session_id + future_pose x 2
        # (jepa + transformer_eigenworms) = 4 entries from single-mode; the
        # ablation path appends neural_prior_ablation_delta_r2 -> 5.
        # Allow ≥ 4 because session_id can degenerate-skip when every fold
        # lacks class diversity (same caveat as the non-ablation e2e).
        assert len(parsed["entries"]) >= 4, parsed["entries"]
        entry_names = {(e["name"], e["producer"]) for e in parsed["entries"]}
        assert ("neural_prior_ablation_delta_r2", "jepa") in entry_names, entry_names
        assert ("neural_probe_partial_r2", "jepa") in entry_names, entry_names
        assert ("future_pose", "jepa") in entry_names, entry_names
        assert ("future_pose", "transformer_eigenworms") in entry_names, entry_names

        # STATUS.md was touched additively.
        assert status_backup.is_file()
        status_text = status_backup.read_text()
        assert "## Gate verdicts" in status_text
        assert f"last_run_id: {primary_run_id}" in status_text
    finally:
        if primary_run_id is not None:
            shutil.rmtree(repo_root / "results" / primary_run_id, ignore_errors=True)
        if control_run_id is not None:
            shutil.rmtree(repo_root / "results" / control_run_id, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if status_backup_bytes is not None:
            status_backup.write_bytes(status_backup_bytes)
