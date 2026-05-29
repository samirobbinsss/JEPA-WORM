"""Smoke test for ``wormjepa run --config configs/baselines/kalman.yaml``.

Verifies Story 3.4's end-to-end CLI wiring: the baseline-run dispatcher loads
the config, instantiates KalmanBaseline, fits + predicts on synthetic data,
computes per-horizon worm-level bootstrap CIs, and writes a contract-compliant
``metrics.json`` to ``results/<run-id>/``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from wormjepa.cli.run_ids import RUN_ID_PATTERN
from wormjepa.eval import MetricsOutput
from wormjepa.reporting import ResultsWriter


def test_wormjepa_run_kalman_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "baselines" / "kalman.yaml"
    assert config_path.is_file()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wormjepa.cli.main",
            "run",
            "--config",
            str(config_path),
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"wormjepa run exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    run_id_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("run-id:")),
        None,
    )
    assert run_id_line is not None
    run_id = run_id_line.split(":", 1)[1].strip()
    assert RUN_ID_PATTERN.match(run_id)

    results_dir = repo_root / "results" / run_id
    try:
        # Contract files present (including the baseline-produced metrics.json).
        for filename in (*ResultsWriter.REQUIRED_INITIAL_FILES, "config.yaml", "metrics.json"):
            assert (results_dir / filename).is_file(), f"missing {filename}"

        # metrics.json validates against the schema and has a kalman future_pose row.
        metrics_text = (results_dir / "metrics.json").read_text(encoding="utf-8")
        metrics = MetricsOutput.from_canonical_json(metrics_text)
        assert metrics.run_id == run_id
        kalman_rows = [e for e in metrics.entries if e.producer == "kalman"]
        assert len(kalman_rows) == 1
        fp_entry = kalman_rows[0]
        assert fp_entry.name == "future_pose"
        # Three pre-registered horizons → three sub-entries.
        assert {se.key for se in fp_entry.sub_entries} == {"0.1s", "1s", "5s"}
        # Each sub-entry has a worm-level CI.
        for sub in fp_entry.sub_entries:
            assert sub.ci.grouping == "worm"
            assert sub.ci.lower <= sub.ci.point <= sub.ci.upper
    finally:
        if results_dir.exists():
            shutil.rmtree(results_dir)
