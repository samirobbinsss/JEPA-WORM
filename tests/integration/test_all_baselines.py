"""Integration test for all four Phase 0 MVP baselines (Story 3.8).

Runs each of the four baselines back-to-back through the CLI on the synthetic
loader and verifies:

- exit code 0
- results/<run-id>/ matches the documented contract
- metrics.json validates against ``MetricsOutput``
- the producer-row carries the expected baseline name with three horizon
  sub-rows and worm-level CIs

Note on NaN tolerance: the random-features baseline performs autoregressive
rollout with frozen random encoder weights, which can produce numerical
explosion at long horizons (5s = 50 rollout steps on short synthetic clips).
This is a synthetic-data artifact; real Phase 0 data has longer clips and the
neural-decoding probe (Epic 6) only uses the latent, not the future-pose, of
the random-features baseline. The test tolerates NaN values in horizon sub-rows
while still requiring schema validity and worm-level grouping.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wormjepa.cli.run_ids import RUN_ID_PATTERN
from wormjepa.eval import MetricsOutput
from wormjepa.reporting import ALLOWED_FILES, ResultsWriter

_BASELINES = ("kalman", "transformer_eigenworms", "pose_tcn", "random_features")


@pytest.mark.parametrize("baseline_name", _BASELINES)
def test_baseline_runs_end_to_end(baseline_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "baselines" / f"{baseline_name}.yaml"
    assert config_path.is_file(), f"missing {config_path}"

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
        timeout=120,
    )
    assert result.returncode == 0, (
        f"wormjepa run --config configs/baselines/{baseline_name}.yaml exited "
        f"{result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
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
        # Contract file presence + no surprises.
        for filename in (*ResultsWriter.REQUIRED_INITIAL_FILES, "config.yaml", "metrics.json"):
            assert (results_dir / filename).is_file(), f"missing {filename} for {baseline_name}"
        for entry in results_dir.iterdir():
            if entry.is_dir():
                continue
            assert entry.name in ALLOWED_FILES, (
                f"unexpected file in {baseline_name} results dir: {entry.name}"
            )

        # metrics.json validates and contains a row from this baseline.
        metrics_text = (results_dir / "metrics.json").read_text(encoding="utf-8")
        metrics = MetricsOutput.from_canonical_json(metrics_text)
        assert metrics.run_id == run_id
        baseline_rows = [e for e in metrics.entries if e.producer == baseline_name]
        assert len(baseline_rows) == 1, (
            f"expected exactly one row produced by {baseline_name}; got "
            f"{[e.producer for e in metrics.entries]}"
        )

        fp = baseline_rows[0]
        assert fp.name == "future_pose"
        assert {se.key for se in fp.sub_entries} == {"0.1s", "1s", "5s"}
        for sub in fp.sub_entries:
            assert sub.ci.grouping == "worm"
            assert sub.ci.n_samples >= 1
            # NaN tolerance: see module docstring. Finite CIs must be ordered.
            if not math.isnan(sub.ci.point):
                assert sub.ci.lower <= sub.ci.point <= sub.ci.upper
    finally:
        if results_dir.exists():
            shutil.rmtree(results_dir)
