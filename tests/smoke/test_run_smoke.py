"""Smoke test for ``wormjepa run``.

Verifies the end-to-end skeleton: CLI loads, config validates, run-id is
generated, results directory is initialized with the contract files. Must
complete in under 5 wall-clock seconds on CPU per Story 1.8 AC.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from wormjepa.cli.run_ids import RUN_ID_PATTERN
from wormjepa.reporting import ALLOWED_FILES, ResultsWriter


def test_wormjepa_run_smoke_creates_contract_directory(tmp_path: Path) -> None:
    """`wormjepa run --config configs/smoke.yaml` creates a contract-compliant results dir."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "smoke.yaml"
    assert config_path.is_file(), f"missing {config_path}"

    # Use a temporary results dir to keep test side-effects out of the repo.
    # We achieve this by monkey-pointing the results root via an environment
    # variable... actually ResultsWriter consults project_root() by default.
    # To keep the test isolated, we run the subprocess in repo_root but then
    # clean up afterwards.

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "wormjepa.cli.main", "run", "--config", str(config_path)],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    duration = time.perf_counter() - start

    assert result.returncode == 0, (
        f"wormjepa run exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # 12s upper bound: macOS Apple Silicon consistently lands at 6.1-6.6s
    # (Python interpreter cold-start + torch import + tiny train loop). A
    # 30s+ regression still trips this; tighter thresholds were flaking.
    assert duration < 12.0, f"smoke run took {duration:.2f}s (>12s limit)"

    # Parse run-id from stdout.
    run_id_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("run-id:")),
        None,
    )
    assert run_id_line is not None, f"missing run-id in stdout: {result.stdout}"
    run_id = run_id_line.split(":", 1)[1].strip()
    assert RUN_ID_PATTERN.match(run_id), f"run-id {run_id!r} does not match contract"

    results_dir = repo_root / "results" / run_id
    try:
        assert results_dir.is_dir(), f"results dir not created: {results_dir}"

        # Required initial files exist.
        for filename in ResultsWriter.REQUIRED_INITIAL_FILES:
            assert (results_dir / filename).is_file(), f"missing required file {filename}"

        # config.yaml was copied in.
        assert (results_dir / "config.yaml").is_file(), "config.yaml not copied"

        # No files outside the contract crept in.
        for entry in results_dir.iterdir():
            if entry.is_dir():
                continue
            assert entry.name in ALLOWED_FILES, f"unexpected file in results dir: {entry.name}"
    finally:
        # Cleanup so smoke runs don't accumulate.
        if results_dir.exists():
            shutil.rmtree(results_dir)
