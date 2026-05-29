"""Smoke test for ``wormjepa run --config configs/jepa_smoke_realdata.yaml``.

Story 8.9 end-to-end check: invokes the CLI against the committed real-format
fixtures (Flavell + WormBehaviorDB + OpenWorm) instead of synthetic data,
verifies the chained-loader path works, and confirms the contract files
(metrics.json, compute.json, manifest_at_run.lock) are all populated.

CPU-only; GPU fields in compute.json are legitimately ``null`` on a CPU host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from wormjepa.eval import MetricsOutput


def test_wormjepa_run_jepa_smoke_realdata() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "jepa_smoke_realdata.yaml"
    assert config_path.is_file()

    result = subprocess.run(
        [sys.executable, "-m", "wormjepa.cli.main", "run", "--config", str(config_path)],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"realdata smoke exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    run_id_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("run-id:")),
        None,
    )
    assert run_id_line is not None
    run_id = run_id_line.split(":", 1)[1].strip()
    results_dir = repo_root / "results" / run_id
    try:
        for filename in ("metrics.json", "compute.json", "manifest_at_run.lock"):
            assert (results_dir / filename).is_file(), f"missing {filename}"

        compute = json.loads((results_dir / "compute.json").read_text(encoding="utf-8"))
        # Story 8.9 AC: compute.json records real wall-clock + python/pytorch versions
        # even on a CPU host. GPU fields may be None.
        assert compute["wall_clock_seconds"] > 0.0
        assert compute["python_version"]
        assert compute["pytorch_version"]

        metrics = MetricsOutput.from_canonical_json(
            (results_dir / "metrics.json").read_text(encoding="utf-8")
        )
        jepa_rows = [e for e in metrics.entries if e.producer == "jepa"]
        assert len(jepa_rows) == 1
        assert jepa_rows[0].name == "jepa_training_loss"

        # manifest_at_run.lock must be byte-identical to current MANIFEST.lock.
        live_lock = (repo_root / "pre-registration" / "MANIFEST.lock").read_bytes()
        copied_lock = (results_dir / "manifest_at_run.lock").read_bytes()
        assert copied_lock == live_lock, (
            "manifest_at_run.lock drifted from pre-registration/MANIFEST.lock"
        )
    finally:
        if results_dir.exists():
            shutil.rmtree(results_dir)
