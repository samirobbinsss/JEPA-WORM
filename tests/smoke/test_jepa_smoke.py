"""Smoke test for ``wormjepa run --config configs/jepa_smoke.yaml``.

Verifies Story 5.11's end-to-end CLI wiring: the JEPA training runner loads
the config, builds the encoder + EMA + predictor + warm-start heads, runs
training steps on the configured loaders, writes contract-compliant
``results/<run-id>/{metrics.json, compute.json, manifest_at_run.lock}``.

Story 8.9 added the ``compute.json`` and config-driven-loader checks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from wormjepa.eval import MetricsOutput
from wormjepa.reporting import ResultsWriter


def _invoke_run(config_path: Path, repo_root: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "wormjepa.cli.main", "run", "--config", str(config_path)],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"wormjepa run {config_path.name} exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    run_id_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("run-id:")),
        None,
    )
    assert run_id_line is not None
    return run_id_line.split(":", 1)[1].strip()


def _assert_contract_files(results_dir: Path) -> None:
    for filename in (
        *ResultsWriter.REQUIRED_INITIAL_FILES,
        "metrics.json",
        "manifest_at_run.lock",
    ):
        assert (results_dir / filename).is_file(), f"missing {filename}"


def _assert_compute_json_populated(results_dir: Path) -> None:
    """Story 8.9: compute.json is no longer the `{}` placeholder."""
    payload = json.loads((results_dir / "compute.json").read_text(encoding="utf-8"))
    # Required keys per ComputeProvenance.to_canonical_json().
    for key in (
        "python_version",
        "pytorch_version",
        "cuda_version",
        "gpu_model",
        "host_machine",
        "wall_clock_seconds",
        "peak_gpu_memory_bytes",
        "gpu_hours",
    ):
        assert key in payload, f"compute.json missing {key!r}"
    # On CPU-only hosts gpu_model / cuda_version / peak_gpu_memory_bytes are
    # legitimately None; wall_clock_seconds and gpu_hours must still be real.
    assert isinstance(payload["wall_clock_seconds"], (int, float))
    assert payload["wall_clock_seconds"] > 0.0
    assert isinstance(payload["gpu_hours"], (int, float))
    assert payload["gpu_hours"] >= 0.0
    assert payload["python_version"]
    assert payload["pytorch_version"]


def test_wormjepa_run_jepa_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "jepa_smoke.yaml"
    assert config_path.is_file()

    run_id = _invoke_run(config_path, repo_root)
    results_dir = repo_root / "results" / run_id
    try:
        _assert_contract_files(results_dir)
        _assert_compute_json_populated(results_dir)
        metrics = MetricsOutput.from_canonical_json(
            (results_dir / "metrics.json").read_text(encoding="utf-8")
        )
        jepa_rows = [e for e in metrics.entries if e.producer == "jepa"]
        assert len(jepa_rows) == 1
        entry = jepa_rows[0]
        assert entry.name == "jepa_training_loss"
        assert any(se.key == "loss_jepa" for se in entry.sub_entries)
        # Story 8.12a: run_jepa now persists a checkpoint so the gate-eval
        # orchestrator (Story 8.12c) can reload the trained state.
        checkpoint = results_dir / "checkpoints" / "checkpoint.pt"
        assert checkpoint.is_file(), f"Expected checkpoint at {checkpoint}; not present."
        assert checkpoint.stat().st_size > 0, "Checkpoint file is empty."
    finally:
        if results_dir.exists():
            shutil.rmtree(results_dir)
