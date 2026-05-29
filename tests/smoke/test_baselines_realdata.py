"""Smoke test for Story 8.10: each of the 4 baselines runs via the CLI
against a chained real-data + synthetic loader composition, producing a
contract-compliant ``results/<run-id>/`` with populated ``metrics.json``,
``compute.json``, and ``manifest_at_run.lock``.

Parameterised over the four baseline configs under ``configs/baselines/``
that carry the ``_realdata.yaml`` suffix.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wormjepa.eval import MetricsOutput

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINES = (
    "kalman",
    "pose_tcn",
    "transformer_eigenworms",
    "random_features",
)


@pytest.mark.parametrize("baseline_name", _BASELINES)
def test_baseline_runs_on_real_data_chain(baseline_name: str) -> None:
    config_path = _REPO_ROOT / "configs" / "baselines" / f"{baseline_name}_realdata.yaml"
    assert config_path.is_file(), f"missing {config_path}"

    result = subprocess.run(
        [sys.executable, "-m", "wormjepa.cli.main", "run", "--config", str(config_path)],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{baseline_name} realdata exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    run_id_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("run-id:")),
        None,
    )
    assert run_id_line is not None
    run_id = run_id_line.split(":", 1)[1].strip()
    results_dir = _REPO_ROOT / "results" / run_id
    try:
        for filename in ("metrics.json", "compute.json", "manifest_at_run.lock"):
            assert (results_dir / filename).is_file(), f"{baseline_name}: missing {filename}"

        # compute.json populated with real fields (Story 8.9 contract).
        compute = json.loads((results_dir / "compute.json").read_text(encoding="utf-8"))
        assert compute["wall_clock_seconds"] > 0.0
        assert compute["python_version"]
        assert compute["pytorch_version"]

        # metrics.json carries one future_pose row producer=baseline_name with 3 horizon sub-rows.
        metrics = MetricsOutput.from_canonical_json(
            (results_dir / "metrics.json").read_text(encoding="utf-8")
        )
        rows = [e for e in metrics.entries if e.producer == baseline_name]
        assert len(rows) == 1
        entry = rows[0]
        assert entry.name == "future_pose"
        # Smoke configs trim long horizons that the 16-frame clips can't cover.
        assert {se.key for se in entry.sub_entries} == {"0.1s", "1s"}
        # The wiring assertion is "metrics were computed at all", not "metrics
        # are numerically sensible" — fit on a synthetic+fixture chain may
        # diverge for random-init baselines, that's a known smoke limitation
        # rather than a wiring bug. n_samples should be >0 in every case,
        # confirming the bootstrap actually ran with real data.
        for se in entry.sub_entries:
            assert se.ci.n_samples > 0, f"{baseline_name}/{se.key}: empty bootstrap"

        # manifest_at_run.lock byte-identical to live MANIFEST.lock (FR3).
        live_lock = (_REPO_ROOT / "pre-registration" / "MANIFEST.lock").read_bytes()
        copied_lock = (results_dir / "manifest_at_run.lock").read_bytes()
        assert copied_lock == live_lock
    finally:
        if results_dir.exists():
            shutil.rmtree(results_dir)
