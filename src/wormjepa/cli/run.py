"""``wormjepa run`` — training/eval orchestrator.

Story 1.8 scope: load and validate the config, generate a run-id, initialize a
contract-compliant ``results/<run-id>/`` directory, and exit cleanly. No actual
training happens — Epic 5 fills that in.

Subsequent stories (Epic 5 Story 5.8, Epic 7 Story 7.9) extend this command to
launch the training loop, evaluate, and produce a published result.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

import torch
import typer
import yaml

from wormjepa import PreRegistrationViolation, WormJEPAError
from wormjepa.baselines._runner import run_baseline
from wormjepa.cli.run_ids import generate_run_id
from wormjepa.configs import WormJEPAConfig, load_config
from wormjepa.configs.baseline_config import BaselineRunConfig
from wormjepa.configs.jepa_config import JEPARunConfig
from wormjepa.paths import project_root
from wormjepa.reporting import ResultsWriter
from wormjepa.reporting.compute_provenance import record_provenance
from wormjepa.training.runner import run_jepa

logger = logging.getLogger(__name__)


def run_command(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to a YAML config file."),
    ],
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Resume a prior run by run-id (not yet implemented)."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help=(
                "Override the config's `jepa.seed` for this run. Used by the "
                "Story 8.11c three-seed headline sweep to launch seeds 1337 + "
                "8675309 against the locked headline.yaml (which carries seed=42 "
                "as the first pre-committed seed). The override is recorded in "
                "results/<run-id>/seed.txt and the config.yaml mirror reflects "
                "the override too."
            ),
        ),
    ] = None,
) -> None:
    """Load + validate the config, initialize a results directory, exit.

    The training loop, eval, and reporting wiring land in later stories. For
    now this command proves that:

    1. the CLI loads;
    2. the config schema validates a YAML file;
    3. the run-id generator produces a contract-compliant id;
    4. the ResultsWriter creates ``results/<run-id>/`` with the contract files.
    """
    if not config.is_file():
        msg = f"Config file not found: {config}"
        raise WormJEPAError(msg)

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Config file {config} did not yield a YAML mapping at the top level."
        raise WormJEPAError(msg)

    baseline_cfg: BaselineRunConfig | None = None
    jepa_cfg: JEPARunConfig | None = None
    if "baseline" in raw:
        baseline_cfg = load_config(raw, BaselineRunConfig)
        logger.info(
            "baseline-run config validated",
            extra={"path": str(config), "baseline": baseline_cfg.baseline.name},
        )
    elif "jepa" in raw:
        jepa_cfg = load_config(raw, JEPARunConfig)
        if seed is not None:
            # Frozen pydantic models — round-trip via model_copy so the override
            # produces a fresh validated instance instead of mutating in place.
            jepa_cfg = jepa_cfg.model_copy(
                update={"jepa": jepa_cfg.jepa.model_copy(update={"seed": seed})}
            )
            logger.info(
                "jepa-run config validated (seed overridden via --seed)",
                extra={"path": str(config), "seed": seed},
            )
        else:
            logger.info("jepa-run config validated", extra={"path": str(config)})
    else:
        _validated: WormJEPAConfig = load_config(raw, WormJEPAConfig)
        logger.info("config validated", extra={"path": str(config)})

    run_id = generate_run_id(config)
    logger.info("run-id generated", extra={"run_id": run_id})

    writer = ResultsWriter(run_id)
    results_dir = writer.initialize()
    # Copy the config verbatim into the results directory for audit (NFR13).
    # If --seed was used, the on-disk config no longer reflects the run's
    # actual seed; surface the effective seed via seed.txt (see below) and a
    # marker in the saved config so audit trail is honest.
    saved_config_text = config.read_text(encoding="utf-8")
    if jepa_cfg is not None and seed is not None:
        saved_config_text = (
            f"# Story 8.11c seed override: --seed {seed} (config file carries "
            f"the pre-committed seed=42; this run was launched with seed={seed} "
            f"as one of the three pre-committed sweep seeds).\n" + saved_config_text
        )
    writer.write_text("config.yaml", saved_config_text)
    # NFR13: results/<run-id>/seed.txt records the actual seed used. With a
    # --seed override, this is the override; otherwise the config's seed.
    if jepa_cfg is not None:
        effective_seed = seed if seed is not None else jepa_cfg.jepa.seed
        writer.write_text("seed.txt", f"{effective_seed}\n")

    # Story 4.9: copy MANIFEST.lock into the run dir so the lock state at
    # run-time is captured even if MANIFEST.lock changes later. Absent lock
    # is tolerated only when the run is *not* baseline_cfg-driven (the smoke
    # config in configs/smoke.yaml exercises the CLI shell before lock).
    manifest_src = project_root() / "pre-registration" / "MANIFEST.lock"
    is_reportable = baseline_cfg is not None or jepa_cfg is not None
    if manifest_src.is_file():
        writer.write_text("manifest_at_run.lock", manifest_src.read_text(encoding="utf-8"))
        logger.info("manifest_at_run.lock copied", extra={"run_id": run_id})
    elif is_reportable:
        msg = (
            "Reportable run requires pre-registration/MANIFEST.lock. "
            "Run `wormjepa preregister` first."
        )
        raise PreRegistrationViolation(msg)

    # Reset accelerator peak-memory counters so compute.json reflects this
    # run only. MPS (Apple Silicon) exposes the same call name as CUDA via
    # torch.mps in PyTorch 2.x; guarded so older builds without it still run.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elif torch.backends.mps.is_available():
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None:
            reset_fn = getattr(mps_module, "empty_cache", None)
            if callable(reset_fn):
                reset_fn()
    start_perf = time.perf_counter()

    if baseline_cfg is not None:
        metrics = run_baseline(baseline_cfg, run_id=run_id)
        writer.write_text("metrics.json", metrics.to_canonical_json())
        logger.info(
            "baseline metrics written",
            extra={"baseline": baseline_cfg.baseline.name, "n_entries": len(metrics.entries)},
        )
    elif jepa_cfg is not None:
        metrics, _state = run_jepa(jepa_cfg, run_id=run_id)
        writer.write_text("metrics.json", metrics.to_canonical_json())
        logger.info("jepa metrics written", extra={"n_entries": len(metrics.entries)})

    # Story 8.9 / Story 7.1: record compute provenance for every run. Even
    # non-reportable smoke runs get a real compute.json so the contract path
    # is exercised end-to-end (the placeholder `{}` from `initialize()` is
    # overwritten here).
    provenance = record_provenance(start_perf)
    writer.write_text("compute.json", provenance.to_canonical_json())
    logger.info(
        "compute provenance written",
        extra={
            "run_id": run_id,
            "gpu_model": provenance.gpu_model,
            "wall_clock_seconds": provenance.wall_clock_seconds,
            "peak_gpu_memory_bytes": provenance.peak_gpu_memory_bytes,
        },
    )

    logger.info(
        "results directory initialized",
        extra={"run_id": run_id, "results_dir": str(results_dir)},
    )

    if resume is not None:
        typer.echo(f"[skeleton] resume not yet implemented (Epic 5 Story 5.10): {resume}")

    typer.echo(f"run-id: {run_id}")
    typer.echo(f"results: {results_dir}")
