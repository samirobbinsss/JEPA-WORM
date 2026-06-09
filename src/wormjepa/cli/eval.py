"""``wormjepa eval`` — evaluation runner (Story 8.12c).

Two invocation modes:

- Single-run: ``wormjepa eval --run <id>`` → :func:`evaluate_run`
  produces a :class:`MetricsOutput` + :class:`GateStatus`, writes
  ``results/<id>/metrics_eval.json``, updates STATUS.md additively.
- Sweep (cross-seed): repeat ``--run`` multiple times. Calls
  :func:`evaluate_sweep` and prints a per-seed verdict table plus
  the NFR9 cross-seed-spread for each gate's point estimate.
  Writes ``metrics_eval.json`` to each per-seed run dir; STATUS.md
  reflects the LAST run only (the writer is single-run today;
  cross-seed verdict aggregation in STATUS.md is a future commit).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Annotated, Any

import typer

from wormjepa import WormJEPAError
from wormjepa.eval.gates import GateStatus
from wormjepa.eval.orchestrator import (
    SweepSummary,
    evaluate_run,
    evaluate_run_with_ablations,
    evaluate_sweep,
)
from wormjepa.manifest.status_writer import update_status_additive, update_status_additive_sweep
from wormjepa.paths import project_root

logger = logging.getLogger(__name__)


def eval_command(
    run_id: Annotated[
        list[str],
        typer.Option(
            "--run",
            help=(
                "Run-id whose checkpoints to evaluate. Pass once for "
                "single-run mode; pass multiple times (one --run per "
                "seed) for cross-seed sweep mode."
            ),
        ),
    ],
    control: Annotated[
        str | None,
        typer.Option(
            "--control",
            help=(
                "Optional control run-id for neural_prior_ablation. "
                "Must have been trained with warm_start.neural=False. "
                "When set, eval runs in ablation mode: primary = first "
                "--run; control = --control; emits a "
                "neural_prior_ablation_delta_r2 MetricEntry. Single-run "
                "mode only (incompatible with multi --run sweep)."
            ),
        ),
    ] = None,
    baaiworm_control: Annotated[
        str | None,
        typer.Option(
            "--baaiworm-control",
            help=(
                "Optional control run-id for the BAAIWorm-augmentation "
                "ablation (PRD row 5, reported-only — no threshold). Must "
                "have been trained with the BAAIWorm augmentation loader "
                "removed. When set, eval emits a "
                "baaiworm_augmentation_ablation_delta_r2 MetricEntry "
                "(primary = WITH augmentation; control = WITHOUT). May be "
                "combined with --control to resolve both ablations in one "
                "invocation. Single-run mode only."
            ),
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit a machine-readable JSON document to stdout instead "
                "of the human-readable banner. Useful for piping eval "
                "results into CI gates, scripts, or dashboards."
            ),
        ),
    ] = False,
) -> None:
    """Evaluate one or more completed runs.

    The expected layout is the contract ``ResultsWriter`` produces:
    ``results/<run_id>/{config.yaml, checkpoints/checkpoint.pt}``.
    """
    if not run_id:
        msg = "wormjepa eval: at least one --run is required."
        raise WormJEPAError(msg)
    root = project_root() / "results"
    run_dirs = []
    for rid in run_id:
        d = root / rid
        if not d.is_dir():
            msg = f"wormjepa eval --run {rid!r}: results directory not found at {d}"
            raise WormJEPAError(msg)
        run_dirs.append(d)

    if control is not None or baaiworm_control is not None:
        if len(run_dirs) != 1:
            msg = "wormjepa eval: --control / --baaiworm-control require exactly one --run."
            raise WormJEPAError(msg)
        control_dir = None
        if control is not None:
            control_dir = root / control
            if not control_dir.is_dir():
                msg = f"wormjepa eval --control {control!r}: not found at {control_dir}"
                raise WormJEPAError(msg)
        baaiworm_control_dir = None
        if baaiworm_control is not None:
            baaiworm_control_dir = root / baaiworm_control
            if not baaiworm_control_dir.is_dir():
                msg = (
                    f"wormjepa eval --baaiworm-control {baaiworm_control!r}: "
                    f"not found at {baaiworm_control_dir}"
                )
                raise WormJEPAError(msg)
        _emit_ablation(
            run_dirs[0],
            run_id[0],
            control_dir,
            control,
            baaiworm_control_dir,
            baaiworm_control,
            json_out=json_out,
        )
    elif len(run_dirs) == 1:
        _emit_single(run_dirs[0], run_id[0], json_out=json_out)
    else:
        _emit_sweep(run_dirs, run_id, json_out=json_out)


def _build_json_payload(
    mode: str,
    run_ids: list[str],
    gate_status: GateStatus,
    metrics_paths: list[str],
    sweep_summary: SweepSummary | None = None,
) -> dict[str, Any]:
    """Assemble the ``--json`` payload contract.

    Schema (stable):
        {"mode": "single"|"sweep"|"ablation",
         "run_ids": [...],
         "outcome": "cleared"|"kill_criterion_fired"|"reframed",
         "gates": {"<gate_name>": "<verdict>"},
         "notes": [...],
         "metrics_eval_paths": [...],
         "sweep_summary": {...}  # only for sweep mode}
    """
    payload: dict[str, Any] = {
        "mode": mode,
        "run_ids": list(run_ids),
        "outcome": gate_status.outcome,
        "gates": {str(gate): str(verdict) for gate, verdict in gate_status.gates.items()},
        "notes": list(gate_status.notes),
        "metrics_eval_paths": list(metrics_paths),
    }
    if sweep_summary is not None:
        # SweepSummary + nested _GateSeedSpread are frozen dataclasses; lift
        # to dicts via dataclasses.asdict for canonical JSON serialisation.
        payload["sweep_summary"] = dataclasses.asdict(sweep_summary)
    return payload


def _emit_json(payload: dict[str, Any]) -> None:
    """Print the JSON payload to stdout with a trailing newline."""
    typer.echo(json.dumps(payload, sort_keys=True, indent=2))


def _emit_ablation(
    primary_dir: object,
    primary_rid: str,
    control_dir: object,
    control_rid: str | None,
    baaiworm_control_dir: object = None,
    baaiworm_control_rid: str | None = None,
    *,
    json_out: bool = False,
) -> None:
    from pathlib import Path

    assert isinstance(primary_dir, Path)
    assert control_dir is None or isinstance(control_dir, Path)
    assert baaiworm_control_dir is None or isinstance(baaiworm_control_dir, Path)
    logger.info(
        "eval ablation invoked",
        extra={
            "primary": primary_rid,
            "control": control_rid,
            "baaiworm_control": baaiworm_control_rid,
        },
    )
    metrics, gate_status = evaluate_run_with_ablations(
        primary_dir,
        control_run_dir=control_dir,
        baaiworm_control_run_dir=baaiworm_control_dir,
    )

    out_path = primary_dir / "metrics_eval.json"
    out_path.write_text(metrics.to_canonical_json(), encoding="utf-8")
    status_path = update_status_additive(primary_rid, gate_status)

    run_ids = [primary_rid]
    if control_rid is not None:
        run_ids.append(control_rid)
    if baaiworm_control_rid is not None:
        run_ids.append(baaiworm_control_rid)

    if json_out:
        _emit_json(
            _build_json_payload(
                mode="ablation",
                run_ids=run_ids,
                gate_status=gate_status,
                metrics_paths=[str(out_path)],
            )
        )
        return

    typer.echo(f"primary: {primary_rid}")
    if control_rid is not None:
        typer.echo(f"control: {control_rid}")
    if baaiworm_control_rid is not None:
        typer.echo(f"baaiworm_control: {baaiworm_control_rid}")
    typer.echo(f"outcome: {gate_status.outcome}")
    for gate, verdict in sorted(gate_status.gates.items()):
        typer.echo(f"  {gate}: {verdict}")
    typer.echo(f"metrics: {len(metrics.entries)} entries")
    typer.echo(f"metrics_eval: {out_path}")
    typer.echo(f"status: {status_path}")


def _emit_single(run_dir: object, rid: str, *, json_out: bool = False) -> None:
    from pathlib import Path

    assert isinstance(run_dir, Path)
    logger.info("eval subcommand invoked", extra={"run_id": rid})
    metrics, gate_status = evaluate_run(run_dir)

    out_path = run_dir / "metrics_eval.json"
    out_path.write_text(metrics.to_canonical_json(), encoding="utf-8")
    status_path = update_status_additive(rid, gate_status)

    if json_out:
        _emit_json(
            _build_json_payload(
                mode="single",
                run_ids=[rid],
                gate_status=gate_status,
                metrics_paths=[str(out_path)],
            )
        )
        return

    typer.echo(f"run-id: {rid}")
    typer.echo(f"outcome: {gate_status.outcome}")
    for gate, verdict in sorted(gate_status.gates.items()):
        typer.echo(f"  {gate}: {verdict}")
    typer.echo(f"metrics: {len(metrics.entries)} entries")
    if not metrics.entries:
        typer.echo(
            "  (probe suite not yet wired — see "
            "_bmad-output/planning-artifacts/epic-9-dataset-pipeline-substance.md "
            "for the Story 8.12c.1+ probe-wiring sub-stories.)"
        )

    typer.echo(f"metrics_eval: {out_path}")
    typer.echo(f"status: {status_path}")


def _emit_sweep(
    run_dirs: list[object],
    run_ids: list[str],
    *,
    json_out: bool = False,
) -> None:
    from pathlib import Path

    assert all(isinstance(d, Path) for d in run_dirs)
    run_paths = [d for d in run_dirs if isinstance(d, Path)]
    logger.info("eval sweep invoked", extra={"run_ids": run_ids})
    metrics_list, gates_list, summary = evaluate_sweep(run_paths)

    metrics_paths: list[str] = []
    for rid, run_dir, metrics in zip(run_ids, run_paths, metrics_list, strict=True):
        out_path = run_dir / "metrics_eval.json"
        out_path.write_text(metrics.to_canonical_json(), encoding="utf-8")
        metrics_paths.append(str(out_path))
        if not json_out:
            typer.echo(f"metrics_eval[{rid}]: {out_path}")

    status_path = update_status_additive_sweep(run_ids, summary, gates_list[-1])

    if json_out:
        _emit_json(
            _build_json_payload(
                mode="sweep",
                run_ids=list(run_ids),
                gate_status=gates_list[-1],
                metrics_paths=metrics_paths,
                sweep_summary=summary,
            )
        )
        return

    typer.echo("=== Cross-seed sweep ===")
    typer.echo(f"runs: {len(run_ids)}")
    for rid in run_ids:
        typer.echo(f"  - {rid}")
    # Per-gate seed-spread table.
    typer.echo("")
    typer.echo("Per-gate cross-seed verdicts + NFR9 point-estimate spread:")
    typer.echo(f"{'gate':<32} {'consensus':<10} {'mean':>9} {'min':>9} {'max':>9} per_seed")
    for s in summary.per_gate:
        per_seed = "[" + ",".join(s.per_seed_verdict) + "]"
        typer.echo(
            f"{s.gate:<32} {s.consensus_verdict:<10} "
            f"{s.point_mean:>9.4f} {s.point_min:>9.4f} {s.point_max:>9.4f} {per_seed}"
        )
    typer.echo(f"status (cross-seed + last-seed): {status_path}")
