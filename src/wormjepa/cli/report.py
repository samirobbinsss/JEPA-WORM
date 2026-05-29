"""``wormjepa report`` — outcome-aware reporting and CI-aware comparison (Story 7.5)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wormjepa import WormJEPAError
from wormjepa.eval.gates import evaluate_gates
from wormjepa.eval.metrics_schema import MetricsOutput
from wormjepa.paths import project_root
from wormjepa.reporting.render import compare_metrics, render_report

logger = logging.getLogger(__name__)
_console = Console()


def _load_metrics(run_id: str) -> tuple[MetricsOutput, Path]:
    """Load probe-suite metrics for ``run_id``.

    Prefers ``results/<run_id>/metrics_eval.json`` (written by ``wormjepa eval``,
    Story 8.12c) over the training-time ``metrics.json``, which is essentially
    empty for the Phase 0 use case.
    """
    results_dir = project_root() / "results" / run_id
    eval_path = results_dir / "metrics_eval.json"
    train_path = results_dir / "metrics.json"
    if eval_path.is_file():
        logger.info("report: using metrics_eval.json", extra={"path": str(eval_path)})
        path = eval_path
    elif train_path.is_file():
        logger.info("report: falling back to metrics.json", extra={"path": str(train_path)})
        path = train_path
    else:
        msg = f"no metrics file found at {eval_path} or {train_path}"
        raise WormJEPAError(msg)
    return MetricsOutput.from_canonical_json(path.read_text(encoding="utf-8")), results_dir


def _render_to_disk(run_id: str) -> None:
    metrics, results_dir = _load_metrics(run_id)
    gate_status = evaluate_gates(metrics)
    text = render_report(metrics, gate_status)
    (results_dir / "report.md").write_text(text, encoding="utf-8")
    _console.print(f"[bold green]report rendered[/bold green]: {results_dir / 'report.md'}")
    _console.print(f"outcome: [bold]{gate_status.outcome}[/bold]")


def _print_compare(run_id: str, published: Path) -> None:
    metrics, _ = _load_metrics(run_id)
    if not published.is_file():
        msg = f"published metrics.json not found at {published}"
        raise WormJEPAError(msg)
    pub_metrics = MetricsOutput.from_canonical_json(published.read_text(encoding="utf-8"))
    diffs = compare_metrics(metrics, pub_metrics)
    table = Table(title=f"CI-aware comparison: {run_id} vs {published}")
    table.add_column("entry", style="cyan")
    table.add_column("status", style="bold")
    table.add_column("message")
    any_outside = False
    for name, status, msg in diffs:
        color = "green" if status == "within_ci" else "red"
        if status != "within_ci":
            any_outside = True
        table.add_row(name, f"[{color}]{status}[/{color}]", msg)
    _console.print(table)
    if any_outside:
        msg = "one or more reported numbers fall outside the published CI"
        raise WormJEPAError(msg)


def _print_gate(run_id: str, gate_name: str) -> None:
    metrics, _ = _load_metrics(run_id)
    status = evaluate_gates(metrics)
    verdict = status.gates.get(gate_name)  # type: ignore[arg-type]
    if verdict is None:
        msg = f"unknown gate {gate_name!r}; known: {sorted(status.gates)}"
        raise WormJEPAError(msg)
    table = Table(title=f"gate: {gate_name}")
    table.add_column("verdict", style="bold")
    color = "green" if verdict == "cleared" else "red" if verdict == "fired" else "yellow"
    table.add_row(f"[{color}]{verdict}[/{color}]")
    _console.print(table)


def report_command(
    run_id: Annotated[
        str | None,
        typer.Option("--run", help="Render the outcome-aware template for this run."),
    ] = None,
    compare: Annotated[
        Path | None,
        typer.Option("--compare", help="Published metrics.json to diff CI-aware against."),
    ] = None,
    gate: Annotated[
        str | None,
        typer.Option("--gate", help="Print the named gate's verdict from metrics.json."),
    ] = None,
) -> None:
    """Render reports, compare runs, inspect gates."""
    if run_id is None:
        msg = "wormjepa report requires --run <run-id>"
        raise WormJEPAError(msg)
    logger.info(
        "report subcommand invoked",
        extra={"run_id": run_id, "compare": str(compare) if compare else None, "gate": gate},
    )

    if compare is not None:
        _print_compare(run_id, compare)
        return

    if gate is not None:
        _print_gate(run_id, gate)
        return

    _render_to_disk(run_id)
