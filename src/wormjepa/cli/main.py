"""Typer CLI application root for JEPA-WORM.

Exposes the ``wormjepa`` console script. Subcommands (``run``, ``eval``,
``report``, ``preregister``) are registered here; their implementations live
in sibling modules and are filled in by their respective stories.

The top-level handler:
- catches :class:`wormjepa.WormJEPAError` and formats it via ``rich``;
- exits with code 1 on domain errors;
- lets programming errors (``KeyError``, ``IndexError``, etc.) propagate uncaught
  so they crash visibly.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.traceback import install as install_rich_traceback

from wormjepa import WormJEPAError, __version__
from wormjepa.cli import eval as eval_cmd
from wormjepa.cli import preregister, report, run
from wormjepa.cli.fetch import fetch_app

_console = Console(stderr=True)


class JSONLFormatter(logging.Formatter):
    """JSON Lines log formatter for ``results/<run-id>/log.jsonl``.

    Emits one JSON object per log record with stable keys (``ts``, ``level``,
    ``module``, ``msg``, ``extra``). Stable schema lets reproducers grep and
    parse training logs without parsing freeform prose.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%f"),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        # Attach any structured fields the caller stuffed under ``extra=``.
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "asctime",
            "taskName",
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in reserved}
        if extras:
            payload["extra"] = extras
        return json.dumps(payload, sort_keys=True, default=str)


def _configure_logging() -> None:
    """Configure stdlib logging once at CLI startup.

    Console handler emits human-readable output via ``rich`` for interactive use.
    A JSONL file handler is *not* attached here; subcommands attach it when they
    create their ``results/<run-id>/`` directory and know the log path.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Idempotency: avoid re-adding handlers if main() is called twice in a session
    # (e.g., test invocation).
    if root.handlers:
        return

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console_handler)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"wormjepa {__version__}")
        raise typer.Exit


app = typer.Typer(
    name="wormjepa",
    help="JEPA-WORM Phase 0: self-supervised vision-only world model for C. elegans.",
    no_args_is_help=True,
    add_completion=False,
)

app.command(
    name="run",
    help="Run a training/eval pipeline from a YAML config (filled in by Epic 5/7).",
)(run.run_command)
app.command(
    name="eval",
    help="Evaluate a completed run against frozen metrics (filled in by Epic 6).",
)(eval_cmd.eval_command)
app.command(
    name="report",
    help="Render a report or compare two runs CI-aware (filled in by Epic 7).",
)(report.report_command)
app.command(
    name="preregister",
    help="Lock or verify the pre-registration manifest (filled in by Epic 4).",
)(preregister.preregister_command)
app.add_typer(
    fetch_app,
    name="fetch",
    help="Fetch DOI-pinned datasets and V-JEPA 2.1 checkpoints.",
)


@app.callback()
def _root(  # pyright: ignore[reportUnusedFunction]  # registered as Typer callback
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the wormjepa version and exit.",
        ),
    ] = False,
) -> None:
    """Root callback: configures logging before any subcommand runs."""
    _configure_logging()


def main() -> NoReturn:
    """Entry point used by the ``wormjepa`` console script.

    Wraps the Typer app in the documented error-handling contract: domain
    errors (``WormJEPAError``) are formatted and exit with code 1; everything
    else propagates uncaught.
    """
    install_rich_traceback(show_locals=False)
    try:
        app()
    except WormJEPAError as exc:
        _console.print(f"[bold red]wormjepa error[/bold red] ({type(exc).__name__}): {exc}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
