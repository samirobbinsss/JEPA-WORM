"""CLI surface for JEPA-WORM.

The CLI is the composition root: every Typer subcommand wires together modules
from across the package. This package's only public entry point is ``app`` (a
Typer instance), registered in ``pyproject.toml`` as the ``wormjepa`` script.
"""

from wormjepa.cli.main import app

__all__ = ["app"]
