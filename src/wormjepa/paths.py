"""Project-root resolution for JEPA-WORM.

Single source of truth for finding the project root. Every path resolution in
the codebase routes through ``project_root()`` so the project is relocatable
and no module hardcodes a working directory.
"""

from __future__ import annotations

from pathlib import Path

from wormjepa import WormJEPAError


class ProjectRootNotFoundError(WormJEPAError):
    """Raised when ``project_root()`` cannot locate a ``pyproject.toml`` ancestor."""


def project_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` to find the directory containing ``pyproject.toml``.

    Args:
        start: Starting directory. Defaults to the current working directory.

    Returns:
        Absolute ``Path`` to the project root.

    Raises:
        ProjectRootNotFoundError: If no ancestor contains ``pyproject.toml``.
    """
    cursor = (start or Path.cwd()).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    msg = f"No pyproject.toml found in {cursor} or any parent."
    raise ProjectRootNotFoundError(msg)
