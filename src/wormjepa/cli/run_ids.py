"""Run-id generation.

Every reportable run is identified by a deterministic, filesystem-safe slug:

    <UTC-isoformat>__<short-git-sha>__<config-slug>

where:

- ``<UTC-isoformat>``  = ``YYYYMMDDTHHMMSSZ`` (no colons; filesystem-safe);
- ``<short-git-sha>``  = first 8 hex characters of ``git rev-parse HEAD``,
                         with ``-dirty`` appended when the working tree has
                         uncommitted changes to tracked files;
- ``<config-slug>``    = the config filename stem normalized to snake_case.

Run-ids must be generated through :func:`generate_run_id` — never assembled
ad-hoc — so the format invariant survives implementation churn.
"""

from __future__ import annotations

import re
import subprocess  # invoked with fixed argv; no shell, no user input
from datetime import UTC, datetime
from pathlib import Path

from wormjepa import WormJEPAError

RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z__[0-9a-f]{8}(?:-dirty)?__[a-z0-9_]+$")
"""Regex that every generated run-id must match.

Exported so callers can validate run-ids parsed from disk (e.g., a
``results/<run-id>/`` directory name).
"""


def _slugify(name: str) -> str:
    """Normalize ``name`` to lowercase, alphanumeric-and-underscore-only.

    Used to derive the config-slug from a config filename stem.
    """
    lowered = name.lower()
    # Replace any run of non-alphanumerics with a single underscore, then trim.
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if not slug:
        msg = f"Cannot derive a non-empty slug from {name!r}."
        raise WormJEPAError(msg)
    return slug


def _git_head_sha() -> str:
    """Return the first 8 chars of HEAD, lowercase hex.

    Raises:
        WormJEPAError: If not inside a git repository or git is unavailable.
    """
    try:
        result = subprocess.run(  # fixed argv list, no shell
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        msg = "git executable not found on PATH."
        raise WormJEPAError(msg) from exc
    except subprocess.CalledProcessError as exc:
        msg = (
            "Run-id generation requires being inside a git repository "
            "(git rev-parse HEAD failed). No fallback is provided per FR / Story 1.5."
        )
        raise WormJEPAError(msg) from exc
    sha = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]+", sha):
        msg = f"Unexpected git HEAD value: {sha!r}."
        raise WormJEPAError(msg)
    return sha[:8]


def _is_working_tree_dirty() -> bool:
    """Return True if the working tree has uncommitted changes to tracked files.

    Untracked files do not count as dirty; only modifications to tracked content
    affect reproducibility.
    """
    try:
        result = subprocess.run(  # fixed argv list, no shell
            ["git", "diff", "--quiet", "HEAD", "--"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        msg = "git executable not found on PATH."
        raise WormJEPAError(msg) from exc
    # `git diff --quiet` exits 0 if no changes, 1 if there are changes.
    return result.returncode != 0


def generate_run_id(config_path: Path) -> str:
    """Produce a deterministic, filesystem-safe run-id for ``config_path``.

    Args:
        config_path: Path to the YAML config file driving the run. The filename
            stem (without ``.yaml``) becomes the config-slug component.

    Returns:
        A string matching :data:`RUN_ID_PATTERN`.

    Raises:
        WormJEPAError: If not inside a git repository, if git is unavailable,
            or if the config filename produces an empty slug.
    """
    config_slug = _slugify(config_path.stem)
    short_sha = _git_head_sha()
    if _is_working_tree_dirty():
        short_sha = f"{short_sha}-dirty"
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}__{short_sha}__{config_slug}"
    if not RUN_ID_PATTERN.match(run_id):
        msg = f"Generated run-id failed format invariant: {run_id!r}."
        raise WormJEPAError(msg)
    return run_id
