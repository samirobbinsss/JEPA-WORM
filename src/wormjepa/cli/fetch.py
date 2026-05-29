"""``wormjepa fetch`` — first-class dataset / checkpoint fetch surface.

Three subcommands, each a thin CLI wrapper around an existing module-level
fetch helper. Replaces the ad-hoc ``scripts/dev/fetch_*.py`` shims with a
discoverable CLI surface while leaving the shims in place for backward
compatibility (their removal is its own follow-up).

  - ``wormjepa fetch zenodo-anchors``: pulls the 4 WBDB + OWMD anchor
    records used by every Phase 0 headline sweep. Delegates to
    :func:`wormjepa.data.download.download_zenodo_record`.
  - ``wormjepa fetch wormid-cohort <dandiset>``: pulls one WormID
    dandiset at the version pinned in ``wormjepa.data.sources.wormid``.
    Delegates to ``scripts/dev/fetch_wormid_cohort.py::main`` so the
    SF-deferral refusal logic, asset enumeration, and resume behaviour
    stay in a single place. ``--override-sf-deferral`` mirrors the
    script's flag.
  - ``wormjepa fetch vjepa-checkpoint <variant>``: pulls a V-JEPA 2.1
    weights file into the user-level checkpoint cache. Delegates to
    :func:`wormjepa.models.vjepa_loader.download_vjepa_checkpoint`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, get_args

import typer

from wormjepa import WormJEPAError
from wormjepa.configs.jepa_config import VJEPAVariant
from wormjepa.data.download import download_zenodo_record
from wormjepa.models.vjepa_loader import download_vjepa_checkpoint
from wormjepa.paths import project_root

logger = logging.getLogger(__name__)


# (record_id, dest_dir relative to repo root) — same set the shim fetches.
_ZENODO_ANCHOR_JOBS: tuple[tuple[str, str], ...] = (
    ("1031550", "data/downloads/wormbehavior_db/1031550"),
    ("1029149", "data/downloads/wormbehavior_db/1029149"),
    ("1031550", "data/downloads/openworm_movement/1031550"),
    ("1033265", "data/downloads/openworm_movement/1033265"),
)


fetch_app = typer.Typer(
    name="fetch",
    help=(
        "Fetch DOI-pinned datasets and V-JEPA 2.1 checkpoints. Each "
        "subcommand is idempotent — re-running skips files already on "
        "disk at the expected size."
    ),
    no_args_is_help=True,
)


@fetch_app.command(
    "zenodo-anchors",
    help=(
        "Fetch the 4 WBDB + OWMD Zenodo anchor records into "
        "data/downloads/. Idempotent: files already present at the "
        "expected size are skipped."
    ),
)
def fetch_zenodo_anchors() -> None:
    """Fetch every Zenodo anchor record used by Phase 0 baseline / smoke runs."""
    repo_root = project_root()
    for record_id, rel_dest in _ZENODO_ANCHOR_JOBS:
        dest = repo_root / rel_dest
        typer.echo(f"fetching zenodo:{record_id} -> {dest}")
        paths = download_zenodo_record(record_id, dest)
        typer.echo(f"  {record_id}: {len(paths)} files at {dest}")


@fetch_app.command(
    "wormid-cohort",
    help=(
        "Fetch one WormID DANDI dandiset at the version pinned in "
        "wormjepa.data.sources.wormid into data/downloads/wormid/. "
        "Refuses 000776 (SF) by default per the 2026-05-18 "
        "materialization deferral; pass --override-sf-deferral to bypass."
    ),
)
def fetch_wormid_cohort(
    dandiset_id: Annotated[
        str,
        typer.Argument(
            help="7-digit DANDI dandiset id (e.g. 000714).",
        ),
    ],
    override_sf_deferral: Annotated[
        bool,
        typer.Option(
            "--override-sf-deferral",
            help=(
                "Force fetch of 000776 (SF). Deferred to Phase 0 Growth "
                "per the 2026-05-18 materialization deferral; pass this "
                "flag only after the deferral is reversed in pre-reg."
            ),
        ),
    ] = False,
    root: Annotated[
        Path | None,
        typer.Option(
            "--root",
            help=(
                "Destination root. Defaults to data/downloads/wormid/ relative to the repo root."
            ),
        ),
    ] = None,
) -> None:
    """Fetch one WormID dandiset by id.

    Delegates to ``scripts/dev/fetch_wormid_cohort.py::main`` so the
    canonical SF-deferral refusal, asset enumeration, and resume logic
    stay in a single source of truth.
    """
    # Imported lazily so that the dev-scripts package doesn't load at
    # CLI import time (it is not part of the installed wheel).
    repo_root = project_root()
    scripts_dev = repo_root / "scripts" / "dev"
    scripts_dev_str = str(scripts_dev)
    if scripts_dev_str not in sys.path:
        sys.path.insert(0, scripts_dev_str)
    try:
        from fetch_wormid_cohort import main as _fetch_main  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            f"Could not import scripts/dev/fetch_wormid_cohort.py from {scripts_dev}: "
            f"{exc}. This CLI command requires the dev-scripts shim to be present."
        )
        raise WormJEPAError(msg) from exc

    argv = [dandiset_id]
    if override_sf_deferral:
        argv.append("--override-sf-deferral")
    if root is not None:
        argv.extend(["--root", str(root)])

    saved_argv = sys.argv
    sys.argv = ["fetch_wormid_cohort.py", *argv]
    try:
        rc = _fetch_main()
    finally:
        sys.argv = saved_argv

    if rc != 0:
        msg = f"wormid-cohort fetch failed for {dandiset_id!r}: exit code {rc}"
        raise WormJEPAError(msg)


@fetch_app.command(
    "vjepa-checkpoint",
    help=(
        "Fetch a V-JEPA 2.1 public checkpoint into the user-level cache "
        "(~/.cache/wormjepa/checkpoints/). Idempotent: cached files are "
        "reused; SHA is logged for later pinning."
    ),
)
def fetch_vjepa_checkpoint(
    variant: Annotated[
        str,
        typer.Argument(
            help=(
                "V-JEPA 2.1 variant. One of: "
                "vjepa2_1_vit_base_384, vjepa2_1_vit_large_384, "
                "vjepa2_1_vit_giant_384, vjepa2_1_vit_gigantic_384."
            ),
        ),
    ],
    expected_sha256: Annotated[
        str | None,
        typer.Option(
            "--sha256",
            help=(
                "Optional pinned SHA-256. When omitted, the observed SHA "
                "is logged so it can be pinned later in configs/headline.yaml."
            ),
        ),
    ] = None,
) -> None:
    """Fetch a single V-JEPA 2.1 checkpoint variant."""
    allowed = set(get_args(VJEPAVariant))
    if variant not in allowed:
        msg = f"Unknown V-JEPA 2.1 variant {variant!r}. Choose one of: {sorted(allowed)}"
        raise WormJEPAError(msg)
    # The pyright Literal narrowing requires us to cast here; the runtime
    # check above guarantees membership.
    path = download_vjepa_checkpoint(variant, expected_sha256)  # type: ignore[arg-type]
    typer.echo(f"vjepa checkpoint: {path}")
