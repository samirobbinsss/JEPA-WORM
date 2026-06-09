"""``wormjepa fetch`` — first-class dataset / checkpoint fetch surface.

Four subcommands, each a thin CLI wrapper around an existing module-level
fetch helper. Replaces the ad-hoc ``scripts/dev/fetch_*.py`` shims with a
discoverable CLI surface while leaving the shims in place for backward
compatibility (their removal is its own follow-up).

  - ``wormjepa fetch zenodo-anchors``: pulls the 3 WBDB + OWMD anchor
    records used by every Phase 0 headline sweep. Delegates to
    :func:`wormjepa.data.download.download_zenodo_record`.
  - ``wormjepa fetch zenodo-subset``: pulls the full WBDB + OWMD
    pre-committed 100-record subsets (~45 GB) by iterating each SPEC's
    ``records``. Idempotent / resumable; per-record failures are isolated.
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

import importlib
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
# OWMD carries a single anchor (the N2 record 1031550): its former mutant
# anchor 1033265 was dropped in Story 9.6 (video-less off-food, silently
# skipped by the loader). OWMD mutant + full-strain coverage now comes from
# `fetch zenodo-subset`, not this minimal anchor set.
_ZENODO_ANCHOR_JOBS: tuple[tuple[str, str], ...] = (
    ("1031550", "data/downloads/wormbehavior_db/1031550"),
    ("1029149", "data/downloads/wormbehavior_db/1029149"),
    ("1031550", "data/downloads/openworm_movement/1031550"),
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
        "Fetch the 3 WBDB + OWMD Zenodo anchor records into "
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


# --dataset choice -> SPEC module exposing a ``records`` list of ZenodoRecordPin.
_ZENODO_SUBSET_SPECS: dict[str, str] = {
    "wormbehavior_db": "wormjepa.data.sources.wormbehavior_db",
    "openworm_movement": "wormjepa.data.sources.openworm_movement",
}


@fetch_app.command(
    "zenodo-subset",
    help=(
        "Fetch the full pre-committed Zenodo subset — every record in the "
        "wormbehavior_db / openworm_movement SPEC (~45 GB for both) — into "
        "data/downloads/<dataset>/<record_id>/. Idempotent: records already "
        "on disk at the expected size are skipped, so a re-run resumes. "
        "Per-record failures are logged and the run continues; a non-zero "
        "exit then reports how many failed."
    ),
)
def fetch_zenodo_subset(
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help=(
                "Which subset to fetch: 'wormbehavior_db', "
                "'openworm_movement', or 'both' (default)."
            ),
        ),
    ] = "both",
    max_records: Annotated[
        int,
        typer.Option(
            "--max-records",
            help=(
                "Cap records fetched per dataset (0 = all). Use a small "
                "value for a cheap connectivity check before the full pull."
            ),
        ),
    ] = 0,
) -> None:
    """Fetch every record in the WBDB / OWMD Zenodo subset SPECs."""
    if dataset == "both":
        targets = list(_ZENODO_SUBSET_SPECS)
    elif dataset in _ZENODO_SUBSET_SPECS:
        targets = [dataset]
    else:
        msg = (
            f"unknown --dataset {dataset!r}; choose 'wormbehavior_db', "
            f"'openworm_movement', or 'both'"
        )
        raise WormJEPAError(msg)

    repo_root = project_root()
    failures: list[tuple[str, str, str]] = []  # (dataset, record_id, error)
    for ds_name in targets:
        spec = importlib.import_module(_ZENODO_SUBSET_SPECS[ds_name]).SPEC
        records = list(spec.records)
        if max_records > 0:
            records = records[:max_records]
        typer.echo(f"=== {ds_name}: {len(records)} records ===")
        for i, pin in enumerate(records, start=1):
            rid = pin.zenodo_record_id
            dest = repo_root / "data" / "downloads" / ds_name / rid
            typer.echo(f"[{ds_name} {i}/{len(records)}] zenodo:{rid} -> {dest}")
            try:
                paths = download_zenodo_record(rid, dest)
            except WormJEPAError as exc:
                logger.error("record %s (%s) failed: %s", rid, ds_name, exc)
                failures.append((ds_name, rid, str(exc)))
                continue
            typer.echo(f"  {rid}: {len(paths)} files")

    if failures:
        typer.echo(f"\n{len(failures)} record(s) FAILED:")
        for ds_name, rid, err in failures:
            typer.echo(f"  {ds_name}/{rid}: {err}")
        msg = f"zenodo-subset fetch: {len(failures)} record(s) failed"
        raise WormJEPAError(msg)
    typer.echo("\nall records present.")


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
