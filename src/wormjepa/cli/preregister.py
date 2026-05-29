"""``wormjepa preregister`` — lock or verify the pre-registration manifest (Story 4.4).

Two modes:

- ``wormjepa preregister``              → compute SHAs for every artifact in
  ``pre-registration/manifest_template.yaml`` (the enumerated artifact list,
  no hashes) and write ``pre-registration/MANIFEST.lock``. Refuses to overwrite
  an existing lock unless ``--force`` is passed (Story 4.5 adds the
  CHANGELOG-frozen-changes pre-commit enforcer that requires a logged reason).
- ``wormjepa preregister --verify``     → call :func:`verify_manifest` and print
  a rich-formatted summary table.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast, get_args

import typer
import yaml
from rich.console import Console
from rich.table import Table

from wormjepa import ManifestLockError, PreRegistrationViolation, WormJEPAError
from wormjepa.manifest.canonicalize import (
    CanonicalizationMethod,
    canonicalize_dandi_federation,
    canonicalize_doi_string,
    canonicalize_github_commit_pin,
    canonicalize_zenodo_subset,
    sha256_of_canonicalized,
)
from wormjepa.manifest.lock import (
    ArtifactEntry,
    DandisetPin,
    Manifest,
    ZenodoRecord,
    read_manifest,
    write_manifest,
)
from wormjepa.manifest.lock_check import verify_manifest
from wormjepa.paths import project_root

logger = logging.getLogger(__name__)
_console = Console()

_MANIFEST_PATH = Path("pre-registration") / "MANIFEST.lock"
_TEMPLATE_PATH = Path("pre-registration") / "manifest_template.yaml"


def _git_user_name() -> str:
    try:
        result = subprocess.run(
            ["git", "config", "user.name"], check=False, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        msg = "git executable not found on PATH."
        raise WormJEPAError(msg) from exc
    name = result.stdout.strip()
    return name or "unknown"


def _git_head_sha_full() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        msg = "wormjepa preregister requires being inside a git repository."
        raise WormJEPAError(msg) from exc
    return result.stdout.strip()


def _build_manifest_from_template(root: Path) -> Manifest:
    """Read ``pre-registration/manifest_template.yaml`` and compute SHAs."""
    template_path = root / _TEMPLATE_PATH
    if not template_path.is_file():
        msg = (
            f"Manifest template missing at {template_path}. Create one listing the "
            f"frozen artifacts (files + datasets) and re-run."
        )
        raise ManifestLockError(msg)
    template = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    raw_artifacts: list[dict[str, str]] = template.get("artifacts", [])
    if not raw_artifacts:
        msg = "manifest_template.yaml has no `artifacts:` entries."
        raise ManifestLockError(msg)

    valid_methods = set(get_args(CanonicalizationMethod))
    entries: list[ArtifactEntry] = []
    for raw in raw_artifacts:
        canon_raw = raw["canonicalization"]
        if canon_raw not in valid_methods:
            msg = (
                f"Unknown canonicalization method {canon_raw!r}; "
                f"valid options: {sorted(valid_methods)}."
            )
            raise ManifestLockError(msg)
        canonicalization = cast(CanonicalizationMethod, canon_raw)
        if "path" in raw:
            path_val = raw["path"]
            file_path = root / path_val
            sha = sha256_of_canonicalized(file_path, canonicalization)
            entries.append(
                ArtifactEntry(
                    path=path_val,
                    sha256=sha,
                    canonicalization=canonicalization,
                    description=raw.get("description", ""),
                )
            )
        elif "dataset" in raw:
            dataset = raw["dataset"]
            entries.append(_build_dataset_entry(dataset, canonicalization, raw))
        else:
            msg = f"Template entry must have `path` or `dataset`: {raw!r}"
            raise ManifestLockError(msg)

    return Manifest(
        locked_at=datetime.now(tz=UTC),
        locked_by=_git_user_name(),
        git_sha_at_lock=_git_head_sha_full(),
        artifacts=entries,
    )


def _load_spec(dataset: str) -> object:
    """Import ``wormjepa.data.sources.<dataset>`` and return its SPEC constant."""
    try:
        module = importlib.import_module(f"wormjepa.data.sources.{dataset}")
    except ModuleNotFoundError as exc:
        msg = (
            f"Manifest template references dataset {dataset!r}, but no "
            f"wormjepa.data.sources.{dataset} module exists."
        )
        raise ManifestLockError(msg) from exc
    spec = getattr(module, "SPEC", None)
    if spec is None:
        msg = f"wormjepa.data.sources.{dataset} does not export SPEC."
        raise ManifestLockError(msg)
    return spec


def _build_dataset_entry(
    dataset: str,
    canonicalization: CanonicalizationMethod,
    raw: dict[str, str],
) -> ArtifactEntry:
    """Construct an :class:`ArtifactEntry` for a dataset by canonicalization.

    Schema v1 (single-DOI) reads ``raw["doi"]`` from the template directly.
    Schema v2 canonicalizations (federation, subset, github pin) read the
    substance from the SPEC module rather than the template — the template
    only enumerates membership, the SPEC carries the values.
    """
    description = raw.get("description", "")

    if canonicalization == "doi_manifest":
        doi = raw.get("doi", "")
        sha = hashlib.sha256(canonicalize_doi_string(doi)).hexdigest()
        return ArtifactEntry(
            dataset=dataset,
            doi=doi,
            sha256=sha,
            canonicalization=canonicalization,
            description=description,
        )

    spec = _load_spec(dataset)

    if canonicalization == "dandi_federation":
        if not hasattr(spec, "dandisets"):
            msg = (
                f"SPEC for {dataset!r} (canonicalization 'dandi_federation') "
                f"is missing required attribute 'dandisets'."
            )
            raise ManifestLockError(msg)
        dandisets_raw = [
            {"dandiset_id": d.dandiset_id, "version": d.version, "doi": d.doi}
            for d in spec.dandisets  # type: ignore[attr-defined]
        ]
        sha = hashlib.sha256(canonicalize_dandi_federation(dandisets_raw)).hexdigest()
        return ArtifactEntry(
            dataset=dataset,
            dandisets=[
                DandisetPin(dandiset_id=d["dandiset_id"], version=d["version"], doi=d["doi"])
                for d in dandisets_raw
            ],
            sha256=sha,
            canonicalization=canonicalization,
            description=description,
        )

    if canonicalization == "zenodo_subset":
        if not hasattr(spec, "records"):
            msg = (
                f"SPEC for {dataset!r} (canonicalization 'zenodo_subset') "
                f"is missing required attribute 'records'."
            )
            raise ManifestLockError(msg)
        records_raw = [
            {
                "zenodo_record_id": r.zenodo_record_id,
                "doi": r.doi,
                "description": r.description,
            }
            for r in spec.records  # type: ignore[attr-defined]
        ]
        sha = hashlib.sha256(canonicalize_zenodo_subset(records_raw)).hexdigest()
        return ArtifactEntry(
            dataset=dataset,
            records=[
                ZenodoRecord(
                    zenodo_record_id=r["zenodo_record_id"],
                    doi=r["doi"],
                    description=r.get("description", ""),
                )
                for r in records_raw
            ],
            sha256=sha,
            canonicalization=canonicalization,
            description=description,
        )

    if canonicalization == "github_commit_pin":
        required = ("repo", "commit_sha", "config_path", "config_sha256")
        missing = [name for name in required if not hasattr(spec, name)]
        if missing:
            msg = (
                f"SPEC for {dataset!r} (canonicalization 'github_commit_pin') "
                f"is missing required attributes: {missing}."
            )
            raise ManifestLockError(msg)
        sha = hashlib.sha256(
            canonicalize_github_commit_pin(
                repo=str(spec.repo),  # type: ignore[attr-defined]
                commit_sha=str(spec.commit_sha),  # type: ignore[attr-defined]
                config_path=str(spec.config_path),  # type: ignore[attr-defined]
                config_sha256=str(spec.config_sha256),  # type: ignore[attr-defined]
            )
        ).hexdigest()
        return ArtifactEntry(
            dataset=dataset,
            repo=str(spec.repo),  # type: ignore[attr-defined]
            commit_sha=str(spec.commit_sha),  # type: ignore[attr-defined]
            config_path=str(spec.config_path),  # type: ignore[attr-defined]
            config_sha256=str(spec.config_sha256),  # type: ignore[attr-defined]
            sha256=sha,
            canonicalization=canonicalization,
            description=description,
        )

    msg = (
        f"Dataset entry for {dataset!r} uses canonicalization "
        f"{canonicalization!r}, which is not a dataset canonicalization method."
    )
    raise ManifestLockError(msg)


_LIT_WATCH_PATH = Path("LIT_WATCH.md")
_LIT_WATCH_MAX_AGE_DAYS = 35


def _print_verification_summary(verified: int, manifest_path: Path) -> None:
    table = Table(title=f"pre-registration verified: {verified} artifacts")
    table.add_column("manifest", style="green")
    table.add_row(str(manifest_path))
    _console.print(table)


def _check_lit_watch_cadence(root: Path) -> str | None:
    """Warn if the most-recent LIT_WATCH.md entry is more than 35 days old (Story 7.11).

    Returns a warning message, or ``None`` if cadence is satisfied.
    """
    import re

    lit_watch = root / _LIT_WATCH_PATH
    if not lit_watch.is_file():
        return None
    text = lit_watch.read_text(encoding="utf-8")
    # Match the documented entry header: "### YYYY-MM-DD" or "### YYYY-MM-DD (...)"
    dates = re.findall(r"^###\s+(\d{4}-\d{2}-\d{2})", text, flags=re.MULTILINE)
    if not dates:
        return None
    latest = max(dates)
    try:
        latest_dt = datetime.fromisoformat(latest).replace(tzinfo=UTC)
    except ValueError:
        return None
    age_days = (datetime.now(tz=UTC) - latest_dt).days
    if age_days > _LIT_WATCH_MAX_AGE_DAYS:
        return (
            f"LIT_WATCH.md most-recent entry is {age_days} days old "
            f"(> {_LIT_WATCH_MAX_AGE_DAYS}). Add a new monthly entry per "
            f"NFR15."
        )
    return None


def _artifact_key(entry: ArtifactEntry) -> str:
    """Identifier used to align entries across two manifests.

    File artifacts are keyed by ``path``; dataset artifacts are keyed by
    ``dataset:<name>`` so the two namespaces cannot collide.
    """
    if entry.path is not None:
        return entry.path
    return f"dataset:{entry.dataset}"


def _load_lock(path: Path) -> dict[str, str]:
    """Parse a ``MANIFEST.lock`` and return ``{artifact_key: sha256}``.

    Re-uses :func:`wormjepa.manifest.lock.read_manifest` so schema validation
    and error reporting stay consistent with the rest of the pipeline.
    """
    manifest = read_manifest(path)
    return {_artifact_key(a): a.sha256 for a in manifest.artifacts}


def _short_sha(sha: str) -> str:
    """First 12 characters of a hex SHA-256."""
    return sha[:12]


def _run_diff(
    current_path: Path,
    prior_path: Path,
    *,
    show_unchanged: bool,
) -> int:
    """Print the diff and return the intended process exit code (0 or 1)."""
    prior = _load_lock(prior_path)
    current = _load_lock(current_path)

    keys = sorted(set(prior) | set(current))
    added: list[str] = []
    removed: list[str] = []
    changed: list[tuple[str, str, str]] = []
    unchanged: list[str] = []
    for key in keys:
        in_prior = key in prior
        in_current = key in current
        if in_current and not in_prior:
            added.append(key)
        elif in_prior and not in_current:
            removed.append(key)
        elif prior[key] != current[key]:
            changed.append((key, prior[key], current[key]))
        else:
            unchanged.append(key)

    for key in added:
        _console.print(f"+ {key}  (new, sha={_short_sha(current[key])})")
    for key in removed:
        _console.print(f"- {key}  (sha={_short_sha(prior[key])})")
    for key, was, now in changed:
        _console.print(f"~ {key}  (was={_short_sha(was)}, now={_short_sha(now)})")
    if show_unchanged:
        for key in unchanged:
            _console.print(f"  {key}  (sha={_short_sha(current[key])})")

    _console.print(
        f"summary: {len(added)} added, {len(removed)} removed, "
        f"{len(changed)} changed, {len(unchanged)} unchanged"
    )

    return 0 if not (added or removed or changed) else 1


def preregister_command(
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Verify MANIFEST.lock against the working tree."),
    ] = False,
    diff: Annotated[
        Path | None,
        typer.Option(
            "--diff",
            help="Diff the current MANIFEST.lock against a prior lock file at "
            "this path. Exit 0 if identical, 1 if any artifacts differ.",
        ),
    ] = None,
    show_unchanged: Annotated[
        bool,
        typer.Option(
            "--show-unchanged",
            help="With --diff, include unchanged artifacts in the output.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing MANIFEST.lock. CHANGELOG-frozen pre-commit "
            "hook (Story 4.5) still requires a logged justification.",
        ),
    ] = False,
) -> None:
    """Lock or verify the pre-registration manifest."""
    root = project_root()
    manifest_path = root / _MANIFEST_PATH

    if diff is not None:
        exit_code = _run_diff(manifest_path, diff, show_unchanged=show_unchanged)
        raise typer.Exit(code=exit_code)

    if verify:
        try:
            result = verify_manifest(manifest_path)
        except PreRegistrationViolation:
            raise
        _print_verification_summary(result.verified, manifest_path)
        cadence_warning = _check_lit_watch_cadence(root)
        if cadence_warning is not None:
            _console.print(f"[bold yellow]warning[/bold yellow]: {cadence_warning}")
        return

    if manifest_path.exists() and not force:
        msg = (
            f"MANIFEST.lock already exists at {manifest_path}. "
            f"Pass --force to overwrite (a CHANGELOG `### Frozen-artifact changes` "
            f"entry is still required at commit time per Story 4.5)."
        )
        raise PreRegistrationViolation(msg)

    manifest = _build_manifest_from_template(root)
    write_manifest(manifest, manifest_path)
    logger.info(
        "manifest locked",
        extra={
            "n_artifacts": len(manifest.artifacts),
            "manifest_path": str(manifest_path),
        },
    )
    _console.print(
        f"[bold green]locked[/bold green]: {len(manifest.artifacts)} artifacts → {manifest_path}"
    )
