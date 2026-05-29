"""Verify the working tree against ``pre-registration/MANIFEST.lock``.

Every reportable computation calls :func:`verify_manifest` before producing
output. SHA mismatches surface as :class:`wormjepa.PreRegistrationViolation`
with enough detail (artifact name + expected + actual + canonicalization) for
the operator to diagnose and either correct the working tree or add an
explicit ``### Frozen-artifact changes`` CHANGELOG entry.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path

from wormjepa import PreRegistrationViolation
from wormjepa.manifest.canonicalize import (
    canonicalize_dandi_federation,
    canonicalize_doi_string,
    canonicalize_github_commit_pin,
    canonicalize_zenodo_subset,
    sha256_of_canonicalized,
)
from wormjepa.manifest.lock import ArtifactEntry, Manifest, read_manifest
from wormjepa.paths import project_root

logger = logging.getLogger(__name__)

_DEFAULT_MANIFEST_PATH = Path("pre-registration") / "MANIFEST.lock"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of :func:`verify_manifest` on a clean working tree.

    Attributes:
        verified: Number of artifacts whose SHA matched the manifest.
        manifest_path: Path that was verified.
    """

    verified: int
    manifest_path: Path


def _load_spec(entry: ArtifactEntry) -> object:
    """Import the SPEC module for ``entry.dataset`` and return its SPEC constant.

    Raises :class:`PreRegistrationViolation` if the module or SPEC is missing.
    """
    assert entry.dataset is not None
    try:
        module = importlib.import_module(f"wormjepa.data.sources.{entry.dataset}")
    except ModuleNotFoundError as exc:
        msg = (
            f"Manifest references dataset {entry.dataset!r} but no module "
            f"wormjepa.data.sources.{entry.dataset} exists."
        )
        raise PreRegistrationViolation(msg) from exc
    spec = getattr(module, "SPEC", None)
    if spec is None:
        msg = f"wormjepa.data.sources.{entry.dataset} does not expose a SPEC constant."
        raise PreRegistrationViolation(msg)
    return spec


def _expected_sha_for_dataset(entry: ArtifactEntry) -> str:
    """Recompute the expected SHA for a dataset entry from the current SPEC.

    Dispatches on the entry's canonicalization method:

    - ``doi_manifest`` — hashes ``SPEC.doi``.
    - ``dandi_federation`` — hashes the sorted list ``SPEC.dandisets``.
    - ``zenodo_subset`` — hashes the sorted list ``SPEC.records``.
    - ``github_commit_pin`` — hashes ``SPEC.{repo, commit_sha, config_path,
      config_sha256}``.

    Any drift between the SPEC's pinned values and what the manifest recorded
    surfaces here as a SHA mismatch, which the caller converts into a
    :class:`PreRegistrationViolation`.
    """
    spec = _load_spec(entry)
    canon = entry.canonicalization

    if canon == "doi_manifest":
        if not hasattr(spec, "doi"):
            msg = (
                f"SPEC for dataset {entry.dataset!r} (canonicalization 'doi_manifest') "
                f"is missing required attribute 'doi'."
            )
            raise PreRegistrationViolation(msg)
        canonical = canonicalize_doi_string(str(spec.doi))  # type: ignore[attr-defined]
        return hashlib.sha256(canonical).hexdigest()

    if canon == "dandi_federation":
        if not hasattr(spec, "dandisets"):
            msg = (
                f"SPEC for dataset {entry.dataset!r} (canonicalization "
                f"'dandi_federation') is missing required attribute 'dandisets'."
            )
            raise PreRegistrationViolation(msg)
        dandisets = [
            {"dandiset_id": d.dandiset_id, "version": d.version, "doi": d.doi}
            for d in spec.dandisets  # type: ignore[attr-defined]
        ]
        canonical = canonicalize_dandi_federation(dandisets)
        return hashlib.sha256(canonical).hexdigest()

    if canon == "zenodo_subset":
        if not hasattr(spec, "records"):
            msg = (
                f"SPEC for dataset {entry.dataset!r} (canonicalization "
                f"'zenodo_subset') is missing required attribute 'records'."
            )
            raise PreRegistrationViolation(msg)
        records = [
            {
                "zenodo_record_id": r.zenodo_record_id,
                "doi": r.doi,
                "description": r.description,
            }
            for r in spec.records  # type: ignore[attr-defined]
        ]
        canonical = canonicalize_zenodo_subset(records)
        return hashlib.sha256(canonical).hexdigest()

    if canon == "github_commit_pin":
        required = ("repo", "commit_sha", "config_path", "config_sha256")
        missing = [name for name in required if not hasattr(spec, name)]
        if missing:
            msg = (
                f"SPEC for dataset {entry.dataset!r} (canonicalization "
                f"'github_commit_pin') is missing required attributes: {missing}."
            )
            raise PreRegistrationViolation(msg)
        canonical = canonicalize_github_commit_pin(
            repo=str(spec.repo),  # type: ignore[attr-defined]
            commit_sha=str(spec.commit_sha),  # type: ignore[attr-defined]
            config_path=str(spec.config_path),  # type: ignore[attr-defined]
            config_sha256=str(spec.config_sha256),  # type: ignore[attr-defined]
        )
        return hashlib.sha256(canonical).hexdigest()

    msg = (
        f"Dataset entry for {entry.dataset!r} uses canonicalization {canon!r}, "
        f"which is not a dataset canonicalization method."
    )
    raise PreRegistrationViolation(msg)


def _verify_entry(entry: ArtifactEntry, root: Path) -> None:
    """Verify one entry; raise :class:`PreRegistrationViolation` on mismatch."""
    if entry.path is not None:
        file_path = root / entry.path
        if not file_path.is_file():
            msg = (
                f"Frozen artifact missing from working tree: {entry.path}\n"
                f"  expected SHA-256: {entry.sha256}"
            )
            raise PreRegistrationViolation(msg)
        actual = sha256_of_canonicalized(file_path, entry.canonicalization)
        if actual != entry.sha256:
            msg = (
                f"Frozen artifact SHA mismatch for {entry.path}\n"
                f"  expected: {entry.sha256}\n"
                f"  actual:   {actual}\n"
                f"  canonicalization: {entry.canonicalization}\n"
                f"  fix: either revert the file or add a `### Frozen-artifact changes` "
                f"CHANGELOG entry and re-lock."
            )
            raise PreRegistrationViolation(msg)
    elif entry.dataset is not None:
        actual = _expected_sha_for_dataset(entry)
        if actual != entry.sha256:
            msg = (
                f"Dataset SPEC drift for {entry.dataset!r}\n"
                f"  expected: {entry.sha256}\n"
                f"  actual:   {actual}\n"
                f"  canonicalization: {entry.canonicalization}\n"
                f"  fix: either revert the SPEC for "
                f"wormjepa.data.sources.{entry.dataset} or add a "
                f"`### Frozen-artifact changes` CHANGELOG entry and re-lock."
            )
            raise PreRegistrationViolation(msg)
    else:  # pragma: no cover  # ArtifactEntry validator forbids this
        msg = "ArtifactEntry has neither path nor dataset; this should be unreachable."
        raise PreRegistrationViolation(msg)


def verify_manifest(
    manifest_path: Path | None = None,
    *,
    root: Path | None = None,
) -> VerificationResult:
    """Verify every artifact listed in ``manifest_path`` against the working tree.

    Args:
        manifest_path: Path to the manifest file. Defaults to
            ``<project_root>/pre-registration/MANIFEST.lock``.
        root: Project root for resolving relative artifact paths. Defaults to
            :func:`wormjepa.paths.project_root`.

    Returns:
        A :class:`VerificationResult` with the count of verified artifacts.

    Raises:
        PreRegistrationViolation: On any mismatch, missing artifact, or
            unreadable dataset SPEC.
    """
    root = root or project_root()
    manifest_path = manifest_path or (root / _DEFAULT_MANIFEST_PATH)

    manifest: Manifest = read_manifest(manifest_path)
    for entry in manifest.artifacts:
        _verify_entry(entry, root)
        logger.debug(
            "manifest entry verified",
            extra={
                "path": entry.path,
                "dataset": entry.dataset,
                "canonicalization": entry.canonicalization,
            },
        )
    return VerificationResult(verified=len(manifest.artifacts), manifest_path=manifest_path)
