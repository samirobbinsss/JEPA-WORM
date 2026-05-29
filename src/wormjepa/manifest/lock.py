"""``MANIFEST.lock`` schema and I/O.

The lockfile lives at ``pre-registration/MANIFEST.lock`` and is the structural
keystone of the pre-registration discipline: it pins a SHA-256 per frozen
artifact plus the canonicalization method used to compute it.

Two top-level entry shapes:

- **File artifacts** — ``path`` is the project-root-relative path; the
  canonicalization is one of ``yaml_sorted_keys_lf`` / ``python_ast_normalized``
  / ``text_lf``.
- **Dataset artifacts** — ``dataset`` is the source name (matches
  ``data/sources/<name>.SPEC.name``). The extra fields required depend on the
  canonicalization method:

  - ``doi_manifest`` — single-DOI corpus; requires ``doi``.
  - ``dandi_federation`` — multi-dandiset corpus; requires ``dandisets``.
  - ``zenodo_subset`` — pre-committed per-experiment Zenodo records;
    requires ``records``.
  - ``github_commit_pin`` — code-only generator repo; requires ``repo``,
    ``commit_sha``, ``config_path``, ``config_sha256``.

``extra='forbid'`` rejects unknown fields. Per-canonicalization validation is
enforced in :meth:`ArtifactEntry._validate_per_canonicalization`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wormjepa import ManifestLockError
from wormjepa.manifest.canonicalize import CanonicalizationMethod

MANIFEST_SCHEMA_VERSION = 2
"""Current schema version. Bumped from 1 to 2 when structured-data
canonicalizations (dandi_federation, zenodo_subset, github_commit_pin) were
introduced. ``read_manifest`` accepts both v1 and v2 payloads for backward
compatibility during transition; new writes default to v2.
"""


class DandisetPin(BaseModel):
    """One DANDI dandiset entry within a ``dandi_federation`` artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dandiset_id: str
    """Six-digit DANDI dandiset identifier (e.g. ``"000715"``)."""

    version: str
    """DANDI version identifier (e.g. ``"0.241009.1514"``)."""

    doi: str
    """Full DOI for this dandiset version (e.g.
    ``"10.48324/dandi.000715/0.241009.1514"``)."""


class ZenodoRecord(BaseModel):
    """One Zenodo record entry within a ``zenodo_subset`` artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    zenodo_record_id: str
    """Numeric Zenodo record id as a string (e.g. ``"1031550"``)."""

    doi: str
    """Full Zenodo DOI (e.g. ``"10.5281/zenodo.1031550"``)."""

    description: str = ""
    """Optional one-line description for human auditors."""


class ArtifactEntry(BaseModel):
    """One row in the manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str | None = None
    """Project-root-relative path to a file artifact."""

    dataset: str | None = None
    """Dataset name (mutually exclusive with ``path``)."""

    # doi_manifest fields:
    doi: str | None = None
    """DOI string for a single-DOI dataset artifact."""

    # dandi_federation fields:
    dandisets: list[DandisetPin] | None = None
    """Ordered list of dandiset pins, for ``dandi_federation`` artifacts."""

    # zenodo_subset fields:
    records: list[ZenodoRecord] | None = None
    """Pre-committed Zenodo records, for ``zenodo_subset`` artifacts."""

    # github_commit_pin fields:
    repo: str | None = None
    """``host/owner/repo``-style identifier (e.g. ``"github.com/Jessie940611/BAAIWorm"``)."""

    commit_sha: str | None = None
    """Full 40-character commit SHA at pin time."""

    config_path: str | None = None
    """Path inside the repo to the generator configuration file."""

    config_sha256: str | None = None
    """SHA-256 of the canonicalized config bytes at pin time."""

    sha256: str
    """Hex SHA-256 of the canonical bytes."""

    canonicalization: CanonicalizationMethod
    """Method used to canonicalize the artifact prior to hashing."""

    description: str = ""
    """Optional free-form annotation."""

    @model_validator(mode="after")
    def _validate_per_canonicalization(self) -> ArtifactEntry:
        # Top-level mutex: path and dataset cannot both be set.
        if self.path is not None and self.dataset is not None:
            msg = "ArtifactEntry cannot specify both 'path' and 'dataset'."
            raise ManifestLockError(msg)

        if self.canonicalization in {
            "yaml_sorted_keys_lf",
            "python_ast_normalized",
            "text_lf",
        }:
            if self.path is None:
                msg = (
                    f"canonicalization {self.canonicalization!r} requires 'path' "
                    f"(file artifact). Got dataset={self.dataset!r}."
                )
                raise ManifestLockError(msg)
        elif self.canonicalization == "doi_manifest":
            if self.dataset is None or self.doi is None:
                msg = (
                    "canonicalization 'doi_manifest' requires 'dataset' and 'doi' "
                    "(single-DOI dataset artifact)."
                )
                raise ManifestLockError(msg)
        elif self.canonicalization == "dandi_federation":
            if self.dataset is None:
                msg = "canonicalization 'dandi_federation' requires 'dataset'."
                raise ManifestLockError(msg)
            if not self.dandisets:
                msg = "canonicalization 'dandi_federation' requires a non-empty 'dandisets' list."
                raise ManifestLockError(msg)
        elif self.canonicalization == "zenodo_subset":
            if self.dataset is None:
                msg = "canonicalization 'zenodo_subset' requires 'dataset'."
                raise ManifestLockError(msg)
            if not self.records:
                msg = "canonicalization 'zenodo_subset' requires a non-empty 'records' list."
                raise ManifestLockError(msg)
        elif self.canonicalization == "github_commit_pin":
            if self.dataset is None:
                msg = "canonicalization 'github_commit_pin' requires 'dataset'."
                raise ManifestLockError(msg)
            missing = [
                name
                for name, value in (
                    ("repo", self.repo),
                    ("commit_sha", self.commit_sha),
                    ("config_path", self.config_path),
                    ("config_sha256", self.config_sha256),
                )
                if not value
            ]
            if missing:
                msg = f"canonicalization 'github_commit_pin' requires non-empty {missing}."
                raise ManifestLockError(msg)
        return self


class Manifest(BaseModel):
    """The full ``MANIFEST.lock`` payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1, 2] = MANIFEST_SCHEMA_VERSION
    locked_at: datetime
    locked_by: str
    git_sha_at_lock: str
    artifacts: list[ArtifactEntry] = Field(default_factory=list)

    def find_file_entry(self, path: str) -> ArtifactEntry | None:
        """Return the entry for ``path``, or ``None``."""
        for a in self.artifacts:
            if a.path == path:
                return a
        return None

    def find_dataset_entry(self, dataset: str) -> ArtifactEntry | None:
        """Return the entry for ``dataset``, or ``None``."""
        for a in self.artifacts:
            if a.dataset == dataset:
                return a
        return None


def _sort_artifacts(artifacts: list[ArtifactEntry]) -> list[ArtifactEntry]:
    """Sort artifacts canonically: file paths first (sorted), then datasets (sorted)."""
    files = sorted((a for a in artifacts if a.path is not None), key=lambda a: a.path or "")
    datasets = sorted(
        (a for a in artifacts if a.dataset is not None), key=lambda a: a.dataset or ""
    )
    return files + datasets


def read_manifest(path: Path) -> Manifest:
    """Read and validate a ``MANIFEST.lock`` file.

    Raises:
        ManifestLockError: If the file is missing, malformed, or fails schema
            validation.
    """
    if not path.is_file():
        msg = f"Manifest file does not exist: {path}"
        raise ManifestLockError(msg)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Manifest YAML parse error in {path}: {exc}"
        raise ManifestLockError(msg) from exc
    if not isinstance(data, dict):
        msg = f"Manifest at {path} did not yield a mapping at the top level."
        raise ManifestLockError(msg)
    try:
        return Manifest.model_validate(data)
    except Exception as exc:
        msg = f"Manifest schema validation failed for {path}: {exc}"
        raise ManifestLockError(msg) from exc


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Write ``manifest`` to ``path`` in canonical YAML.

    Canonical form: sorted artifact list (files first, then datasets), sorted
    keys per row, LF newlines. Round-tripping (``read_manifest(write_manifest(m))``)
    is bit-stable for a given ``Manifest`` instance.
    """
    sorted_artifacts = _sort_artifacts(list(manifest.artifacts))
    payload = manifest.model_dump(mode="json")
    payload["artifacts"] = [a.model_dump(mode="json", exclude_none=True) for a in sorted_artifacts]
    # Sort top-level keys too for stable output.
    text = yaml.safe_dump(
        payload,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        line_break="\n",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
