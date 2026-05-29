"""SPEC types for every public dataset JEPA-WORM consumes.

A SPEC module under ``src/wormjepa/data/sources/<name>.py`` exposes a
module-level ``SPEC`` constant of one of the types below. Which type is used
depends on the canonicalization method used to lock the dataset in
``pre-registration/MANIFEST.lock``:

- :class:`DatasetSource` — for single-DOI corpora (``doi_manifest``).
- :class:`DandiFederationSource` — for multi-dandiset DANDI federations
  (``dandi_federation``).
- :class:`ZenodoSubsetSource` — for pre-committed subsets of per-experiment
  Zenodo archives (``zenodo_subset``).
- :class:`GithubGeneratorSource` — for code-only generator repos pinned by
  commit SHA + config hash (``github_commit_pin``).

:data:`AnyDatasetSource` is the union used in places that accept any SPEC
shape (e.g. ``data/SOURCES.md`` rendering, manifest verification).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Single-DOI dataset SPEC. Used with ``canonicalization: doi_manifest``.

    Attributes:
        name: Stable dataset key (matches ``data/sources/<name>.py`` filename
            without ``.py``). Used as ``SourceDataset`` value downstream.
        url: HTTPS download URL. The download infra appends Range headers and
            retries 429/503 with exponential backoff.
        dest_filename: Filename the payload lands as inside ``data/downloads/``.
            Kept stable across versions so resumable downloads find the right
            partial file.
        sha256: Hex SHA-256 of the full payload. Verified after download.
            Mismatch raises :class:`wormjepa.DatasetIntegrityError`.
        doi: Canonical DOI. Recorded in MANIFEST.lock under ``doi_manifest``
            canonicalization (Epic 4).
        license: SPDX-style or human-readable license identifier.
        citation: Citation key matching an entry in ``CITATIONS.bib``.
        redistribution_restrictions: One-line note on any restrictions. The
            project never redistributes payloads; this documents *why* for
            future reviewers.
    """

    name: str
    url: str
    dest_filename: str
    sha256: str
    doi: str
    license: str
    citation: str
    redistribution_restrictions: str = ""


@dataclass(frozen=True, slots=True)
class DandisetPin:
    """One dandiset entry in a federation."""

    dandiset_id: str
    version: str
    doi: str


@dataclass(frozen=True, slots=True)
class DandiFederationSource:
    """Federated DANDI corpus SPEC. Used with ``canonicalization: dandi_federation``.

    Attributes:
        name: Stable dataset key (e.g. ``"wormid"``).
        dandisets: Ordered list of dandiset pins. Each entry pins one DANDI
            dandiset by ``(dandiset_id, version, doi)``. The canonical hash is
            computed by :func:`wormjepa.manifest.canonicalize.canonicalize_dandi_federation`
            over the sorted-by-id list.
        license: SPDX-style or human-readable license identifier (federation
            license; per-dandiset licensing is the dandiset's own concern).
        citation: Citation key matching an entry in ``CITATIONS.bib``.
        redistribution_restrictions: One-line note on any restrictions.
    """

    name: str
    dandisets: list[DandisetPin] = field(default_factory=list)
    license: str = ""
    citation: str = ""
    redistribution_restrictions: str = ""


@dataclass(frozen=True, slots=True)
class ZenodoRecordPin:
    """One Zenodo record in a pre-committed subset.

    Attributes:
        zenodo_record_id: Numeric Zenodo record identifier.
        doi: Canonical DOI for the record.
        description: Optional human-readable label (e.g., strain + date).
        sha256: Optional hex SHA-256 of the primary extracted HDF5 file under
            the record's local subdirectory. When non-empty, the loader
            verifies file bytes against this SHA on first iteration (FR7).
            Defaults to empty for backward compatibility — populate when the
            canonical bytes-on-disk SHAs are known. Excluded from the
            ``zenodo_subset`` canonicalization output, so adding values to
            existing records does NOT shift the dataset-level manifest hash.
    """

    zenodo_record_id: str
    doi: str
    description: str = ""
    sha256: str = ""


@dataclass(frozen=True, slots=True)
class ZenodoSubsetSource:
    """Per-experiment Zenodo subset SPEC. Used with ``canonicalization: zenodo_subset``.

    Attributes:
        name: Stable dataset key (e.g. ``"wormbehavior_db"``).
        records: Ordered list of pre-committed Zenodo records. The canonical
            hash is computed by
            :func:`wormjepa.manifest.canonicalize.canonicalize_zenodo_subset`
            over the sorted-by-id list. Adding or removing a record is a
            substantive frozen-artifact change.
        license: SPDX-style or human-readable license identifier.
        citation: Citation key matching an entry in ``CITATIONS.bib``.
        redistribution_restrictions: One-line note on any restrictions.
    """

    name: str
    records: list[ZenodoRecordPin] = field(default_factory=list)
    license: str = ""
    citation: str = ""
    redistribution_restrictions: str = ""


@dataclass(frozen=True, slots=True)
class GithubGeneratorSource:
    """GitHub generator/code-only SPEC. Used with ``canonicalization: github_commit_pin``.

    Attributes:
        name: Stable dataset key (e.g. ``"baaiworm"``).
        repo: ``host/owner/repo``-style identifier (e.g.
            ``"github.com/Jessie940611/BAAIWorm"``).
        commit_sha: Full 40-character commit SHA at pin time.
        config_path: Path inside the repo to the generator configuration file
            (relative to repo root).
        config_sha256: SHA-256 of the canonicalized config bytes at pin time.
            Canonicalization of the config follows whatever format the config
            uses (YAML → ``canonicalize_yaml``-equivalent).
        license: SPDX-style or human-readable license identifier (typically
            the repo's LICENSE).
        citation: Citation key matching an entry in ``CITATIONS.bib``.
        redistribution_restrictions: One-line note on any restrictions.
    """

    name: str
    repo: str
    commit_sha: str
    config_path: str
    config_sha256: str
    license: str = ""
    citation: str = ""
    redistribution_restrictions: str = ""


AnyDatasetSource = (
    DatasetSource | DandiFederationSource | ZenodoSubsetSource | GithubGeneratorSource
)
"""Union of every SPEC dataclass. Use where any SPEC shape is acceptable."""
