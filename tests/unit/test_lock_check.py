"""Unit tests for ``wormjepa.manifest.lock_check``."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wormjepa import PreRegistrationViolation
from wormjepa.manifest.canonicalize import (
    canonicalize_doi_string,
    sha256_of_canonicalized,
)
from wormjepa.manifest.lock import ArtifactEntry, Manifest, write_manifest
from wormjepa.manifest.lock_check import verify_manifest


def _yaml_artifact(path: Path, root: Path) -> ArtifactEntry:
    """Create an ArtifactEntry for a YAML file under ``root``."""
    sha = sha256_of_canonicalized(path, "yaml_sorted_keys_lf")
    return ArtifactEntry(
        path=str(path.relative_to(root)),
        sha256=sha,
        canonicalization="yaml_sorted_keys_lf",
    )


def _python_artifact(path: Path, root: Path) -> ArtifactEntry:
    sha = sha256_of_canonicalized(path, "python_ast_normalized")
    return ArtifactEntry(
        path=str(path.relative_to(root)),
        sha256=sha,
        canonicalization="python_ast_normalized",
    )


def _set_up_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")


def test_verify_manifest_happy_path(tmp_path: Path) -> None:
    _set_up_pyproject(tmp_path)
    yaml_file = tmp_path / "pre-registration" / "splits" / "split.yaml"
    yaml_file.parent.mkdir(parents=True)
    yaml_file.write_text("schema_version: 1\n", encoding="utf-8")

    py_file = tmp_path / "pre-registration" / "probes" / "probe.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("def probe():\n    return 0\n", encoding="utf-8")

    manifest = Manifest(
        locked_at=datetime(2026, 5, 12, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="abc",
        artifacts=[
            _yaml_artifact(yaml_file, tmp_path),
            _python_artifact(py_file, tmp_path),
        ],
    )
    manifest_path = tmp_path / "pre-registration" / "MANIFEST.lock"
    write_manifest(manifest, manifest_path)

    result = verify_manifest(manifest_path, root=tmp_path)
    assert result.verified == 2


def test_verify_manifest_sha_mismatch_raises(tmp_path: Path) -> None:
    _set_up_pyproject(tmp_path)
    yaml_file = tmp_path / "pre-registration" / "splits" / "split.yaml"
    yaml_file.parent.mkdir(parents=True)
    yaml_file.write_text("schema_version: 1\n", encoding="utf-8")

    entry = _yaml_artifact(yaml_file, tmp_path)
    bad_entry = entry.model_copy(update={"sha256": "f" * 64})

    manifest = Manifest(
        locked_at=datetime(2026, 5, 12, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="abc",
        artifacts=[bad_entry],
    )
    manifest_path = tmp_path / "pre-registration" / "MANIFEST.lock"
    write_manifest(manifest, manifest_path)

    with pytest.raises(PreRegistrationViolation, match="SHA mismatch"):
        verify_manifest(manifest_path, root=tmp_path)


def test_verify_manifest_missing_file_raises(tmp_path: Path) -> None:
    _set_up_pyproject(tmp_path)
    manifest = Manifest(
        locked_at=datetime(2026, 5, 12, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="abc",
        artifacts=[
            ArtifactEntry(
                path="pre-registration/nonexistent.yaml",
                sha256="0" * 64,
                canonicalization="yaml_sorted_keys_lf",
            )
        ],
    )
    manifest_path = tmp_path / "pre-registration" / "MANIFEST.lock"
    write_manifest(manifest, manifest_path)
    with pytest.raises(PreRegistrationViolation, match="missing from working tree"):
        verify_manifest(manifest_path, root=tmp_path)


def test_verify_manifest_dataset_entry_uses_current_spec(tmp_path: Path) -> None:
    """Dataset entries hash the current SPEC's DOI string.

    Uses ``flavell_2023`` because it is the only Phase 0 dataset that
    remains a single-DOI source (``doi_manifest`` canonicalization).
    The other four switched to v2 canonicalizations at the 2026-05-14
    re-lock and no longer expose a ``SPEC.doi`` attribute.
    """
    _set_up_pyproject(tmp_path)
    from wormjepa.data.sources import flavell_2023 as flavell_module

    doi = flavell_module.SPEC.doi
    expected_sha = hashlib.sha256(canonicalize_doi_string(doi)).hexdigest()

    manifest = Manifest(
        locked_at=datetime(2026, 5, 14, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="abc",
        artifacts=[
            ArtifactEntry(
                dataset="flavell_2023",
                doi=doi,
                sha256=expected_sha,
                canonicalization="doi_manifest",
            )
        ],
    )
    manifest_path = tmp_path / "pre-registration" / "MANIFEST.lock"
    write_manifest(manifest, manifest_path)
    result = verify_manifest(manifest_path, root=tmp_path)
    assert result.verified == 1


def test_verify_manifest_dataset_drift_raises(tmp_path: Path) -> None:
    """Manifest pinning a stale dataset SHA raises."""
    _set_up_pyproject(tmp_path)
    manifest = Manifest(
        locked_at=datetime(2026, 5, 14, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="abc",
        artifacts=[
            ArtifactEntry(
                dataset="flavell_2023",
                doi="10.0/some-stale-doi",
                sha256="0" * 64,
                canonicalization="doi_manifest",
            )
        ],
    )
    manifest_path = tmp_path / "pre-registration" / "MANIFEST.lock"
    write_manifest(manifest, manifest_path)
    with pytest.raises(PreRegistrationViolation, match="SPEC drift"):
        verify_manifest(manifest_path, root=tmp_path)


def test_verify_manifest_unknown_dataset_raises(tmp_path: Path) -> None:
    _set_up_pyproject(tmp_path)
    manifest = Manifest(
        locked_at=datetime(2026, 5, 14, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="abc",
        artifacts=[
            ArtifactEntry(
                dataset="definitely_not_a_real_dataset",
                doi="10.0/x",
                sha256="0" * 64,
                canonicalization="doi_manifest",
            )
        ],
    )
    manifest_path = tmp_path / "pre-registration" / "MANIFEST.lock"
    write_manifest(manifest, manifest_path)
    with pytest.raises(PreRegistrationViolation, match="no module"):
        verify_manifest(manifest_path, root=tmp_path)
