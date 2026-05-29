"""Unit tests for ``wormjepa.manifest.lock``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wormjepa import ManifestLockError
from wormjepa.manifest.lock import (
    MANIFEST_SCHEMA_VERSION,
    ArtifactEntry,
    DandisetPin,
    Manifest,
    ZenodoRecord,
    read_manifest,
    write_manifest,
)


def _file_entry(path: str = "pre-registration/splits/wormid_train_eval.yaml") -> ArtifactEntry:
    return ArtifactEntry(
        path=path,
        sha256="a" * 64,
        canonicalization="yaml_sorted_keys_lf",
    )


def _dataset_entry(name: str = "wormid") -> ArtifactEntry:
    return ArtifactEntry(
        dataset=name,
        doi="10.48324/dandi.000xxx/0.260512",
        sha256="b" * 64,
        canonicalization="doi_manifest",
    )


def _manifest(*entries: ArtifactEntry) -> Manifest:
    return Manifest(
        locked_at=datetime(2026, 5, 12, 14, 23, 1, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="a1b2c3d4e5f6789012345678901234567890abcd",
        artifacts=list(entries),
    )


def test_artifact_entry_requires_path_or_dataset() -> None:
    with pytest.raises(ManifestLockError, match="path"):
        ArtifactEntry(sha256="0" * 64, canonicalization="yaml_sorted_keys_lf")


def test_artifact_entry_rejects_both_path_and_dataset() -> None:
    with pytest.raises(ManifestLockError, match="both"):
        ArtifactEntry(
            path="x.yaml",
            dataset="wormid",
            doi="10.0/x",
            sha256="0" * 64,
            canonicalization="yaml_sorted_keys_lf",
        )


def test_artifact_entry_dataset_requires_doi() -> None:
    with pytest.raises(ManifestLockError, match="doi"):
        ArtifactEntry(
            dataset="wormid",
            sha256="0" * 64,
            canonicalization="doi_manifest",
        )


def test_manifest_round_trip(tmp_path: Path) -> None:
    m = _manifest(_file_entry(), _dataset_entry())
    out = tmp_path / "MANIFEST.lock"
    write_manifest(m, out)
    loaded = read_manifest(out)
    assert loaded == m


def test_manifest_round_trip_is_byte_stable(tmp_path: Path) -> None:
    m = _manifest(_file_entry(), _dataset_entry())
    out1 = tmp_path / "a.lock"
    out2 = tmp_path / "b.lock"
    write_manifest(m, out1)
    write_manifest(m, out2)
    assert out1.read_bytes() == out2.read_bytes()


def test_manifest_sorts_artifacts_canonically(tmp_path: Path) -> None:
    """Files listed before datasets; each group sorted alphabetically."""
    m = _manifest(
        _dataset_entry("wormid"),
        _file_entry("pre-registration/probes/neural_decoding.py"),
        _file_entry("pre-registration/configs/headline.yaml"),
        _dataset_entry("flavell_2023"),
    )
    out = tmp_path / "MANIFEST.lock"
    write_manifest(m, out)
    text = out.read_text(encoding="utf-8")
    # configs/headline.yaml appears before probes/neural_decoding.py (file path sort).
    assert text.index("configs/headline.yaml") < text.index("probes/neural_decoding.py")
    # File paths appear before datasets.
    assert text.index("probes/neural_decoding.py") < text.index("dataset: flavell_2023")
    # Within datasets: flavell_2023 < wormid alphabetically.
    assert text.index("dataset: flavell_2023") < text.index("dataset: wormid")


def test_manifest_schema_version_is_pinned() -> None:
    m = _manifest()
    assert m.schema_version == MANIFEST_SCHEMA_VERSION


def test_read_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestLockError, match="does not exist"):
        read_manifest(tmp_path / "nonexistent.lock")


def test_read_manifest_malformed_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.lock"
    p.write_text("not a mapping at top level", encoding="utf-8")
    with pytest.raises(ManifestLockError):
        read_manifest(p)


def test_find_file_entry() -> None:
    m = _manifest(_file_entry("pre-registration/x.yaml"))
    assert m.find_file_entry("pre-registration/x.yaml") is not None
    assert m.find_file_entry("missing.yaml") is None


def test_find_dataset_entry() -> None:
    m = _manifest(_dataset_entry("wormid"))
    assert m.find_dataset_entry("wormid") is not None
    assert m.find_dataset_entry("missing") is None


# -- Schema v2 dataset canonicalizations -------------------------------


def _federation_entry(name: str = "wormid") -> ArtifactEntry:
    return ArtifactEntry(
        dataset=name,
        canonicalization="dandi_federation",
        dandisets=[
            DandisetPin(
                dandiset_id="000715",
                version="0.241009.1514",
                doi="10.48324/dandi.000715/0.241009.1514",
            ),
            DandisetPin(
                dandiset_id="000472",
                version="0.241009.1502",
                doi="10.48324/dandi.000472/0.241009.1502",
            ),
        ],
        sha256="c" * 64,
    )


def _zenodo_subset_entry(name: str = "wormbehavior_db") -> ArtifactEntry:
    return ArtifactEntry(
        dataset=name,
        canonicalization="zenodo_subset",
        records=[
            ZenodoRecord(zenodo_record_id="1031550", doi="10.5281/zenodo.1031550"),
            ZenodoRecord(
                zenodo_record_id="1029149",
                doi="10.5281/zenodo.1029149",
                description="MT1078 egl-13(n483)X",
            ),
        ],
        sha256="d" * 64,
    )


def _github_pin_entry(name: str = "baaiworm") -> ArtifactEntry:
    return ArtifactEntry(
        dataset=name,
        canonicalization="github_commit_pin",
        repo="github.com/Jessie940611/BAAIWorm",
        commit_sha="a" * 40,
        config_path="configs/default.yaml",
        config_sha256="b" * 64,
        sha256="e" * 64,
    )


def test_dandi_federation_entry_constructs_and_round_trips(tmp_path: Path) -> None:
    m = _manifest(_federation_entry())
    out = tmp_path / "MANIFEST.lock"
    write_manifest(m, out)
    loaded = read_manifest(out)
    entry = loaded.find_dataset_entry("wormid")
    assert entry is not None
    assert entry.canonicalization == "dandi_federation"
    assert entry.dandisets is not None
    assert len(entry.dandisets) == 2


def test_zenodo_subset_entry_constructs_and_round_trips(tmp_path: Path) -> None:
    m = _manifest(_zenodo_subset_entry())
    out = tmp_path / "MANIFEST.lock"
    write_manifest(m, out)
    loaded = read_manifest(out)
    entry = loaded.find_dataset_entry("wormbehavior_db")
    assert entry is not None
    assert entry.canonicalization == "zenodo_subset"
    assert entry.records is not None
    assert {r.zenodo_record_id for r in entry.records} == {"1029149", "1031550"}


def test_github_commit_pin_entry_constructs_and_round_trips(tmp_path: Path) -> None:
    m = _manifest(_github_pin_entry())
    out = tmp_path / "MANIFEST.lock"
    write_manifest(m, out)
    loaded = read_manifest(out)
    entry = loaded.find_dataset_entry("baaiworm")
    assert entry is not None
    assert entry.canonicalization == "github_commit_pin"
    assert entry.repo == "github.com/Jessie940611/BAAIWorm"
    assert entry.commit_sha == "a" * 40
    assert entry.config_sha256 == "b" * 64


def test_dandi_federation_requires_non_empty_dandisets() -> None:
    with pytest.raises(ManifestLockError, match="non-empty"):
        ArtifactEntry(
            dataset="wormid",
            canonicalization="dandi_federation",
            dandisets=[],
            sha256="0" * 64,
        )


def test_zenodo_subset_requires_non_empty_records() -> None:
    with pytest.raises(ManifestLockError, match="non-empty"):
        ArtifactEntry(
            dataset="wormbehavior_db",
            canonicalization="zenodo_subset",
            records=[],
            sha256="0" * 64,
        )


def test_github_commit_pin_requires_all_fields() -> None:
    with pytest.raises(ManifestLockError, match="non-empty"):
        ArtifactEntry(
            dataset="baaiworm",
            canonicalization="github_commit_pin",
            repo="github.com/x/y",
            commit_sha="a" * 40,
            config_path="",  # empty -> violation
            config_sha256="b" * 64,
            sha256="0" * 64,
        )


def test_schema_version_2_is_default() -> None:
    """After bumping for the v2 canonicalizations, MANIFEST_SCHEMA_VERSION == 2."""
    assert MANIFEST_SCHEMA_VERSION == 2
    m = _manifest()
    assert m.schema_version == 2


def test_schema_version_1_still_accepted_for_backward_compat() -> None:
    """Existing on-disk v1 manifests must continue to read until re-locked."""
    m = Manifest(
        schema_version=1,
        locked_at=datetime(2026, 5, 12, 14, 23, 1, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="a1b2c3d4e5f6789012345678901234567890abcd",
        artifacts=[_file_entry()],
    )
    assert m.schema_version == 1
