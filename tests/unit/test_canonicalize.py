"""Unit tests for ``wormjepa.manifest.canonicalize``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wormjepa.manifest.canonicalize import (
    canonicalize_dandi_federation,
    canonicalize_doi_manifest,
    canonicalize_github_commit_pin,
    canonicalize_python,
    canonicalize_yaml,
    canonicalize_zenodo_subset,
    sha256_of_canonicalized,
)


def test_yaml_sorted_keys_invariant(tmp_path: Path) -> None:
    """Two YAML files differing only in key order produce identical bytes."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("name: headline\nschema_version: 1\n", encoding="utf-8")
    b.write_text("schema_version: 1\nname: headline\n", encoding="utf-8")
    assert canonicalize_yaml(a) == canonicalize_yaml(b)


def test_yaml_comments_dropped(tmp_path: Path) -> None:
    """Comments do not affect canonical bytes (semantic content only)."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("# this is a comment\nname: headline\n", encoding="utf-8")
    b.write_text("name: headline\n", encoding="utf-8")
    assert canonicalize_yaml(a) == canonicalize_yaml(b)


def test_yaml_canonical_uses_lf(tmp_path: Path) -> None:
    p = tmp_path / "f.yaml"
    p.write_text("a: 1\r\nb: 2\r\n", encoding="utf-8")
    text = canonicalize_yaml(p).decode("utf-8")
    assert "\r" not in text


def test_python_ast_whitespace_invariant(tmp_path: Path) -> None:
    """Whitespace-only edits do not change the canonical bytes."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    b.write_text("def    f(x):\n  return x + 1\n", encoding="utf-8")
    assert canonicalize_python(a) == canonicalize_python(b)


def test_python_ast_comment_invariant(tmp_path: Path) -> None:
    """Comment-only edits do not change the canonical bytes (ast.unparse drops them)."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def f(x):\n    # frozen\n    return x + 1\n", encoding="utf-8")
    b.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    assert canonicalize_python(a) == canonicalize_python(b)


def test_python_semantic_change_changes_hash(tmp_path: Path) -> None:
    """A real semantic edit changes the canonical bytes."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    b.write_text("def f(x):\n    return x + 2\n", encoding="utf-8")
    assert canonicalize_python(a) != canonicalize_python(b)


def test_doi_manifest_sorted_keys(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"doi": "10.1/x", "version": 1}', encoding="utf-8")
    b.write_text('{"version": 1, "doi": "10.1/x"}', encoding="utf-8")
    assert canonicalize_doi_manifest(a) == canonicalize_doi_manifest(b)


def test_doi_manifest_bare_string_wrapped(tmp_path: Path) -> None:
    p = tmp_path / "doi.txt"
    p.write_text("10.48324/dandi.000xxx/0.260512", encoding="utf-8")
    canonical = canonicalize_doi_manifest(p)
    assert b'"doi"' in canonical
    assert b"10.48324/dandi.000xxx/0.260512" in canonical


def test_sha256_dispatch(tmp_path: Path) -> None:
    p = tmp_path / "f.yaml"
    p.write_text("k: v\n", encoding="utf-8")
    sha_yaml = sha256_of_canonicalized(p, "yaml_sorted_keys_lf")
    assert len(sha_yaml) == 64
    assert all(c in "0123456789abcdef" for c in sha_yaml)


def test_sha256_stable_across_runs(tmp_path: Path) -> None:
    p = tmp_path / "f.yaml"
    p.write_text("schema_version: 1\nname: x\n", encoding="utf-8")
    h1 = sha256_of_canonicalized(p, "yaml_sorted_keys_lf")
    h2 = sha256_of_canonicalized(p, "yaml_sorted_keys_lf")
    assert h1 == h2


# -- dandi_federation -------------------------------------------------


def test_dandi_federation_sort_invariant() -> None:
    """Order of input dandisets does not affect the canonical bytes."""
    a = [
        {"dandiset_id": "000715", "version": "0.1", "doi": "10.1/715"},
        {"dandiset_id": "000472", "version": "0.1", "doi": "10.1/472"},
    ]
    b = list(reversed(a))
    assert canonicalize_dandi_federation(a) == canonicalize_dandi_federation(b)


def test_dandi_federation_changes_with_membership() -> None:
    """Adding a dandiset to the federation changes the canonical bytes."""
    a = [{"dandiset_id": "000715", "version": "0.1", "doi": "10.1/715"}]
    b = [*a, {"dandiset_id": "000472", "version": "0.1", "doi": "10.1/472"}]
    assert canonicalize_dandi_federation(a) != canonicalize_dandi_federation(b)


def test_dandi_federation_emits_sorted_compact_json() -> None:
    """Canonical output is sorted-by-id compact JSON with LF terminator."""
    payload = canonicalize_dandi_federation(
        [
            {"dandiset_id": "000715", "version": "0.1", "doi": "10.1/715"},
            {"dandiset_id": "000472", "version": "0.1", "doi": "10.1/472"},
        ]
    )
    text = payload.decode("utf-8")
    assert text.endswith("\n")
    parsed = json.loads(text)
    assert [d["dandiset_id"] for d in parsed] == ["000472", "000715"]


# -- zenodo_subset ----------------------------------------------------


def test_zenodo_subset_sort_invariant() -> None:
    a = [
        {"zenodo_record_id": "1031550", "doi": "10.5281/zenodo.1031550"},
        {"zenodo_record_id": "1029149", "doi": "10.5281/zenodo.1029149"},
    ]
    b = list(reversed(a))
    assert canonicalize_zenodo_subset(a) == canonicalize_zenodo_subset(b)


def test_zenodo_subset_description_changes_hash() -> None:
    """An added description on a record changes the canonical bytes."""
    a = [{"zenodo_record_id": "1031550", "doi": "10.5281/zenodo.1031550"}]
    b = [
        {
            "zenodo_record_id": "1031550",
            "doi": "10.5281/zenodo.1031550",
            "description": "N2 Schafer Lab",
        }
    ]
    assert canonicalize_zenodo_subset(a) != canonicalize_zenodo_subset(b)


def test_zenodo_subset_empty_description_omitted() -> None:
    """Empty descriptions are omitted so they hash identically to absent."""
    a = [{"zenodo_record_id": "1031550", "doi": "10.5281/zenodo.1031550"}]
    b = [
        {
            "zenodo_record_id": "1031550",
            "doi": "10.5281/zenodo.1031550",
            "description": "",
        }
    ]
    assert canonicalize_zenodo_subset(a) == canonicalize_zenodo_subset(b)


# -- github_commit_pin ------------------------------------------------


def test_github_commit_pin_deterministic() -> None:
    payload = canonicalize_github_commit_pin(
        repo="github.com/Jessie940611/BAAIWorm",
        commit_sha="a" * 40,
        config_path="configs/default.yaml",
        config_sha256="b" * 64,
    )
    text = payload.decode("utf-8")
    assert text.endswith("\n")
    parsed = json.loads(text)
    assert parsed["repo"] == "github.com/Jessie940611/BAAIWorm"
    assert parsed["commit_sha"] == "a" * 40
    assert parsed["config_path"] == "configs/default.yaml"
    assert parsed["config_sha256"] == "b" * 64


def test_github_commit_pin_changes_with_commit() -> None:
    a = canonicalize_github_commit_pin(
        repo="r", commit_sha="a" * 40, config_path="c", config_sha256="d" * 64
    )
    b = canonicalize_github_commit_pin(
        repo="r", commit_sha="b" * 40, config_path="c", config_sha256="d" * 64
    )
    assert a != b


# -- sha256_of_canonicalized rejects structured methods ---------------


@pytest.mark.parametrize(
    "method",
    ["dandi_federation", "zenodo_subset", "github_commit_pin"],
)
def test_sha256_of_canonicalized_rejects_structured_methods(tmp_path: Path, method: str) -> None:
    """Structured-data canonicalizations cannot be invoked via a file path."""
    p = tmp_path / "anything.txt"
    p.write_text("doesn't matter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="structured-data-based"):
        sha256_of_canonicalized(p, method)  # type: ignore[arg-type]
