"""Unit tests for ``wormjepa.cli.run_ids``."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from wormjepa import WormJEPAError
from wormjepa.cli.run_ids import RUN_ID_PATTERN, generate_run_id


def test_run_id_pattern_constant() -> None:
    """The exported pattern matches the documented format."""
    sample = "20260512T143201Z__a1b2c3d4__headline"
    assert RUN_ID_PATTERN.match(sample)
    sample_dirty = "20260512T143201Z__a1b2c3d4-dirty__headline"
    assert RUN_ID_PATTERN.match(sample_dirty)


def test_generate_run_id_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated run-ids match the documented pattern."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    run_id = generate_run_id(Path("configs/headline.yaml"))
    assert RUN_ID_PATTERN.match(run_id), f"run_id={run_id!r} does not match pattern"


def test_generate_run_id_includes_config_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    """The config-slug component is derived from the config filename stem."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    run_id = generate_run_id(Path("configs/headline.yaml"))
    assert run_id.endswith("__headline")


def test_generate_run_id_slugifies_non_alphanumerics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hyphens, dots, and other punctuation become underscores in the slug."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    run_id = generate_run_id(Path("configs/baselines/transformer-eigenworms.yaml"))
    assert run_id.endswith("__transformer_eigenworms")


def test_generate_run_id_marks_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uncommitted modifications to tracked files mark the run-id as ``-dirty``."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    # Mark this test as dirty: temporarily modify a tracked file. We add then
    # restore via subprocess so we don't depend on the test's own working tree
    # state at the start of the run.
    target = repo_root / "pyproject.toml"
    original = target.read_text()
    try:
        target.write_text(original + "# transient test marker\n")
        run_id = generate_run_id(Path("configs/headline.yaml"))
        # We can't assert the dirty/clean state of the repo prior to the test,
        # so we accept either: the run-id must contain a valid sha and may
        # contain '-dirty'. The key assertion: when we know we made it dirty,
        # the run-id must show '-dirty'.
        assert "-dirty" in run_id
    finally:
        target.write_text(original)


def test_generate_run_id_outside_git_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside any git repository, run-id generation raises WormJEPAError."""
    monkeypatch.chdir(tmp_path)
    # Confirm we're not inside a repo:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        pytest.skip("tmp_path is inside a git repo; cannot test the no-repo failure path")
    with pytest.raises(WormJEPAError):
        generate_run_id(Path("headline.yaml"))


def test_generate_run_id_empty_slug_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config filename that slugifies to empty raises WormJEPAError."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    with pytest.raises(WormJEPAError, match="non-empty slug"):
        generate_run_id(Path("---.yaml"))


def test_generate_run_id_timestamp_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timestamp component is YYYYMMDDTHHMMSSZ (no separators except T and Z)."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    run_id = generate_run_id(Path("configs/headline.yaml"))
    timestamp = run_id.split("__")[0]
    assert re.match(r"^\d{8}T\d{6}Z$", timestamp)
