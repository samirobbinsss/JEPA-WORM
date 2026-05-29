"""Unit tests for ``wormjepa.paths``."""

from __future__ import annotations

from pathlib import Path

import pytest

from wormjepa.paths import ProjectRootNotFoundError, project_root


def test_project_root_from_repo_root() -> None:
    """From the repo root, ``project_root()`` returns the repo root itself."""
    expected = Path(__file__).resolve().parents[2]
    assert project_root(expected) == expected


def test_project_root_from_src_directory() -> None:
    """From inside ``src/wormjepa/``, ``project_root()`` walks up to the repo root."""
    repo_root = Path(__file__).resolve().parents[2]
    inner = repo_root / "src" / "wormjepa"
    assert project_root(inner) == repo_root


def test_project_root_from_test_file_path() -> None:
    """Passing a file path (not a directory) resolves to the file's containing repo root."""
    repo_root = Path(__file__).resolve().parents[2]
    assert project_root(Path(__file__)) == repo_root


def test_project_root_uses_cwd_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no argument, ``project_root()`` walks up from the current working directory."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root / "src" / "wormjepa")
    assert project_root() == repo_root


def test_project_root_raises_outside_project(tmp_path: Path) -> None:
    """``project_root()`` raises ``ProjectRootNotFoundError`` outside any project tree."""
    with pytest.raises(ProjectRootNotFoundError):
        project_root(tmp_path)


def test_project_root_error_subclasses_wormjepa_error() -> None:
    """``ProjectRootNotFoundError`` is a ``WormJEPAError`` (caught by CLI top-level)."""
    from wormjepa import WormJEPAError

    assert issubclass(ProjectRootNotFoundError, WormJEPAError)
