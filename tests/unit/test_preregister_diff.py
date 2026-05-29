"""Unit tests for ``wormjepa preregister --diff`` (Story 4.x).

Covers the audit-tool semantics: added/removed/changed/unchanged classification,
short-SHA truncation, summary line, exit codes, and the ``--show-unchanged``
flag.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer

from wormjepa.cli import preregister as preregister_cli
from wormjepa.manifest.lock import ArtifactEntry, Manifest, write_manifest


def _file_entry(path: str, sha_char: str = "a") -> ArtifactEntry:
    return ArtifactEntry(
        path=path,
        sha256=sha_char * 64,
        canonicalization="yaml_sorted_keys_lf",
    )


def _manifest(*entries: ArtifactEntry) -> Manifest:
    return Manifest(
        locked_at=datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC),
        locked_by="sami",
        git_sha_at_lock="a1b2c3d4e5f6789012345678901234567890abcd",
        artifacts=list(entries),
    )


def _write_lock(tmp_path: Path, name: str, manifest: Manifest) -> Path:
    out = tmp_path / name
    write_manifest(manifest, out)
    return out


def _patch_current_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, current_lock: Path
) -> None:
    """Point ``preregister_command`` at ``current_lock`` as the "current" manifest.

    The CLI resolves the current lock via ``project_root() / pre-registration / MANIFEST.lock``,
    so we lay out a matching directory and monkey-patch ``project_root``.
    """
    root = tmp_path / "fake_root"
    (root / "pre-registration").mkdir(parents=True)
    target = root / "pre-registration" / "MANIFEST.lock"
    target.write_bytes(current_lock.read_bytes())
    monkeypatch.setattr(preregister_cli, "project_root", lambda: root)


def _invoke_diff(prior: Path, *, show_unchanged: bool = False) -> int:
    """Invoke ``preregister_command(diff=prior, ...)`` and return the exit code."""
    with pytest.raises(typer.Exit) as exc_info:
        preregister_cli.preregister_command(
            verify=False,
            diff=prior,
            show_unchanged=show_unchanged,
            force=False,
        )
    return int(exc_info.value.exit_code)


def test_diff_no_changes_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same lock content on both sides → no diff lines, exit 0."""
    manifest = _manifest(
        _file_entry("pre-registration/configs/headline.yaml", "a"),
        _file_entry("pre-registration/neuron_subset.yaml", "b"),
    )
    prior = _write_lock(tmp_path, "prior.lock", manifest)
    current = _write_lock(tmp_path, "current.lock", manifest)
    _patch_current_lock(monkeypatch, tmp_path, current)

    code = _invoke_diff(prior)

    captured = capsys.readouterr()
    out = captured.out
    assert code == 0
    assert "summary: 0 added, 0 removed, 0 changed, 2 unchanged" in out
    # No diff markers should appear on no-change.
    assert "+ " not in out
    assert "- " not in out
    assert "~ " not in out


def test_diff_added_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Current lock has an extra entry that prior lacks."""
    prior_manifest = _manifest(
        _file_entry("pre-registration/configs/headline.yaml", "a"),
    )
    current_manifest = _manifest(
        _file_entry("pre-registration/configs/headline.yaml", "a"),
        _file_entry("pre-registration/neuron_subset.yaml", "c"),
    )
    prior = _write_lock(tmp_path, "prior.lock", prior_manifest)
    current = _write_lock(tmp_path, "current.lock", current_manifest)
    _patch_current_lock(monkeypatch, tmp_path, current)

    code = _invoke_diff(prior)

    out = capsys.readouterr().out
    assert code == 1
    assert "+ pre-registration/neuron_subset.yaml  (new, sha=cccccccccccc)" in out
    assert "summary: 1 added, 0 removed, 0 changed, 1 unchanged" in out


def test_diff_changed_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same artifact path on both sides, different SHA."""
    path = "pre-registration/configs/headline.yaml"
    prior_manifest = _manifest(_file_entry(path, "a"))
    current_manifest = _manifest(_file_entry(path, "b"))
    prior = _write_lock(tmp_path, "prior.lock", prior_manifest)
    current = _write_lock(tmp_path, "current.lock", current_manifest)
    _patch_current_lock(monkeypatch, tmp_path, current)

    code = _invoke_diff(prior)

    out = capsys.readouterr().out
    assert code == 1
    assert f"~ {path}  (was=aaaaaaaaaaaa, now=bbbbbbbbbbbb)" in out
    assert "summary: 0 added, 0 removed, 1 changed, 0 unchanged" in out


def test_diff_removed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prior lock has an entry the current one no longer carries."""
    prior_manifest = _manifest(
        _file_entry("pre-registration/configs/headline.yaml", "a"),
        _file_entry("pre-registration/neuron_subset.yaml", "d"),
    )
    current_manifest = _manifest(
        _file_entry("pre-registration/configs/headline.yaml", "a"),
    )
    prior = _write_lock(tmp_path, "prior.lock", prior_manifest)
    current = _write_lock(tmp_path, "current.lock", current_manifest)
    _patch_current_lock(monkeypatch, tmp_path, current)

    code = _invoke_diff(prior)

    out = capsys.readouterr().out
    assert code == 1
    assert "- pre-registration/neuron_subset.yaml  (sha=dddddddddddd)" in out
    assert "summary: 0 added, 1 removed, 0 changed, 1 unchanged" in out


def test_show_unchanged_includes_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--show-unchanged`` adds unchanged rows to the output."""
    path_unchanged = "pre-registration/configs/headline.yaml"
    path_changed = "pre-registration/neuron_subset.yaml"
    prior_manifest = _manifest(
        _file_entry(path_unchanged, "a"),
        _file_entry(path_changed, "b"),
    )
    current_manifest = _manifest(
        _file_entry(path_unchanged, "a"),
        _file_entry(path_changed, "c"),
    )
    prior = _write_lock(tmp_path, "prior.lock", prior_manifest)
    current = _write_lock(tmp_path, "current.lock", current_manifest)
    _patch_current_lock(monkeypatch, tmp_path, current)

    # Without --show-unchanged: unchanged path absent.
    code_default = _invoke_diff(prior, show_unchanged=False)
    out_default = capsys.readouterr().out
    assert code_default == 1
    assert path_unchanged not in out_default
    assert path_changed in out_default

    # With --show-unchanged: unchanged path present.
    code_shown = _invoke_diff(prior, show_unchanged=True)
    out_shown = capsys.readouterr().out
    assert code_shown == 1
    assert f"  {path_unchanged}  (sha=aaaaaaaaaaaa)" in out_shown
    assert "summary: 0 added, 0 removed, 1 changed, 1 unchanged" in out_shown
