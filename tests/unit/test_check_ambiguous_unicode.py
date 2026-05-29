"""Unit tests for ``scripts/dev/check_ambiguous_unicode.py``.

Runs the script as a subprocess (mirroring how pre-commit invokes it) and
verifies exit codes + stdout messages for clean, dirty, and skipped inputs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_ambiguous_unicode.py"

# Use \N{} escapes so this test source file is pure ASCII while the *content*
# we write into fixture files is the ambiguous codepoint under test.
GAMMA = "\N{GREEK SMALL LETTER GAMMA}"  # U+03B3 — confusable with 'y'
MUL = "\N{MULTIPLICATION SIGN}"  # U+00D7 — confusable with 'x'


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the hook script with ``args`` and capture output."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_file_passes(tmp_path: Path) -> None:
    """Pure-ASCII Python file produces no output and exits 0."""
    src = tmp_path / "clean.py"
    src.write_text("# weight = y\nx = 1 * 2\n", encoding="utf-8")
    result = _run([str(src)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_gamma_flagged(tmp_path: Path) -> None:
    """Greek gamma in a comment is flagged with codepoint + ASCII confusable."""
    src = tmp_path / "bad_gamma.py"
    src.write_text(f"# weight = {GAMMA}\n", encoding="utf-8")
    result = _run([str(src)])
    assert result.returncode == 1
    assert "U+03B3" in result.stdout
    assert "'y'" in result.stdout


def test_multiplication_sign_flagged(tmp_path: Path) -> None:
    """Multiplication sign in a comment is flagged with U+00D7."""
    src = tmp_path / "bad_mul.py"
    src.write_text(f"# A {MUL} B\n", encoding="utf-8")
    result = _run([str(src)])
    assert result.returncode == 1
    assert "U+00D7" in result.stdout


def test_skips_markdown(tmp_path: Path) -> None:
    """Markdown files are skipped even when they contain ambiguous unicode."""
    src = tmp_path / "notes.md"
    src.write_text(f"weight = {GAMMA}\n", encoding="utf-8")
    result = _run([str(src)])
    assert result.returncode == 0
    assert result.stdout == ""


def test_skips_third_party(tmp_path: Path) -> None:
    """Files under ``third_party/`` are skipped regardless of suffix."""
    nested = tmp_path / "third_party" / "foo.py"
    nested.parent.mkdir(parents=True)
    nested.write_text(f"# weight = {GAMMA}\n", encoding="utf-8")
    result = _run([str(nested)])
    assert result.returncode == 0
    assert result.stdout == ""
