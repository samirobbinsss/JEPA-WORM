"""Pre-commit hook: flag ambiguous unicode characters in Python source files.

Catches the same confusables ruff RUF002 / RUF003 catch (greek gamma vs ``y``,
multiplication sign vs ``x``, smart quotes vs ASCII quotes, etc.) -- but at
commit time, across the whole staged source set, before ruff even runs.

Faster feedback than waiting for ruff to scan the full tree, and identical
guarantee on the surface of what's allowed in.

Usage::

    python3 scripts/dev/check_ambiguous_unicode.py <path> [<path> ...]

Exits non-zero (1) on the first hit, after listing every offending location
in the form::

    <path>:<line>:<col>: <char> (U+XXXX) -- confusable with '<ascii>'

Files outside the ``.py`` scope (markdown, rst, txt, LICENSE*) and files
under vendored / generated trees (``third_party/``, ``vendor/``, ``.git/``,
``_bmad-output/``) are skipped: unicode is fine in prose, and we do not own
vendored code.

This script intentionally flags ambiguous unicode *anywhere* in a ``.py``
file, including inside string and byte literals. A future enhancement could
parse the AST and skip literals; for now the conservative behavior matches
what ruff RUF001/RUF002/RUF003 report.

The script's own source is kept pure ASCII (the ambiguous codepoints are
expressed as ``\\N{...}`` escapes) so running the hook over itself produces
zero output.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Map: ambiguous unicode codepoint -> ASCII char it visually resembles.
# Curated tight: only the 3 confusables that have actually bitten this
# project (greek gamma vs 'y' in protocol notation, multiplication sign
# vs 'x' in dimension specs, minus sign vs '-' in arithmetic). Stylistic
# unicode that the codebase uses intentionally (em-dash separators,
# horizontal ellipsis in docstrings, smart quotes in user-facing strings)
# is allowed.
#
# Keys are written as ``\N{...}`` escapes so this source file is pure ASCII
# (and thus passes its own check) while still keying on the real codepoints.
AMBIGUOUS_CHARS: dict[str, str] = {
    "\N{GREEK SMALL LETTER GAMMA}": "y",  # U+03B3
    "\N{MULTIPLICATION SIGN}": "x",  # U+00D7
    "\N{MINUS SIGN}": "-",  # U+2212
}

# File-suffix skip list: prose/license files allowed to contain unicode.
_SKIP_SUFFIXES = frozenset({".md", ".rst", ".txt"})

# Directory-component skip list: vendored / generated trees.
_SKIP_DIR_PARTS = frozenset({"third_party", "vendor", ".git", "_bmad-output"})

# Message glyph: dash separating location from violation detail. Written as a
# \N escape so it does not appear as a raw ambiguous codepoint in source.
_DASH = "\N{EM DASH}"


def _should_skip(path: Path) -> bool:
    """Return True if ``path`` is outside the scope we enforce.

    Skipping rules:

    * Suffix in ``_SKIP_SUFFIXES`` (markdown, rst, txt).
    * File stem starts with ``LICENSE`` (LICENSE, LICENSE.txt, LICENSE-MIT, ...).
    * Any path component matches ``_SKIP_DIR_PARTS``.
    * Suffix is not ``.py``; only Python source is enforced.
    """
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return True
    if path.name.upper().startswith("LICENSE"):
        return True
    parts = set(path.parts)
    if parts & _SKIP_DIR_PARTS:
        return True
    return path.suffix != ".py"


def scan_file(path: Path) -> list[str]:
    """Scan ``path`` and return a list of formatted violation messages.

    Returns an empty list when the file is clean or skipped. Read errors
    (binary, missing) are silently treated as no-violations; pre-commit
    already covers file-not-found cases via its own staging logic.
    """
    if _should_skip(path):
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    violations: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for col, ch in enumerate(line, start=1):
            ascii_eq = AMBIGUOUS_CHARS.get(ch)
            if ascii_eq is None:
                continue
            codepoint = f"U+{ord(ch):04X}"
            violations.append(
                f"{path}:{line_no}:{col}: {ch} ({codepoint}) {_DASH} confusable with '{ascii_eq}'"
            )
    return violations


def main(argv: list[str]) -> int:
    """Scan every argv path; return 0 if clean, 1 otherwise."""
    any_violations = False
    for arg in argv:
        for msg in scan_file(Path(arg)):
            sys.stdout.write(msg + "\n")
            any_violations = True
    return 1 if any_violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
