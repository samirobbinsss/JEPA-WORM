"""Pre-commit hook: require a CHANGELOG `### Frozen-artifact changes` entry
whenever a commit modifies any file listed in pre-registration/MANIFEST.lock.

Algorithm:
1. Read MANIFEST.lock; collect the set of frozen file paths.
2. Compare the staged file set to the frozen set.
3. If any staged file is frozen:
   - require CHANGELOG.md to also be staged;
   - require the staged CHANGELOG diff to add content under
     ``### Frozen-artifact changes``.
4. Otherwise: silently allow the commit.

The hook is intentionally conservative: false positives (commit blocked when
it shouldn't be) are easier to recover from than false negatives (silent drift).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    cursor = Path.cwd()
    while cursor != cursor.parent:
        if (cursor / "pyproject.toml").is_file():
            return cursor
        cursor = cursor.parent
    return Path.cwd()


def _staged_files() -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _is_initial_lock_commit() -> bool:
    """Return True if MANIFEST.lock is being added (not modified) in this commit.

    Initial-lock commits have no prior frozen state, so the
    ``### Frozen-artifact changes`` requirement does not apply.
    """
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-status"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        status, path = parts
        if path.strip() == "pre-registration/MANIFEST.lock" and status.startswith("A"):
            return True
    return False


def _frozen_paths(manifest_path: Path) -> set[str]:
    """Read MANIFEST.lock and return the set of project-relative frozen file paths.

    Dataset entries are skipped — they don't correspond to working-tree files.
    """
    if not manifest_path.is_file():
        return set()
    # Avoid pulling pydantic for a hook. Use plain YAML.
    import yaml

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    paths: set[str] = set()
    for entry in data.get("artifacts", []):
        path = entry.get("path")
        if path:
            paths.add(path)
    return paths


def _changelog_diff_has_frozen_section_entry(changelog_path: Path) -> bool:
    """Return True if the staged diff for CHANGELOG.md adds lines under
    ``### Frozen-artifact changes``.

    We look for a hunk that contains a non-comment, non-header added line in
    the subsection. The check is heuristic — false positives on weird hunks
    are tolerable; false negatives are not.
    """
    if not changelog_path.is_file():
        return False
    result = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", str(changelog_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout:
        return False

    in_section = False
    found_added_line = False
    for line in result.stdout.splitlines():
        # Track section context via hunk content lines (no +/- prefix).
        if line.startswith("@@"):
            in_section = False
            continue
        # Hunk content lines: leading space, +, or -.
        if not line:
            continue
        marker = line[0]
        body = line[1:] if marker in (" ", "+", "-") else line

        if body.strip().startswith("### Frozen-artifact changes"):
            in_section = True
            continue
        if body.strip().startswith("### ") or body.strip().startswith("## "):
            # Entering a different section.
            if in_section:
                in_section = False
            continue

        if in_section and marker == "+":
            stripped = body.strip()
            # Skip blank lines and HTML comment lines.
            if not stripped or stripped.startswith("<!--") or stripped.startswith("-->"):
                continue
            found_added_line = True
            break
    return found_added_line


def main() -> int:
    root = _project_root()
    manifest_path = root / "pre-registration" / "MANIFEST.lock"
    frozen = _frozen_paths(manifest_path)
    if not frozen:
        return 0

    # Initial-lock commit: no prior frozen state → nothing to log a "change" from.
    if _is_initial_lock_commit():
        return 0

    staged = _staged_files()
    frozen_staged = staged & frozen
    if not frozen_staged:
        return 0

    changelog_path = root / "CHANGELOG.md"
    if "CHANGELOG.md" not in staged:
        sys.stderr.write(
            "Commit modifies frozen artifacts but CHANGELOG.md is not staged:\n"
            + "\n".join(f"  - {p}" for p in sorted(frozen_staged))
            + "\nAdd a `### Frozen-artifact changes` entry to CHANGELOG.md and re-stage.\n"
        )
        return 1

    if not _changelog_diff_has_frozen_section_entry(changelog_path):
        sys.stderr.write(
            "Commit modifies frozen artifacts but CHANGELOG.md's "
            "`### Frozen-artifact changes` subsection has no new content:\n"
            + "\n".join(f"  - {p}" for p in sorted(frozen_staged))
            + "\nAdd a Keep-a-Changelog-style entry under that subsection (artifact "
            "name, prior SHA, new SHA, justification, date) and re-stage.\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
