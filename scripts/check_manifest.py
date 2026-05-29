"""Pre-commit hook: verify pre-registration/MANIFEST.lock against the working tree.

Wraps :func:`wormjepa.manifest.lock_check.verify_manifest` with graceful
no-manifest tolerance (the lock does not exist until Story 4.8). Once the
lock is in place, this hook fails any commit that introduces drift.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    # Avoid importing wormjepa eagerly — keeps the hook fast when the manifest
    # is absent (no torch import overhead).
    project_root = Path.cwd()
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").is_file():
            break
        project_root = project_root.parent
    manifest_path = project_root / "pre-registration" / "MANIFEST.lock"
    if not manifest_path.is_file():
        # Lock does not exist yet (Story 4.8 hasn't run). Skip silently.
        return 0

    from wormjepa import PreRegistrationViolation
    from wormjepa.manifest.lock_check import verify_manifest

    try:
        result = verify_manifest(manifest_path)
    except PreRegistrationViolation as exc:
        sys.stderr.write(f"pre-registration check failed:\n{exc}\n")
        return 1
    sys.stdout.write(f"pre-registration verified: {result.verified} artifacts ({manifest_path})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
