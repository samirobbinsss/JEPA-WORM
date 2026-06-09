"""Idempotently fetch the WBDB + OWMD anchor records to local disk.

Stories 9.5 / 9.6 grew each SPEC to 100 records; this script is the minimal
anchor-record baseline (2 WBDB + 1 OWMD) used by every Phase 0 headline
sweep. For the full 100-record subsets use ``wormjepa fetch zenodo-subset``.
Called from scripts/dev/remote_sweep.sh after the repo + venv are on the
remote pod.
"""
# ruff: noqa: T201

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

# Zenodo record id, destination root (relative to repo root).
JOBS: tuple[tuple[str, str], ...] = (
    ("1031550", "data/downloads/wormbehavior_db/1031550"),
    ("1029149", "data/downloads/wormbehavior_db/1029149"),
    ("1031550", "data/downloads/openworm_movement/1031550"),
    # OWMD mutant anchor 1033265 dropped in Story 9.6 (video-less off-food).
)


def main() -> None:
    for rec, dst in JOBS:
        out = Path(dst)
        out.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}") as r:
            data = json.load(r)
        skipped = 0
        fetched = 0
        for f in data["files"]:
            dest = out / f["key"]
            if dest.exists() and dest.stat().st_size == f["size"]:
                skipped += 1
                continue
            urllib.request.urlretrieve(f["links"]["self"], dest)
            fetched += 1
        print(f"  {rec} -> {dst}: fetched {fetched}, skipped {skipped}")


if __name__ == "__main__":
    main()
