"""Regenerate the committed loader smoke-test fixtures.

Outputs five minimal real-format HDF5 files under ``tests/fixtures/``:

- ``flavell_2023/worm_anchor.h5``                — Flavell schema (video/neural/behavior)
- ``wormbehavior_db/<record_id>/experiment_<id>.hdf5``  — Schafer schema, both anchor records
- ``openworm_movement/<record_id>/experiment_<id>.hdf5`` — Schafer schema, both anchor records

Run from the repo root:

    uv run python scripts/dev/regenerate_loader_fixtures.py

Fixtures are deterministic (fixed seeds) so re-running produces bit-identical
bytes; verify with ``git diff`` and commit only when the underlying loader
contract changes.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def write_flavell(path: Path, seed: int = 20260515) -> None:
    rng = np.random.default_rng(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "neural_activity",
            data=rng.standard_normal((16, 5)).astype(np.float32),
        )
        f.create_dataset(
            "behavioral_state",
            data=rng.integers(0, 4, size=16, dtype=np.int32),
        )
        vid = f.create_dataset(
            "video",
            data=rng.integers(0, 255, size=(16, 16, 16), dtype=np.uint8),
        )
        vid.attrs["frame_rate"] = 3.0


def write_schafer(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        mask = f.create_dataset(
            "mask",
            data=rng.integers(0, 255, (16, 16, 16), dtype=np.uint8),
        )
        mask.attrs["fps"] = 30.0
        f.create_dataset(
            "skeleton",
            data=rng.standard_normal((16, 49, 2)).astype(np.float32),
        )


def main() -> None:
    write_flavell(FIXTURE_ROOT / "flavell_2023" / "worm_anchor.h5")
    write_schafer(
        FIXTURE_ROOT / "wormbehavior_db" / "1031550" / "experiment_1031550.hdf5",
        20260515,
    )
    write_schafer(
        FIXTURE_ROOT / "wormbehavior_db" / "1029149" / "experiment_1029149.hdf5",
        20260516,
    )
    write_schafer(
        FIXTURE_ROOT / "openworm_movement" / "1031550" / "experiment_1031550.hdf5",
        20260517,
    )
    write_schafer(
        FIXTURE_ROOT / "openworm_movement" / "1033265" / "experiment_1033265.hdf5",
        20260518,
    )
    for f in sorted(FIXTURE_ROOT.rglob("*.h5")) + sorted(FIXTURE_ROOT.rglob("*.hdf5")):
        print(f"{f.relative_to(FIXTURE_ROOT.parent.parent)}: {f.stat().st_size} bytes")  # noqa: T201


if __name__ == "__main__":
    main()
