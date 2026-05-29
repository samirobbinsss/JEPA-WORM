"""Writer for the ``results/<run-id>/`` directory contract.

Every reportable run gets a results directory that contains exactly the files
documented in the architecture's results contract. Untracked, ad-hoc, or
typo'd filenames are rejected at write time so the contract cannot drift.

Contract (top-level files):

==========================  ===========================================
File                        Purpose
==========================  ===========================================
``config.yaml``             Exact config used (Epic 3+)
``metrics.json``            All metric values + worm-level CIs (Epic 6)
``compute.json``            GPU/CUDA/wall-time/peak-mem (Epic 7 Story 7.1)
``seed.txt``                Random seeds used (Epic 5 Story 5.9)
``manifest_at_run.lock``    Copy of MANIFEST.lock at run start (Epic 4 Story 4.9)
``report.md``               Outcome-aware report (Epic 7 Story 7.3+)
``log.jsonl``               JSON Lines training/eval log
==========================  ===========================================

Plus two optional subdirectories:

- ``checkpoints/`` — gitignored; model checkpoints
- ``bootstrap_samples.parquet`` — gitignored; large intermediate samples
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from wormjepa import WormJEPAError
from wormjepa.paths import project_root

ALLOWED_FILES: frozenset[str] = frozenset(
    {
        "config.yaml",
        "metrics.json",
        # Story 8.12c-skeleton: gate-evaluation orchestrator output. Named
        # distinctly from `metrics.json` (which holds training-time losses)
        # so the two stay independently inspectable.
        "metrics_eval.json",
        "compute.json",
        "seed.txt",
        "manifest_at_run.lock",
        "report.md",
        "log.jsonl",
        # bootstrap_samples is a single parquet file at top level (gitignored).
        "bootstrap_samples.parquet",
    }
)
"""Filenames the ``results/<run-id>/`` contract allows at the top level."""

ALLOWED_SUBDIRS: frozenset[str] = frozenset({"checkpoints"})
"""Subdirectories the contract allows. Contents inside these are unrestricted."""


class ResultsContractViolation(WormJEPAError):  # noqa: N818  # naming convention: discipline breach
    """Raised on any attempt to write a file outside the ``results/<run-id>/`` contract."""


class ResultsWriter:
    """Manages a single ``results/<run-id>/`` directory.

    Lifecycle:

    1. Construct with a ``run_id`` (and optionally a custom ``results_root``).
    2. Call :meth:`initialize` once to create the directory plus placeholder
       required files (``seed.txt``, ``compute.json``, ``log.jsonl``).
    3. Use :meth:`write_text` and :meth:`write_bytes` for every subsequent
       file write — both reject filenames outside :data:`ALLOWED_FILES`.

    Direct filesystem writes to the results directory are an anti-pattern and
    will be flagged by integration tests (the smoke test verifies the directory
    shape after a run).
    """

    REQUIRED_INITIAL_FILES: ClassVar[tuple[str, ...]] = (
        "seed.txt",
        "compute.json",
        "log.jsonl",
    )
    """Files :meth:`initialize` creates with placeholder content.

    Other contract files (``config.yaml``, ``metrics.json``, ``report.md``,
    ``manifest_at_run.lock``) are written later by their respective modules.
    """

    def __init__(self, run_id: str, results_root: Path | None = None) -> None:
        self.run_id = run_id
        if results_root is None:
            results_root = project_root() / "results"
        self.results_root: Path = results_root
        self.path: Path = results_root / run_id

    def initialize(self) -> Path:
        """Create the results directory and seed the placeholder required files.

        Returns:
            The absolute path to the new ``results/<run-id>/`` directory.

        Raises:
            ResultsContractViolation: If the directory already exists.
        """
        if self.path.exists():
            msg = f"results directory already exists: {self.path}"
            raise ResultsContractViolation(msg)
        self.path.mkdir(parents=True, exist_ok=False)

        # Seed placeholder content. Production callers will overwrite these.
        (self.path / "seed.txt").write_text("placeholder\n", encoding="utf-8")
        (self.path / "compute.json").write_text("{}\n", encoding="utf-8")
        (self.path / "log.jsonl").write_text("", encoding="utf-8")
        return self.path

    def _check_relative(self, relative: str) -> None:
        """Reject writes to filenames outside the contract."""
        # Strip leading "./" if a caller passes it.
        relative = relative.removeprefix("./")
        if "/" in relative:
            head, _ = relative.split("/", 1)
            if head not in ALLOWED_SUBDIRS:
                msg = (
                    f"results contract forbids writes outside ALLOWED_SUBDIRS; "
                    f"got prefix {head!r} for {relative!r}. Allowed subdirs: "
                    f"{sorted(ALLOWED_SUBDIRS)}."
                )
                raise ResultsContractViolation(msg)
            return
        if relative not in ALLOWED_FILES:
            msg = (
                f"results contract forbids file {relative!r}. "
                f"Allowed files: {sorted(ALLOWED_FILES)}."
            )
            raise ResultsContractViolation(msg)

    def write_text(self, relative: str, content: str) -> Path:
        """Write text to ``self.path / relative``, enforcing the contract.

        Args:
            relative: Filename relative to the results directory. Must be in
                :data:`ALLOWED_FILES` or under an :data:`ALLOWED_SUBDIRS` prefix.
            content: Text content.

        Returns:
            The absolute path to the written file.

        Raises:
            ResultsContractViolation: If ``relative`` is outside the contract.
        """
        self._check_relative(relative)
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def write_bytes(self, relative: str, content: bytes) -> Path:
        """Write bytes to ``self.path / relative``, enforcing the contract.

        See :meth:`write_text` for parameter and error documentation.
        """
        self._check_relative(relative)
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target
