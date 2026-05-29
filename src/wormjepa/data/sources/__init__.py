"""Per-dataset source specifications.

Each module under this package exports a single ``SPEC: DatasetSource`` constant
describing where to fetch the dataset, the expected SHA-256 of the payload, the
license, and the canonical citation. The download infrastructure in
:mod:`wormjepa.data.download` consumes these specs.

Specs are *not* frozen by MANIFEST.lock directly — instead, the DOI hash field
of the spec gets recorded in MANIFEST.lock under canonicalization
``doi_manifest`` (Epic 4). Changing a spec post-lock requires a CHANGELOG
``### Frozen-artifact changes`` entry.

Story 2.3 populates the real SPEC values for all five datasets. This skeleton
provides the contract.
"""

from wormjepa.data.sources.base import DatasetSource

__all__ = ["DatasetSource"]
