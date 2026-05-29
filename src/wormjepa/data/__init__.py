"""Data ingestion for JEPA-WORM.

All five public dataset loaders return the same :class:`DatasetSample` named
tuple defined in :mod:`wormjepa.data.contract`. Worm-level identifiers
(``worm_id``, ``session_id``) propagate from loader through training and
evaluation to the bootstrap-CI API — never dropped.
"""

from wormjepa.data.contract import (
    DatasetSample,
    SessionID,
    SourceDataset,
    WormID,
)

__all__ = [
    "DatasetSample",
    "SessionID",
    "SourceDataset",
    "WormID",
]
