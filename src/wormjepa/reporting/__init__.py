"""Reporting infrastructure for JEPA-WORM.

This package is the only sanctioned producer of files under ``results/<run-id>/``.
Every reportable run writes through :class:`ResultsWriter`, which enforces the
documented filename contract — any attempt to write a file outside the contract
raises :class:`wormjepa.WormJEPAError`.
"""

from wormjepa.reporting.results_writer import (
    ALLOWED_FILES,
    ALLOWED_SUBDIRS,
    ResultsContractViolation,
    ResultsWriter,
)

__all__ = [
    "ALLOWED_FILES",
    "ALLOWED_SUBDIRS",
    "ResultsContractViolation",
    "ResultsWriter",
]
