"""JEPA-WORM Phase 0: self-supervised vision-only world model for C. elegans.

Public release readiness: this package is structured to support a private→public
visibility flip without scrubbing (per FR45). Do not commit secrets or local paths.

Error hierarchy: every domain error in the codebase subclasses ``WormJEPAError``.
The CLI top-level catches ``WormJEPAError`` and formats it for the user; non-domain
errors (programming bugs) propagate uncaught so they crash visibly.
"""

__version__ = "0.1.0"


class WormJEPAError(Exception):
    """Base class for every domain error in JEPA-WORM."""


class PreRegistrationViolation(WormJEPAError):  # noqa: N818  # spec'd name: discipline breach, not Error
    """Raised when a reportable computation would proceed against an invalid lock state.

    Examples: a frozen artifact's SHA does not match MANIFEST.lock, a frozen artifact
    is missing, or a reportable run is attempted before pre-registration is locked.
    """


class BootstrapGroupingError(WormJEPAError):
    """Raised when the bootstrap-CI API is called without valid worm-level grouping.

    Frame-level bootstrap is forbidden architecturally (NFR16 / FR28). Any failure
    to construct or pass a ``WormGrouping`` instance surfaces as this error.
    """


class ConfigSchemaError(WormJEPAError):
    """Raised when a YAML configuration fails pydantic schema validation.

    Triggered by missing ``schema_version``, unknown fields, type mismatches, or a
    ``schema_version`` newer than the current code's supported version.
    """


class DatasetIntegrityError(WormJEPAError):
    """Raised when a downloaded dataset's hash does not match the DOI-pinned manifest.

    Per FR7, dataset version drift fails loudly rather than silently producing
    different numbers.
    """


class ManifestLockError(WormJEPAError):
    """Raised when MANIFEST.lock cannot be read, written, or verified.

    Distinct from ``PreRegistrationViolation`` (which is about *content* mismatch);
    this covers I/O, parse, and schema-shape problems with the lock file itself.
    """
