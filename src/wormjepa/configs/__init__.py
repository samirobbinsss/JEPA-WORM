"""Configuration schema infrastructure for JEPA-WORM.

All hyperparameters live in YAML configs validated against pydantic models. No
hyperparameter is settable via command-line flags or hardcoded constants (FR12).
Schema failures are caught at config load, not at runtime (NFR12).
"""

from wormjepa.configs.models import (
    CURRENT_SCHEMA_VERSION,
    WormJEPAConfig,
    load_config,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "WormJEPAConfig",
    "load_config",
]
