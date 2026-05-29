"""Config schema migration registry.

Migrations are explicit, named functions registered under
``MIGRATIONS[(from_version, to_version)]``. There is no implicit migration,
no monkey-patched coercion, no "best-effort" upgrade. If a config's
``schema_version`` is older than the current code's and no path of registered
migrations connects them, the load fails with :class:`ConfigSchemaError`.

This module is intentionally a skeleton in Phase 0 v0: no migrations have
been registered yet because no schema-breaking change has shipped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from wormjepa import ConfigSchemaError

MigrationFn: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]

MIGRATIONS: dict[tuple[int, int], MigrationFn] = {
    # Format: (from_version, to_version) -> function that takes the config dict
    # and returns the migrated dict. Add entries as schema-breaking changes ship.
    #
    # Example (commented; uncomment when first migration is needed):
    # (1, 2): migrate_v1_to_v2,
}


def migrate(config: dict[str, Any], from_version: int, to_version: int) -> dict[str, Any]:
    """Apply registered migrations to walk ``config`` from ``from_version`` to ``to_version``.

    Currently implements only direct (single-step) migrations: a migration from
    version N to version M must be registered explicitly under ``MIGRATIONS[(N, M)]``.
    Multi-step chains are out of scope until a second schema version exists.

    Args:
        config: The raw config dict (already known to have ``schema_version``).
        from_version: The config's declared version.
        to_version: The target version (typically :data:`CURRENT_SCHEMA_VERSION`).

    Returns:
        The migrated config dict.

    Raises:
        ConfigSchemaError: If no migration is registered for the requested path.
    """
    if from_version == to_version:
        return config

    fn = MIGRATIONS.get((from_version, to_version))
    if fn is None:
        msg = (
            f"No migration registered from schema_version={from_version} to "
            f"schema_version={to_version}. Either add an explicit migration "
            f"to wormjepa.configs.migrations.MIGRATIONS, or pin the config to "
            f"its declared version."
        )
        raise ConfigSchemaError(msg)

    return fn(config)
