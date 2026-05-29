"""Unit tests for ``wormjepa.configs``."""

from __future__ import annotations

import pytest
from pydantic import ConfigDict

from wormjepa import ConfigSchemaError
from wormjepa.configs import CURRENT_SCHEMA_VERSION, WormJEPAConfig, load_config
from wormjepa.configs.migrations import MIGRATIONS, migrate


class _ToyConfig(WormJEPAConfig):
    """Minimal subclass used to exercise the loader."""

    name: str
    factor: float = 1.0


def test_load_config_accepts_current_schema_version() -> None:
    cfg = load_config({"schema_version": CURRENT_SCHEMA_VERSION, "name": "headline"}, _ToyConfig)
    assert cfg.name == "headline"
    assert cfg.factor == 1.0
    assert cfg.schema_version == CURRENT_SCHEMA_VERSION


def test_load_config_missing_schema_version_raises() -> None:
    with pytest.raises(ConfigSchemaError, match="schema_version"):
        load_config({"name": "headline"}, _ToyConfig)


def test_load_config_schema_version_must_be_int() -> None:
    with pytest.raises(ConfigSchemaError, match="must be an int"):
        load_config({"schema_version": "1", "name": "headline"}, _ToyConfig)


def test_load_config_unknown_field_raises() -> None:
    with pytest.raises(ConfigSchemaError):
        load_config(
            {"schema_version": CURRENT_SCHEMA_VERSION, "name": "h", "unknown": 42},
            _ToyConfig,
        )


def test_load_config_newer_schema_version_raises() -> None:
    with pytest.raises(ConfigSchemaError, match="only supports"):
        load_config({"schema_version": CURRENT_SCHEMA_VERSION + 1, "name": "h"}, _ToyConfig)


def test_load_config_type_mismatch_raises() -> None:
    with pytest.raises(ConfigSchemaError):
        load_config(
            {"schema_version": CURRENT_SCHEMA_VERSION, "name": "h", "factor": "notanumber"},
            _ToyConfig,
        )


def test_load_config_strict_mode_rejects_int_for_string() -> None:
    """Strict mode: integers passed where strings are required are rejected."""
    with pytest.raises(ConfigSchemaError):
        load_config({"schema_version": CURRENT_SCHEMA_VERSION, "name": 42}, _ToyConfig)


def test_loaded_config_is_frozen() -> None:
    cfg = load_config({"schema_version": CURRENT_SCHEMA_VERSION, "name": "h"}, _ToyConfig)
    with pytest.raises(ValueError, match="frozen"):
        cfg.name = "changed"  # type: ignore[misc]


def test_migrate_identity_returns_input() -> None:
    """Migration from version N to version N is a no-op."""
    data = {"schema_version": 1, "x": 2}
    assert migrate(data, 1, 1) == data


def test_migrate_unregistered_path_raises() -> None:
    """Asking for an unregistered migration is a hard failure."""
    with pytest.raises(ConfigSchemaError, match="No migration registered"):
        migrate({"schema_version": 99}, 99, 100)


def test_load_config_with_older_version_and_registered_migration() -> None:
    """End-to-end: an older config flows through a registered migration before validation."""

    class _BumpedConfig(WormJEPAConfig):
        model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
        renamed_field: str

    def _migrate_v0_to_current(data: dict[str, object]) -> dict[str, object]:
        # Simulate a rename: old key "legacy_field" → new key "renamed_field".
        new_data = dict(data)
        if "legacy_field" in new_data:
            new_data["renamed_field"] = new_data.pop("legacy_field")
        new_data["schema_version"] = CURRENT_SCHEMA_VERSION
        return new_data

    MIGRATIONS[(0, CURRENT_SCHEMA_VERSION)] = _migrate_v0_to_current
    try:
        cfg = load_config({"schema_version": 0, "legacy_field": "value"}, _BumpedConfig)
        assert cfg.renamed_field == "value"
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION
    finally:
        del MIGRATIONS[(0, CURRENT_SCHEMA_VERSION)]


def test_load_config_with_older_version_and_no_migration_raises() -> None:
    """Older version without a registered migration fails loudly."""
    with pytest.raises(ConfigSchemaError, match="No migration registered"):
        load_config({"schema_version": 0, "name": "h"}, _ToyConfig)
