"""Base pydantic config schema for JEPA-WORM YAML configurations.

Every config file the system reads must:
- include a top-level ``schema_version: <int>`` field;
- match the schema of a ``WormJEPAConfig`` subclass exactly (``extra='forbid'``);
- declare a ``schema_version`` no newer than this code's ``CURRENT_SCHEMA_VERSION``.

Missing version, unknown fields, type mismatches, and newer versions all surface
as :class:`wormjepa.ConfigSchemaError` (caught by the CLI top-level).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from wormjepa import ConfigSchemaError
from wormjepa.configs.migrations import migrate

if TYPE_CHECKING:
    from collections.abc import Mapping

CURRENT_SCHEMA_VERSION = 1
"""Highest config schema version this code understands.

Bumping this constant requires a corresponding migration function in
:mod:`wormjepa.configs.migrations`.
"""


class WormJEPAConfig(BaseModel):
    """Base class for every YAML config schema in JEPA-WORM.

    Subclasses define their own fields. The base contributes:

    - ``schema_version``: int — required; mismatched versions trigger migration or fail.
    - ``extra='forbid'``: unknown fields raise :class:`ConfigSchemaError` at load.
    - ``frozen=True``: configs are immutable after construction.

    Pydantic's strict mode ensures that, e.g., a string passed where an int is
    expected raises rather than silently coercing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: int


T = TypeVar("T", bound=WormJEPAConfig)


def load_config(data: Mapping[str, Any], model_cls: type[T]) -> T:
    """Validate and construct a config from a mapping (typically loaded from YAML).

    This function is the *only* sanctioned path from raw mapping data to a typed
    config. It rejects:

    - missing ``schema_version`` (the loader cannot guess the schema);
    - unknown fields (``extra='forbid'`` is the structural defense against drift);
    - a ``schema_version`` strictly newer than :data:`CURRENT_SCHEMA_VERSION`;
    - any pydantic validation error (type mismatch, wrong shape, etc.).

    If ``schema_version`` is older than the current version, ``migrate()`` runs
    first; if no migration is registered, :class:`ConfigSchemaError` is raised.

    Args:
        data: Raw mapping (e.g., the result of ``yaml.safe_load(path.read_text())``).
        model_cls: The expected pydantic subclass of :class:`WormJEPAConfig`.

    Returns:
        A validated instance of ``model_cls``.

    Raises:
        ConfigSchemaError: For any of the failure modes listed above.
    """
    if "schema_version" not in data:
        msg = (
            "Configuration is missing the required 'schema_version' field. "
            "Every YAML config must declare its schema version explicitly."
        )
        raise ConfigSchemaError(msg)

    version = data["schema_version"]
    if not isinstance(version, int):
        msg = f"'schema_version' must be an int; got {type(version).__name__}: {version!r}."
        raise ConfigSchemaError(msg)

    if version > CURRENT_SCHEMA_VERSION:
        msg = (
            f"Config declares schema_version={version} but this code only supports "
            f"versions up to {CURRENT_SCHEMA_VERSION}. Upgrade the package or pin "
            f"to a version that understands schema_version={version}."
        )
        raise ConfigSchemaError(msg)

    if version < CURRENT_SCHEMA_VERSION:
        data = migrate(dict(data), version, CURRENT_SCHEMA_VERSION)

    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ConfigSchemaError(str(exc)) from exc
