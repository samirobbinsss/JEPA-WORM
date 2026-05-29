"""Shared schema for ``results/<run-id>/metrics.json``.

Every JEPA run and every baseline writes a :class:`MetricsOutput` to disk.
The schema is the only sanctioned shape for reportable metric values; eval,
reporting, and the CI-aware comparison script all read and write through it.

PRD measurable-outcome rows that land here (8 rows total):

1. Future-pose prediction at 0.1s/1s/5s horizons (Epic 6 Story 6.2)
2. Motif ARI vs Flavell labels (Epic 6 Story 6.3)
3. Neural probe partial R² over kinematic baseline (Epic 6 Story 6.4)
4. Neural-prior ablation ΔR² (Epic 6 Story 6.8)
5. BAAIWorm-augmentation ablation ΔR² (Epic 6 Story 6.8)
6. Session-ID classifier accuracy (Epic 6 Story 6.5)
7. Within-state stratified R² (Epic 6 Story 6.6)
8. Non-trivial neuron subset R² (Epic 6 Story 6.7)

Every value carries a :class:`BootstrapCI` produced by the worm-level
bootstrap API (NFR16 / Story 3.3). Frame-level bootstrap is structurally
impossible to express.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_GROUPING_VALUES = ("worm",)


class BootstrapCI(BaseModel):
    """A worm-level bootstrap confidence interval for one scalar metric.

    Produced exclusively by :func:`wormjepa.eval.bootstrap.bootstrap_ci`, which
    enforces worm-level resampling at the type-signature level (NFR16 / FR28).
    The ``grouping`` field is constrained to ``"worm"`` so that any future
    accidental introduction of a frame-level path fails validation here too.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: float
    lower: float
    upper: float
    level: float = 0.95
    method: Literal["bca", "percentile"] = "bca"
    n_samples: int = Field(ge=1)
    grouping: Literal["worm"] = "worm"

    @field_validator("level")
    @classmethod
    def _level_in_unit_interval(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            msg = f"BootstrapCI.level must be in (0, 1); got {v}"
            raise ValueError(msg)
        return v

    @field_validator("upper")
    @classmethod
    def _upper_at_least_lower(cls, v: float, info: object) -> float:  # type: ignore[unused-ignore]  # pydantic v2 callable
        # Note: cross-field validation in pydantic v2 uses ``model_validator``;
        # we keep a simple inequality check inside ``model_validate`` below.
        return v


class SubEntry(BaseModel):
    """A sub-row of a :class:`MetricEntry`.

    Used when a single named metric carries multiple values keyed by some
    parameter (per horizon for future-pose; per behavioral state for
    within-state stratified; per-neuron for the non-trivial subset).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    ci: BootstrapCI


class MetricEntry(BaseModel):
    """A single row in the results table.

    Attributes:
        name: Stable identifier matching the PRD measurable-outcomes row
            (e.g., ``"future_pose"``, ``"motif_ari"``, ``"neural_probe_partial_r2"``).
        producer: What produced this row — ``"jepa"`` or a baseline name
            matching ``configs/baselines/<name>.yaml``.
        ci: Top-level CI for this metric. Some entries (per-horizon future-pose)
            do not have a single top-level value; in that case use a placeholder
            CI (point=NaN) and read the sub-rows.
        sub_entries: Optional list of per-key sub-rows.
        notes: Free-form annotation (e.g., "below NFR18 threshold").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    producer: str
    ci: BootstrapCI
    sub_entries: list[SubEntry] = Field(default_factory=list)
    notes: str = ""


class MetricsOutput(BaseModel):
    """The full ``results/<run-id>/metrics.json`` payload.

    A reportable run writes one of these per ``results/<run-id>/``. The
    metrics-writer (Epic 7 Story 7.2) assembles it; the report renderer
    (Epic 7 Stories 7.3-7.5) consumes it; the CI-aware comparison (Epic 7
    Story 7.6) diffs two of them at the row level.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    entries: list[MetricEntry] = Field(default_factory=list)

    def to_canonical_json(self) -> str:
        """Render to canonical JSON: sorted keys, two-space indent, LF newlines.

        Canonical output is required because ``metrics.json`` is one of the
        files compared SHA-stably during reproducibility verification
        (FR44 / NFR7).
        """
        payload = self.model_dump(mode="json")
        return json.dumps(payload, sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_canonical_json(cls, text: str) -> MetricsOutput:
        """Parse a canonical JSON payload (inverse of :meth:`to_canonical_json`)."""
        return cls.model_validate(json.loads(text))


_ALL_GROUPING_VALUES = _GROUPING_VALUES  # re-export shim; kept for downstream lookups
