"""Unit tests for ``wormjepa.eval.metrics_schema``."""

from __future__ import annotations

import math

import pytest

from wormjepa.eval import BootstrapCI, MetricEntry, MetricsOutput, SubEntry


def _ci(point: float = 0.1, lower: float = 0.0, upper: float = 0.2) -> BootstrapCI:
    return BootstrapCI(point=point, lower=lower, upper=upper, n_samples=1000)


def test_bootstrap_ci_basic_construction() -> None:
    ci = _ci()
    assert ci.point == 0.1
    assert ci.grouping == "worm"
    assert ci.method == "bca"
    assert ci.level == 0.95


def test_bootstrap_ci_rejects_level_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="level must be in"):
        BootstrapCI(point=0.0, lower=-0.1, upper=0.1, level=0.0, n_samples=1000)
    with pytest.raises(ValueError, match="level must be in"):
        BootstrapCI(point=0.0, lower=-0.1, upper=0.1, level=1.0, n_samples=1000)


def test_bootstrap_ci_rejects_zero_or_negative_samples() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        BootstrapCI(point=0.0, lower=-0.1, upper=0.1, n_samples=0)


def test_bootstrap_ci_grouping_is_worm_only() -> None:
    """The ``grouping`` field is Literal['worm'] — alternative values are rejected."""
    with pytest.raises(ValueError, match="grouping"):
        BootstrapCI.model_validate(
            {
                "point": 0.1,
                "lower": 0.0,
                "upper": 0.2,
                "n_samples": 1000,
                "grouping": "frame",
            }
        )


def test_bootstrap_ci_is_frozen() -> None:
    ci = _ci()
    with pytest.raises(ValueError, match="frozen"):
        ci.point = 99.0  # type: ignore[misc]


def test_metric_entry_with_sub_entries() -> None:
    entry = MetricEntry(
        name="future_pose",
        producer="kalman",
        ci=_ci(point=float("nan"), lower=float("nan"), upper=float("nan")),
        sub_entries=[
            SubEntry(key="0.1s", ci=_ci(point=2.0, lower=1.5, upper=2.5)),
            SubEntry(key="1.0s", ci=_ci(point=4.0, lower=3.5, upper=4.5)),
            SubEntry(key="5.0s", ci=_ci(point=9.0, lower=8.5, upper=9.5)),
        ],
    )
    assert len(entry.sub_entries) == 3
    assert entry.sub_entries[1].key == "1.0s"
    assert entry.sub_entries[1].ci.point == 4.0


def test_metric_entry_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        MetricEntry.model_validate(
            {
                "name": "x",
                "producer": "y",
                "ci": _ci().model_dump(),
                "unknown_field": 42,
            }
        )


def test_metrics_output_roundtrip_canonical_json() -> None:
    out = MetricsOutput(
        run_id="20260512T000000Z__abcdef12__test",
        entries=[
            MetricEntry(
                name="neural_probe_partial_r2",
                producer="jepa",
                ci=_ci(point=0.08, lower=0.06, upper=0.10),
            ),
            MetricEntry(
                name="motif_ari",
                producer="jepa",
                ci=_ci(point=0.40, lower=0.35, upper=0.45),
            ),
        ],
    )
    text = out.to_canonical_json()
    parsed = MetricsOutput.from_canonical_json(text)
    assert parsed == out


def test_canonical_json_has_sorted_keys_and_lf_endings() -> None:
    out = MetricsOutput(run_id="r", entries=[])
    text = out.to_canonical_json()
    # Sorted keys: "entries" comes before "run_id".
    assert text.index('"entries"') < text.index('"run_id"')
    # LF newline at end, no CRLF anywhere.
    assert text.endswith("\n")
    assert "\r\n" not in text


def test_metrics_output_is_frozen() -> None:
    out = MetricsOutput(run_id="r", entries=[])
    with pytest.raises(ValueError, match="frozen"):
        out.run_id = "other"  # type: ignore[misc]


def test_metric_entry_with_nan_top_level_ci() -> None:
    """Sub-row-only metrics (e.g., per-horizon future-pose) use NaN top-level CIs.

    The schema must accept NaN values — pydantic v2 does by default.
    """
    nan_ci = BootstrapCI(point=math.nan, lower=math.nan, upper=math.nan, n_samples=1)
    entry = MetricEntry(name="future_pose", producer="kalman", ci=nan_ci)
    assert math.isnan(entry.ci.point)
