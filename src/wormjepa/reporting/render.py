"""Render outcome-aware reports + CI-aware comparison (Stories 7.3-7.6 / 7.8)."""

from __future__ import annotations

import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from wormjepa.eval.gates import GateStatus
from wormjepa.eval.metrics_schema import MetricEntry, MetricsOutput
from wormjepa.reporting.template_selector import select_template


def render_report(
    metrics: MetricsOutput,
    gate_status: GateStatus,
    *,
    templates_dir: Path | None = None,
) -> str:
    """Render the outcome-aware report.md content (Story 7.3)."""
    template_path = select_template(gate_status, templates_dir=templates_dir)
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(disabled_extensions=("md", "j2")),
        keep_trailing_newline=True,
    )
    template = env.get_template(template_path.name)
    return template.render(
        run_id=metrics.run_id,
        entries=metrics.entries,
        outcome=gate_status.outcome,
        gates=gate_status.gates,
        notes=gate_status.notes,
    )


def compare_metrics(local: MetricsOutput, published: MetricsOutput) -> list[tuple[str, str, str]]:
    """CI-aware diff between two ``MetricsOutput`` (Story 7.6 / FR38).

    For each entry in ``published``, find the matching entry in ``local`` by
    ``(producer, name)``; check whether the local point estimate falls inside
    the published CI. Returns a list of ``(name, status, message)`` tuples
    where ``status`` is one of: ``"within_ci"``, ``"outside_ci"``, ``"missing"``.
    """

    def _key(entry: MetricEntry) -> tuple[str, str]:
        return entry.producer, entry.name

    local_lookup = {_key(e): e for e in local.entries}
    results: list[tuple[str, str, str]] = []
    for pub_entry in published.entries:
        key = _key(pub_entry)
        loc_entry = local_lookup.get(key)
        if loc_entry is None:
            results.append(
                (
                    f"{pub_entry.producer}/{pub_entry.name}",
                    "missing",
                    "not produced locally",
                )
            )
            continue
        loc_point = loc_entry.ci.point
        pub_lo, pub_hi = pub_entry.ci.lower, pub_entry.ci.upper
        if math.isnan(pub_lo) or math.isnan(pub_hi):
            # Published row has no scalar CI — pass on top-level, sub-rows handled below.
            status = "within_ci"
            msg = "top-level CI is NaN; check sub-rows"
        elif pub_lo <= loc_point <= pub_hi:
            status = "within_ci"
            msg = f"{loc_point:.3f} in [{pub_lo:.3f}, {pub_hi:.3f}]"
        else:
            status = "outside_ci"
            msg = f"{loc_point:.3f} not in [{pub_lo:.3f}, {pub_hi:.3f}]"
        results.append((f"{pub_entry.producer}/{pub_entry.name}", status, msg))
    return results
