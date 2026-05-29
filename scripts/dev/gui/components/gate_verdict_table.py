"""C5 — GateVerdictTable component (Phase 3).

Two variants per the UX spec §"C5 — `GateVerdictTable`":

- ``render_gate_verdict_table(run_dir, run_id)`` — single-run variant.
- ``render_gate_verdict_diff(rows)`` — cross-run-diff variant: rows are gates,
  columns are runs; rows where any pair of runs disagrees on verdict are
  flagged in the caption.

Gate verdicts come from :func:`wormjepa.eval.gates.evaluate_gates`, which is
the single source of truth for the four implemented gates. The PRD lists
nine gates; the remaining five surface as ``pending`` rows here (data is
not yet wired through the metrics writer for them) so the table always
shows the full pre-registered set — Sami can see at a glance which gates
have producers and which are still on the to-do list.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from scripts.dev.gui import cache, theme
from wormjepa.eval.gates import GateStatus, evaluate_gates
from wormjepa.eval.metrics_schema import MetricEntry, MetricsOutput

logger = logging.getLogger(__name__)

# The full PRD set of nine pre-registered gates. The four in `gates.py`
# evaluate live; the remaining five surface as "pending" rows until their
# producers land in the metrics writer.
_FULL_GATE_ORDER: tuple[str, ...] = (
    "kill_criterion",
    "neural_probe_partial_r2",
    "neural_prior_ablation",
    "session_id_at_chance",
    "motif_ari",
    "baaiworm_augmentation",
    "within_state_stratified",
    "non_trivial_neuron_subset",
    "future_pose_at_horizons",
)

# Map gate name → display label + canonical MetricEntry name (best effort).
_GATE_DISPLAY: dict[str, tuple[str, str | None, str | None]] = {
    # gate_name: (label, metric_entry_name, threshold_text)
    "kill_criterion": ("Future-pose 1s vs Transformer", "future_pose", "JEPA < Transformer"),
    "neural_probe_partial_r2": ("Neural probe partial R²", "neural_probe_partial_r2", "≥ 0.05"),
    "neural_prior_ablation": (
        "Neural prior ablation ΔR²",
        "neural_prior_ablation_delta_r2",
        "≥ 0.02",
    ),
    "session_id_at_chance": ("Session-ID at chance", "session_id_classifier", "CI ∋ chance"),
    "motif_ari": ("Motif ARI vs Flavell", "motif_ari", "—"),
    "baaiworm_augmentation": (
        "BAAIWorm augmentation ΔR²",
        "baaiworm_augmentation_delta_r2",
        "—",
    ),
    "within_state_stratified": (
        "Within-state stratified R²",
        "within_state_stratified_r2",
        "—",
    ),
    "non_trivial_neuron_subset": (
        "Non-trivial neuron subset R²",
        "non_trivial_neuron_subset_r2",
        "—",
    ),
    "future_pose_at_horizons": ("Future-pose multi-horizon", "future_pose", "—"),
}

_VERDICT_ICON: dict[str, str] = {
    "cleared": "✓",
    "fired": "✗",
    "pending": "—",
    "warning": "⚠",
}


def _find_entry(metrics: MetricsOutput, name: str) -> MetricEntry | None:
    for entry in metrics.entries:
        if entry.name == name:
            return entry
    return None


def _row_for_gate(
    gate_name: str,
    status: GateStatus | None,
    metrics: MetricsOutput | None,
    source_run_id: str,
) -> dict[str, Any]:
    """Build a single dataframe row for one gate.

    A row is "pending" if either the GateStatus has it pending, the gate
    is not in the implemented four (so it stays pending until the producer
    lands), or metrics.json is missing entirely.
    """
    label, entry_name, threshold = _GATE_DISPLAY[gate_name]
    verdict: str
    if status is not None and gate_name in status.gates:
        verdict = status.gates[gate_name]  # type: ignore[assignment]
    else:
        verdict = "pending"

    point: str = "—"
    ci_text: str = "—"
    if metrics is not None and entry_name is not None:
        entry = _find_entry(metrics, entry_name)
        if entry is not None:
            point = f"{entry.ci.point:.4g}"
            ci_text = f"[{entry.ci.lower:.4g}, {entry.ci.upper:.4g}]"

    return {
        "gate": label,
        "verdict": f"{_VERDICT_ICON.get(verdict, '?')} {verdict}",
        "point": point,
        "CI": ci_text,
        "threshold": threshold or "—",
        "source": source_run_id,
    }


def _load_metrics_and_status(
    run_dir: Path,
) -> tuple[MetricsOutput | None, GateStatus | None]:
    metrics = cache.metrics(run_dir)
    if metrics is None:
        return None, None
    try:
        return metrics, evaluate_gates(metrics)
    except (ValueError, KeyError, AttributeError) as exc:
        logger.warning("evaluate_gates failed on %s: %s", run_dir, exc)
        return metrics, None


def render_gate_verdict_table(run_dir: Path, run_id: str) -> None:
    """Render the single-run gate-verdict table.

    Missing ``metrics.json`` → render an info banner with the full set of
    gates as ``pending`` rows; this is the most common state during early
    runs and the table should still communicate "9 gates exist."
    """
    metrics, status = _load_metrics_and_status(run_dir)
    if metrics is None:
        st.info(f"`{run_dir.name}/metrics.json` not present yet — all gates pending.")

    rows = [_row_for_gate(name, status, metrics, run_id) for name in _FULL_GATE_ORDER]

    st.markdown("##### Gate verdicts")
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "gate": st.column_config.TextColumn("gate", help="Pre-registered gate name"),
            "verdict": st.column_config.TextColumn("verdict"),
            "point": st.column_config.TextColumn("point"),
            "CI": st.column_config.TextColumn("95% CI [lower, upper]"),
            "threshold": st.column_config.TextColumn("threshold"),
            "source": st.column_config.TextColumn("run-id"),
        },
    )
    if status is not None:
        st.caption(f"Overall outcome: **{status.outcome}** (from {run_id}).")
    else:
        st.caption(f"No gate evaluation available for {run_id}.")


def render_gate_verdict_diff(rows: list[tuple[str, Path]]) -> None:
    """Render a side-by-side gate-diff across two-or-more runs.

    Args:
        rows: ``(run_id, run_dir)`` pairs to compare.
    """
    if len(rows) < 2:
        st.info("Cross-run gate diff requires at least two runs.")
        return

    statuses: list[tuple[str, MetricsOutput | None, GateStatus | None]] = []
    for run_id, run_dir in rows:
        metrics, status = _load_metrics_and_status(run_dir)
        statuses.append((run_id, metrics, status))

    table_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for gate_name in _FULL_GATE_ORDER:
        label = _GATE_DISPLAY[gate_name][0]
        row: dict[str, Any] = {"gate": label}
        verdicts: set[str] = set()
        for run_id, _metrics, status in statuses:
            if status is not None and gate_name in status.gates:
                verdict = status.gates[gate_name]  # type: ignore[assignment]
            else:
                verdict = "pending"
            row[run_id] = f"{_VERDICT_ICON.get(verdict, '?')} {verdict}"
            verdicts.add(verdict)
        if len(verdicts) > 1 and verdicts != {"pending"}:
            mismatches.append(label)
        table_rows.append(row)

    st.markdown("##### Gate-verdict diff")
    st.dataframe(table_rows, use_container_width=True, hide_index=True)
    if mismatches:
        st.markdown(
            f"<span style='color: {theme.COLOR_WARNING}'>⚠ verdict disagreement on: "
            f"{', '.join(mismatches)}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("All gate verdicts agree across the selected runs.")
