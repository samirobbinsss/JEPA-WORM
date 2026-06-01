"""Phase 0 verdict dashboard — read-only Streamlit inspector.

Invocation:

    uv run streamlit run scripts/dev/gui_verdict_dashboard.py
    uv run streamlit run scripts/dev/gui_verdict_dashboard.py -- /path/to/results/

This is a standalone Streamlit app separate from the dev-loop GUI under
``scripts/dev/gui/`` (the live-tail run-table cockpit). It targets the
post-eval audit workflow: open one or more ``results/<run-id>/`` dirs,
inspect their ``metrics_eval.json`` payloads (produced by
``wormjepa eval``), and surface every signal a Phase 0 gate-verdict
review needs in one screen:

1. Run selector — pick any run with a ``metrics_eval.json`` on disk.
2. Outcome banner — render the recomputed :class:`GateStatus.outcome`
   (cleared / kill_criterion_fired / reframed) with a status colour.
3. Per-gate verdict table — parsed from the chosen run's ``STATUS.md``
   ``## Gate verdicts`` writer-owned section.
4. MetricEntry inspector — selectable list of probe outputs; shows
   name, producer, point estimate, CI bounds, notes (newline-preserved).
5. Cross-seed sweep panel — when ≥ 2 runs are multi-selected, renders
   the per-gate consensus + spread (mirrors ``wormjepa eval --run a
   --run b`` console output).
6. Holm correction pre-render — parses ``gate_status.notes`` for
   ``Holm correction at alpha=…`` lines and surfaces the per-gate
   p / holm_adj_p / reject_null in a small table.

The dashboard is strictly read-only against ``results/`` and STATUS.md.
It never writes to disk or mutates session state outside Streamlit's
default rerun semantics.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Streamlit launches this script as ``__main__``; make the project root
# importable so ``from wormjepa.eval...`` and ``from scripts.dev.gui...``
# both resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from scripts.dev.gui import theme  # noqa: E402
from wormjepa.eval.gates import evaluate_gates  # noqa: E402
from wormjepa.eval.metrics_schema import MetricEntry, MetricsOutput  # noqa: E402

logger = logging.getLogger(__name__)


# Verdict → (icon, colour) for the per-gate table cell renderer. Mirrors
# the dev-loop GUI's GateVerdictTable component so the eye-training carries
# across the two screens.
_VERDICT_ICON: dict[str, str] = {
    "cleared": "✓",
    "fired": "✗",
    "pending": "—",
    "warning": "⚠",
    "split": "≠",
}

# Outcome category → banner colour. Picks from theme.COLOR_HEALTHY /
# COLOR_CRITICAL / COLOR_WARNING for the three categories the gate
# evaluator can emit.
_OUTCOME_COLOR: dict[str, str] = {
    "cleared": theme.COLOR_HEALTHY,
    "kill_criterion_fired": theme.COLOR_CRITICAL,
    "reframed": theme.COLOR_WARNING,
}

_HOLM_HEADER_RE = re.compile(r"Holm correction at alpha=([0-9.]+)")
_HOLM_GATE_RE = re.compile(
    r"^\s*(?P<gate>[a-z_]+):\s*p=(?P<p>[0-9.eE+-]+),\s*"
    r"holm_adj_p=(?P<padj>[0-9.eE+-]+),\s*reject_null=(?P<rej>True|False)\s*$"
)


# ---------------------------------------------------------------------------
# Loaders — strictly read-only against results/<run-id>/.
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class StatusGateRow:
    """One parsed row from STATUS.md ``## Gate verdicts``.

    The writer-owned table is ``| Gate | Verdict |``; some hand-curated
    sections (the ``## Gate states`` table above the writer-owned one) also
    carry a ``Source run-id`` column. We tolerate both shapes — extra cells
    land in ``source_run_id`` when present.
    """

    gate: str
    verdict: str
    source_run_id: str | None


def list_runs_with_metrics_eval(results_root: Path) -> list[str]:
    """Return run-ids whose directory contains a ``metrics_eval.json``.

    Sorted by mtime descending so the most-recently-evaluated run lands
    at the top of the dropdown. Hidden dirs are skipped.
    """
    if not results_root.is_dir():
        return []
    candidates: list[tuple[float, str]] = []
    for child in results_root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        metrics_eval = child / "metrics_eval.json"
        if metrics_eval.is_file():
            candidates.append((metrics_eval.stat().st_mtime, child.name))
    candidates.sort(key=lambda kv: kv[0], reverse=True)
    return [name for _, name in candidates]


def load_metrics_eval(run_dir: Path) -> MetricsOutput | None:
    """Parse ``<run_dir>/metrics_eval.json`` into :class:`MetricsOutput`.

    Returns ``None`` if the file is missing or fails canonical-JSON
    validation (the dashboard renders a friendly "missing/unreadable"
    hint rather than crashing).
    """
    path = run_dir / "metrics_eval.json"
    if not path.is_file():
        return None
    try:
        return MetricsOutput.from_canonical_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("metrics_eval.json unreadable at %s: %s", path, exc)
        return None


def parse_status_gate_verdicts(status_path: Path) -> list[StatusGateRow]:
    """Parse the ``## Gate verdicts`` (or ``## Gate states``) table from STATUS.md.

    Returns the rows as parsed; empty when no recognisable gate-verdict
    table is present. Accepts both the writer-owned 2-column shape
    (``| Gate | Verdict |``) and the hand-curated 3-column shape
    (``| Gate | Status | Source run-id |``) used by Sami's milestone
    table in STATUS.md.

    The function does NOT distinguish "table missing" from "table empty"
    — both surface as ``[]`` and the dashboard falls back to the
    recomputed verdicts from :func:`evaluate_gates`.
    """
    if not status_path.is_file():
        return []
    text = status_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[StatusGateRow] = []
    in_table = False
    in_target_section = False
    for line in lines:
        if line.startswith("## "):
            heading = line.strip()
            in_target_section = heading in ("## Gate verdicts", "## Gate states")
            in_table = False
            continue
        if not in_target_section:
            continue
        # Heuristic table-row detection: starts and ends with `|`.
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        # Skip header + separator rows.
        if not cells:
            continue
        if all(set(c.replace("-", "").replace(":", "")) <= {""} for c in cells):
            in_table = True
            continue
        if cells[0].lower() == "gate":
            in_table = True
            continue
        if not in_table:
            continue
        gate = cells[0]
        verdict = cells[1] if len(cells) >= 2 else "?"
        source = cells[2] if len(cells) >= 3 and cells[2] not in ("—", "-") else None
        rows.append(StatusGateRow(gate=gate, verdict=verdict, source_run_id=source))
    return rows


def parse_holm_block(notes: list[str]) -> tuple[float | None, list[dict[str, Any]]]:
    """Extract Holm-corrected p-values from a :class:`GateStatus.notes` list.

    The orchestrator emits a header line ``Holm correction at alpha=0.05
    over directional-threshold gates (family: …)`` followed by one
    line per gate of the form ``  <gate>: p=…, holm_adj_p=…,
    reject_null=…``. We parse both and return ``(alpha, rows)`` where
    ``rows`` is a list of dicts ready for ``st.dataframe``.

    Returns ``(None, [])`` when no Holm header is found — the dashboard
    suppresses the section in that case.
    """
    alpha: float | None = None
    rows: list[dict[str, Any]] = []
    found_header = False
    for raw in notes:
        m = _HOLM_HEADER_RE.search(raw)
        if m is not None:
            try:
                alpha = float(m.group(1))
            except ValueError:
                alpha = None
            found_header = True
            continue
        if not found_header:
            continue
        m2 = _HOLM_GATE_RE.match(raw)
        if m2 is None:
            continue
        rows.append(
            {
                "gate": m2.group("gate"),
                "p": float(m2.group("p")),
                "holm_adj_p": float(m2.group("padj")),
                "reject_null": m2.group("rej") == "True",
            }
        )
    return alpha, rows


# ---------------------------------------------------------------------------
# Panel renderers — each takes plain data, calls Streamlit primitives.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jepa-worm-verdict-dashboard",
        description="Phase 0 gate-verdict dashboard.",
    )
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path("results"),
        help="Path to results/ root (default: ./results).",
    )
    return parser.parse_args(argv)


def render_outcome_banner(metrics: MetricsOutput, run_id: str) -> None:
    """Panel 2 — render the recomputed :class:`GateStatus.outcome` banner.

    Colour-coded via :data:`_OUTCOME_COLOR`. Failure mode: ``evaluate_gates``
    may raise on a corrupt MetricsOutput; we catch and surface an info
    banner instead of crashing the whole dashboard.
    """
    try:
        status = evaluate_gates(metrics)
    except (ValueError, KeyError, AttributeError) as exc:
        st.error(f"Gate evaluation failed for `{run_id}`: {exc}")
        return
    color = _OUTCOME_COLOR.get(status.outcome, theme.COLOR_NEUTRAL)
    st.markdown(
        f"""<div style="padding: {theme.SPACE_MD_PX}px {theme.SPACE_LG_PX}px;
        border-radius: 6px; background: {color}22;
        border-left: 6px solid {color}; font-size: {theme.FONT_SIZE_H2_PX}px;
        font-weight: 600; color: {color}; margin-bottom: {theme.SPACE_MD_PX}px;">
        Outcome: {status.outcome}
        <span style="font-size: {theme.FONT_SIZE_CAPTION_PX}px;
        font-weight: 400; color: {theme.COLOR_NEUTRAL}; margin-left: 12px;">
        (recomputed from metrics_eval.json via evaluate_gates)
        </span></div>""",
        unsafe_allow_html=True,
    )
    # Quick per-gate summary directly under the banner so the eye can
    # tie outcome → verdict mix without scrolling.
    cleared = sum(1 for v in status.gates.values() if v == "cleared")
    fired = sum(1 for v in status.gates.values() if v == "fired")
    pending = sum(1 for v in status.gates.values() if v == "pending")
    st.caption(
        f"{cleared} cleared · {fired} fired · {pending} pending ({len(status.gates)} primary gates)"
    )


def render_status_gate_table(status_path: Path, run_id: str) -> None:
    """Panel 3 — render the per-gate verdict table parsed from STATUS.md.

    Hand-curated and writer-owned tables coexist in STATUS.md (the
    pre-Phase-0 hand table at ``## Gate states`` carries the substantive
    "where we are" view; the writer-owned ``## Gate verdicts`` table is
    populated each ``wormjepa eval`` call). This panel parses whichever
    is present, with the writer-owned section winning when both exist.
    """
    rows = parse_status_gate_verdicts(status_path)
    st.markdown("##### Gate verdict table (from STATUS.md)")
    if not rows:
        st.info(
            f"No `## Gate verdicts` / `## Gate states` table found in `{status_path}`. "
            f"The outcome banner above reflects the recomputed verdicts."
        )
        return
    table_rows: list[dict[str, Any]] = []
    for r in rows:
        icon = _VERDICT_ICON.get(r.verdict.lower().split()[0], "")
        verdict_cell = f"{icon} {r.verdict}".strip()
        table_rows.append(
            {
                "gate": r.gate,
                "verdict": verdict_cell,
                "source run-id": r.source_run_id or "—",
            }
        )
    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "gate": st.column_config.TextColumn("gate"),
            "verdict": st.column_config.TextColumn("verdict"),
            "source run-id": st.column_config.TextColumn("source run-id"),
        },
    )
    st.caption(f"Source: `{status_path.name}` · sortable by clicking column headers.")
    # Mark which run-id was selected so reviewers can cross-check
    # whether the STATUS.md table is stale vs the selected metrics_eval.
    st.caption(f"Selected run-id: `{run_id}` (table source is the repo's STATUS.md).")


def render_metric_entry_inspector(metrics: MetricsOutput) -> None:
    """Panel 4 — selectable list of :class:`MetricEntry`; expanded detail.

    Each row carries point + CI lower/upper + producer + notes. Notes
    often span multiple sentences (Phase 0 v0 captions explaining
    proxy cohorts / Holm caveats) so the inspector pre-renders them
    with newlines preserved via ``st.text``.
    """
    st.markdown("##### MetricEntry inspector")
    if not metrics.entries:
        st.info("`metrics_eval.json` carries no entries — the probe suite produced no rows.")
        return
    labels = [f"{e.name} ({e.producer})" for e in metrics.entries]
    selected_label = st.selectbox(
        "Select a MetricEntry to inspect",
        options=labels,
        key="verdict_dashboard.metric_selector",
    )
    idx = labels.index(selected_label) if selected_label in labels else 0
    entry: MetricEntry = metrics.entries[idx]

    point_col, lower_col, upper_col, n_col = st.columns(4)
    with point_col:
        st.metric("point", _format_float(entry.ci.point))
    with lower_col:
        st.metric("CI lower", _format_float(entry.ci.lower))
    with upper_col:
        st.metric("CI upper", _format_float(entry.ci.upper))
    with n_col:
        st.metric("n_bootstrap", str(entry.ci.n_samples))

    # CI method + level are useful audit metadata; surface them as
    # smaller captions rather than the prominent metric tiles.
    st.caption(f"method={entry.ci.method} · level={entry.ci.level} · grouping={entry.ci.grouping}")

    if entry.sub_entries:
        st.markdown("**Sub-entries**")
        sub_rows = [
            {
                "key": s.key,
                "point": _format_float(s.ci.point),
                "CI lower": _format_float(s.ci.lower),
                "CI upper": _format_float(s.ci.upper),
                "method": s.ci.method,
            }
            for s in entry.sub_entries
        ]
        st.dataframe(sub_rows, use_container_width=True, hide_index=True)

    if entry.notes:
        st.markdown("**Notes**")
        st.text(entry.notes)  # `st.text` preserves whitespace + newlines.


def render_cross_seed_sweep(results_root: Path, run_ids: list[str]) -> None:
    """Panel 5 — cross-seed sweep summary across multi-selected runs.

    Mirrors the shape of ``wormjepa eval --run a --run b ...`` console
    output: per-gate consensus + NFR9 spread (point-mean / min / max).

    Computation reuses :func:`evaluate_gates` on each run's
    ``metrics_eval.json``; the orchestrator's ``evaluate_sweep`` is NOT
    invoked here because the dashboard must stay read-only (the
    orchestrator re-runs probes, which writes side-effects).
    """
    st.markdown("##### Cross-seed sweep (≥ 2 runs)")
    if len(run_ids) < 2:
        st.info("Select 2+ runs in the sidebar to engage the cross-seed sweep panel.")
        return
    metrics_per_run: list[MetricsOutput] = []
    missing: list[str] = []
    for rid in run_ids:
        m = load_metrics_eval(results_root / rid)
        if m is None:
            missing.append(rid)
        else:
            metrics_per_run.append(m)
    if missing:
        st.warning("Skipping runs without `metrics_eval.json`: " + ", ".join(missing))
    if len(metrics_per_run) < 2:
        st.info("Need ≥ 2 runs with `metrics_eval.json` for a sweep view.")
        return

    statuses = []
    for m in metrics_per_run:
        try:
            statuses.append(evaluate_gates(m))
        except (ValueError, KeyError, AttributeError) as exc:
            logger.warning("evaluate_gates failed for %s: %s", m.run_id, exc)
            statuses.append(None)
    # Build the per-gate row: gate, consensus, per-seed verdict array,
    # per-seed point estimate, point mean/min/max.
    all_gates: list[str] = []
    for s in statuses:
        if s is None:
            continue
        for g in s.gates:
            if g not in all_gates:
                all_gates.append(g)

    table_rows: list[dict[str, Any]] = []
    for gate in sorted(all_gates):
        per_seed_verdicts = [
            (s.gates.get(gate, "pending") if s is not None else "?") for s in statuses
        ]
        per_seed_points: list[float] = []
        for m in metrics_per_run:
            per_seed_points.append(_point_estimate_for_gate(m, gate))
        finite = [p for p in per_seed_points if p == p]  # NaN filter
        consensus = _consensus_verdict(per_seed_verdicts)
        row = {
            "gate": gate,
            "consensus": f"{_VERDICT_ICON.get(consensus, '?')} {consensus}",
            "point_mean": _format_float(sum(finite) / len(finite) if finite else float("nan")),
            "point_min": _format_float(min(finite) if finite else float("nan")),
            "point_max": _format_float(max(finite) if finite else float("nan")),
            "per_seed_verdicts": "[" + ",".join(per_seed_verdicts) + "]",
        }
        table_rows.append(row)
    st.dataframe(table_rows, use_container_width=True, hide_index=True)
    st.caption(
        f"Runs in sweep: {', '.join(m.run_id for m in metrics_per_run)} "
        f"({len(metrics_per_run)} seeds)."
    )


def render_holm_panel(metrics: MetricsOutput, run_id: str) -> None:
    """Panel 6 — pre-render the Holm-correction block from gate notes.

    ``GateStatus.notes`` is the orchestrator's audit log; the Holm pass
    appends a header + one line per gate. We parse and surface as a
    table so the per-gate adjusted p-values are scannable without
    eyeballing notes prose.
    """
    st.markdown("##### Holm correction")
    try:
        status = evaluate_gates(metrics)
    except (ValueError, KeyError, AttributeError) as exc:
        st.info(f"Holm panel skipped — gate evaluation failed: {exc}")
        return
    alpha, rows = parse_holm_block(status.notes)
    if alpha is None or not rows:
        st.caption(
            "No Holm-correction block in this run's gate notes "
            "(the orchestrator only emits one when at least one "
            "directional-threshold gate produces a finite p-value)."
        )
        return
    st.caption(
        f"Family-wise error correction at alpha = {alpha} (parsed from `{run_id}` gate notes)."
    )
    table_rows = [
        {
            "gate": r["gate"],
            "p": _format_float(r["p"]),
            "holm_adj_p": _format_float(r["holm_adj_p"]),
            "reject_null": "✓" if r["reject_null"] else "—",
        }
        for r in rows
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_float(x: float) -> str:
    """Compact float formatter that handles NaN gracefully."""
    if x != x:  # NaN
        return "—"
    return f"{x:.4g}"


def _point_estimate_for_gate(metrics: MetricsOutput, gate: str) -> float:
    """Return the per-seed point estimate for a gate from its MetricEntry.

    Mirrors ``orchestrator._point_for_gate`` — kept inline so the
    dashboard does not import from a ``_private`` symbol path. The
    mapping is small and stable; if either side grows a new gate the
    dashboard surfaces NaN rather than crashing.
    """
    gate_to_entry = {
        "neural_probe_partial_r2": "neural_probe_partial_r2",
        "neural_prior_ablation": "neural_prior_ablation_delta_r2",
        "session_id_at_chance": "session_id_classifier",
    }
    entry_name = gate_to_entry.get(gate)
    if entry_name is not None:
        for e in metrics.entries:
            if e.name == entry_name:
                return float(e.ci.point)
    if gate == "kill_criterion":
        for e in metrics.entries:
            if e.name == "future_pose" and e.producer == "jepa":
                for sub in e.sub_entries:
                    if sub.key == "1s":
                        return float(sub.ci.point)
    return float("nan")


def _consensus_verdict(per_seed: list[str]) -> str:
    """Aggregate per-seed verdicts into a single consensus label.

    Mirrors ``orchestrator._consensus``. Kept inline (private symbol
    avoidance + the rule is two lines).
    """
    s = set(per_seed)
    if s == {"cleared"}:
        return "cleared"
    if "fired" in s:
        return "fired"
    if s == {"pending"}:
        return "pending"
    return "split"


def _streamlit_argv_tail() -> list[str]:
    return sys.argv[1:]


# ---------------------------------------------------------------------------
# Page composition
# ---------------------------------------------------------------------------


def run() -> None:
    """Compose the dashboard page (single-column with side-by-side panels).

    Layout: outcome banner spans the page top; below that a two-column
    split places STATUS.md gate table + Holm panel on the left, and
    MetricEntry inspector on the right. Cross-seed sweep sits below
    the split, full-width, since it spans multiple runs.
    """
    args = _parse_args(_streamlit_argv_tail())
    results_root: Path = args.results_root.expanduser().resolve()

    st.set_page_config(
        page_title="JEPA-WORM Phase 0 verdict dashboard",
        page_icon="🪱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Phase 0 verdict dashboard")
    st.caption(f"Read-only inspector for `metrics_eval.json` + `STATUS.md` under `{results_root}`.")

    # Guard: STATUS.md is gitignored / local-only (generated at runtime by
    # `wormjepa eval` as an additive merge). On a fresh public-repo clone
    # the file is absent and Panel 3 / parse_status_gate_verdicts would
    # silently render an empty table. Surface an actionable explanation
    # once at app load instead.
    status_path = _PROJECT_ROOT / "STATUS.md"
    if not status_path.is_file():
        st.error(
            "This is an author-tooling dev dashboard that depends on a "
            "local-only `STATUS.md` at the project root — the file is "
            "gitignored (generated at runtime by `wormjepa eval` as an "
            "additive merge) and therefore absent from a fresh public-repo "
            "clone. For public users, the gate-verdict layout and the "
            "recomputed outcome semantics are documented in `REPRODUCE.md` "
            "and the inline docstrings of `wormjepa.eval.gates`; produce "
            "a `STATUS.md` locally via `wormjepa eval --run <id>` to "
            "engage this dashboard."
        )
        st.stop()

    run_ids = list_runs_with_metrics_eval(results_root)
    if not run_ids:
        st.warning(
            f"No `metrics_eval.json` files found under `{results_root}`. "
            f"Run `wormjepa eval --run <id>` to produce one."
        )
        return

    with st.sidebar:
        st.markdown("### Run selection")
        selected_run = st.selectbox(
            "Primary run",
            options=run_ids,
            key="verdict_dashboard.primary_run",
        )
        st.markdown("---")
        st.markdown("### Cross-seed sweep")
        sweep_runs = st.multiselect(
            "Compare ≥ 2 runs",
            options=run_ids,
            default=[],
            key="verdict_dashboard.sweep_runs",
            help="Select 2+ runs to render the cross-seed sweep panel below.",
        )

    run_dir = results_root / selected_run
    metrics = load_metrics_eval(run_dir)
    if metrics is None:
        st.error(f"Failed to load `metrics_eval.json` for `{selected_run}`.")
        return

    # Panel 2: outcome banner (top).
    render_outcome_banner(metrics, selected_run)

    # Split below the banner: left = STATUS.md table + Holm; right = inspector.
    left, right = st.columns([6, 6])
    with left:
        # STATUS.md lives at the project root, not inside the run dir.
        # The dashboard reads the repo's single STATUS.md (the authoritative
        # writer target) rather than a per-run copy.
        status_path = _PROJECT_ROOT / "STATUS.md"
        render_status_gate_table(status_path, selected_run)
        st.markdown("---")
        render_holm_panel(metrics, selected_run)
    with right:
        render_metric_entry_inspector(metrics)

    st.markdown("---")
    render_cross_seed_sweep(results_root, sweep_runs)


# Streamlit imports this module; calling run() at module scope is the
# canonical Streamlit entry pattern.
run()
