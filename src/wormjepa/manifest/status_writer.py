"""STATUS.md writer (Story 6.11 / FR39 / FR46).

After every reportable run, the gate-status writer updates ``STATUS.md`` with
the current phase, the run-id that produced the gate outcomes, and the
verdict for each gate. The file is the single human-readable source of truth
for "where Phase 0 stands."
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from wormjepa.eval.gates import GateStatus

if TYPE_CHECKING:
    from wormjepa.eval.orchestrator import SweepSummary

_STATUS_PATH = Path("STATUS.md")


def render_status(
    run_id: str,
    gate_status: GateStatus,
    *,
    phase: str = "0",
    now: datetime | None = None,
) -> str:
    """Render the canonical STATUS.md content for the given gate status."""
    now = now or datetime.now(tz=UTC)
    lines: list[str] = [
        "# JEPA-WORM Status",
        "",
        f"phase: {phase}",
        f"gate_status: {gate_status.outcome}",
        f"last_updated: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"last_run_id: {run_id}",
        "",
        "## Gate verdicts",
        "",
        "| Gate | Verdict |",
        "|---|---|",
    ]
    for gate_name, verdict in sorted(gate_status.gates.items()):
        lines.append(f"| {gate_name} | {verdict} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in gate_status.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def update_status(
    run_id: str,
    gate_status: GateStatus,
    *,
    status_path: Path | None = None,
    phase: str = "0",
    now: datetime | None = None,
) -> Path:
    """Write the rendered status to ``status_path`` (default: ``STATUS.md``).

    Wholesale rewrite — destroys any out-of-scope sections in the
    existing file. Use :func:`update_status_additive` when the file
    has hand-curated content outside the writer's authoritative scope
    (e.g. the Phase 0 milestone-progress table).

    The writer is idempotent: writing the same status twice produces the
    same file (modulo the timestamp, which is the only intentional source
    of churn).
    """
    target = status_path or (Path.cwd() / _STATUS_PATH)
    text = render_status(run_id, gate_status, phase=phase, now=now)
    target.write_text(text, encoding="utf-8")
    return target


def update_status_additive(
    run_id: str,
    gate_status: GateStatus,
    *,
    status_path: Path | None = None,
    phase: str = "0",
    now: datetime | None = None,
) -> Path:
    """Additive variant of :func:`update_status`: preserves out-of-scope
    sections of an existing STATUS.md.

    Scope ownership:

    - **Writer-owned (overwritten on every call):** the header block
      (``phase``, ``gate_status``, ``last_updated``, ``last_run_id``)
      and the ``## Gate verdicts`` table. Plus a sibling ``## Notes``
      section if one is rendered by :func:`render_status`.
    - **Caller-owned (preserved verbatim):** every other section. In
      practice this is the ``## Phase 0 milestone progress`` table the
      Story 8.11b commit curated by hand, plus any narrative the
      project lead has added (e.g. story-outcome reflections).

    Implementation strategy: parse the existing file into
    ``(top-of-document-header-block, sections-by-heading)``; replace
    only the writer-owned slices; re-serialise. On a fresh file (no
    existing STATUS.md), behaviour matches :func:`update_status`.

    The writer is idempotent at the writer-owned slice; caller-owned
    slices are untouched and thus also idempotent.
    """
    target = status_path or (Path.cwd() / _STATUS_PATH)
    rendered = render_status(run_id, gate_status, phase=phase, now=now)
    if not target.is_file():
        target.write_text(rendered, encoding="utf-8")
        return target

    existing = target.read_text(encoding="utf-8")
    merged = _merge_status(existing, rendered)
    target.write_text(merged, encoding="utf-8")
    return target


# Heading markers the additive merger owns. Sections starting with one
# of these get replaced from the freshly-rendered text; everything else
# survives.
_OWNED_HEADINGS: frozenset[str] = frozenset({"## Gate verdicts", "## Notes", "## Cross-seed sweep"})


def render_sweep_status(
    run_ids: list[str],
    summary: SweepSummary,
    last_gate_status: GateStatus,
    *,
    phase: str = "0",
    now: datetime | None = None,
) -> str:
    """Render STATUS.md content for a cross-seed sweep.

    Emits the standard single-run header plus a writer-owned
    ``## Cross-seed sweep`` section carrying per-gate consensus
    verdict + per-seed verdict array + NFR9 point-estimate
    mean/min/max. The ``## Gate verdicts`` / ``## Notes`` sections
    are rendered from ``last_gate_status`` (the last seed's
    GateStatus); they remain present so single-run consumers keep
    working.
    """
    base = render_status(run_ids[-1] if run_ids else "", last_gate_status, phase=phase, now=now)
    lines: list[str] = [base.rstrip("\n"), "", "## Cross-seed sweep", "", "run-ids:"]
    for rid in run_ids:
        lines.append(f"- {rid}")
    lines.append("")
    lines.append("| gate | consensus | point_mean | point_min | point_max | per_seed_verdicts |")
    lines.append("|---|---|---|---|---|---|")
    for s in summary.per_gate:
        per_seed = ",".join(s.per_seed_verdict)
        lines.append(
            f"| {s.gate} | {s.consensus_verdict} | "
            f"{s.point_mean:.4f} | {s.point_min:.4f} | {s.point_max:.4f} | "
            f"[{per_seed}] |"
        )
    lines.append("")
    return "\n".join(lines)


def update_status_additive_sweep(
    run_ids: list[str],
    summary: SweepSummary,
    last_gate_status: GateStatus,
    *,
    status_path: Path | None = None,
    phase: str = "0",
    now: datetime | None = None,
) -> Path:
    """Cross-seed variant of :func:`update_status_additive`.

    Merges :func:`render_sweep_status` into the existing STATUS.md,
    preserving caller-owned sections (milestone progress, etc.). The
    writer-owned set extends to include ``## Cross-seed sweep``.
    """
    target = status_path or (Path.cwd() / _STATUS_PATH)
    rendered = render_sweep_status(run_ids, summary, last_gate_status, phase=phase, now=now)
    if not target.is_file():
        target.write_text(rendered, encoding="utf-8")
        return target
    existing = target.read_text(encoding="utf-8")
    merged = _merge_status(existing, rendered)
    target.write_text(merged, encoding="utf-8")
    return target


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into ``(heading_or_empty, body)`` sections.

    The first chunk (before any ``## `` heading) gets the heading
    ``""`` — treated as the document's header block.
    """
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []
    for line in lines:
        if line.startswith("## "):
            sections.append((current_heading, "".join(current_body)))
            current_heading = line.rstrip("\n")
            current_body = [line]
        else:
            current_body.append(line)
    sections.append((current_heading, "".join(current_body)))
    return sections


def _merge_status(existing: str, rendered: str) -> str:
    """Merge ``rendered`` into ``existing``, preserving caller-owned sections."""
    existing_sections = _split_into_sections(existing)
    rendered_sections = _split_into_sections(rendered)
    rendered_by_heading: dict[str, str] = {
        heading: body for heading, body in rendered_sections if heading
    }
    rendered_header = next(
        (body for heading, body in rendered_sections if heading == ""),
        "",
    )

    out_parts: list[str] = []
    # Header block: always overwrite from the rendered output.
    out_parts.append(rendered_header)
    # Track which writer-owned headings have been emitted so we don't
    # duplicate when an existing section gets replaced.
    emitted: set[str] = set()
    for heading, body in existing_sections:
        if heading == "":
            continue  # already replaced above
        if heading in _OWNED_HEADINGS:
            replacement = rendered_by_heading.get(heading)
            if replacement is not None:
                out_parts.append(replacement)
                emitted.add(heading)
            # If rendered output omits this writer-owned section, drop it.
            continue
        # Caller-owned — preserve verbatim.
        out_parts.append(body)
    # Append any writer-owned sections that existed in the rendered
    # output but not in the existing file.
    for heading, body in rendered_sections:
        if heading in _OWNED_HEADINGS and heading not in emitted:
            out_parts.append(body)
    return "".join(out_parts)
