"""Outcome-aware report template selector (Story 7.4 / FR37)."""

from __future__ import annotations

from pathlib import Path

from wormjepa.eval.gates import GateStatus


def select_template(gate_status: GateStatus, *, templates_dir: Path | None = None) -> Path:
    """Return the path to the outcome-aware report template.

    Args:
        gate_status: The :class:`GateStatus` produced by :func:`evaluate_gates`.
        templates_dir: Override directory. Defaults to the package-bundled
            ``templates/`` directory.

    Returns:
        Path to the chosen ``.md.j2`` template.
    """
    if templates_dir is None:
        templates_dir = Path(__file__).resolve().parent / "templates"
    mapping = {
        "cleared": "cleared.md.j2",
        "kill_criterion_fired": "kill_criterion_fired.md.j2",
        "reframed": "reframed.md.j2",
    }
    return templates_dir / mapping[gate_status.outcome]
