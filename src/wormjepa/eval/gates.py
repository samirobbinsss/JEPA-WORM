"""Gate evaluation: map MetricsOutput → GateStatus (Story 6.10 / NFR18 / FR39).

All effect-size thresholds use the **lower bound of the worm-level 95% CI**
per NFR18 — a threshold "clears" only if the CI lower bound exceeds the
threshold.

Phase 0 v0 implements the gates as flat functions taking the relevant
``MetricsOutput`` rows by ``name``. The selector :func:`evaluate_gates` walks
a :class:`MetricsOutput`, computes each gate's verdict, and returns a
:class:`GateStatus` with an overall ``outcome`` category (cleared /
kill_criterion_fired / reframed) consumed by the outcome-aware report
template selector (Epic 7 Story 7.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from wormjepa.eval.metrics_schema import MetricEntry, MetricsOutput

GateName = Literal[
    "kill_criterion",
    "neural_probe_partial_r2",
    "neural_prior_ablation",
    "session_id_at_chance",
]

GateVerdict = Literal["cleared", "fired", "pending"]

OutcomeCategory = Literal["cleared", "kill_criterion_fired", "reframed"]


# Pre-registered thresholds (see pre-registration/PRE-REGISTRATION.md).
_HEADLINE_PARTIAL_R2_THRESHOLD = 0.05
_NEURAL_PRIOR_DELTA_R2_THRESHOLD = 0.02


@dataclass(frozen=True, slots=True)
class GateStatus:
    """Per-gate verdicts + overall outcome category."""

    gates: dict[GateName, GateVerdict]
    outcome: OutcomeCategory
    notes: list[str] = field(default_factory=list)


def _find_entry(
    metrics: MetricsOutput, name: str, producer: str | None = None
) -> MetricEntry | None:
    for e in metrics.entries:
        if e.name == name and (producer is None or e.producer == producer):
            return e
    return None


def _eval_kill_criterion(metrics: MetricsOutput) -> tuple[GateVerdict, str]:
    """JEPA must beat transformer-on-eigenworms at the 1 s future-pose horizon.

    Heuristic: both producers ('jepa' or any baseline matching the JEPA
    headline run) report a `future_pose` entry with a `1s` sub-row. JEPA's
    CI lower bound on per-clip error must be *below* the Transformer-on-
    eigenworms point estimate (lower error is better).
    """
    jepa = _find_entry(metrics, "future_pose", producer="jepa")
    transformer = _find_entry(metrics, "future_pose", producer="transformer_eigenworms")
    if jepa is None or transformer is None:
        return "pending", "future_pose entries for both jepa and transformer_eigenworms required"
    jepa_1s = next((s for s in jepa.sub_entries if s.key == "1s"), None)
    transformer_1s = next((s for s in transformer.sub_entries if s.key == "1s"), None)
    if jepa_1s is None or transformer_1s is None:
        return "pending", "1s sub-row missing"
    # Lower error is better. Kill-criterion fires if JEPA does NOT beat the
    # transformer at 1s.
    jepa_u = jepa_1s.ci.upper
    trans_p = transformer_1s.ci.point
    if jepa_u < trans_p:
        return "cleared", f"jepa upper {jepa_u:.3f} < transformer point {trans_p:.3f}"
    return "fired", f"jepa upper {jepa_u:.3f} not < transformer point {trans_p:.3f}"


def _eval_neural_probe_partial_r2(metrics: MetricsOutput) -> tuple[GateVerdict, str]:
    entry = _find_entry(metrics, "neural_probe_partial_r2", producer="jepa")
    if entry is None:
        return "pending", "neural_probe_partial_r2 (producer=jepa) entry missing"
    if entry.ci.lower >= _HEADLINE_PARTIAL_R2_THRESHOLD:
        return "cleared", f"CI lower {entry.ci.lower:.3f} >= {_HEADLINE_PARTIAL_R2_THRESHOLD}"
    return "fired", f"CI lower {entry.ci.lower:.3f} < {_HEADLINE_PARTIAL_R2_THRESHOLD}"


def _eval_neural_prior_ablation(metrics: MetricsOutput) -> tuple[GateVerdict, str]:
    entry = _find_entry(metrics, "neural_prior_ablation_delta_r2", producer="jepa")
    if entry is None:
        return "pending", "neural_prior_ablation_delta_r2 entry missing"
    if entry.ci.lower >= _NEURAL_PRIOR_DELTA_R2_THRESHOLD:
        return "cleared", f"CI lower {entry.ci.lower:.3f} >= {_NEURAL_PRIOR_DELTA_R2_THRESHOLD}"
    return "fired", f"CI lower {entry.ci.lower:.3f} < {_NEURAL_PRIOR_DELTA_R2_THRESHOLD}"


def _eval_session_id_at_chance(metrics: MetricsOutput) -> tuple[GateVerdict, str]:
    entry = _find_entry(metrics, "session_id_classifier", producer="jepa")
    if entry is None:
        return "pending", "session_id_classifier entry missing"
    # The chance baseline is encoded in the notes (e.g. "chance=0.125; ...").
    # We require the 95% CI of accuracy to *contain* the chance baseline.
    import re

    match = re.search(r"chance=(\d*\.?\d+)", entry.notes or "")
    if not match:
        return "pending", "could not parse chance baseline from notes"
    chance = float(match.group(1))
    lower, upper = entry.ci.lower, entry.ci.upper
    if lower <= chance <= upper:
        return "cleared", f"CI [{lower:.3f}, {upper:.3f}] contains chance={chance:.3f}"
    return "fired", f"CI [{lower:.3f}, {upper:.3f}] excludes chance={chance:.3f}"


def evaluate_gates(metrics: MetricsOutput) -> GateStatus:
    """Run all four primary gates and produce an overall outcome category."""
    kill_verdict, kill_note = _eval_kill_criterion(metrics)
    headline_verdict, headline_note = _eval_neural_probe_partial_r2(metrics)
    ablation_verdict, ablation_note = _eval_neural_prior_ablation(metrics)
    session_verdict, session_note = _eval_session_id_at_chance(metrics)

    gates: dict[GateName, GateVerdict] = {
        "kill_criterion": kill_verdict,
        "neural_probe_partial_r2": headline_verdict,
        "neural_prior_ablation": ablation_verdict,
        "session_id_at_chance": session_verdict,
    }

    notes = [kill_note, headline_note, ablation_note, session_note]

    if kill_verdict == "fired":
        outcome: OutcomeCategory = "kill_criterion_fired"
    elif (
        headline_verdict == "cleared"
        and ablation_verdict == "cleared"
        and session_verdict == "cleared"
    ):
        outcome = "cleared"
    else:
        outcome = "reframed"

    return GateStatus(gates=gates, outcome=outcome, notes=notes)
