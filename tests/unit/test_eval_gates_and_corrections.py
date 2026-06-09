"""Unit tests for ablations + multiple-comparison + gates + STATUS writer (Stories 6.8-6.11)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wormjepa.eval.ablations import (
    baaiworm_ablation_entry,
    neural_prior_ablation_entry,
)
from wormjepa.eval.gates import evaluate_gates
from wormjepa.eval.metrics_schema import (
    BootstrapCI,
    MetricEntry,
    MetricsOutput,
    SubEntry,
)
from wormjepa.eval.multiple_comparison import apply_correction
from wormjepa.manifest.status_writer import render_status, update_status

# --- ablations ---


def test_neural_prior_ablation_positive_when_full_better() -> None:
    full = [0.20, 0.21, 0.19, 0.22]
    kin_only = [0.10, 0.11, 0.09, 0.12]
    entry = neural_prior_ablation_entry(
        full, kin_only, [f"w{i}" for i in range(4)], n_bootstrap=200
    )
    assert entry.name == "neural_prior_ablation_delta_r2"
    assert entry.ci.point > 0.05


def test_baaiworm_ablation_can_be_negative() -> None:
    with_aug = [0.10, 0.11, 0.09, 0.12]
    without = [0.15, 0.14, 0.16, 0.13]
    entry = baaiworm_ablation_entry(with_aug, without, [f"w{i}" for i in range(4)], n_bootstrap=200)
    assert entry.name == "baaiworm_augmentation_ablation_delta_r2"
    assert entry.ci.point < 0.0  # without is better than with — reported either way


def test_ablation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="share length"):
        neural_prior_ablation_entry([0.1, 0.2], [0.1], ["w1", "w2"])


# --- multiple-comparison correction ---


def test_apply_correction_holm() -> None:
    p_values = [0.001, 0.01, 0.04, 0.5]
    result = apply_correction(p_values, method="holm", alpha=0.05)
    assert len(result.corrected_pvalues) == 4
    assert len(result.reject) == 4
    assert result.reject[0] is True  # 0.001 always rejects at alpha 0.05


def test_apply_correction_bh() -> None:
    p_values = [0.001, 0.01, 0.04, 0.5]
    result = apply_correction(p_values, method="bh", alpha=0.05)
    assert result.method == "bh"


def test_apply_correction_empty() -> None:
    result = apply_correction([], method="holm")
    assert result.corrected_pvalues == []


def test_apply_correction_alpha_validated() -> None:
    with pytest.raises(ValueError, match="alpha"):
        apply_correction([0.1], alpha=1.0)


# --- gates ---


def _ci(point: float, lower: float, upper: float) -> BootstrapCI:
    return BootstrapCI(point=point, lower=lower, upper=upper, n_samples=200, method="bca")


def _build_metrics(
    *,
    jepa_1s_point: float,
    jepa_1s_upper: float,
    transformer_1s_point: float,
    headline_lower: float,
    ablation_lower: float,
    session_lower: float,
    session_upper: float,
    chance: float = 0.125,
) -> MetricsOutput:
    jepa_fp = MetricEntry(
        name="future_pose",
        producer="jepa",
        ci=_ci(float("nan"), float("nan"), float("nan")),
        sub_entries=[
            SubEntry(
                key="1s",
                ci=_ci(jepa_1s_point, max(0.0, jepa_1s_point - 0.05), jepa_1s_upper),
            ),
        ],
    )
    trans_fp = MetricEntry(
        name="future_pose",
        producer="transformer_eigenworms",
        ci=_ci(float("nan"), float("nan"), float("nan")),
        sub_entries=[
            SubEntry(
                key="1s",
                ci=_ci(
                    transformer_1s_point,
                    transformer_1s_point - 0.05,
                    transformer_1s_point + 0.05,
                ),
            ),
        ],
    )
    headline = MetricEntry(
        name="neural_probe_partial_r2",
        producer="jepa",
        ci=_ci(headline_lower + 0.02, headline_lower, headline_lower + 0.05),
    )
    ablation = MetricEntry(
        name="neural_prior_ablation_delta_r2",
        producer="jepa",
        ci=_ci(ablation_lower + 0.01, ablation_lower, ablation_lower + 0.03),
    )
    session = MetricEntry(
        name="session_id_classifier",
        producer="jepa",
        ci=_ci((session_lower + session_upper) / 2, session_lower, session_upper),
        notes=f"chance={chance:.3f}; at-chance gate requires CI to contain chance",
    )
    return MetricsOutput(
        run_id="20260512T000000Z__abcdef12__test",
        entries=[jepa_fp, trans_fp, headline, ablation, session],
    )


def test_evaluate_gates_all_cleared() -> None:
    metrics = _build_metrics(
        jepa_1s_point=1.0,
        jepa_1s_upper=1.2,
        transformer_1s_point=1.5,  # JEPA's upper 1.2 < transformer point 1.5
        headline_lower=0.08,  # >= 0.05
        ablation_lower=0.05,  # >= 0.02
        session_lower=0.10,  # contains chance=0.125
        session_upper=0.15,
    )
    status = evaluate_gates(metrics)
    assert status.outcome == "cleared"
    assert status.gates["kill_criterion"] == "cleared"
    assert status.gates["neural_probe_partial_r2"] == "cleared"


def test_evaluate_gates_kill_criterion_fired() -> None:
    metrics = _build_metrics(
        jepa_1s_point=1.6,
        jepa_1s_upper=1.8,  # JEPA upper 1.8 >= transformer point 1.5 → kill fired
        transformer_1s_point=1.5,
        headline_lower=0.10,
        ablation_lower=0.05,
        session_lower=0.10,
        session_upper=0.15,
    )
    status = evaluate_gates(metrics)
    assert status.outcome == "kill_criterion_fired"
    assert status.gates["kill_criterion"] == "fired"


def test_evaluate_gates_reframed_when_headline_misses() -> None:
    metrics = _build_metrics(
        jepa_1s_point=1.0,
        jepa_1s_upper=1.2,
        transformer_1s_point=1.5,
        headline_lower=0.01,  # below 0.05 → headline gate fired
        ablation_lower=0.05,
        session_lower=0.10,
        session_upper=0.15,
    )
    status = evaluate_gates(metrics)
    assert status.outcome == "reframed"


# --- STATUS.md writer ---


def test_render_status_contains_all_gates() -> None:
    metrics = _build_metrics(
        jepa_1s_point=1.0,
        jepa_1s_upper=1.2,
        transformer_1s_point=1.5,
        headline_lower=0.08,
        ablation_lower=0.05,
        session_lower=0.10,
        session_upper=0.15,
    )
    status = evaluate_gates(metrics)
    text = render_status(
        "20260512T000000Z__abcdef12__test",
        status,
        now=datetime(2026, 5, 12, 14, 23, 1, tzinfo=UTC),
    )
    assert "phase: 0" in text
    assert "gate_status: cleared" in text
    for gate in ("kill_criterion", "neural_probe_partial_r2", "neural_prior_ablation"):
        assert gate in text


def test_update_status_writes_file(tmp_path: Path) -> None:
    metrics = _build_metrics(
        jepa_1s_point=1.0,
        jepa_1s_upper=1.2,
        transformer_1s_point=1.5,
        headline_lower=0.08,
        ablation_lower=0.05,
        session_lower=0.10,
        session_upper=0.15,
    )
    status = evaluate_gates(metrics)
    target = tmp_path / "STATUS.md"
    update_status("rid", status, status_path=target, now=datetime(2026, 5, 12, tzinfo=UTC))
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert "last_run_id: rid" in content


def test_update_status_idempotent_modulo_now(tmp_path: Path) -> None:
    metrics = _build_metrics(
        jepa_1s_point=1.0,
        jepa_1s_upper=1.2,
        transformer_1s_point=1.5,
        headline_lower=0.08,
        ablation_lower=0.05,
        session_lower=0.10,
        session_upper=0.15,
    )
    status = evaluate_gates(metrics)
    fixed_time = datetime(2026, 5, 12, tzinfo=UTC)
    p1 = tmp_path / "a.md"
    p2 = tmp_path / "b.md"
    update_status("rid", status, status_path=p1, now=fixed_time)
    update_status("rid", status, status_path=p2, now=fixed_time)
    assert p1.read_bytes() == p2.read_bytes()


def test_evaluate_gates_pending_when_metrics_missing() -> None:
    """When required entries aren't present, gates report pending and outcome is reframed."""
    metrics = MetricsOutput(run_id="r", entries=[])
    status = evaluate_gates(metrics)
    assert all(v == "pending" for v in status.gates.values())
    # outcome falls through to "reframed" since headline is not cleared
    assert status.outcome == "reframed"


def test_nan_resilient() -> None:
    """NaN values in CIs should not crash the gate evaluation (they're allowed in the schema)."""
    nan_ci = BootstrapCI(
        point=math.nan,
        lower=math.nan,
        upper=math.nan,
        n_samples=1,
        method="percentile",
    )
    metrics = MetricsOutput(
        run_id="r",
        entries=[
            MetricEntry(name="neural_probe_partial_r2", producer="jepa", ci=nan_ci),
        ],
    )
    status = evaluate_gates(metrics)
    # NaN comparisons return False, so the gate "fires" rather than clearing — acceptable behavior.
    assert "neural_probe_partial_r2" in status.gates


def test_baaiworm_ablation_entry_name_matches_pre_reg() -> None:
    """The emitted key must match the STATUS/pre-reg row-5 identity exactly."""
    entry = baaiworm_ablation_entry(
        [0.20, 0.21, 0.19], [0.10, 0.11, 0.09], ["w0", "w1", "w2"], n_bootstrap=100
    )
    assert entry.name == "baaiworm_augmentation_ablation_delta_r2"
    # Reported-only: positive or negative is fine, no threshold is applied.
    assert entry.ci.point > 0.0


# --- baaiworm-augmentation ablation wiring (control-arm plumbing) ---


def _primary_metrics_with_partial_r2(point: float, lower: float, upper: float) -> MetricsOutput:
    """A minimal single-run MetricsOutput carrying the primary partial-R² entry."""
    return MetricsOutput(
        run_id="primary_run",
        entries=[
            MetricEntry(
                name="neural_probe_partial_r2",
                producer="jepa",
                ci=_ci(point, lower, upper),
            ),
        ],
    )


def _install_ablation_seams(
    monkeypatch: pytest.MonkeyPatch,
    primary: MetricsOutput,
    *,
    control_point: float | None,
) -> None:
    """Patch orchestrator seams so no training / cache build runs.

    ``evaluate_run`` returns ``primary``; ``_build_eval_cache_and_run_partial_r2``
    returns a control partial-R² entry (or ``(None, None)`` to simulate a
    control that produced no usable partial-R²).
    """
    from wormjepa.eval import orchestrator

    def _fake_evaluate_run(_run_dir: Path) -> tuple[MetricsOutput, object]:
        return primary, evaluate_gates(primary)

    def _fake_cache(_run_dir: Path) -> tuple[MetricEntry | None, None]:
        if control_point is None:
            return None, None
        entry = MetricEntry(
            name="neural_probe_partial_r2",
            producer="jepa",
            ci=_ci(control_point, control_point - 0.02, control_point + 0.02),
        )
        return entry, None

    monkeypatch.setattr(orchestrator, "evaluate_run", _fake_evaluate_run)
    monkeypatch.setattr(orchestrator, "_build_eval_cache_and_run_partial_r2", _fake_cache)


def test_baaiworm_ablation_resolves_with_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """A baaiworm control arm appends the reported ΔR² entry (with - without)."""
    from wormjepa.eval import orchestrator

    primary = _primary_metrics_with_partial_r2(0.30, 0.25, 0.35)
    _install_ablation_seams(monkeypatch, primary, control_point=0.20)

    metrics, status = orchestrator.evaluate_run_with_ablations(
        Path("primary"), baaiworm_control_run_dir=Path("baaiworm_off")
    )
    entry = next(
        (e for e in metrics.entries if e.name == "baaiworm_augmentation_ablation_delta_r2"), None
    )
    assert entry is not None, [e.name for e in metrics.entries]
    # delta = primary - control = 0.30 - 0.20 = 0.10 (reported either sign).
    assert entry.ci.point == pytest.approx(0.10, abs=1e-9)
    # Reported-only: it is NOT a gate, never fires, and is not in any Holm
    # family — re-running gate evaluation on the augmented metrics must not
    # introduce a baaiworm verdict, and the gate set stays the four primary
    # gates only.
    assert "baaiworm_augmentation" not in status.gates
    assert set(evaluate_gates(metrics).gates) == {
        "kill_criterion",
        "neural_probe_partial_r2",
        "neural_prior_ablation",
        "session_id_at_chance",
    }


def test_baaiworm_ablation_pending_without_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """No baaiworm control → no fabricated entry; ablation stays pending."""
    from wormjepa.eval import orchestrator

    primary = _primary_metrics_with_partial_r2(0.30, 0.25, 0.35)
    _install_ablation_seams(monkeypatch, primary, control_point=0.20)

    metrics, _status = orchestrator.evaluate_run_with_ablations(Path("primary"))
    assert all(e.name != "baaiworm_augmentation_ablation_delta_r2" for e in metrics.entries), [
        e.name for e in metrics.entries
    ]


def test_baaiworm_ablation_pending_when_control_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A control run that yields no partial-R² leaves the ablation unresolved."""
    from wormjepa.eval import orchestrator

    primary = _primary_metrics_with_partial_r2(0.30, 0.25, 0.35)
    _install_ablation_seams(monkeypatch, primary, control_point=None)

    metrics, _status = orchestrator.evaluate_run_with_ablations(
        Path("primary"), baaiworm_control_run_dir=Path("baaiworm_off")
    )
    assert all(e.name != "baaiworm_augmentation_ablation_delta_r2" for e in metrics.entries)


def test_both_ablation_arms_resolve_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supplying both controls appends both ablation entries in one pass."""
    from wormjepa.eval import orchestrator

    primary = _primary_metrics_with_partial_r2(0.30, 0.25, 0.35)
    _install_ablation_seams(monkeypatch, primary, control_point=0.20)

    metrics, status = orchestrator.evaluate_run_with_ablations(
        Path("primary"),
        control_run_dir=Path("neural_off"),
        baaiworm_control_run_dir=Path("baaiworm_off"),
    )
    names = {e.name for e in metrics.entries}
    assert "neural_prior_ablation_delta_r2" in names
    assert "baaiworm_augmentation_ablation_delta_r2" in names
    # neural_prior IS a gate (threshold 0.02); it resolves to a verdict here.
    assert status.gates.get("neural_prior_ablation") in {"cleared", "fired"}
    # baaiworm remains reported-only — not a gate key.
    assert "baaiworm_augmentation" not in status.gates


def test_neural_prior_back_compat_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retained evaluate_run_with_ablation wrapper still wires neural_prior only."""
    from wormjepa.eval import orchestrator

    primary = _primary_metrics_with_partial_r2(0.30, 0.25, 0.35)
    _install_ablation_seams(monkeypatch, primary, control_point=0.20)

    metrics, _status = orchestrator.evaluate_run_with_ablation(Path("primary"), Path("neural_off"))
    names = {e.name for e in metrics.entries}
    assert "neural_prior_ablation_delta_r2" in names
    assert "baaiworm_augmentation_ablation_delta_r2" not in names
