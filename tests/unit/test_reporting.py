"""Unit tests for reporting (Stories 7.1-7.6 / 7.8)."""

from __future__ import annotations

import time

import pytest

from wormjepa.eval.gates import GateStatus
from wormjepa.eval.metrics_schema import BootstrapCI, MetricEntry, MetricsOutput
from wormjepa.reporting.compute_provenance import record_provenance
from wormjepa.reporting.render import compare_metrics, render_report
from wormjepa.reporting.template_selector import select_template


def _ci(point: float, lower: float, upper: float) -> BootstrapCI:
    return BootstrapCI(point=point, lower=lower, upper=upper, n_samples=200, method="bca")


def _gate_status(outcome: str) -> GateStatus:
    return GateStatus(
        gates={
            "kill_criterion": "cleared" if outcome != "kill_criterion_fired" else "fired",
            "neural_probe_partial_r2": "cleared" if outcome == "cleared" else "fired",
            "neural_prior_ablation": "cleared",
            "session_id_at_chance": "cleared",
        },
        outcome=outcome,  # type: ignore[arg-type]
        notes=[],
    )


def _metrics() -> MetricsOutput:
    return MetricsOutput(
        run_id="20260512T000000Z__abcdef12__headline",
        entries=[
            MetricEntry(
                name="neural_probe_partial_r2",
                producer="jepa",
                ci=_ci(0.10, 0.07, 0.13),
                notes="r2_jepa=0.30, r2_kin=0.20",
            ),
            MetricEntry(
                name="motif_ari",
                producer="jepa",
                ci=_ci(0.42, 0.35, 0.50),
                notes="Hungarian-matched ARI",
            ),
        ],
    )


# --- compute provenance ---


def test_record_provenance_basic() -> None:
    t0 = time.perf_counter()
    time.sleep(0.01)
    p = record_provenance(t0)
    assert p.wall_clock_seconds > 0.005
    assert p.gpu_hours >= 0.0
    assert p.pytorch_version
    assert p.python_version


def test_compute_provenance_canonical_json() -> None:
    t0 = time.perf_counter()
    p = record_provenance(t0)
    text = p.to_canonical_json()
    # Sorted keys: cuda_version < gpu_hours < ...
    assert '"cuda_version"' in text
    assert text.endswith("\n")


def test_record_provenance_detects_mps_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When CUDA is unavailable but MPS is, gpu_model carries the MPS marker."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    p = record_provenance(time.perf_counter())
    assert p.gpu_model is not None
    assert "MPS" in p.gpu_model
    # cuda_version field is repurposed to carry the accelerator-version marker
    # on MPS — keeps the canonical-json shape stable across backends.
    assert p.cuda_version is not None
    assert p.cuda_version.startswith("mps-")


def test_record_provenance_cpu_only_returns_none_gpu_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither CUDA nor MPS is available, gpu fields are None."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    p = record_provenance(time.perf_counter())
    assert p.gpu_model is None
    assert p.cuda_version is None
    assert p.peak_gpu_memory_bytes is None


# --- template selector ---


def test_select_template_cleared() -> None:
    path = select_template(_gate_status("cleared"))
    assert path.name == "cleared.md.j2"
    assert path.is_file()


def test_select_template_kill_criterion_fired() -> None:
    path = select_template(_gate_status("kill_criterion_fired"))
    assert path.name == "kill_criterion_fired.md.j2"


def test_select_template_reframed() -> None:
    path = select_template(_gate_status("reframed"))
    assert path.name == "reframed.md.j2"


# --- render ---


def test_render_report_includes_run_id_and_outcome() -> None:
    text = render_report(_metrics(), _gate_status("cleared"))
    assert "20260512T000000Z__abcdef12__headline" in text
    assert "cleared" in text


def test_render_report_includes_all_entries() -> None:
    text = render_report(_metrics(), _gate_status("cleared"))
    assert "neural_probe_partial_r2" in text
    assert "motif_ari" in text


def test_render_report_kill_criterion_fired_language() -> None:
    text = render_report(_metrics(), _gate_status("kill_criterion_fired"))
    assert "Negative Result" in text


# --- compare_metrics ---


def test_compare_metrics_within_ci() -> None:
    local = MetricsOutput(
        run_id="local",
        entries=[
            MetricEntry(
                name="neural_probe_partial_r2",
                producer="jepa",
                ci=_ci(0.10, 0.07, 0.13),
            )
        ],
    )
    pub = MetricsOutput(
        run_id="published",
        entries=[
            MetricEntry(
                name="neural_probe_partial_r2",
                producer="jepa",
                ci=_ci(0.12, 0.08, 0.16),
            )
        ],
    )
    diffs = compare_metrics(local, pub)
    assert diffs[0][1] == "within_ci"


def test_compare_metrics_outside_ci() -> None:
    local = MetricsOutput(
        run_id="local",
        entries=[
            MetricEntry(
                name="neural_probe_partial_r2",
                producer="jepa",
                ci=_ci(0.50, 0.45, 0.55),
            )
        ],
    )
    pub = MetricsOutput(
        run_id="published",
        entries=[
            MetricEntry(
                name="neural_probe_partial_r2",
                producer="jepa",
                ci=_ci(0.10, 0.08, 0.12),
            )
        ],
    )
    diffs = compare_metrics(local, pub)
    assert diffs[0][1] == "outside_ci"


def test_compare_metrics_missing() -> None:
    local = MetricsOutput(run_id="local", entries=[])
    pub = MetricsOutput(
        run_id="published",
        entries=[
            MetricEntry(
                name="motif_ari",
                producer="jepa",
                ci=_ci(0.4, 0.3, 0.5),
            )
        ],
    )
    diffs = compare_metrics(local, pub)
    assert diffs[0][1] == "missing"
