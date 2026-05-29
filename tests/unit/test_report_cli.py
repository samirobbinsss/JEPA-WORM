"""Unit tests for ``wormjepa.cli.report._load_metrics`` precedence (Story 8.12c follow-up)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wormjepa import WormJEPAError
from wormjepa.cli import report as report_cli
from wormjepa.eval.metrics_schema import BootstrapCI, MetricEntry, MetricsOutput


def _metrics(run_id: str, point: float) -> MetricsOutput:
    return MetricsOutput(
        run_id=run_id,
        entries=[
            MetricEntry(
                name="neural_probe_partial_r2",
                producer="jepa",
                ci=BootstrapCI(
                    point=point,
                    lower=point - 0.02,
                    upper=point + 0.02,
                    n_samples=200,
                    method="bca",
                ),
            )
        ],
    )


def _patch_project_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(report_cli, "project_root", lambda: root)


def test_load_metrics_prefers_metrics_eval_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``metrics_eval.json`` is present, it takes precedence over ``metrics.json``."""
    run_id = "20260512T000000Z__deadbeef__headline"
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics_eval.json").write_text(
        _metrics(run_id, 0.42).to_canonical_json(), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        _metrics(run_id, 0.10).to_canonical_json(), encoding="utf-8"
    )
    _patch_project_root(monkeypatch, tmp_path)

    metrics, results_dir = report_cli._load_metrics(run_id)

    assert results_dir == run_dir
    assert metrics.entries[0].ci.point == pytest.approx(0.42)


def test_load_metrics_falls_back_to_metrics_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When only ``metrics.json`` is present, the loader falls back to it."""
    run_id = "20260512T000000Z__cafebabe__headline"
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        _metrics(run_id, 0.10).to_canonical_json(), encoding="utf-8"
    )
    _patch_project_root(monkeypatch, tmp_path)

    metrics, results_dir = report_cli._load_metrics(run_id)

    assert results_dir == run_dir
    assert metrics.entries[0].ci.point == pytest.approx(0.10)


def test_load_metrics_missing_both_paths_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When neither metrics file exists, the error message mentions both paths."""
    run_id = "20260512T000000Z__feedface__headline"
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    _patch_project_root(monkeypatch, tmp_path)

    with pytest.raises(WormJEPAError) as exc_info:
        report_cli._load_metrics(run_id)

    msg = str(exc_info.value)
    assert "metrics_eval.json" in msg
    assert "metrics.json" in msg
